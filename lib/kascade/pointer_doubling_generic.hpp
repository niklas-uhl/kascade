#pragma once
#include <ranges>

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <fmt/ranges.h>
#include <kamping/collectives/allreduce.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/communicator.hpp>
#include <kamping/measurements/timer.hpp>
#include <kamping/utils/flatten.hpp>
#include <spdlog/spdlog.h>

#include "kascade/bits.hpp"
#include "kascade/configuration.hpp"
#include "kascade/distribution.hpp"
#include "kascade/grid_communicator.hpp"
#include "kascade/list_ranking.hpp"
#include "kascade/request_aggregation_scheme.hpp"
#include "kascade/types.hpp"
#include "successor_utils.hpp"

namespace kascade {
namespace {

struct Request {
  kascade::idx_t write_back_idx;
  kascade::idx_t succ;
  friend std::ostream& operator<<(std::ostream& os, const Request& r) {
    return os << "Request{write_back_idx=" << r.write_back_idx << ", succ=" << r.succ
              << "}";
  }
};

struct Reply {
  kascade::idx_t write_back_idx;
  kascade::idx_t succ;
  kascade::rank_t rank;
  friend std::ostream& operator<<(std::ostream& os, const Reply& r) {
    return os << "Reply{write_back_idx=" << r.write_back_idx << ", succ=" << r.succ
              << ", rank=" << r.rank << "}";
  }
};

auto update_via_writeback(std::ranges::forward_range auto const& recv_replies,
                          std::span<kascade::rank_t> rank_array,
                          std::span<kascade::idx_t> root_array,
                          std::span<kascade::idx_t> local_request_array) -> std::size_t {
  std::size_t unfinished_elems = 0;
  for (const auto& reply : recv_replies) {
    auto write_back_idx = reply.write_back_idx;
    auto succ = reply.succ;
    auto rank = reply.rank;
    // local index
    root_array[write_back_idx] = succ;
    rank_array[write_back_idx] += rank;
    if (!bits::has_root_flag(succ)) {
      local_request_array[unfinished_elems++] = write_back_idx;
    }
  }
  return unfinished_elems;
}

auto update_via_lookup(std::ranges::forward_range auto const& recv_replies,
                       std::span<kascade::rank_t> rank_array,
                       std::span<kascade::idx_t> root_array,
                       std::span<kascade::idx_t> local_request_array,
                       absl::flat_hash_map<idx_t, std::pair<idx_t, rank_t>>& lookup_table)
    -> std::size_t {
  // prepare lookup table
  lookup_table.clear();
  for (const auto [req_succ, succ, rank] : recv_replies) {
    lookup_table.emplace(req_succ, std::make_pair(succ, rank));
  }
  std::size_t unfinished_elems = 0;
  for (auto& local_elem_idx : local_request_array) {
    // local index
    auto cur_succ = root_array[local_elem_idx];
    auto it = lookup_table.find(cur_succ);
    KASSERT(it != lookup_table.end());
    auto const& [succ, rank] = it->second;
    root_array[local_elem_idx] = succ;
    rank_array[local_elem_idx] += rank;
    if (!bits::has_root_flag(succ)) {
      local_request_array[unfinished_elems++] = local_elem_idx;
    }
  }
  return unfinished_elems;
}

auto make_requests_with_writeback_and_target(std::span<idx_t> root_array,
                                             std::span<idx_t> local_request_array,
                                             kascade::Distribution const& dist) {
  return local_request_array | std::views::transform([=](idx_t local_elem_idx) {
           auto succ = root_array[local_elem_idx];
           KASSERT(!bits::has_root_flag(succ),
                   "Do not continue on already finised elements.");
           return std::make_pair<int, Request>(
               dist.get_owner_signed(succ),
               Request{.write_back_idx = local_elem_idx, .succ = succ});
         });
}

auto make_aggregated_requests(std::span<idx_t> root_array,
                              std::span<idx_t> local_request_array,
                              absl::flat_hash_set<idx_t>& request_buffer) {
  request_buffer.clear();
  for (auto& local_elem_idx : local_request_array) {
    auto succ = root_array[local_elem_idx];
    KASSERT(!bits::has_root_flag(succ), "Do not continue on already finised elements.");
    request_buffer.emplace(succ);
  }
}

class DoublingStrategy {
public:
  DoublingStrategy(kamping::Communicator<> const& comm,
                   std::optional<TopologyAwareGridCommunicator> const& grid_comm)
      : comm_{&comm}, grid_comm_{&grid_comm} {}
  virtual auto execute_doubling_step(std::span<kascade::rank_t> rank_array,
                                     std::span<kascade::idx_t> root_array,
                                     std::span<kascade::idx_t> local_request_array,
                                     kascade::Distribution const& dist)
      -> std::span<kascade::idx_t> = 0;
  virtual ~DoublingStrategy() = default;

protected:
  kamping::Communicator<> const* comm_;
  std::optional<TopologyAwareGridCommunicator> const* grid_comm_;
};

class DoublingWithoutAggregation : public DoublingStrategy {
public:
  using DoublingStrategy::DoublingStrategy;
  auto execute_doubling_step(std::span<kascade::rank_t> rank_array,
                             std::span<kascade::idx_t> root_array,
                             std::span<kascade::idx_t> local_request_array,
                             kascade::Distribution const& dist)
      -> std::span<kascade::idx_t> final {
    auto make_reply = [&](const Request& request) {
      auto local_idx = dist.get_local_idx(request.succ, comm_->rank());
      return Reply{.write_back_idx = request.write_back_idx,
                   .succ = root_array[local_idx],
                   .rank = rank_array[local_idx]};
    };
    auto requests =
        make_requests_with_writeback_and_target(root_array, local_request_array, dist);

    request_reply_without_remote_aggregation(requests, make_reply, requests_send_buffer,
                                             requests_recv_buffer, replies_send_buffer,
                                             replies_recv_buffer, *comm_);

    std::size_t const num_remaining_elems = update_via_writeback(
        replies_recv_buffer, rank_array, root_array, local_request_array);
    return local_request_array.first(num_remaining_elems);
  }

private:
  MPIBuffer<Request> requests_send_buffer;
  MPIBuffer<Request> requests_recv_buffer;
  std::vector<Reply> replies_send_buffer;
  std::vector<Reply> replies_recv_buffer;
};

class GridDoublingWithoutAggregation : public DoublingStrategy {
public:
  using DoublingStrategy::DoublingStrategy;

  auto execute_doubling_step(std::span<kascade::rank_t> rank_array,
                             std::span<kascade::idx_t> root_array,
                             std::span<kascade::idx_t> local_request_array,
                             kascade::Distribution const& dist)
      -> std::span<kascade::idx_t> final {
    KASSERT(grid_comm_->has_value());
    auto make_reply = [&](const Request& request) {
      auto local_idx = dist.get_local_idx(request.succ, comm_->rank());
      return Reply{.write_back_idx = request.write_back_idx,
                   .succ = root_array[local_idx],
                   .rank = rank_array[local_idx]};
    };
    auto requests =
        make_requests_with_writeback_and_target(root_array, local_request_array, dist);

    auto replies = request_reply_without_remote_aggregation(requests, make_reply,
                                                            grid_comm_->value());

    std::size_t const num_remaining_elems =
        update_via_writeback(replies, rank_array, root_array, local_request_array);
    return local_request_array.first(num_remaining_elems);
  }
};

class DoublingWithAggregation : public DoublingStrategy {
public:
  using DoublingStrategy::DoublingStrategy;
  DoublingWithAggregation(kamping::Communicator<> const& comm,
                          std::optional<TopologyAwareGridCommunicator> const& grid_comm,
                          bool use_grid_communication,
                          bool use_remote_aggregation)
      : DoublingStrategy(comm, grid_comm),
        use_grid_communication_{use_grid_communication},
        use_remote_aggregation_{use_remote_aggregation} {}

  auto execute_doubling_step(std::span<kascade::rank_t> rank_array,
                             std::span<kascade::idx_t> root_array,
                             std::span<kascade::idx_t> local_request_array,
                             kascade::Distribution const& dist)
      -> std::span<kascade::idx_t> final {
    struct Reply {
      kascade::idx_t req_succ;
      kascade::idx_t succ;
      kascade::rank_t rank;
    };
    auto make_reply = [&](const idx_t& request) {
      auto local_idx = dist.get_local_idx(request, comm_->rank());
      return Reply{.req_succ = request,
                   .succ = root_array[local_idx],
                   .rank = rank_array[local_idx]};
    };
    make_aggregated_requests(root_array, local_request_array, local_aggreation_buf);

    auto get_target_rank = [&](idx_t const& request) { return dist.get_owner(request); };

    auto get_request_key = [](const idx_t& req) { return req; };
    auto get_reply_key = [](const Reply& reply) { return reply.req_succ; };
    std::vector<Reply> replies;

    // do request reply pattern
    if (use_remote_aggregation_) {
      replies = request_with_remote_aggreation<idx_t, Reply>(
          local_aggreation_buf, get_target_rank, get_request_key, get_reply_key,
          make_reply, grid_comm_->value());
    } else {
      // only local aggregation
      if (use_grid_communication_) {
        auto targets =
            local_aggreation_buf | std::views::transform([&](auto const& request) {
              return dist.get_owner(request);
            });
        replies = request_reply_without_remote_aggregation(
            std::views::zip(targets, local_aggreation_buf), make_reply,
            grid_comm_->value());
      } else {
        auto noop = []() {};
        replies = request_without_remote_aggregation<idx_t, Reply>(
            local_aggreation_buf, get_target_rank, noop, noop, make_reply, *comm_);
      }
    }

    std::size_t const num_remaining_elems = update_via_lookup(
        replies, rank_array, root_array, local_request_array, lookup_table);
    return local_request_array.first(num_remaining_elems);
  }

private:
  bool use_grid_communication_;
  bool use_remote_aggregation_;
  absl::flat_hash_set<idx_t> local_aggreation_buf;
  absl::flat_hash_map<idx_t, std::pair<idx_t, rank_t>> lookup_table;
};

auto is_finished(std::size_t unfinished_elems, kamping::Communicator<> const& comm)
    -> bool {
  namespace kmp = kamping::params;
  size_t const global_unfinished_elems =
      comm.allreduce_single(kmp::send_buf(unfinished_elems), kmp::op(std::plus<>{}));
  return global_unfinished_elems == 0U;
}

auto make_grid_comm(kamping::Communicator<> const& comm,
                    AggregationLevel level,
                    bool use_grid_comm) -> std::optional<TopologyAwareGridCommunicator> {
  if (use_grid_comm) {
    return TopologyAwareGridCommunicator{comm};
  }
  switch (level) {
    case kascade::AggregationLevel::invalid:
    case kascade::AggregationLevel::none:
    case kascade::AggregationLevel::local:
      return std::nullopt;
    case kascade::AggregationLevel::all:
      return TopologyAwareGridCommunicator{comm};
  }
  return std::nullopt;
}

auto initialize_active_vertices(std::span<idx_t> succ_array,
                                std::span<rank_t> rank_array,
                                Distribution const& dist,
                                auto const& active_local_indices,
                                kamping::Communicator<> const& comm) {
  std::vector<idx_t> active_vertices;
  active_vertices.reserve(std::ranges::size(active_local_indices));
  for (std::size_t i : active_local_indices) {
    idx_t global_idx = dist.get_global_idx(i, comm.rank());
    if (succ_array[i] == global_idx) {
      KASSERT(rank_array[i] == 0);
      succ_array[i] = bits::set_root_flag(global_idx);
    } else {
      active_vertices.emplace_back(i);
    }
  }
  return active_vertices;
}

auto select_doubling_strategy(
    PointerDoublingConfig const& config,
    kamping::Communicator<> const& comm,
    std::optional<TopologyAwareGridCommunicator> const& grid_comm)
    -> std::unique_ptr<DoublingStrategy> {
  switch (config.aggregation_level) {
    case kascade::AggregationLevel::invalid:
      throw std::runtime_error("invalid aggregation level");
      return nullptr;
    case kascade::AggregationLevel::none: {
      if (!config.use_grid_communication) {
        return std::make_unique<DoublingWithoutAggregation>(comm, grid_comm);
      }
      return std::make_unique<GridDoublingWithoutAggregation>(comm, grid_comm);
    }
    case kascade::AggregationLevel::local:
      return std::make_unique<DoublingWithAggregation>(
          comm, grid_comm, config.use_grid_communication, false);
    case kascade::AggregationLevel::all:
      return std::make_unique<DoublingWithAggregation>(comm, grid_comm, true, true);
  }
  return nullptr;
}

}  // namespace

template <typename R>
void pointer_doubling_generic(PointerDoublingConfig config,
                              std::span<idx_t> succ_array,
                              std::span<rank_t> rank_array,
                              Distribution const& dist,
                              R const& active_local_indices,
                              kamping::Communicator<> const& comm) {
  if (config.use_local_preprocessing) {
    kamping::measurements::timer().synchronize_and_start("local_preprocessing");
    local_pointer_chasing(succ_array, rank_array, comm.rank(), dist);
    kamping::measurements::timer().stop();
  }

  kamping::measurements::timer().synchronize_and_start("create_grid_comm");
  kamping::measurements::timer().stop_and_append();

  kamping::measurements::timer().synchronize_and_start("pointer_doubling_alltoall");

  auto active_vertices_storage = initialize_active_vertices(succ_array, rank_array, dist,
                                                            active_local_indices, comm);
  std::span<idx_t> active_vertices = active_vertices_storage;
  std::optional<TopologyAwareGridCommunicator> grid_comm =
      make_grid_comm(comm, config.aggregation_level, config.use_grid_communication);
  auto doubling_strategy = select_doubling_strategy(config, comm, grid_comm);

  while (!is_finished(active_vertices.size(), comm)) {
    kamping::measurements::timer().synchronize_and_start("pointer_doubling_step");
    active_vertices = doubling_strategy->execute_doubling_step(rank_array, succ_array,
                                                               active_vertices, dist);
    kamping::measurements::timer().stop_and_append();
  }

  // clear result
  for (std::size_t local_idx : active_local_indices) {
    succ_array[local_idx] = bits::clear_root_flag(succ_array[local_idx]);
  }
  kamping::measurements::timer().stop();
}
}  // namespace kascade
