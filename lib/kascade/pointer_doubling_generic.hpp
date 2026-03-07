#pragma once
#include <ranges>

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <fmt/ranges.h>
#include <kamping/collectives/allreduce.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/communicator.hpp>
#include <kamping/measurements/counter.hpp>
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

struct Reply {
  kascade::idx_t succ;
  kascade::rank_t rank;
  friend std::ostream& operator<<(std::ostream& os, const Reply& r) {
    return os << "Reply{ succ=" << r.succ << ", rank=" << r.rank << "}";
  }
};

struct ExtendedReply {
  kascade::idx_t succ;
  kascade::rank_t rank;
  int succ_owner;
};

auto update_via_read_offsets(std::ranges::forward_range auto const& recv_replies,
                             std::span<kascade::rank_t> rank_array,
                             std::span<kascade::idx_t> root_array,
                             std::span<kascade::idx_t> local_request_array,
                             std::span<int> read_offsets,
                             Distribution const& dist) -> std::size_t {
  std::size_t unfinished_elems = 0;
  for (idx_t local_idx : local_request_array) {
    auto target = dist.get_owner(root_array[local_idx]);
    auto pos = read_offsets[target]++;
    root_array[local_idx] = recv_replies[pos].succ;
    rank_array[local_idx] += recv_replies[pos].rank;
    if (!bits::has_root_flag(root_array[local_idx])) {
      local_request_array[unfinished_elems++] = local_idx;
    }
  }
  return unfinished_elems;
}

auto update_via_read_offsets(std::ranges::forward_range auto const& recv_replies,
                             std::span<kascade::rank_t> rank_array,
                             std::span<kascade::idx_t> root_array,
                             std::span<int> root_owner_array,
                             std::span<kascade::idx_t> local_request_array,
                             std::span<int> read_offsets,
                             Distribution const& /*dist*/) -> std::size_t {
  std::size_t unfinished_elems = 0;
  for (idx_t local_idx : local_request_array) {
    auto target = root_owner_array[local_idx];  // dist.get_owner(root_array[local_idx]);
    auto pos = read_offsets[target]++;
    root_array[local_idx] = recv_replies[pos].succ;
    rank_array[local_idx] += recv_replies[pos].rank;
    root_owner_array[local_idx] = recv_replies[pos].succ_owner;
    if (!bits::has_root_flag(root_array[local_idx])) {
      local_request_array[unfinished_elems++] = local_idx;
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

auto make_requests_with_target(std::span<idx_t> root_array,
                               std::span<idx_t> local_request_array,
                               kascade::Distribution const& dist) {
  return local_request_array | std::views::transform([=](idx_t local_elem_idx) {
           idx_t succ = root_array[local_elem_idx];
           KASSERT(!bits::has_root_flag(succ),
                   "Do not continue on already finised elements.");
           return std::make_pair(dist.get_owner_signed(succ), succ);
         }) |
         std::ranges::to<std::vector>();
}

auto make_requests_with_target(std::span<idx_t> root_array,
                               std::span<idx_t> local_request_array,
                               std::span<int> root_owner_array) {
  return local_request_array | std::views::transform([=](idx_t local_elem_idx) {
           idx_t succ = root_array[local_elem_idx];
           KASSERT(!bits::has_root_flag(succ),
                   "Do not continue on already finised elements.");
           return std::make_pair(root_owner_array[local_elem_idx], succ);
         }) |
         std::ranges::to<std::vector>();
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
  DoublingStrategy(PointerDoublingConfig config,
                   kamping::Communicator<> const& comm,
                   std::optional<TopologyAwareGridCommunicator> const& grid_comm)
      : config_{config}, comm_{&comm}, grid_comm_{&grid_comm} {}
  virtual auto execute_doubling_step(std::span<kascade::rank_t> rank_array,
                                     std::span<kascade::idx_t> root_array,
                                     std::span<kascade::idx_t> local_request_array,
                                     kascade::Distribution const& dist)
      -> std::span<kascade::idx_t> = 0;
  virtual auto execute_doubling_step(std::span<kascade::rank_t> rank_array,
                                     std::span<kascade::idx_t> root_array,
                                     std::span<kascade::idx_t> local_request_array,
                                     std::span<int> root_owner_array,
                                     kascade::Distribution const& dist)
      -> std::span<kascade::idx_t> = 0;

  virtual ~DoublingStrategy() = default;

protected:
  PointerDoublingConfig config_;
  kamping::Communicator<> const* comm_;
  std::optional<TopologyAwareGridCommunicator> const* grid_comm_;
};

class DoublingWithoutAggregation : public DoublingStrategy {
public:
  using DoublingStrategy::DoublingStrategy;
  DoublingWithoutAggregation(
      PointerDoublingConfig config,
      kamping::Communicator<> const& comm,
      std::optional<TopologyAwareGridCommunicator> const& grid_comm)
      : DoublingStrategy(config, comm, grid_comm) {
    if (config.cache_succ_owners) {
      replies_send_buffer.emplace<std::vector<ExtendedReply>>();
      replies_recv_buffer.emplace<std::vector<ExtendedReply>>();
    } else {
      replies_send_buffer.emplace<std::vector<Reply>>();
      replies_recv_buffer.emplace<std::vector<Reply>>();
    }
  }
  auto execute_doubling_step(std::span<kascade::rank_t> rank_array,
                             std::span<kascade::idx_t> root_array,
                             std::span<kascade::idx_t> local_request_array,
                             kascade::Distribution const& dist)
      -> std::span<kascade::idx_t> final {
    auto make_reply = [&](idx_t request) {
      auto local_idx = dist.get_local_idx(request, comm_->rank());
      return Reply{.succ = root_array[local_idx], .rank = rank_array[local_idx]};
    };
    kamping::measurements::timer().start("make_requests");
    auto requests = make_requests_with_target(root_array, local_request_array, dist);
    kamping::measurements::timer().stop_and_append();
    kamping::measurements::timer().synchronize_and_start("request_reply");
    request_reply_without_remote_aggregation(
        requests, make_reply, requests_send_buffer, requests_recv_buffer,
        std::get<0>(replies_send_buffer), std::get<0>(replies_recv_buffer), *comm_);
    kamping::measurements::timer().stop_and_append();

    kamping::measurements::timer().start("update");
    // std::cout << "i'm here" << std::endl;
    std::size_t const num_remaining_elems =
        update_via_read_offsets(std::get<0>(replies_recv_buffer), rank_array, root_array,
                                local_request_array, requests_send_buffer.displs, dist);
    kamping::measurements::timer().stop_and_append();
    return local_request_array.first(num_remaining_elems);
  }

  auto execute_doubling_step(std::span<kascade::rank_t> rank_array,
                             std::span<kascade::idx_t> root_array,
                             std::span<kascade::idx_t> local_request_array,
                             std::span<int> root_owner_array,
                             kascade::Distribution const& dist)
      -> std::span<kascade::idx_t> final {
    auto make_reply = [&](idx_t request) {
      auto local_idx = dist.get_local_idx(request, comm_->rank());
      return ExtendedReply{.succ = root_array[local_idx],
                           .rank = rank_array[local_idx],
                           .succ_owner = root_owner_array[local_idx]};
    };
    kamping::measurements::timer().start("make_requests");
    auto requests =
        make_requests_with_target(root_array, local_request_array, root_owner_array);
    kamping::measurements::timer().stop_and_append();
    kamping::measurements::timer().synchronize_and_start("request_reply");
    request_reply_without_remote_aggregation(
        requests, make_reply, requests_send_buffer, requests_recv_buffer,
        std::get<1>(replies_send_buffer), std::get<1>(replies_recv_buffer), *comm_);
    kamping::measurements::timer().stop_and_append();

    kamping::measurements::timer().start("update");
    // std::cout << "i'm here with root owner caching" << std::endl;
    std::size_t const num_remaining_elems = update_via_read_offsets(
        std::get<1>(replies_recv_buffer), rank_array, root_array, root_owner_array,
        local_request_array, requests_send_buffer.displs, dist);
    kamping::measurements::timer().stop_and_append();
    return local_request_array.first(num_remaining_elems);
  }

private:
  MPIBuffer<idx_t> requests_send_buffer;
  MPIBuffer<idx_t> requests_recv_buffer;
  std::variant<std::vector<Reply>, std::vector<ExtendedReply>> replies_send_buffer;
  std::variant<std::vector<Reply>, std::vector<ExtendedReply>> replies_recv_buffer;
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
    auto make_reply = [&](idx_t request) {
      auto local_idx = dist.get_local_idx(request, comm_->rank());
      return Reply{.succ = root_array[local_idx], .rank = rank_array[local_idx]};
    };

    kamping::measurements::timer().start("make_requests");
    auto requests = make_requests_with_target(root_array, local_request_array, dist);
    kamping::measurements::timer().stop_and_append();

    kamping::measurements::timer().synchronize_and_start("request_reply");
    auto [recv_replies, write_offsets] = [&]() {
      if (config_.use_local_first_request_scheme) {
        return request_reply_local_first_without_remote_aggregation(
            requests, make_reply, grid_comm_->value(),
            request_reply_mode::reorder_output);
      }
      return request_reply_without_remote_aggregation(
          requests, make_reply, grid_comm_->value(), request_reply_mode::reorder_output);
    }();
    kamping::measurements::timer().stop_and_append();

    kamping::measurements::timer().start("update");
    // std::cout << "i'm grid" << std::endl;
    std::size_t const num_remaining_elems = update_via_read_offsets(
        recv_replies, rank_array, root_array, local_request_array, write_offsets, dist);
    kamping::measurements::timer().stop_and_append();
    return local_request_array.first(num_remaining_elems);
  }

  auto execute_doubling_step(std::span<kascade::rank_t> rank_array,
                             std::span<kascade::idx_t> root_array,
                             std::span<kascade::idx_t> local_request_array,
                             std::span<int> root_owner_array,
                             kascade::Distribution const& dist)
      -> std::span<kascade::idx_t> final {
    KASSERT(grid_comm_->has_value());
    auto make_reply = [&](idx_t request) {
      auto local_idx = dist.get_local_idx(request, comm_->rank());
      return ExtendedReply{.succ = root_array[local_idx],
                           .rank = rank_array[local_idx],
                           .succ_owner = root_owner_array[local_idx]};
    };
    kamping::measurements::timer().start("make_requests");
    auto requests =
        make_requests_with_target(root_array, local_request_array, root_owner_array);
    kamping::measurements::timer().stop_and_append();

    kamping::measurements::timer().synchronize_and_start("request_reply");
    auto [recv_replies, write_offsets] = [&]() {
      if (config_.use_local_first_request_scheme) {
        return request_reply_local_first_without_remote_aggregation(
            requests, make_reply, grid_comm_->value(),
            request_reply_mode::reorder_output);
      }
      return request_reply_without_remote_aggregation(
          requests, make_reply, grid_comm_->value(), request_reply_mode::reorder_output);
    }();
    kamping::measurements::timer().stop_and_append();

    // std::cout << "i'm grid with owner caching" << std::endl;
    kamping::measurements::timer().start("update");
    std::size_t const num_remaining_elems =
        update_via_read_offsets(recv_replies, rank_array, root_array, root_owner_array,
                                local_request_array, write_offsets, dist);
    kamping::measurements::timer().stop_and_append();
    return local_request_array.first(num_remaining_elems);
  }
};

class DoublingWithAggregation : public DoublingStrategy {
public:
  using DoublingStrategy::DoublingStrategy;
  DoublingWithAggregation(PointerDoublingConfig config,
                          kamping::Communicator<> const& comm,
                          std::optional<TopologyAwareGridCommunicator> const& grid_comm,
                          bool use_grid_communication,
                          bool use_remote_aggregation)
      : DoublingStrategy(config, comm, grid_comm),
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
  auto execute_doubling_step(std::span<kascade::rank_t> /*rank_array*/,
                             std::span<kascade::idx_t> /*root_array*/,
                             std::span<kascade::idx_t> /*local_request_array*/,
                             std::span<int> /*root_owner_array*/,
                             kascade::Distribution const& /*dist*/)
      -> std::span<kascade::idx_t> final {
    throw std::runtime_error{"not implemented"};
    return {};
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

auto make_grid_comm(kamping::Communicator<> const& comm, GridCommunicatorMode mode)
    -> std::optional<TopologyAwareGridCommunicator> {
  switch (mode) {
    case GridCommunicatorMode::balanced: {
      auto [first_dim, second_dim] = compute_grid_dimensions(comm.size());
      if (first_dim < second_dim) {
        std::swap(first_dim, second_dim);
      }
      return TopologyAwareGridCommunicator{comm, first_dim};
    }
    case GridCommunicatorMode::topology_aware:
      return TopologyAwareGridCommunicator{comm};
    case GridCommunicatorMode::invalid: {
      throw std::runtime_error("invalid parameter for grid communicator mode");
    }
  };
  return std::nullopt;
}

auto make_grid_comm(kamping::Communicator<> const& comm,
                    bool use_grid_communication,
                    AggregationLevel level,
                    GridCommunicatorMode grid_mode)
    -> std::optional<TopologyAwareGridCommunicator> {
  if (use_grid_communication) {
    return make_grid_comm(comm, grid_mode);
  }
  switch (level) {
    case kascade::AggregationLevel::invalid:
    case kascade::AggregationLevel::none:
    case kascade::AggregationLevel::local:
      return std::nullopt;
    case kascade::AggregationLevel::all:
      KASSERT(grid_mode != GridCommunicatorMode::invalid);
      return make_grid_comm(comm, grid_mode);
  }
  return std::nullopt;
}

auto initialize_active_vertices(std::span<idx_t> succ_array,
                                std::span<rank_t> rank_array,
                                Distribution const& dist,
                                auto const& is_active,
                                kamping::Communicator<> const& comm) {
  std::vector<idx_t> active_vertices;
  active_vertices.reserve(is_active.size());
  for (std::size_t i = 0; i < succ_array.size(); ++i) {
    if (!is_active[i]) {
      continue;
    }
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
      if (config.use_grid_communication) {
        return std::make_unique<GridDoublingWithoutAggregation>(config, comm, grid_comm);
      }
      return std::make_unique<DoublingWithoutAggregation>(config, comm, grid_comm);
    }
    case kascade::AggregationLevel::local:
      return std::make_unique<DoublingWithAggregation>(
          config, comm, grid_comm, config.use_grid_communication, false);
    case kascade::AggregationLevel::all:
      return std::make_unique<DoublingWithAggregation>(config, comm, grid_comm, true,
                                                       true);
  }
  return nullptr;
}

struct LocalChainInfo {
  idx_t local_chain_start;
  idx_t next;
  rank_t dist_to_next;
};

auto local_contraction(std::span<idx_t> succ_array,
                       std::span<rank_t> rank_array,
                       std::vector<bool>& is_active,
                       Distribution const& dist,
                       kamping::Communicator<> const& comm) {
  std::vector<bool> has_local_pred(dist.get_local_size(comm.rank()), false);
  for (auto const& succ : succ_array) {
    if (dist.is_local(succ, comm.rank())) {
      auto succ_local = dist.get_local_idx(succ, comm.rank());
      has_local_pred[succ_local] = true;
    }
  }

  auto local_chain_starts =
      dist.local_indices(comm.rank()) |
      std::views::filter([&](idx_t idx_local) { return !has_local_pred[idx_local]; });
  std::vector<LocalChainInfo> local_chain_info;
  std::size_t num_masked = 0;
  for (idx_t local_chain_start : local_chain_starts) {
    auto current_node = succ_array[local_chain_start];
    auto is_root = [&](idx_t idx) {
      idx_t global_idx = dist.get_global_idx(idx, comm.rank());
      return succ_array[idx] == global_idx;
    };
    auto is_end_of_local_chain = [&](idx_t idx) {
      if (!dist.is_local(idx, comm.rank())) {
        return true;
      }
      auto idx_local = dist.get_local_idx(idx, comm.rank());
      return is_root(idx_local);
    };
    rank_t chain_length = rank_array[local_chain_start];
    while (!is_end_of_local_chain(current_node)) {
      auto current_node_local = dist.get_local_idx(current_node, comm.rank());
      chain_length += rank_array[current_node_local];
      is_active[current_node_local] = false;
      num_masked++;
      current_node = succ_array[current_node_local];
    }
    if (current_node == succ_array[local_chain_start]) {
      continue;
    }
    local_chain_info.emplace_back(local_chain_start, succ_array[local_chain_start],
                                  rank_array[local_chain_start]);
    succ_array[local_chain_start] = current_node;
    rank_array[local_chain_start] = chain_length;
  }
  return std::pair{std::move(local_chain_info), num_masked};
}

auto local_uncontraction(std::vector<LocalChainInfo> const& local_chain_info,
                         std::span<idx_t> succ_array,
                         std::span<rank_t> rank_array,
                         Distribution const& dist,
                         kamping::Communicator<> const& comm) {
  auto is_end_of_local_chain = [&](idx_t idx) {
    if (!dist.is_local(idx, comm.rank())) {
      return true;
    }
    auto idx_local = dist.get_local_idx(idx, comm.rank());
    return bits::has_root_flag(succ_array[idx_local]);
  };
  for (auto const& [local_chain_start, next, dist_to_next] : local_chain_info) {
    auto current_node = next;
    auto current_dist = rank_array[local_chain_start] - dist_to_next;
    while (!is_end_of_local_chain(current_node)) {
      auto current_node_local = dist.get_local_idx(current_node, comm.rank());
      auto next_node = succ_array[current_node_local];
      auto next_dist = rank_array[current_node_local];
      succ_array[current_node_local] = succ_array[local_chain_start];
      rank_array[current_node_local] = current_dist;
      current_node = next_node;
      current_dist -= next_dist;
    }
  }
}

}  // namespace

template <typename R>
void pointer_doubling_generic(PointerDoublingConfig config,
                              std::span<idx_t> succ_array,
                              std::span<rank_t> rank_array,
                              Distribution const& dist,
                              R const& active_local_indices,
                              kamping::Communicator<> const& comm) {
  if (dist.get_global_size() <
      static_cast<std::size_t>(config.fallback_allgather_size_ratio *
                               static_cast<double>(comm.size()))) {
    SPDLOG_LOGGER_DEBUG(spdlog::get("root"),
                        "global size smaller than {}*p, falling back to allgather",
                        config.fallback_allgather_size_ratio);
    kamping::measurements::timer().synchronize_and_start("rank_on_root_fallback");
    rank_on_root(succ_array, rank_array, dist, comm);
    kamping::measurements::timer().stop();
    return;
  }

  std::vector<bool> is_active(succ_array.size(), true);
  std::vector<LocalChainInfo> local_chain_info;
  std::size_t num_masked = 0;
  if (config.use_local_preprocessing) {
    kamping::measurements::timer().synchronize_and_start("local_preprocessing");
    std::tie(local_chain_info, num_masked) =
        local_contraction(succ_array, rank_array, is_active, dist, comm);
    kamping::measurements::counter().append(
        "pointer-doubling-num-masked-vertices", static_cast<std::int64_t>(num_masked),
        {kamping::measurements::GlobalAggregationMode::min,
         kamping::measurements::GlobalAggregationMode::max,
         kamping::measurements::GlobalAggregationMode::sum});
    kamping::measurements::timer().stop();
  }
  kamping::measurements::timer().synchronize_and_start("pointer_doubling_alltoall");

  auto active_vertices_storage =
      initialize_active_vertices(succ_array, rank_array, dist, is_active, comm);
  std::optional<std::vector<int>> succ_owner_array;
  if (config.cache_succ_owners) {
    succ_owner_array.emplace();
    succ_owner_array->reserve(succ_array.size());
    for (const auto succ : succ_array) {
      succ_owner_array->push_back(dist.get_owner_signed(bits::clear_root_flag(succ)));
    }
  }
  std::span<idx_t> active_vertices = active_vertices_storage;
  kamping::measurements::timer().synchronize_and_start("create_grid_comm");
  std::optional<TopologyAwareGridCommunicator> grid_comm =
      make_grid_comm(comm, config.use_grid_communication, config.aggregation_level,
                     config.grid_communicator_mode);
  std::size_t intra_comm_size = 1;
  if (grid_comm.has_value()) {
    intra_comm_size = grid_comm->ranks_per_compute_node();
  }
  kamping::measurements::counter().append(
      "intra-comm-size", static_cast<std::int64_t>(intra_comm_size),
      {kamping::measurements::GlobalAggregationMode::min,
       kamping::measurements::GlobalAggregationMode::max});
  kamping::measurements::timer().stop_and_append();
  auto doubling_strategy = select_doubling_strategy(config, comm, grid_comm);

  while (!is_finished(active_vertices.size(), comm)) {
    kamping::measurements::timer().synchronize_and_start("pointer_doubling_step");
    if (config.cache_succ_owners) {
      KASSERT(succ_owner_array.has_value());
      active_vertices = doubling_strategy->execute_doubling_step(
          rank_array, succ_array, active_vertices, succ_owner_array.value(), dist);
    } else {
      active_vertices = doubling_strategy->execute_doubling_step(rank_array, succ_array,
                                                                 active_vertices, dist);
    }
    kamping::measurements::timer().stop_and_append();
  }

  kamping::measurements::timer().start("local_uncontraction");
  local_uncontraction(local_chain_info, succ_array, rank_array, dist, comm);
  kamping::measurements::timer().stop();

  //  clear result
  for (std::size_t local_idx : active_local_indices) {
    succ_array[local_idx] = bits::clear_root_flag(succ_array[local_idx]);
  }
  kamping::measurements::timer().stop();
}
}  // namespace kascade
