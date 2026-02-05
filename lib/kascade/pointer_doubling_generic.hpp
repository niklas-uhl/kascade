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

namespace kascade {
namespace {

template <typename MakeRequestsFn, typename SendRequestsFn, typename UpdateFn>
auto do_doubling_step_skeleton(std::span<kascade::idx_t> rank_array,
                               std::span<kascade::idx_t> root_array,
                               std::span<kascade::idx_t> local_request_array,
                               MakeRequestsFn const& make_requests,
                               SendRequestsFn const& send_requests,
                               UpdateFn const& update) {
  auto request_view = make_requests(root_array, local_request_array);
  auto recv_replies = send_requests(request_view);
  std::size_t const num_remaining_elems =
      update(recv_replies, rank_array, root_array, local_request_array);
  return local_request_array.first(num_remaining_elems);
}

struct Request {
  kascade::idx_t write_back_idx;
  kascade::idx_t succ;
};

struct Reply {
  kascade::idx_t write_back_idx;
  kascade::idx_t succ;
  kascade::idx_t rank;
};

struct update_via_writeback {
  auto operator()(std::ranges::forward_range auto const& recv_replies,
                  std::span<kascade::idx_t> rank_array,
                  std::span<kascade::idx_t> root_array,
                  std::span<kascade::idx_t> local_request_array) const -> std::size_t {
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
};

struct update_via_lookup {
  auto operator()(std::ranges::forward_range auto const& recv_replies,
                  std::span<kascade::idx_t> rank_array,
                  std::span<kascade::idx_t> root_array,
                  std::span<kascade::idx_t> local_request_array) const -> std::size_t {
    // prepare lookup table
    absl::flat_hash_map<idx_t, std::pair<idx_t, idx_t>> lookup_table;
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
};

auto make_requests(std::span<idx_t> root_array, std::span<idx_t> local_request_array) {
  return local_request_array | std::views::transform([=](idx_t local_elem_idx) {
           auto succ = root_array[local_elem_idx];
           KASSERT(!bits::has_root_flag(succ), "Do not continue on already finised elements.");
           return succ;
         });
}

auto make_requests_with_writeback(std::span<idx_t> root_array,
                                  std::span<idx_t> local_request_array) {
  return local_request_array | std::views::transform([=](idx_t local_elem_idx) {
           auto succ = root_array[local_elem_idx];
           KASSERT(!bits::has_root_flag(succ), "Do not continue on already finised elements.");
           return Request{.write_back_idx = local_elem_idx, .succ = succ};
         });
}

auto make_aggregated_requests(std::span<idx_t> root_array,
                              std::span<idx_t> local_request_array)
    -> absl::flat_hash_set<idx_t> {
  absl::flat_hash_set<idx_t> request_buffer;
  for (auto& local_elem_idx : local_request_array) {
    auto succ = root_array[local_elem_idx];
    KASSERT(!bits::has_root_flag(succ), "Do not continue on already finised elements.");
    request_buffer.emplace(succ);
  }
  return request_buffer;
}

auto do_doubling_step_without_aggregation(std::span<kascade::idx_t> rank_array,
                                          std::span<kascade::idx_t> root_array,
                                          std::span<kascade::idx_t> local_request_array,
                                          kascade::Distribution const& dist,
                                          kamping::Communicator<> const& comm)
    -> std::span<kascade::idx_t> {
  auto make_reply = [&](const Request& request) {
    auto local_idx = dist.get_local_idx(request.succ, comm.rank());
    return Reply{.write_back_idx = request.write_back_idx,
                 .succ = root_array[local_idx],
                 .rank = rank_array[local_idx]};
  };
  auto get_target_rank = [&](Request const& request) {
    return dist.get_owner(request.succ);
  };
  auto noop = []() {};

  auto send_requests = [&](auto const& requests) {
    return request_without_remote_aggregation<idx_t, Reply>(requests, get_target_rank,
                                                            noop, noop, make_reply, comm);
  };

  return do_doubling_step_skeleton(rank_array, root_array, local_request_array,
                                   make_requests_with_writeback, send_requests,
                                   update_via_writeback{});
}

auto do_doubling_step_with_remote_aggregation_only(
    std::span<kascade::idx_t> rank_array,
    std::span<kascade::idx_t> root_array,
    std::span<kascade::idx_t> local_request_array,
    kascade::Distribution const& dist,
    TopologyAwareGridCommunicator const& grid_comm) -> std::span<kascade::idx_t> {
  struct Reply {
    kascade::idx_t req_succ;
    kascade::idx_t succ;
    kascade::idx_t rank;
  };

  auto make_reply = [&](const idx_t& request) {
    auto local_idx = dist.get_local_idx(request, grid_comm.global_comm().rank());
    return Reply{.req_succ = request,
                 .succ = root_array[local_idx],
                 .rank = rank_array[local_idx]};
  };
  auto get_target_rank = [&](const idx_t& request) { return dist.get_owner(request); };
  auto get_request_key = [](const idx_t& req) { return req; };
  auto get_reply_key = [](const Reply& reply) { return reply.req_succ; };

  auto send_requests = [&](auto const& requests) {
    return request_with_remote_aggreation<idx_t, Reply>(
        requests, get_target_rank, get_request_key, get_reply_key, make_reply, grid_comm);
  };

  return do_doubling_step_skeleton(rank_array, root_array, local_request_array,
                                   make_requests, send_requests, update_via_lookup{});
}

auto do_doubling_step_with_local_aggregation(
    bool use_remote_aggregation,
    std::span<kascade::idx_t> rank_array,
    std::span<kascade::idx_t> root_array,
    std::span<kascade::idx_t> local_request_array,
    kascade::Distribution const& dist,
    kamping::Communicator<> const& comm,
    TopologyAwareGridCommunicator const* grid_comm) -> std::span<kascade::idx_t> {
  struct Reply {
    kascade::idx_t req_succ;
    kascade::idx_t succ;
    kascade::idx_t rank;
  };
  auto make_reply = [&](const idx_t& request) {
    auto local_idx = dist.get_local_idx(request, comm.rank());
    return Reply{.req_succ = request,
                 .succ = root_array[local_idx],
                 .rank = rank_array[local_idx]};
  };
  auto get_target_rank = [&](idx_t const& request) { return dist.get_owner(request); };

  auto get_request_key = [](const idx_t& req) { return req; };
  auto get_reply_key = [](const Reply& reply) { return reply.req_succ; };

  if (use_remote_aggregation) {
    auto send_requests_with_aggreation = [&](auto const& requests) {
      KASSERT(grid_comm != nullptr);
      return request_with_remote_aggreation<idx_t, Reply>(requests, get_target_rank,
                                                          get_request_key, get_reply_key,
                                                          make_reply, *grid_comm);
    };
    return do_doubling_step_skeleton(rank_array, root_array, local_request_array,
                                     make_aggregated_requests,
                                     send_requests_with_aggreation, update_via_lookup{});
  }

  auto send_requests = [&](auto const& requests) {
    auto noop = []() {};
    return request_without_remote_aggregation<idx_t, Reply>(requests, get_target_rank,
                                                            noop, noop, make_reply, comm);
  };
  return do_doubling_step_skeleton(rank_array, root_array, local_request_array,
                                   make_requests, send_requests, update_via_lookup{});
}

auto do_doubling_step(kascade::PointerDoublingConfig const& config,
                      std::span<kascade::idx_t> rank_array,
                      std::span<kascade::idx_t> root_array,
                      std::span<kascade::idx_t> local_request_array,
                      kascade::Distribution const& dist,
                      kamping::Communicator<> const& comm,
                      TopologyAwareGridCommunicator const* grid_comm)
    -> std::span<kascade::idx_t> {
  switch (config.aggregation_level) {
    case kascade::AggregationLevel::none:
      return do_doubling_step_without_aggregation(rank_array, root_array,
                                                  local_request_array, dist, comm);
    case kascade::AggregationLevel::local:
      return do_doubling_step_with_local_aggregation(
          false, rank_array, root_array, local_request_array, dist, comm, grid_comm);
    case kascade::AggregationLevel::remote: {
      KASSERT(grid_comm != nullptr);
      auto res = do_doubling_step_with_remote_aggregation_only(
          rank_array, root_array, local_request_array, dist, *grid_comm);
      return res;
    }
    case kascade::AggregationLevel::all:
      KASSERT(grid_comm != nullptr);
      return do_doubling_step_with_local_aggregation(
          true, rank_array, root_array, local_request_array, dist, comm, grid_comm);
    default: {
      throw std::runtime_error("not supported");
    }
  }
  return do_doubling_step_without_aggregation(rank_array, root_array, local_request_array,
                                              dist, comm);
}

auto is_finished(std::size_t unfinished_elems, kamping::Communicator<> const& comm)
    -> bool {
  namespace kmp = kamping::params;
  size_t const global_unfinished_elems =
      comm.allreduce_single(kmp::send_buf(unfinished_elems), kmp::op(std::plus<>{}));
  return global_unfinished_elems == 0U;
}

auto make_grid_comm(kamping::Communicator<> const& comm, AggregationLevel level)
    -> std::unique_ptr<TopologyAwareGridCommunicator> {
  switch (level) {
    case kascade::AggregationLevel::invalid:
    case kascade::AggregationLevel::none:
    case kascade::AggregationLevel::local:
      return nullptr;
    case kascade::AggregationLevel::remote:
    case kascade::AggregationLevel::all:
      return std::make_unique<TopologyAwareGridCommunicator>(comm);
  }
  return nullptr;
}

auto initialize_active_vertices(std::span<idx_t> succ_array,
                                std::span<idx_t> rank_array,
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

}  // namespace

template <typename R>
void pointer_doubling_generic(PointerDoublingConfig config,
                              std::span<idx_t> succ_array,
                              std::span<idx_t> rank_array,
                              Distribution const& dist,
                              R const& active_local_indices,
                              kamping::Communicator<> const& comm) {
  if (config.use_local_preprocessing) {
    kamping::measurements::timer().synchronize_and_start("local_preprocessing");
    local_pointer_chasing(succ_array, rank_array, comm.rank(), dist);
    kamping::measurements::timer().stop();
  }

  kamping::measurements::timer().synchronize_and_start("create_grid_comm");
  std::unique_ptr<TopologyAwareGridCommunicator> grid_comm_ptr =
      make_grid_comm(comm, config.aggregation_level);
  kamping::measurements::timer().stop_and_append();

  kamping::measurements::timer().synchronize_and_start("pointer_doubling_alltoall");

  std::size_t rounds = 0;
  auto active_vertices_storage = initialize_active_vertices(succ_array, rank_array, dist,
                                                            active_local_indices, comm);
  std::span<idx_t> active_vertices = active_vertices_storage;

  while (!is_finished(active_vertices.size(), comm)) {
    kamping::measurements::timer().synchronize_and_start("pointer_doubling_step_" +
                                                         std::to_string(rounds++));
    active_vertices = do_doubling_step(config, rank_array, succ_array, active_vertices,
                                       dist, comm, grid_comm_ptr.get());
    kamping::measurements::timer().stop_and_append();
  }

  // clear result
  for (std::size_t local_idx : active_local_indices) {
    succ_array[local_idx] = bits::clear_root_flag(succ_array[local_idx]);
  }
  kamping::measurements::timer().stop();
}
}  // namespace kascade
