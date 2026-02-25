#pragma once

#include <ranges>
#include <utility>

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <fmt/ranges.h>
#include <kamping/communicator.hpp>
#include <kamping/measurements/counter.hpp>
#include <kamping/measurements/timer.hpp>
#include <kassert/kassert.hpp>
#include <spdlog/spdlog.h>

#include "kascade/bits.hpp"
#include "kascade/configuration.hpp"
#include "kascade/distribution.hpp"
#include "kascade/packed_index.hpp"
#include "kascade/sparse_ruling_set_detail/ruler_chasing_engine.hpp"
#include "kascade/sparse_ruling_set_detail/types.hpp"
#include "kascade/types.hpp"
#include "request_aggregation_scheme.hpp"

namespace kascade::sparse_ruling_set_detail {
namespace propagation_mode {
struct pull_tag {};
struct push_tag {};
constexpr pull_tag pull{};
constexpr push_tag push{};

struct local_aggregation_tag {};
constexpr local_aggregation_tag local_aggregation{};
}  // namespace propagation_mode

inline auto ruler_propagation(
    SparseRulingSetConfig const& config,
    std::span<idx_t> succ_array,
    std::span<rank_t> rank_array,
    std::vector<NodeType> const& node_type,
    Distribution const& dist,
    kamping::Communicator<> const& comm,
    std::optional<TopologyAwareGridCommunicator> const& grid_comm,
    propagation_mode::local_aggregation_tag /*tag*/,
    propagation_mode::pull_tag /* tag */) {
  auto needs_to_request_ruler = [&](idx_t local_idx) {
    // if the msb is set, this node was reached from a leaf, so root and rank are already
    // correct
    // rulers and leafs also have the correct result already from the base algorithm
    return !bits::has_root_flag(succ_array[local_idx]) &&
           node_type[local_idx] != NodeType::ruler &&
           node_type[local_idx] != NodeType::leaf;
  };
  kamping::measurements::timer().start("collect_requests");
  std::size_t deduped_requests = 0;
  absl::flat_hash_set<packed_index> requested_rulers;
  for (auto local_idx : dist.local_indices(comm.rank())) {
    if (!needs_to_request_ruler(local_idx)) {
      continue;
    }
    auto ruler = succ_array[local_idx];
    if (config.use_locality_aware_in_ruler_propagation &&
        dist.is_local(ruler, comm.rank())) {
      continue;
    }
    packed_index packed_ruler{ruler, static_cast<std::uint32_t>(dist.get_owner(ruler))};
    bool inserted = requested_rulers.insert(packed_ruler).second;
    if (inserted) {
      SPDLOG_TRACE("Requesting info for ruler {} from PE {}", ruler,
                   dist.get_owner(ruler));
    } else {
      deduped_requests++;
    }
  }
  kamping::measurements::timer().stop();
  SPDLOG_DEBUG("[ruler_propagation] removed {} duplicate requests", deduped_requests);

  kamping::measurements::timer().start("pack_requests");
  auto requests =
      requested_rulers | std::views::transform([&](auto const& requested_ruler) {
        return std::make_pair(requested_ruler.get_owner(), requested_ruler.get_index());
      }) |
      std::ranges::to<std::vector>();
  struct ruler_reply {
    idx_t ruler;
    idx_t root;
    rank_t dist_to_root;
  };

  kamping::measurements::timer().stop();
  kamping::measurements::counter().append(
      "ruler_propagation_requests", static_cast<std::int64_t>(requests.size()),
      {
          kamping::measurements::GlobalAggregationMode::min,
          kamping::measurements::GlobalAggregationMode::max,
      });

  kamping::measurements::timer().start("request_reply");
  auto make_reply = [&](const auto& requested_ruler) {
    auto local_idx = dist.get_local_idx(requested_ruler, comm.rank());
    return ruler_reply{.ruler = requested_ruler,
                       .root = succ_array[local_idx],
                       .dist_to_root = rank_array[local_idx]};
  };
  auto recv_replies = [&]() {
    if (config.use_grid_communication) {
      KASSERT(grid_comm.has_value());
      return request_reply_without_remote_aggregation(requests, make_reply,
                                                      grid_comm.value());
    }
    return request_reply_without_remote_aggregation(requests, make_reply, comm);
  }();

  kamping::measurements::timer().stop();

  // store replies
  kamping::measurements::timer().start("process_replies");
  absl::flat_hash_map<idx_t, ruler_reply> ruler_info;
  for (auto const& reply : recv_replies) {
    ruler_info[reply.ruler] = reply;
  }
  for (auto local_idx : dist.local_indices(comm.rank())) {
    if (!needs_to_request_ruler(local_idx)) {
      // this node might have been reached by a leaf, so its msb might be still be set,
      // fix that
      succ_array[local_idx] = bits::clear_root_flag(succ_array[local_idx]);
      continue;
    }
    auto ruler = succ_array[local_idx];
    if (config.use_locality_aware_in_ruler_propagation &&
        dist.is_local(ruler, comm.rank())) {
      auto ruler_local = dist.get_local_idx(ruler, comm.rank());
      succ_array[local_idx] = bits::clear_root_flag(succ_array[ruler_local]);
      rank_array[local_idx] = rank_array[ruler_local] + rank_array[local_idx];
      continue;
    }
    auto info_it = ruler_info.find(ruler);
    KASSERT(info_it != ruler_info.end(),
            fmt::format("Did not receive info for ruler {}. This should not happen, "
                        "since we requested it.",
                        ruler));
    succ_array[local_idx] = info_it->second.root;
    rank_array[local_idx] += info_it->second.dist_to_root;
  }
  kamping::measurements::timer().stop();
}

inline auto ruler_propagation(
    SparseRulingSetConfig const& config,
    std::span<idx_t> succ_array,
    std::span<rank_t> rank_array,
    std::vector<NodeType> const& node_type,
    Distribution const& dist,
    kamping::Communicator<> const& comm,
    std::optional<TopologyAwareGridCommunicator> const& grid_comm,
    propagation_mode::pull_tag /* tag */ = {}) {
  auto needs_to_request_ruler = [&](idx_t local_idx) {
    // if the msb is set, this node was reached from a leaf, so root and rank are already
    // correct
    // rulers and leafs also have the correct result already from the base algorithm
    return !bits::has_root_flag(succ_array[local_idx]) &&
           node_type[local_idx] != NodeType::ruler &&
           node_type[local_idx] != NodeType::leaf;
  };
  kamping::measurements::timer().start("cache_owners");
  std::optional<std::vector<std::size_t>> succ_owner;
  if (config.cache_owners) {
    succ_owner.emplace(succ_array.size());
    for (std::size_t i = 0; i < succ_owner->size(); i++) {
      auto succ = succ_array[i];
      if (dist.is_local(succ, comm.rank())) {
        (*succ_owner)[i] = comm.rank();
      } else {
        (*succ_owner)[i] = dist.get_owner(bits::clear_root_flag(succ));
      }
    }
  }
  kamping::measurements::timer().stop();
  auto get_succ_owner = [&](idx_t local_idx, idx_t succ_global) {
    if (!config.cache_owners) {
      return dist.get_owner(succ_global);
    }
    return (*succ_owner)[local_idx];
  };
  kamping::measurements::timer().start("collect_requests");
  std::vector<std::pair<int, idx_t>> requests;
  std::size_t const max_size_requests = dist.local_indices(comm.rank()).size();
  requests.reserve(max_size_requests);
  for (auto local_idx : dist.local_indices(comm.rank())) {
    if (!needs_to_request_ruler(local_idx)) {
      continue;
    }
    auto ruler = succ_array[local_idx];
    if (config.use_locality_aware_in_ruler_propagation &&
        dist.is_local(ruler, comm.rank())) {
      continue;
    }
    requests.emplace_back(get_succ_owner(local_idx, ruler), ruler);
  }
  kamping::measurements::timer().stop();

  struct ruler_reply {
    idx_t root;
    rank_t dist_to_root;
  };

  kamping::measurements::timer().start("request_reply");

  auto make_reply = [&](const auto& requested_ruler) {
    auto local_idx = dist.get_local_idx(requested_ruler, comm.rank());
    return ruler_reply{.root = succ_array[local_idx],
                       .dist_to_root = rank_array[local_idx]};
  };

  // execute the actual request step either using direct or indirect (grid) communication
  auto [recv_replies, send_displs] = [&]() {
    if (config.use_grid_communication) {
      KASSERT(grid_comm.has_value());
      return request_reply_without_remote_aggregation(
          requests, make_reply, grid_comm.value(), request_reply_mode::reorder_output);
    }
    return request_reply_without_remote_aggregation(requests, make_reply, comm,
                                                    request_reply_mode::reorder_output);
  }();

  kamping::measurements::timer().stop();
  kamping::measurements::timer().start("postprocessing");
  for (auto local_idx : dist.local_indices(comm.rank())) {
    if (!needs_to_request_ruler(local_idx)) {
      // this node might have been reached by a leaf, so its msb might be still be set,
      // fix that
      succ_array[local_idx] = bits::clear_root_flag(succ_array[local_idx]);
      continue;
    }
    if (config.use_locality_aware_in_ruler_propagation &&
        dist.is_local(succ_array[local_idx], comm.rank())) {
      auto ruler = succ_array[local_idx];
      auto ruler_local = dist.get_local_idx(ruler, comm.rank());
      succ_array[local_idx] = bits::clear_root_flag(succ_array[ruler_local]);
      rank_array[local_idx] = rank_array[ruler_local] + rank_array[local_idx];
      continue;
    }
    auto target = get_succ_owner(local_idx, succ_array[local_idx]);
    auto pos = send_displs[target]++;
    succ_array[local_idx] = recv_replies[pos].root;
    rank_array[local_idx] += recv_replies[pos].dist_to_root;
  }
  kamping::measurements::timer().stop();
}

inline auto ruler_propagation(
    SparseRulingSetConfig const& config,
    std::span<idx_t> succ_array,
    std::span<rank_t> rank_array,
    std::span<const idx_t> initial_succ_array,
    std::vector<NodeType> const& node_type,
    std::span<const idx_t> rulers,
    Distribution const& dist,
    kamping::Communicator<> const& comm,
    std::optional<TopologyAwareGridCommunicator> const& grid_comm,
    propagation_mode::push_tag /* tag */) {
  auto init = [&](auto&& enqueue_locally, auto&& send_to) {
    for (auto const& ruler_local : rulers) {
      if (node_type[ruler_local] == NodeType::leaf) {
        // nodes reached from leafs already have the correct root and rank, so we can just
        // skip them
        continue;
      }
      auto succ = initial_succ_array[ruler_local];
      auto ruler_root = succ_array[ruler_local];
      auto ruler_rank = rank_array[ruler_local];
      RulerMessage msg{
          .target_idx = succ, .ruler = ruler_root, .dist_from_ruler = ruler_rank};
      if (dist.is_local(succ, comm.rank())) {
        enqueue_locally(msg);
        continue;
      }
      send_to(msg, dist.get_owner(succ));
    }
  };
  auto work_on_item = [&](RulerMessage const& msg, auto&& enqueue_locally,
                          auto&& send_to) {
    const auto& [idx, ruler_root, ruler_rank] = msg;
    KASSERT(dist.is_local(idx, comm.rank()));
    auto idx_local = dist.get_local_idx(idx, comm.rank());
    if (node_type[idx_local] == NodeType::ruler) {
      return;
    }
    succ_array[idx_local] = ruler_root;
    rank_array[idx_local] += ruler_rank;
    if (node_type[idx_local] == NodeType::root) {
      return;
    }
    KASSERT(node_type[idx_local] == NodeType::reached);
    auto succ = initial_succ_array[idx_local];
    RulerMessage forward_msg = msg;
    forward_msg.target_idx = succ;
    if (dist.is_local(succ, comm.rank())) {
      enqueue_locally(forward_msg);
      return;
    }
    send_to(forward_msg, dist.get_owner(succ));
  };
  if (config.sync) {
    ruler_chasing_engine(config, init, work_on_item, dist, comm, grid_comm,
                         ruler_chasing::sync);
  } else {
    ruler_chasing_engine(config, init, work_on_item, dist, comm, ruler_chasing::async);
  }
}

inline auto ruler_propagation(
    SparseRulingSetConfig const& config,
    std::span<idx_t> succ_array,
    std::span<rank_t> rank_array,
    std::optional<std::span<const idx_t>> initial_succ_array,
    std::vector<NodeType> const& node_type,
    std::span<const idx_t> rulers,
    Distribution const& dist,
    kamping::Communicator<> const& comm,
    std::optional<TopologyAwareGridCommunicator> const& grid_comm) {
  switch (config.ruler_propagation_mode) {
    case RulerPropagationMode::pull:
      if (config.use_aggregation_in_ruler_propagation) {
        ruler_propagation(config, succ_array, rank_array, node_type, dist, comm,
                          grid_comm,
                          sparse_ruling_set_detail::propagation_mode::local_aggregation,
                          sparse_ruling_set_detail::propagation_mode::pull);
      } else {
        ruler_propagation(config, succ_array, rank_array, node_type, dist, comm,
                          grid_comm, sparse_ruling_set_detail::propagation_mode::pull);
      }
      break;
    case RulerPropagationMode::push:
      ruler_propagation(config, succ_array, rank_array, initial_succ_array.value(),
                        node_type, rulers, dist, comm, grid_comm,
                        sparse_ruling_set_detail::propagation_mode::push);
      break;
    case RulerPropagationMode::invalid:
      throw std::runtime_error("Invalid ruler propagation mode");
      break;
  }
}

}  // namespace kascade::sparse_ruling_set_detail
