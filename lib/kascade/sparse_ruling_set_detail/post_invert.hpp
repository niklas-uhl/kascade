#pragma once

#include <ranges>
#include <span>
#include <vector>

#include <absl/container/flat_hash_set.h>
#include <kamping/communicator.hpp>
#include <spdlog/spdlog.h>

#include "kascade/configuration.hpp"
#include "kascade/distribution.hpp"
#include "kascade/grid_alltoall.hpp"
#include "kascade/grid_communicator.hpp"
#include "kascade/request_aggregation_scheme.hpp"
#include "kascade/types.hpp"
#include "sparse_ruling_set_detail/types.hpp"

namespace kascade::sparse_ruling_set_detail {
namespace {
auto post_invert(SparseRulingSetConfig const& config,
                 std::span<idx_t> succ_array,
                 std::span<rank_t> rank_array,
                 std::vector<NodeType>& node_type,
                 Distribution const& dist,
                 kamping::Communicator<> const& comm,
                 std::optional<TopologyAwareGridCommunicator> const& grid_comm) {
  auto roots = dist.local_indices(comm.rank()) | std::views::filter([&](idx_t local_idx) {
                 return node_type[local_idx] == NodeType::root;
               });
  struct LeafMessage {
    idx_t leaf;
    idx_t root;
    rank_t dist_from_root_to_leaf;
  };
  AlltoallDispatcher<LeafMessage> dispatcher{config.use_grid_communication, comm,
                                             grid_comm};
  std::vector<std::pair<int, LeafMessage>> messages;
  for (auto root_local : roots) {
    auto leaf = succ_array[root_local];
    auto root = dist.get_global_idx(root_local, comm.rank());
    if (leaf == root) {  // this a singular node, so we do nothing
      continue;
    }
    auto dist_from_root_to_leaf = rank_array[root_local];
    succ_array[root_local] = root;
    rank_array[root_local] = 0;
    if (dist.is_local(leaf, comm.rank())) {
      auto leaf_local = dist.get_local_idx(leaf, comm.rank());
      KASSERT(succ_array[leaf_local] == leaf);
      KASSERT(rank_array[leaf_local] == 0);
      succ_array[leaf_local] = root;
      rank_array[leaf_local] = dist_from_root_to_leaf;
      // the leaf might not know yet that it's a leaf, so we set it
      node_type[leaf_local] = NodeType::leaf;
      continue;
    }
    auto leaf_owner = dist.get_owner(leaf);
    messages.emplace_back(leaf_owner,
                          LeafMessage{.leaf = leaf,
                                      .root = root,
                                      .dist_from_root_to_leaf = dist_from_root_to_leaf});
  }

  auto leaf_messages = dispatcher.alltoallv(messages);
  for (auto const& leaf_message : leaf_messages) {
    KASSERT(dist.is_local(leaf_message.leaf, comm.rank()));
    auto leaf_local = dist.get_local_idx(leaf_message.leaf, comm.rank());
    KASSERT(succ_array[leaf_local] == leaf_message.leaf);
    KASSERT(rank_array[leaf_local] == 0);
    succ_array[leaf_local] = leaf_message.root;
    rank_array[leaf_local] = leaf_message.dist_from_root_to_leaf;
    // the leaf might not know yet that it's a leaf, so we set it
    node_type[leaf_local] = NodeType::leaf;
  }

  absl::flat_hash_set<idx_t> leafs_to_query;
  for (auto local_idx : dist.local_indices(comm.rank())) {
    if (node_type[local_idx] == NodeType::root ||
        node_type[local_idx] == NodeType::leaf) {
      continue;
    }
    auto leaf = succ_array[local_idx];
    if (dist.is_local(leaf, comm.rank())) {
      continue;
    }
    leafs_to_query.insert(leaf);
  }
  auto leaf_requests = leafs_to_query | std::views::transform([&](idx_t leaf) {
                         return std::pair{dist.get_owner(leaf), leaf};
                       }) |
                       std::ranges::to<std::vector>();
  struct LeafReply {
    idx_t root;
    rank_t dist_to_root;
  };
  auto make_reply = [&](idx_t const& requested_leaf) {
    KASSERT(dist.is_local(requested_leaf, comm.rank()));
    auto leaf_local = dist.get_local_idx(requested_leaf, comm.rank());
    KASSERT(node_type[leaf_local] == NodeType::leaf);
    return LeafReply{.root = succ_array[leaf_local],
                     .dist_to_root = rank_array[leaf_local]};
  };
  auto [leaf_replies, displs] = [&] {
    if (config.use_grid_communication) {
      KASSERT(grid_comm.has_value());
      return request_reply_without_remote_aggregation(
          leaf_requests, make_reply, *grid_comm, request_reply_mode::reorder_output);
    }
    return request_reply_without_remote_aggregation(leaf_requests, make_reply, comm,
                                                    request_reply_mode::reorder_output);
  }();
  absl::flat_hash_map<idx_t, LeafReply> leaf_info;
  leaf_info.reserve(leaf_replies.size());
  for (auto const& [leaf_owner, leaf] : leaf_requests) {
    auto const& reply = leaf_replies[displs[leaf_owner]++];
    leaf_info[leaf] = reply;
  }
  for (auto local_idx : dist.local_indices(comm.rank())) {
    if (node_type[local_idx] == NodeType::root ||
        node_type[local_idx] == NodeType::leaf) {
      continue;
    }
    auto leaf = succ_array[local_idx];
    if (dist.is_local(leaf, comm.rank())) {
      auto leaf_local = dist.get_local_idx(leaf, comm.rank());
      succ_array[local_idx] = succ_array[leaf_local];  // points to root
      rank_array[local_idx] = rank_array[leaf_local] - rank_array[local_idx];
      continue;
    }
    auto const& info = leaf_info[leaf];
    succ_array[local_idx] = info.root;
    rank_array[local_idx] = info.dist_to_root - rank_array[local_idx];
  }
}
}  // namespace
}  // namespace kascade::sparse_ruling_set_detail
