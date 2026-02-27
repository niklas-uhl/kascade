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

auto fixup_unreached(SparseRulingSetConfig const& config,
                     std::size_t num_unreached,
                     std::span<idx_t> succ_array,
                     std::span<rank_t> rank_array,
                     std::vector<NodeType>& node_type,
                     Distribution const& dist,
                     kamping::Communicator<> const& comm,
                     std::optional<TopologyAwareGridCommunicator> const& grid_comm)
    -> std::vector<idx_t> {
  std::vector<idx_t> elements_to_fix;
  if (num_unreached > 0) {
    auto unreached =
        dist.local_indices(comm.rank()) | std::views::filter([&](idx_t local_idx) {
          return node_type[local_idx] == NodeType::unreached;
        });
    elements_to_fix.reserve(num_unreached);
    elements_to_fix.insert_range(elements_to_fix.end(), unreached);
    KASSERT(elements_to_fix.size() == num_unreached);
  }
  SPDLOG_DEBUG("Fixup {} unreached nodes.", elements_to_fix.size());

  struct message_type {
    idx_t pred;
    idx_t succ;
    rank_t dist_pred_succ;
  };
  std::vector<std::pair<int, message_type>> message_buf;
  message_buf.reserve(elements_to_fix.size());
  std::vector<message_type> local_message_buf;
  local_message_buf.reserve(elements_to_fix.size());
  for (idx_t unreached_local : elements_to_fix) {
    // all unreached nodes are marked as leafs initially, and get relabeled as ruler once
    // they receive a message, so we can identify real leafs among the unreached nodes
    // later
    node_type[unreached_local] = NodeType::leaf;
    auto unreached = dist.get_global_idx(unreached_local, comm.rank());
    auto succ = succ_array[unreached_local];
    auto dist_to_succ = rank_array[unreached_local];
    succ_array[unreached_local] = unreached;
    rank_array[unreached_local] = 0;
    message_type msg{.pred = unreached, .succ = succ, .dist_pred_succ = dist_to_succ};
    if (dist.is_local(succ, comm.rank())) {
      local_message_buf.emplace_back(msg);
      continue;
    }
    auto owner = dist.get_owner(succ);
    message_buf.emplace_back(owner, msg);
  }
  AlltoallDispatcher<message_type> dispatcher(config.use_grid_communication, comm,
                                              grid_comm);
  auto recv_buf = dispatcher.alltoallv(message_buf);
  auto handle_message = [&](message_type const& msg) {
    KASSERT(dist.is_local(msg.succ, comm.rank()));
    auto local_idx = dist.get_local_idx(msg.succ, comm.rank());
    succ_array[local_idx] = msg.pred;
    rank_array[local_idx] = msg.dist_pred_succ;
    if (node_type[local_idx] == NodeType::root) {
      // this happens when a list has not been reached at all, so the root is still
      // unreached. In this case, we add it to the list of elements to fix, so it gets
      // handled in the base case.
      elements_to_fix.push_back(local_idx);
    }
    if (node_type[local_idx] == NodeType::leaf) {
      node_type[local_idx] = NodeType::ruler;
    }
  };
  std::ranges::for_each(local_message_buf, handle_message);
  std::ranges::for_each(recv_buf, handle_message);
  return elements_to_fix;
}

}  // namespace
}  // namespace kascade::sparse_ruling_set_detail
