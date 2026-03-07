#pragma once

#include <span>
#include <string>

#include <fmt/format.h>
#include <kamping/communicator.hpp>

#include "kascade/distribution.hpp"
#include "kascade/types.hpp"
#include "sparse_ruling_set_detail/trace.hpp"
#include "sparse_ruling_set_detail/types.hpp"

namespace kascade::sparse_ruling_set_detail {
namespace {
struct LocalChainInfo {
  idx_t local_chain_start;
  idx_t next;
  rank_t dist_to_next;
};

auto format_as(LocalChainInfo const& local_chain_info) -> std::string {
  return fmt::format("{{local_chain_start={}, next={}, dist_to_next={}}}",
                     local_chain_info.local_chain_start, local_chain_info.next,
                     local_chain_info.dist_to_next);
}

auto local_contraction(std::span<idx_t> succ_array,
                       std::span<rank_t> rank_array,
                       std::span<NodeType> node_type,
                       Distribution const& dist,
                       kamping::Communicator<> const& comm, RulerTrace& trace) {
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
    auto current_node =
        succ_array[local_chain_start];  // sucdist.get_global_idx(local_chain_start,
                                        // comm.rank());
    auto is_end_of_local_chain = [&](idx_t idx) {
      if (!dist.is_local(idx, comm.rank())) {
        return true;
      }
      auto idx_local = dist.get_local_idx(idx, comm.rank());
      return node_type[idx_local] == NodeType::root;
    };
    rank_t chain_length = rank_array[local_chain_start];
    while (!is_end_of_local_chain(current_node)) {
      auto current_node_local = dist.get_local_idx(current_node, comm.rank());
      chain_length += rank_array[current_node_local];
      node_type[current_node_local] = NodeType::masked;
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
    trace.track_local_contraction(chain_length);
  }
  return std::pair{std::move(local_chain_info), num_masked};
}

auto local_uncontraction(std::vector<LocalChainInfo> const& local_chain_info,
                         std::span<idx_t> succ_array,
                         std::span<rank_t> rank_array,
                         std::span<NodeType> node_type,
                         Distribution const& dist,
                         kamping::Communicator<> const& comm) {
  auto is_end_of_local_chain = [&](idx_t idx) {
    if (!dist.is_local(idx, comm.rank())) {
      return true;
    }
    auto idx_local = dist.get_local_idx(idx, comm.rank());
    return node_type[idx_local] == NodeType::root;
  };
  for (auto const& [local_chain_start, next, dist_to_next] : local_chain_info) {
    auto current_node = next;
    auto current_dist = rank_array[local_chain_start] + dist_to_next;
    while (!is_end_of_local_chain(current_node)) {
      auto current_node_local = dist.get_local_idx(current_node, comm.rank());
      auto next_node = succ_array[current_node_local];
      auto next_dist = rank_array[current_node_local];
      succ_array[current_node_local] = succ_array[local_chain_start];
      rank_array[current_node_local] = current_dist;
      node_type[current_node_local] = NodeType::reached;
      current_node = next_node;
      current_dist += next_dist;
    }
  }
}
}  // namespace
}  // namespace kascade::sparse_ruling_set_detail
