#pragma once

#include <concepts>
#include <ranges>
#include <span>
#include <vector>

#include <kassert/kassert.hpp>

#include "kascade/types.hpp"

namespace kascade {

/// Collapse the successor graph `succ_array` so that every vertex
/// directly points to its root or to the successor of its last local anchestor.
/// While doing that the function also accumulates the rank
/// values.
void local_pointer_chasing_generic(std::span<idx_t> succ_array,
                                   std::span<idx_t> rank_array,
                                   std::invocable<idx_t> auto& to_local,
                                   std::invocable<idx_t> auto& to_global,
                                   std::predicate<idx_t> auto& is_local_vertex) {
  std::vector<bool> has_pred(succ_array.size(), false);
  // find leaves, verify (pseudo)roots
  for (idx_t idx{0}; idx < succ_array.size(); idx++) {
    auto const succ = succ_array[idx];
    if (succ == to_global(idx)) {
      KASSERT(rank_array[idx] == 0);
    }
    if (is_local_vertex(succ)) {
      has_pred[to_local(succ)] = true;
    }
  }
  auto leaves = std::views::iota(std::size_t{0}, succ_array.size()) |
                std::views::filter([&](auto idx) { return !has_pred[idx]; });

  std::vector<idx_t> path;
  path.reserve(succ_array.size());
  // we go up from each leaf, until we reach a root, then assign back down the path and
  // contract the path
  // since we contract paths as we go, each node is only visited once

  for (auto leaf : leaves) {
    idx_t cur = to_global(leaf);
    // follow successors until we reach a root or cur is not longer on local rank
    while (is_local_vertex(cur) && succ_array[to_local(cur)] != cur) {
      path.push_back(cur);
      cur = succ_array[to_local(cur)];
    }
    // now assign back along the path
    idx_t root = cur;
    for (auto idx : std::ranges::reverse_view(path) | std::views::drop(1)) {
      KASSERT(is_local_vertex(succ_array[to_local(idx)]));
      rank_array[to_local(idx)] += rank_array[to_local(succ_array[to_local(idx)])];
      succ_array[to_local(idx)] = root;
    }
    path.clear();
  }
}
}  // namespace kascade
