#include <ranges>

#include <kascade/list_ranking.hpp>

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <fmt/ranges.h>
#include <kamping/collectives/allreduce.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/collectives/gather.hpp>
#include <kamping/collectives/scatter.hpp>
#include <kamping/communicator.hpp>
#include <kamping/measurements/timer.hpp>
#include <kamping/named_parameters.hpp>
#include <kamping/utils/flatten.hpp>
#include <spdlog/spdlog.h>

#include "kascade/distribution.hpp"
#include "kascade/types.hpp"

namespace kascade {

void rank(std::span<const idx_t> /*succ_array*/,
          std::span<idx_t> /*rank_array*/,
          std::span<idx_t> /*root_array*/,
          kamping::Communicator<> const& /*comm*/) {}

namespace {
void local_pointer_chasing(std::span<idx_t> succ_array, std::span<idx_t> rank_array) {
  std::vector<bool> has_pred(succ_array.size(), false);
  // find leaves, verify roots
  for (idx_t idx{0}; idx < succ_array.size(); idx++) {
    if (succ_array[idx] == idx) {
      KASSERT(rank_array[idx] == 0);
    }
    has_pred[succ_array[idx]] = true;
  }
  auto leaves = std::views::iota(std::size_t{0}, succ_array.size()) |
                std::views::filter([&](auto idx) { return !has_pred[idx]; });

  std::vector<idx_t> path;
  path.reserve(succ_array.size());
  // we go up from each leaf, until we reach a root, then assign back down the path and
  // contract the path
  // since we contract paths as we go, each node is only visited once
  for (auto leaf : leaves) {
    idx_t cur = leaf;
    // follow successors until we reach a root
    while (succ_array[cur] != cur) {
      path.push_back(cur);
      cur = succ_array[cur];
    }
    // now assign back along the path
    idx_t root = cur;
    for (auto idx : std::ranges::reverse_view(path)) {
      rank_array[idx] += rank_array[succ_array[idx]];
      succ_array[idx] = root;
    }
    path.clear();
  }
}
}  // namespace

void rank_on_root(std::span<idx_t> succ_array,
                  std::span<idx_t> rank_array,
                  Distribution const& dist,
                  kamping::Communicator<> const& comm) {
  namespace kmp = kamping::params;

  kamping::measurements::timer().synchronize_and_start("gather_input");
  std::vector<int> recv_counts{dist.counts().begin(), dist.counts().end()};
  auto [global_succ_array, recv_displs] = comm.gatherv(
      kmp::send_buf(succ_array), kmp::recv_counts(recv_counts), kmp::recv_displs_out());
  auto global_rank_array =
      comm.gatherv(kmp::send_buf(rank_array), kmp::recv_counts(recv_counts),
                   kmp::recv_displs(recv_displs));
  kamping::measurements::timer().stop();

  kamping::measurements::timer().synchronize_and_start("local_ranking");
  if (comm.is_root()) {
    local_pointer_chasing(global_succ_array, global_rank_array);
  }
  kamping::measurements::timer().stop();

  kamping::measurements::timer().synchronize_and_start("scatter_result");
  comm.scatterv(kmp::send_buf(global_rank_array), kmp::recv_buf(rank_array),
                kmp::send_counts(recv_counts), kmp::send_displs(recv_displs),
                kmp::recv_count(static_cast<int>(succ_array.size())));
  comm.scatterv(kmp::send_buf(global_succ_array), kmp::recv_buf(succ_array),
                kmp::send_counts(recv_counts), kmp::send_displs(recv_displs),
                kmp::recv_count(static_cast<int>(succ_array.size())));
  kamping::measurements::timer().stop();
}

auto set_initial_ranking_state(std::span<const idx_t> succ_array,
                               std::span<idx_t> root_array,
                               std::span<idx_t> rank_array,
                               kamping::Communicator<> const& comm) -> Distribution {
  KASSERT(root_array.size() == succ_array.size());
  KASSERT(rank_array.size() == succ_array.size());
  Distribution dist{succ_array.size(), comm};
  std::ranges::copy(succ_array, root_array.begin());
  std::ranges::fill(rank_array, 1);
  auto global_indices =
      std::views::iota(idx_t{0}, static_cast<idx_t>(succ_array.size())) |
      std::views::transform(
          [&](auto local_idx) { return dist.get_global_idx(local_idx, comm.rank()); });
  for (auto [idx, root, rank] : std::views::zip(global_indices, root_array, rank_array)) {
    if (idx == root) {
      rank = 0;
    }
  }

  return dist;
}

}  // namespace kascade
