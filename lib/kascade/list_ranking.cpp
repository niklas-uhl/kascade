#include <ranges>

#include <kascade/list_ranking.hpp>

#include <kamping/collectives/gather.hpp>
#include <kamping/collectives/scatter.hpp>
#include <kamping/measurements/timer.hpp>
#include <kamping/named_parameters.hpp>

#include "kascade/types.hpp"

namespace kascade {

void rank(std::span<const idx_t> /*succ_array*/,
          std::span<idx_t> /*rank_array*/,
          std::span<idx_t> /*root_array*/,
          kamping::Communicator<> const& /*comm*/) {}

namespace {
void local_pointer_chasing(std::span<const idx_t> succ_array,
                           std::span<idx_t> rank_array,
                           std::span<idx_t> root_array) {
  std::vector<bool> has_pred(succ_array.size(), false);
  auto const UNASSIGNED = static_cast<idx_t>(succ_array.size());
  std::ranges::fill(rank_array, UNASSIGNED);
  std::ranges::fill(root_array, UNASSIGNED);
  // label roots and find leaves
  for (idx_t idx{0}; idx < succ_array.size(); idx++) {
    if (succ_array[idx] == idx) {
      root_array[idx] = idx;
      rank_array[idx] = 0;
    }
    has_pred[succ_array[idx]] = true;
  }
  auto leaves = std::views::iota(std::size_t {0}, succ_array.size()) |
                std::views::filter([&](auto idx) { return !has_pred[idx]; });

  std::vector<idx_t> path;
  path.reserve(succ_array.size());
  for (auto leaf : leaves) {
    idx_t cur = leaf;
    // follow successors until we reach a node whose rank is known
    while (rank_array[cur] == UNASSIGNED) {
      path.push_back(cur);
      cur = succ_array[cur];
    }
    // now assign back along the path
    idx_t rank = rank_array[cur] + 1;
    idx_t root = root_array[cur];
    for (auto idx : std::ranges::reverse_view(path)) {
      rank_array[idx] = rank;
      root_array[idx] = root;
      rank++;
    }
    path.clear();
  }
}
}  // namespace

void rank_on_root(std::span<const idx_t> succ_array,
                  std::span<idx_t> rank_array,
                  std::span<idx_t> root_array,
                  kamping::Communicator<> const& comm) {
  namespace kmp = kamping::params;
  kamping::measurements::timer().synchronize_and_start("gather_succ_array");
  auto [global_succ_array, recv_counts, recv_displs] = comm.gatherv(
      kmp::send_buf(succ_array), kmp::recv_counts_out(), kmp::recv_displs_out());
  kamping::measurements::timer().stop();
  std::vector<idx_t> global_rank_array;
  std::vector<idx_t> global_root_array;
  kamping::measurements::timer().synchronize_and_start("local_ranking");
  if (comm.is_root()) {
    global_rank_array.resize(global_succ_array.size());
    global_root_array.resize(global_succ_array.size());
    local_pointer_chasing(global_succ_array, global_rank_array, global_root_array);
  }
  kamping::measurements::timer().stop();
  kamping::measurements::timer().synchronize_and_start("scatter_result");
  comm.scatterv(kmp::send_buf(global_rank_array), kmp::recv_buf(rank_array),
                kmp::send_counts(recv_counts), kmp::send_displs(recv_displs),
                kmp::recv_count(static_cast<int>(succ_array.size())));
  comm.scatterv(kmp::send_buf(global_root_array), kmp::recv_buf(root_array),
                kmp::send_counts(recv_counts), kmp::send_displs(recv_displs),
                kmp::recv_count(static_cast<int>(succ_array.size())));
  kamping::measurements::timer().stop();
}


}  // namespace kascade
