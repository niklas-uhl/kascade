#include <ranges>

#include <kascade/list_ranking.hpp>

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <kamping/collectives/allreduce.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/collectives/gather.hpp>
#include <kamping/collectives/scatter.hpp>
#include <kamping/communicator.hpp>
#include <kamping/measurements/timer.hpp>
#include <kamping/utils/flatten.hpp>

#include "kascade/distribution.hpp"
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
  auto leaves = std::views::iota(std::size_t{0}, succ_array.size()) |
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

/// Check if the given successor array represents a list (no node has more than one
/// predecessor)
auto is_list(std::span<const idx_t> succ_array, kamping::Communicator<> const& comm)
    -> bool {
  Distribution dist{succ_array.size(), comm};
  absl::flat_hash_map<int, std::vector<idx_t>> requests;
  auto indices = std::views::iota(idx_t{0}, static_cast<idx_t>(succ_array.size())) |
                 std::views::transform([&](auto local_idx) {
                   return dist.get_global_idx(local_idx, comm.rank());
                 });
  for (auto [idx, succ] : std::views::zip(indices, succ_array)) {
    if (succ == idx) {
      continue;
    }
    auto owner = dist.get_owner_signed(succ);
    requests[owner].push_back(succ);
  }
  auto [send_buf, send_counts, send_displs] = kamping::flatten(requests, comm.size());
  requests.clear();
  auto recv_buf =
      comm.alltoallv(kamping::send_buf(send_buf), kamping::send_counts(send_counts),
                     kamping::send_displs(send_displs));
  // if we receive a duplicate request, it's not a list
  absl::flat_hash_set<idx_t> received;
  bool duplicate_found = false;
  for (auto succ : recv_buf) {
    if (!received.insert(succ).second) {
      duplicate_found = true;
      break;
    }
  }
  bool is_list = !duplicate_found;
  comm.allreduce(kamping::send_recv_buf(is_list), kamping::op(std::logical_and{}));
  return is_list;
}

}  // namespace kascade
