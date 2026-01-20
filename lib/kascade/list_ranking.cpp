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
  auto const n = static_cast<idx_t>(succ_array.size());
  std::ranges::fill(rank_array, static_cast<idx_t>(-1));
  std::ranges::fill(root_array, static_cast<idx_t>(-1));

  std::vector<idx_t> stack;
  stack.reserve(n);
  for (idx_t cur = 0; cur < n; ++cur) {
    if (rank_array[cur] != static_cast<idx_t>(-1)) {
      continue;
    }
    stack.clear();

    // Follow predecessors until we reach a node whose dist is known
    while (rank_array[cur] == static_cast<idx_t>(-1)) {
      stack.push_back(cur);
      idx_t succ = succ_array[cur];

      if (succ == cur) {  // found root
        rank_array[cur] = 0;
        root_array[cur] = cur;
        break;
      }

      // parent already knows its dist and root
      if (rank_array[succ] != static_cast<idx_t>(-1)) {
        break;
      }
      cur = succ;
    }

    // Now dist[cur] and root_of[cur] are known
    idx_t rank = rank_array[cur];
    idx_t root = root_array[cur];

    // Assign distances going back along the stack
    for (unsigned long& it : std::ranges::reverse_view(stack)) {
      ++rank;
      rank_array[it] = rank;
      root_array[it] = root;
    }
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
