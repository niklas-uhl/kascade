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
          std::span<idx_t> /*root_array*/,
          std::span<rank_t> /*rank_array*/,
          kamping::Communicator<> const& /*comm*/) {}

void rank_on_root(std::span<idx_t> succ_array,
                  std::span<rank_t> rank_array,
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
                               std::span<rank_t> rank_array,
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
