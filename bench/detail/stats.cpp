#include "detail/stats.hpp"

#include <ranges>

#include <kamping/collectives/allreduce.hpp>

#include "kascade/distribution.hpp"

namespace {
auto is_root(std::size_t prefix_count,
             std::size_t local_idx,
             std::span<kascade::idx_t const> root_array) -> bool {
  return root_array[local_idx] == static_cast<kascade::idx_t>(prefix_count + local_idx);
}

}  // namespace

Stats::Stats(std::span<kascade::idx_t const> root_array,
             std::span<kascade::idx_t const> rank_array,
             kamping::Communicator<> const& comm) {
  namespace kmp = kamping::params;
  kascade::Distribution dist(root_array.size(), comm);
  std::size_t const prefix_count = dist.get_exclusive_prefix(comm.rank());

  std::size_t const local_num_roots = std::ranges::count_if(
      std::ranges::views::iota(0U, root_array.size()),
      [&](std::size_t i) { return is_root(prefix_count, i, root_array); });

  std::size_t const local_max_rank =
      std::ranges::max(rank_array, std::ranges::less{},
                       [](auto v) { return static_cast<std::size_t>(v); });
  std::size_t const local_rank_sum = std::ranges::fold_left(
      rank_array, std::size_t{0},
      [](std::size_t acc, auto v) { return acc + static_cast<std::size_t>(v); });

  num_roots =
      comm.allreduce_single(kmp::send_buf(local_num_roots), kmp::op(std::plus<>{}));

  std::size_t const global_rank_sum =
      comm.allreduce_single(kmp::send_buf(local_rank_sum), kmp::op(std::plus<>{}));

  avg_rank =
      static_cast<double>(global_rank_sum) / static_cast<double>(dist.get_global_size());

  max_rank = comm.allreduce_single(kmp::send_buf(local_max_rank),
                                   kmp::op(kamping::ops::max<>{}));
}
