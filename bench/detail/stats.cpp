#include "detail/stats.hpp"

#include <ranges>

#include <absl/container/flat_hash_map.h>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/collectives/gather.hpp>
#include <kamping/collectives/reduce.hpp>
#include <kamping/types/unsafe/utility.hpp>
#include <kamping/utils/flatten.hpp>
#include <spdlog/spdlog.h>

#include "kascade/distribution.hpp"

namespace {
auto is_root(std::size_t prefix_count,
             std::size_t local_idx,
             std::span<kascade::idx_t const> root_array) -> bool {
  return root_array[local_idx] == static_cast<kascade::idx_t>(prefix_count + local_idx);
}

std::vector<TreeStats> compute_per_tree_stats(std::span<kascade::idx_t const> root_array,
                                              std::span<kascade::idx_t const> rank_array,
                                              kascade::Distribution const& dist,
                                              kamping::Communicator<> const& comm) {
  namespace kmp = kamping::params;
  absl::flat_hash_map<kascade::idx_t, TreeStats> local_stats;
  for (std::size_t i = 0; i < root_array.size(); ++i) {
    auto const root = root_array[i];
    auto const rank = rank_array[i];
    ++local_stats[root].size;
    local_stats[root].rank_sum += rank;
    local_stats[root].max_rank = std::max(local_stats[root].max_rank, rank);
  }
  absl::flat_hash_map<int, std::vector<std::pair<kascade::idx_t, TreeStats>>>
      nested_send_buf;
  for (auto const& [root, stats] : local_stats) {
    int owner = dist.get_owner_signed(root);
    nested_send_buf[owner].emplace_back(root, stats);
  }
  auto [send_buf, send_counts, send_displs] =
      kamping::flatten(nested_send_buf, comm.size());
  auto recv_buf = comm.alltoallv(kmp::send_buf(send_buf), kmp::send_counts(send_counts),
                                 kmp::send_displs(send_displs));
  local_stats.clear();
  for (auto const& [root, stats] : recv_buf) {
    local_stats[root].size += stats.size;
    local_stats[root].rank_sum += stats.rank_sum;
    local_stats[root].max_rank = std::max(local_stats[root].max_rank, stats.max_rank);
  }
  std::vector<TreeStats> aggregated_stats;
  for (auto const& [root, stats] : local_stats) {
    aggregated_stats.emplace_back(stats.size, stats.max_rank, stats.rank_sum);
  }
  return comm.gatherv(kmp::send_buf(aggregated_stats));
}

}  // namespace

Stats::Stats(StatsLevel stats_level,
             std::span<kascade::idx_t const> root_array,
             std::span<kascade::idx_t const> rank_array,
             kamping::Communicator<> const& comm) {
  if (stats_level == StatsLevel::none) {
    return;
  }
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

  num_roots = comm.reduce_single(kmp::send_buf(local_num_roots), kmp::op(std::plus<>{}))
                  .value_or(0U);

  std::size_t const global_rank_sum =
      comm.reduce_single(kmp::send_buf(local_rank_sum), kmp::op(std::plus<>{}))
          .value_or(0U);

  avg_rank =
      static_cast<double>(global_rank_sum) / static_cast<double>(dist.get_global_size());

  max_rank =
      comm.reduce_single(kmp::send_buf(local_max_rank), kmp::op(kamping::ops::max<>{}))
          .value_or(0U);
  if (stats_level == StatsLevel::basic) {
    return;
  }
  tree_stats = compute_per_tree_stats(root_array, rank_array, dist, comm);
}
