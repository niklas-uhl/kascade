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
#include "kascade/successor_utils.hpp"

namespace {

template <typename T>
constexpr auto safe_division(T dividend, T divisor) noexcept -> double {
  return (divisor == T{0}) ? 0.0
                           : static_cast<double>(dividend) / static_cast<double>(divisor);
}

auto compute_per_tree_stats(bool gather_tree_stats,
                            std::span<kascade::idx_t const> root_array,
                            std::span<kascade::rank_t const> rank_array,
                            kascade::Distribution const& dist,
                            kamping::Communicator<> const& comm) -> ExtensiveStats {
  namespace kmp = kamping::params;
  ExtensiveStats stats;
  stats.per_tree_stats = std::vector<TreeStats>{};
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
  for (auto const& [root, tree_stats] : local_stats) {
    int owner = dist.get_owner_signed(root);
    nested_send_buf[owner].emplace_back(root, tree_stats);
  }
  auto [send_buf, send_counts, send_displs] =
      kamping::flatten(nested_send_buf, comm.size());
  auto recv_buf = comm.alltoallv(kmp::send_buf(send_buf), kmp::send_counts(send_counts),
                                 kmp::send_displs(send_displs));
  local_stats.clear();
  for (auto const& [root, tree_stats] : recv_buf) {
    local_stats[root].size += tree_stats.size;
    local_stats[root].rank_sum += tree_stats.rank_sum;
    local_stats[root].max_rank =
        std::max(local_stats[root].max_rank, tree_stats.max_rank);
  }

  for (auto const& [root, tree_stats] : local_stats) {
    ++stats.num_trees;
    stats.max_size = std::max(stats.max_size, tree_stats.size);
    if (tree_stats.size > 1) {
      ++stats.num_nontrivial_trees;
      stats.nontrivial_size_sum += tree_stats.size;
      stats.nontrivial_rank_sum += tree_stats.rank_sum;
      stats.per_tree_stats->emplace_back(tree_stats.size, tree_stats.max_rank,
                                         tree_stats.rank_sum);
    }
  }
  stats.num_trees =
      comm.reduce_single(kmp::send_buf(stats.num_trees), kmp::op(std::plus<>{}))
          .value_or(0U);

  stats.num_nontrivial_trees =
      comm.reduce_single(kmp::send_buf(stats.num_nontrivial_trees),
                         kmp::op(std::plus<>{}))
          .value_or(0U);

  stats.nontrivial_size_sum =
      comm.reduce_single(kmp::send_buf(stats.nontrivial_size_sum), kmp::op(std::plus<>{}))
          .value_or(0U);

  stats.nontrivial_rank_sum =
      comm.reduce_single(kmp::send_buf(stats.nontrivial_rank_sum), kmp::op(std::plus<>{}))
          .value_or(0U);

  stats.max_size =
      comm.reduce_single(kmp::send_buf(stats.max_size), kmp::op(kamping::ops::max<>{}))
          .value_or(0U);

  if (comm.is_root()) {
    std::size_t num_trivial_trees = stats.num_trees - stats.num_nontrivial_trees;
    stats.avg_size =
        safe_division(stats.nontrivial_size_sum + num_trivial_trees, stats.num_trees);
    stats.nontrivial_avg_size =
        safe_division(stats.nontrivial_size_sum, stats.num_nontrivial_trees);

    stats.nontrivial_avg_rank = safe_division(
        stats.nontrivial_rank_sum, static_cast<std::int64_t>(stats.nontrivial_size_sum));
  }

  if (gather_tree_stats) {
    stats.per_tree_stats.value() =
        comm.gatherv(kmp::send_buf(stats.per_tree_stats.value()));
  } else {
    stats.per_tree_stats = std::nullopt;
  }
  return stats;
}

auto compute_basic_stats(std::span<kascade::idx_t const> root_array,
                         std::span<kascade::rank_t const> rank_array,
                         kascade::Distribution const& dist,
                         kamping::Communicator<> const& comm) -> BasicStats {
  namespace kmp = kamping::params;

  BasicStats stats;
  std::size_t const local_num_roots = std::ranges::count_if(
      std::ranges::views::iota(0U, root_array.size()),
      [&](std::size_t i) { return is_root(i, root_array, dist, comm); });

  std::size_t const local_max_rank =
      std::ranges::max(rank_array, std::ranges::less{},
                       [](auto v) { return static_cast<std::int64_t>(v); });
  std::size_t const local_rank_sum = std::ranges::fold_left(
      rank_array, std::int64_t{0},
      [](std::int64_t acc, auto v) { return acc + static_cast<std::int64_t>(v); });

  stats.num_trees =
      comm.reduce_single(kmp::send_buf(local_num_roots), kmp::op(std::plus<>{}))
          .value_or(0U);

  std::size_t const global_rank_sum =
      comm.reduce_single(kmp::send_buf(local_rank_sum), kmp::op(std::plus<>{}))
          .value_or(0U);

  stats.avg_rank =
      static_cast<double>(global_rank_sum) / static_cast<double>(dist.get_global_size());

  stats.max_rank =
      comm.reduce_single(kmp::send_buf(local_max_rank), kmp::op(kamping::ops::max<>{}))
          .value_or(0U);
  return stats;
}
}  // namespace

auto compute_stats(StatsLevel stats_level,
                   std::span<kascade::idx_t const> root_array,
                   std::span<kascade::rank_t const> rank_array,
                   kamping::Communicator<> const& comm) -> Stats {
  if (stats_level == StatsLevel::none) {
    return Stats{};
  }
  kascade::Distribution dist(root_array.size(), comm);
  BasicStats basic_stats = compute_basic_stats(root_array, rank_array, dist, comm);
  if (stats_level == StatsLevel::basic) {
    return Stats{.basic_stats = basic_stats, .extensive_stats = std::nullopt};
  }
  bool const gather_per_tree_stats = stats_level == StatsLevel::extensive;
  ExtensiveStats extensive_stats =
      compute_per_tree_stats(gather_per_tree_stats, root_array, rank_array, dist, comm);
  return Stats{.basic_stats = basic_stats, .extensive_stats = extensive_stats};
}
