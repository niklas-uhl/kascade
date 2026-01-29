#pragma once

#include <span>

#include <kamping/communicator.hpp>

#include "kascade/types.hpp"

enum class StatsLevel : std::uint8_t {
  invalid,
  none,
  basic,
  reduced_extensive,
  extensive,
};

struct BasicStats {
  std::size_t num_trees = 0U;
  std::size_t max_rank = 0U;
  double avg_rank = 0.;
};

struct TreeStats {
  std::size_t size = 0;
  std::size_t max_rank = 0;
  std::size_t rank_sum = 0;
};

struct ExtensiveStats {
  std::size_t num_trees = 0;
  std::size_t num_nontrivial_trees = 0;
  std::size_t nontrivial_size_sum = 0;
  std::size_t nontrivial_rank_sum = 0;
  std::size_t max_size = 0;
  double avg_size = 0.;             // somewhat redundant remove?
  double nontrivial_avg_size = 0.;  // somewhat redundant remove?
  double nontrivial_avg_rank = 0.;  // somewhat redundant remove?
  std::optional<std::vector<TreeStats>> per_tree_stats;
};

struct Stats {
  std::optional<BasicStats> basic_stats;
  std::optional<ExtensiveStats> extensive_stats;
};

auto compute_stats(StatsLevel stats_level,
                   std::span<kascade::idx_t const> root_array,
                   std::span<kascade::idx_t const> rank_array,
                   kamping::Communicator<> const& comm) -> Stats;
