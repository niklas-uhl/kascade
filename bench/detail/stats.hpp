#pragma once

#include <span>

#include <kamping/communicator.hpp>

#include "kascade/types.hpp"

enum class StatsLevel : std::uint8_t {
  invalid,
  none,
  basic,
  extensive,
};

struct TreeStats {
  std::size_t size = 0;
  std::size_t max_rank = 0;
  std::size_t rank_sum = 0;
};

struct Stats {
  Stats(StatsLevel stats_level,
        std::span<kascade::idx_t const> root_array,
        std::span<kascade::idx_t const> rank_array,
        kamping::Communicator<> const& comm);
  std::vector<TreeStats> tree_stats;
  std::size_t num_roots = 0U;
  std::size_t max_rank = 0U;
  double avg_rank = 0.;
};
