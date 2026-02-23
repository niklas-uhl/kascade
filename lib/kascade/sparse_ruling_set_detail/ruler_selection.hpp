#pragma once

#include <algorithm>
#include <cmath>
#include <ranges>
#include <utility>

#include <kamping/communicator.hpp>
#include <spdlog/spdlog.h>

#include "kascade/configuration.hpp"
#include "kascade/distribution.hpp"
#include "kascade/types.hpp"

namespace kascade::sparse_ruling_set_detail {
namespace {
auto compute_local_num_rulers(SparseRulingSetConfig const& config,
                              Distribution const& dist,
                              kamping::Communicator<> const& comm) -> std::size_t {
  // NOLINTBEGIN(readability-identifier-length)
  auto n = dist.get_global_size();
  auto p = comm.size();
  auto rel_local_size =
      static_cast<double>(dist.get_local_size(comm.rank())) / static_cast<double>(n);
  // NOLINTEND(readability-identifier-length)

  switch (config.ruler_selection) {
    case RulerSelectionStrategy::dehne:
      // pick O(n/p) rulers in total
      return static_cast<std::size_t>(config.dehne_factor *
                                      (static_cast<double>(n) / static_cast<double>(p))) /
             p;
    case RulerSelectionStrategy::heuristic:
      // pick heuristic_factor * local_num_leaves per PE
      return static_cast<std::size_t>(
          config.heuristic_factor *
          static_cast<double>(dist.get_local_size(comm.rank())));
    case RulerSelectionStrategy::sanders:
      return static_cast<std::size_t>(config.sanders_factor * std::sqrt(n) *
                                      static_cast<double>(p) / std::log(n) *
                                      rel_local_size);
    case RulerSelectionStrategy::limit_rounds: {
      if (!config.spawn) {
        SPDLOG_LOGGER_WARN(spdlog::get("root"),
                           "limit-rounds ruler selection strategy is only effective if "
                           "spawn is enabled");
      }
      auto total_num_rulers = n / config.round_limit;
      return static_cast<std::size_t>(static_cast<double>(total_num_rulers) *
                                      rel_local_size);
    }
    case RulerSelectionStrategy::invalid:
      throw std::runtime_error("Invalid ruler selection strategy");
      break;
  }
  std::unreachable();
}

auto pick_rulers(std::span<const idx_t> succ_array,
                 std::size_t local_num_rulers,
                 auto& rng,
                 std::predicate<idx_t> auto const& idx_predicate) -> std::vector<idx_t> {
  std::vector<idx_t> rulers(local_num_rulers);
  auto indices = std::views::iota(idx_t{0}, static_cast<idx_t>(succ_array.size())) |
                 std::views::filter(idx_predicate);
  auto it = std::ranges::sample(
      indices, rulers.begin(),
      static_cast<std::ranges::range_difference_t<decltype(indices)>>(local_num_rulers),
      rng);
  rulers.erase(it, rulers.end());
  return rulers;
}
}  // namespace
}  // namespace kascade::sparse_ruling_set_detail
