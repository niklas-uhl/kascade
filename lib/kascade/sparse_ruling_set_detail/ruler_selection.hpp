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

template <typename R>
auto pick_rulers(SparseRulingSetConfig const& config,
                 std::size_t local_num_rulers,
                 Distribution const& dist,
                 auto& rng,
                 R&& local_indices_permuted,
                 std::predicate<idx_t> auto const& idx_predicate,
                 kamping::Communicator<> const& comm) -> std::pair<std::vector<idx_t>, std::ranges::iterator_t<R>> {
  std::vector<idx_t> rulers(local_num_rulers);
  if (!config.no_precompute_rulers) {
    auto current = std::ranges::begin(local_indices_permuted);
    std::size_t ruler_idx = 0;
    for (; current != std::ranges::end(local_indices_permuted); ++current) {
      idx_t idx = *current;
      if (idx_predicate(idx)) {
        rulers[ruler_idx++] = idx;
        if (ruler_idx == local_num_rulers) {
          break;
        }
      }
    }
    rulers.resize(ruler_idx);
    return {rulers, current};
  }
  auto indices = dist.local_indices(comm.rank()) | std::views::filter(idx_predicate);
  auto it = std::ranges::sample(
      indices, rulers.begin(),
      static_cast<std::ranges::range_difference_t<decltype(indices)>>(local_num_rulers),
      rng);
  rulers.erase(it, rulers.end());
  return {rulers, std::ranges::end(local_indices_permuted)};
}
}  // namespace
}  // namespace kascade::sparse_ruling_set_detail

