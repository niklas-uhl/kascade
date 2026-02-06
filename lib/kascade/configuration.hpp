#pragma once

#include <any>
#include <cstddef>
#include <cstdint>

namespace kascade {

enum class Algorithm : std::uint8_t {
  GatherChase,
  PointerDoubling,
  AsyncPointerDoubling,
  RMAPointerDoubling,
  SparseRulingSet,
  EulerTour,
  invalid,
};

enum class AggregationLevel : std::uint8_t { none, local, remote, all, invalid };

struct AsyncPointerChasingConfig {
  bool use_caching = false;
};

enum class RMASyncMode : std::uint8_t { passive_target, fenced, invalid };

struct RMAPointerChasingConfig {
  RMASyncMode sync_mode = RMASyncMode::passive_target;
  std::size_t batch_size = 1;
};

struct PointerDoublingConfig {
  bool use_local_preprocessing = false;
  AggregationLevel aggregation_level = AggregationLevel::none;
};

enum class RulerSelectionStrategy : std::uint8_t { dehne, heuristic, invalid };

struct SparseRulingSetConfig {
  RulerSelectionStrategy ruler_selection = RulerSelectionStrategy::dehne;
  double dehne_factor = 1.0;
  double heuristic_factor = 0.01;
  bool sync = false;
  bool spawn = false;
};

struct EulerTourConfig {
  Algorithm algorithm;
  std::any algo_config;
};
}  // namespace kascade
