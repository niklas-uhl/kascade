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
  MPLR,
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

enum class RulerSelectionStrategy : std::uint8_t { dehne, heuristic, sanders, invalid };

struct BriefkastenConfig {
  std::size_t local_threshold = 32ULL * 1024;
  std::size_t poll_skip_threshold = 100;
};

struct SparseRulingSetConfig {
  RulerSelectionStrategy ruler_selection = RulerSelectionStrategy::dehne;
  double dehne_factor = 1.0;
  double heuristic_factor = 0.01;
  double sanders_factor = 1.0;
  bool sync = false;
  bool sync_locality_aware = false;
  bool spawn = false;
  BriefkastenConfig briefkasten {};
  bool use_grid_communication = false;
};

struct EulerTourConfig {
  Algorithm algorithm;
  bool use_high_degree_handling = false;
  std::any algo_config;
};
}  // namespace kascade
