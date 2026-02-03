#pragma once

#include <cstddef>
#include <cstdint>

namespace kascade {
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
}  // namespace kascade
