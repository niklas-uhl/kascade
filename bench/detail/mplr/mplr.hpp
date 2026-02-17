#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <kamping/communicator.hpp>

namespace mplr {

enum class Algorithm : std::uint8_t { PointerDoubling, ForestRulingSet, invalid };

struct Configuration {
  Algorithm algorithm;
  std::size_t comm_rounds = 1;
  std::size_t recursion_levels = 1;
  bool use_grid = false;
  bool use_aggregation = false;
};

auto forest_ruling_set(Configuration config,
                       std::vector<std::uint64_t>& s,
                       kamping::Communicator<>& comm)
    -> std::pair<std::vector<std::uint64_t>, std::vector<std::int64_t>>;

auto forest_pointer_doubling(Configuration config,
                             std::vector<std::uint64_t>& s,
                             kamping::Communicator<>& comm)
    -> std::pair<std::vector<std::uint64_t>, std::vector<std::int64_t>>;
}  // namespace mplr
