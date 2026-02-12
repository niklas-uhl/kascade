#pragma once

#include <absl/container/flat_hash_set.h>
#include <kamping/collectives/alltoall.hpp>

#include "configuration.hpp"
#include "distribution.hpp"
#include "kascade/graph/graph.hpp"
#include "kascade/types.hpp"
namespace kascade {

struct EulerTour {
  using Edge = std::pair<idx_t, idx_t>;
  std::vector<idx_t> succ_array;
  std::vector<rank_t> rank_array;  // distance to root
  std::vector<bool> is_upward_edge;
  std::vector<Edge> index_to_edge;
  Distribution distribution;
};

void rank_via_euler_tour(EulerTourConfig const& config,
                         std::span<idx_t> succ_array,
                         std::span<rank_t> rank_array,
                         Distribution const& dist,
                         kamping::Communicator<> const& comm);

auto format_as(std::pair<EulerTour const&, kamping::Communicator<> const&> obj)
    -> std::string;
}  // namespace kascade
