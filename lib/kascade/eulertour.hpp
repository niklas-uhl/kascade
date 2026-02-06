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
  std::vector<bool> is_upward_edge;
  std::vector<std::int64_t> rank_array;  // distance to root
  absl::flat_hash_map<Edge, idx_t> edge_to_index;
  std::vector<Edge> index_to_edge;
  Distribution distribution;
};

void rank_via_euler_tour(EulerTourConfig const& config, std::span<idx_t> succ_array,
                         std::span<rank_t> rank_array,
                         Distribution const& dist,
                         kamping::Communicator<> const& comm);

/// for each edge in this (undirected tree), we use the following successor function
///
/// succ[u_i, v] = (v, u_{next}}) where u_{next} = u_{i + 1 % degree(v)}
///
/// the edge (v,u) is stored on owner(v) as a global id
/// we set dist_to_root(v, u) = 1 if u is parent of v and -1 otherwise
/// If v is a root, we determine exactly one edge (u, v), which has itself as successor.
auto compute_euler_tour(graph::DistributedCSRGraph const& forest,
                        std::span<idx_t> parent_array,
                        kamping::Communicator<> const& comm) -> EulerTour;

auto format_as(std::pair<EulerTour const&, kamping::Communicator<> const&> obj)
    -> std::string;

void map_euler_tour_back(EulerTour const& euler_tour,
                         std::span<idx_t> root_array,
                         std::span<rank_t> rank_array,
                         kamping::Communicator<> const& comm);
}  // namespace kascade
