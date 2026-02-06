#pragma once

#include <kamping/collectives/alltoall.hpp>
#include <kamping/types/unsafe/tuple.hpp>

#include "distribution.hpp"
#include "kascade/graph/graph.hpp"
#include "kascade/types.hpp"
namespace kascade {

inline void compute_euler_tour(graph::DistributedCSRGraph const& forest,
                        kamping::Communicator<> const& comm) {
  using Edge = std::pair<idx_t, idx_t>;
  namespace kmp = kamping::params;

  Distribution dist(forest.num_local_edges(), comm);
  std::vector<Edge> index_to_edge;
  absl::flat_hash_map<Edge, idx_t> edge_to_index;
  absl::flat_hash_map<int, std::vector<std::tuple<idx_t, idx_t, idx_t>>> send_bufs;
  std::size_t next_id = 0;
  for (auto const& v : forest.vertices()) {
    std::size_t const degree = forest.degree(v);
    auto neighbors = forest.neighbors(v);
    for (std::size_t i = 0; i < degree; ++i) {
      const auto u_prev = neighbors[(i + degree - 1) % degree];
      const auto u = neighbors[i];
      edge_to_index[std::make_pair(v, u)] = next_id;
      index_to_edge.emplace_back(v, u);
      // succ[std::make_pair(u_prev, v)] = std::make_pair(v, u);
      int owner_u_prev = forest.get_rank(u_prev);
      send_bufs[owner_u_prev].emplace_back(u_prev, v,
                                           dist.get_global_idx(next_id, comm.rank()));
      next_id++;
    }
  }

  auto [send_buf, send_counts, send_displs] = kamping::flatten(send_bufs, comm.size());
  auto recv_buf = comm.alltoallv(kmp::send_buf(send_buf), kmp::send_counts(send_counts),
                                 kmp::send_displs(send_displs));
  std::vector<idx_t> succ_array(forest.num_local_edges());
  for (const auto& [u, v, succ] : recv_buf) {
    auto it = edge_to_index.find(std::make_pair(u, v));
    KASSERT(it != edge_to_index.end());
    succ_array[it->second] = succ;
  }

  // compute list of euler tour
}

}  // namespace kascade
