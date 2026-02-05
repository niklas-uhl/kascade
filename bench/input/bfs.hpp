// taken and adapted from KaCCv2, Tim Niklas Uhl, 2026
#pragma once

#include <cassert>
#include <concepts>
#include <functional>
#include <vector>

#include <absl/container/flat_hash_map.h>
#include <fmt/ranges.h>
#include <kamping/collectives/allreduce.hpp>  // IWYU pragma: keep
#include <kamping/collectives/alltoall.hpp>   // IWYU pragma: keep
#include <kamping/communicator.hpp>
#include <kamping/utils/flatten.hpp>
#include <spdlog/spdlog.h>

#include "kascade/graph/graph.hpp"

namespace kascade::input {

template <typename T, typename VertexType>
concept VertexVisitor = std::invocable<T, VertexType, VertexType, std::size_t>;

template <typename F, typename VertexType>
concept EdgePredicate = std::predicate<F, VertexType, VertexType>;

template <typename F>
concept LevelCompletionHook = std::invocable<F, std::size_t>;

auto distributed_bfs(
    kascade::graph::DistributedCSRGraph const& G,
    kascade::graph::DistributedCSRGraph::VId start_vertex,
    VertexVisitor<kascade::graph::DistributedCSRGraph::VId> auto&& visit_vertex,
    EdgePredicate<kascade::graph::DistributedCSRGraph::VId> auto&& edge_predicate,
    LevelCompletionHook auto&& complete_level,
    std::vector<bool>& visited,
    kamping::Communicator<> const& comm) {
  std::vector<std::pair<kascade::graph::DistributedCSRGraph::VId,
                        kascade::graph::DistributedCSRGraph::VId>>
      frontier;
  assert(visited.size() == G.num_local_vertices());
  if (G.is_local(start_vertex)) {
    frontier.emplace_back(start_vertex, start_vertex);
  }
  std::size_t round = 0;
  absl::flat_hash_map<int,
                      std::vector<std::pair<kascade::graph::DistributedCSRGraph::VId,
                                            kascade::graph::DistributedCSRGraph::VId>>>
      next_frontier;

  while (!comm.allreduce_single(kamping::send_buf(frontier.empty()),
                                kamping::op(std::logical_and<>{}))) {
    for (auto const& [parent, u] : frontier) {
      if (!visited[G.to_local(u)]) {
        visited[G.to_local(u)] = true;
        visit_vertex(u, parent, round);
      } else {
        continue;
      }
      for (auto const& v : G.neighbors(u)) {
        if (!edge_predicate(u, v)) {
          continue;
        }
        next_frontier[G.get_rank(v)].emplace_back(u, v);
      }
    }
    frontier =
        kamping::with_flattened(next_frontier, comm.size()).call([&](auto... flattened) {
          return comm.alltoallv(std::move(flattened)...);
        });
    SPDLOG_TRACE("next_frontier={}", next_frontier);
    next_frontier.clear();
    SPDLOG_TRACE("next_frontier={}", next_frontier);
    SPDLOG_TRACE("frontier.size()={}", frontier.size());
    complete_level(round);
    round++;
  }
}

auto distributed_bfs(
    kascade::graph::DistributedCSRGraph const& G,
    kascade::graph::DistributedCSRGraph::VId start_vertex,
    VertexVisitor<kascade::graph::DistributedCSRGraph::VId> auto&& visit_vertex,
    std::vector<bool>& visited,
    kamping::Communicator<> const& comm) {
  distributed_bfs(
      G, start_vertex, std::forward<decltype(visit_vertex)>(visit_vertex),
      [](auto, auto) { return true; }, [](auto) {}, visited, comm);
}

auto distributed_bfs(
    kascade::graph::DistributedCSRGraph const& G,
    kascade::graph::DistributedCSRGraph::VId start_vertex,
    VertexVisitor<kascade::graph::DistributedCSRGraph::VId> auto&& visit_vertex,
    kamping::Communicator<> const& comm) {
  std::vector<bool> visited(G.num_local_vertices(), false);
  distributed_bfs(G, start_vertex, std::forward<decltype(visit_vertex)>(visit_vertex),
                  visited, comm);
}
}  // namespace kascade::input
