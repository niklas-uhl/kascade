
#include <kagen.h>
#include <kamping/collectives/allreduce.hpp>
#include <kamping/communicator.hpp>
#include <kamping/types/unsafe/utility.hpp>

#include "input/bfs.hpp"
#include "input/graph.hpp"
#include "kascade/types.hpp"

namespace {
auto next_unvisited(std::size_t start, std::vector<bool> const& visited)
    -> std::optional<std::size_t> {
  while (start < visited.size() && visited[start]) {
    ++start;
  }
  if (start == visited.size()) {
    return std::nullopt;
  }
  return start;
}

auto get_next_global_start(std::size_t local_start,
                           std::vector<bool> const& visited,
                           kacc::DistributedCSRGraph const& graph,
                           kamping::Communicator<> const& comm) -> std::size_t {
  namespace kmp = kamping::params;

  auto const next_local = next_unvisited(local_start, visited);
  auto const global_idx =
      next_local ? graph.to_global(*next_local) : graph.num_global_vertices();

  return comm.allreduce_single(kmp::send_buf(global_idx), kmp::op(kamping::ops::min<>{}));
}

/// @brief Mark isolated vertices as roots/visited to prevent global termination checks on
/// them
void handle_isolated_vertices(std::vector<bool>& visited,
                              std::vector<kascade::idx_t>& parent_array,
                              kacc::DistributedCSRGraph const& graph) {
  for (std::size_t i = 0; i < graph.num_local_vertices(); ++i) {
    auto global_id = graph.to_global(i);
    if (graph.neighbors(global_id).empty()) {
      visited[i] = true;
      parent_array[i] = static_cast<kascade::idx_t>(global_id);
    }
  }
}
}  // namespace
//
namespace kascade::input::internal {
auto generate_bfs_tree(kagen::Graph const& kagen_graph,
                       kamping::Communicator<> const& comm)
    -> std::vector<kascade::idx_t> {
  std::vector<bool> visited(kagen_graph.NumberOfLocalVertices(), false);

  kacc::DistributedCSRGraph graph(kagen_graph, comm);
  std::vector<kascade::idx_t> parent_array(graph.num_local_vertices());
  handle_isolated_vertices(visited, parent_array, graph);

  std::size_t next_local_unvisited = 0;
  while (true) {
    std::size_t next_global_start =
        get_next_global_start(next_local_unvisited, visited, graph, comm);
    if (next_global_start >= graph.num_global_vertices()) {
      break;
    }
    kacc::distributed_bfs(
        graph, next_global_start,
        [&](auto u, auto parent, auto) {
          parent_array[graph.to_local(u)] = static_cast<kascade::idx_t>(parent);
        },
        visited, comm);
  }
  return parent_array;
}
}  // namespace kascade::input::internal
