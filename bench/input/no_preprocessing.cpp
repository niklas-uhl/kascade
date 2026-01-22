#include "input/generation.hpp"

namespace kascade::input::internal {
auto write_graph_to_succ_array(kagen::Graph const& kagen_graph,
                               kamping::Communicator<> const& comm)
    -> std::vector<kascade::idx_t> {
  kacc::DistributedCSRGraph graph(kagen_graph, comm);
  std::vector<kascade::idx_t> succ_array(graph.num_local_vertices());
  for (std::size_t i = 0; i < graph.num_local_vertices(); ++i) {
    auto global_id = graph.to_global(i);
    succ_array[i] = static_cast<kascade::idx_t>(global_id);
    for (auto const& u : graph.neighbors(global_id)) {
      succ_array[i] = static_cast<kascade::idx_t>(u);
    }
  }
  return succ_array;
}
}  // namespace kascade::input::internal
