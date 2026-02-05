// taken and adapted from kascadev2, Tim Niklas Uhl, 2026

#include "graph.hpp"

#include <utility>

#include <kagen/kagen.h>
#include <kagen/tools/converter.h>
#include <kamping/collectives/allreduce.hpp>

namespace kascade::graph {

DistributedCSRGraph::DistributedCSRGraph(kagen::Graph G,
                                         kamping::Communicator<> const& comm)
    : kagen::Graph(std::move(G)), partition_offset(xadj.size() - 1), comm_(&comm) {
  vtx_dist = kagen::BuildVertexDistribution<kagen::SInt>(*this, KAGEN_MPI_SINT,
                                                         comm_->mpi_communicator());
  for (std::size_t idx = 0; idx < partition_offset.size(); idx++) {
    partition_offset[idx] = xadj[idx + 1] - xadj[idx];
  }
}

auto DistributedCSRGraph::num_local_vertices() const -> std::size_t {
  return xadj.size() - 1;
}

auto DistributedCSRGraph::num_local_edges() const -> std::size_t {
  return adjncy.size();
}

auto DistributedCSRGraph::num_global_vertices() const -> std::size_t {
  return vtx_dist.back();
}

auto DistributedCSRGraph::num_global_edges() const -> std::size_t {
  return comm_->allreduce_single(kamping::send_buf(num_local_edges()),
                                 kamping::op(std::plus<>{}));
}

auto DistributedCSRGraph::is_local(VId const& v) const -> bool {
  return kagen::Graph::vertex_range.first <= v && v < kagen::Graph::vertex_range.second;
}

auto DistributedCSRGraph::to_local(VId const& v) const -> std::size_t {
  if (is_local(v)) {
    return v - kagen::Graph::vertex_range.first;
  }
  return v - this->vtx_dist[get_rank(v)];
}

auto DistributedCSRGraph::to_global(std::size_t idx) const -> std::size_t {
  return idx + kagen::Graph::vertex_range.first;
}

auto DistributedCSRGraph::get_rank(VId const& v) const -> kagen::PEID {
  auto iter = std::ranges::upper_bound(vtx_dist, v);
  return static_cast<int>(std::distance(vtx_dist.begin(), iter)) - 1;
}

auto DistributedCSRGraph::vertices() const
    -> decltype(std::views::iota(kagen::Graph::vertex_range.first,
                                 kagen::Graph::vertex_range.second)) {
  return std::views::iota(kagen::Graph::vertex_range.first,
                          kagen::Graph::vertex_range.second);
}

auto DistributedCSRGraph::neighbors(VId const& v) const -> std::span<const VId> {
  return neighborhood_span(v);
}

auto DistributedCSRGraph::degree(DistributedCSRGraph::VId const& v) const -> std::size_t {
  return neighborhood_span<NeighborPartition::full>(v).size();
}

auto format_as(DistributedCSRGraph const& G) -> std::string {
  auto range = G.vertices() | std::views::transform([&](auto const& v) {
                 return std::make_tuple(v, G.neighbors(v));
               });
  return fmt::format("{}", range);
}
}  // namespace kascade::graph
