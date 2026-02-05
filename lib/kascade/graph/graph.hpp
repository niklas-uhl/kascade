// taken and adapted from kascadev2, Tim Niklas Uhl, 2026
#pragma once

#include <absl/container/flat_hash_map.h>  // for flat_hash_map
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <kagen/kagen.h>  // for Graph, PEID, SInt
#include <mpi.h>
#include <algorithm>                         // for __is_sorted_fn, is_sorted
#include <cassert>                           // for assert
#include <cstddef>                           // for size_t
#include <cstdint>                           // for uint8_t
#include <functional>                        // for identity, less
#include <iterator>                          // for pair, distance
#include <kamping/collectives/alltoall.hpp>  // IWYU pragma: keep
#include <kamping/communicator.hpp>          // for Communicator
#include <kamping/mpi_datatype.hpp>
#include <kamping/utils/flatten.hpp>  // for with_flattened
#include <ranges>                     // for _Iota, iota, iota_view, views
#include <span>                       // for span
#include <type_traits>                // for invoke_result_t
#include <utility>                    // for move, make_pair, pair
#include <vector>                     // for vector

namespace kascade::graph {

enum class NeighborPartition : std::uint8_t { first, second, full };

class DistributedCSRGraph : kagen::Graph {
public:
  using VId = kagen::SInt;
  DistributedCSRGraph(kagen::Graph G, kamping::Communicator<> const& comm);

  [[nodiscard]] auto num_local_vertices() const -> std::size_t;

  [[nodiscard]] auto num_local_edges() const -> std::size_t;

  [[nodiscard]] auto num_global_vertices() const -> std::size_t;

  [[nodiscard]] auto num_global_edges() const -> std::size_t;

  [[nodiscard]] auto is_local(VId const& v) const -> bool;

  [[nodiscard]] auto get_rank(VId const& v) const -> kagen::PEID;

  [[nodiscard]] auto to_local(VId const& v) const -> std::size_t;

  [[nodiscard]] auto to_global(std::size_t idx) const -> std::size_t;

  [[nodiscard]] auto degree(VId const& v) const -> std::size_t;

  [[nodiscard]] auto neighbors(VId const& v) const -> std::span<const VId>;

  template <NeighborPartition part = NeighborPartition::full,
            class Comp = std::ranges::less,
            class Proj = std::identity>
  void sort_neighborhood(VId const& v, Comp comp = {}, Proj proj = {}) {
    std::ranges::sort(neighborhood_span<part>(v), std::move(comp), std::move(proj));
  }

  template <class Proj = std::identity, class Pred>
  void partition_neighborhood(VId const& v, Pred pred, Proj proj = {}) {
    auto tail =
        std::ranges::partition(neighborhood_span(v), std::move(pred), std::move(proj));
    partition_offset[to_local(v)] =
        std::distance(neighborhood_span(v).begin(), tail.begin());
  }

  template <class VertexToData, class MetaData = std::invoke_result_t<VertexToData, VId>>
  [[nodiscard]] auto exchange_ghost_metadata(VertexToData vertex_to_data) const
      -> std::vector<std::pair<VId, MetaData>> {
    std::vector<int> last_rank(this->num_local_vertices(), -1);
    absl::flat_hash_map<int, std::vector<std::pair<DistributedCSRGraph::VId, MetaData>>>
        ghost_data_buffer;
    for (auto u : this->vertices()) {
      assert(std::ranges::is_sorted(this->neighbors(u)));
      auto u_local = this->to_local(u);
      MetaData u_data = vertex_to_data(u);
      for (auto v : this->neighbors(u)) {
        if (!this->is_local(v)) {
          auto v_rank = this->get_rank(v);
          if (last_rank[u_local] != v_rank) {
            ghost_data_buffer[v_rank].emplace_back(u, u_data);
            last_rank[u_local] = v_rank;
          }
        }
      }
    }
    auto recv_buf = kamping::with_flattened(ghost_data_buffer, comm_->size())
                        .call([&](auto... flattened) {
                          return comm_->alltoallv(std::move(flattened)...);
                        });
    return recv_buf;
  }

  [[nodiscard]] auto vertices() const
      -> decltype(std::views::iota(kagen::Graph::vertex_range.first,
                                   kagen::Graph::vertex_range.second));

  [[nodiscard]] auto vertex_range() const -> auto const& {
    return kagen::Graph::vertex_range;
  }

  [[nodiscard]] auto vertex_distribution() const -> auto const& { return vtx_dist; }

  template <typename T>
  [[nodiscard]] auto to_edge_list() const -> std::vector<std::pair<T, T>> {
    std::vector<std::pair<T, T>> edge_list;
    edge_list.reserve(adjncy.size());

    for (kagen::SInt u = 0; u + 1 < xadj.size(); ++u) {
      for (kagen::SInt e = xadj[u]; e < xadj[u + 1]; ++e) {
        edge_list.emplace_back(static_cast<T>(u + vertex_range().first),
                               static_cast<T>(adjncy[e]));
      }
    }

    return edge_list;
  }

  [[nodiscard]] auto comm() const -> kamping::Communicator<> const& { return *comm_; }

private:
  template <NeighborPartition part = NeighborPartition::full>
  [[nodiscard]] auto neighborhood_begin_end(VId const& v) const -> std::pair<VId, VId> {
    auto local_vertex_id = v - kagen::Graph::vertex_range.first;
    auto nb_begin = xadj[local_vertex_id];
    if constexpr (part == NeighborPartition::second) {
      nb_begin += partition_offset[local_vertex_id];
    }
    auto nb_end = xadj[local_vertex_id + 1];
    if constexpr (part == NeighborPartition::first) {
      nb_end = nb_begin + partition_offset[local_vertex_id];
    }
    return std::make_pair(nb_begin, nb_end);
  }

  template <NeighborPartition part = NeighborPartition::full>
  [[nodiscard]] auto neighborhood_span(VId const& v) const -> std::span<const VId> {
    auto [nb_begin, nb_end] = neighborhood_begin_end<part>(v);
    return std::span(adjncy).subspan(nb_begin, nb_end - nb_begin);
  }

  template <NeighborPartition part = NeighborPartition::full>
  auto neighborhood_span(VId const& v) -> std::span<VId> {
    auto [nb_begin, nb_end] = neighborhood_begin_end<part>(v);
    return std::span(adjncy).subspan(nb_begin, nb_end - nb_begin);
  }

  std::vector<VId> vtx_dist;
  std::vector<VId> partition_offset;
  kamping::Communicator<> const* comm_;
};

auto format_as(DistributedCSRGraph const& G) -> std::string;

}  // namespace kascade
