// taken and adapted from kascadev2, Tim Niklas Uhl, 2026
#pragma once

#include <concepts>
#include <ranges>
#include <type_traits>

namespace kascade::graph {

template <typename G>
struct vertex_id {
  using type = G::VId;
};

template <typename G>
using vertex_id_t = typename vertex_id<std::remove_cvref_t<G>>::type;

template <typename G>
struct vertex_range {
  using type = decltype(std::declval<G>().vertices());
};

template <typename G>
using vertex_range_t = typename vertex_range<std::remove_cvref_t<G>>::type;

template <typename G>
struct neighbor_range {
  using type = decltype(std::declval<G>().neighbors(std::declval<vertex_id_t<G>>()));
};

template <typename G>
using neighbor_range_t = typename neighbor_range<std::remove_cvref_t<G>>::type;

template <typename G>
concept Graph = requires { typename vertex_id_t<G>; } &&
                requires(G graph, vertex_id_t<G> v) {
                  { graph.vertices() } -> std::ranges::range;
                  { graph.neighbors(v) } -> std::ranges::range;
                } &&
                std::same_as<typename std::ranges::range_value_t<vertex_range_t<G>>,
                             vertex_id_t<G>> &&
                std::same_as<typename std::ranges::range_value_t<neighbor_range_t<G>>,
                             vertex_id_t<G>>;

template <typename G>
concept DistributedGraph = Graph<G> && requires(G graph, vertex_id_t<G> v) {
  { graph.get_rank(v) } -> std::convertible_to<int>;
  { graph.is_local(v) } -> std::convertible_to<bool>;
  { graph.to_local(v) } -> std::convertible_to<std::size_t>;
  { graph.num_local_vertices() } -> std::convertible_to<std::size_t>;
};

auto to_local(Graph auto const& G, vertex_id_t<decltype(G)> v) -> std::size_t {
  if constexpr (DistributedGraph<decltype(G)>) {
    return G.to_local(v);
  } else {
    return static_cast<std::size_t>(v);
  }
}
}  // namespace kascade::graph
