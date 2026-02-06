#include "kascade/eulertour.hpp"

#include <absl/container/flat_hash_set.h>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/types/unsafe/tuple.hpp>
#include <kamping/types/unsafe/utility.hpp>
#include <spdlog/spdlog.h>

#include "distribution.hpp"
#include "kascade/graph/graph.hpp"
#include "kascade/types.hpp"
#include "list_ranking.hpp"
#include "pointer_doubling.hpp"
#include "sparse_ruling_set.hpp"
#include "successor_utils.hpp"

namespace kascade {
auto compute_euler_tour(graph::DistributedCSRGraph const& forest,
                        std::span<idx_t> parent_array,
                        kamping::Communicator<> const& comm) -> EulerTour {
  using Edge = EulerTour::Edge;
  namespace kmp = kamping::params;

  auto assign_weight = [&](idx_t v, idx_t u) {
    if (parent_array[forest.to_local(v)] == u) {
      return 1;
    }
    return -1;
  };

  absl::flat_hash_set<idx_t> is_root;
  for (const auto& v : forest.vertices()) {
    if (parent_array[forest.to_local(v)] == v) {
      is_root.emplace(v);
    }
  }

  Distribution euler_tour_dist(forest.num_local_edges() + is_root.size(), comm);
  absl::flat_hash_map<Edge, idx_t> edge_to_index;
  absl::flat_hash_map<int, std::vector<std::tuple<idx_t, idx_t, idx_t>>> send_bufs;
  std::vector<std::int64_t> dist_to_root(forest.num_local_edges() + is_root.size(), 0);
  std::vector<bool> is_upward_edge(forest.num_local_edges() + is_root.size());
  std::vector<Edge> index_to_edge(forest.num_local_edges() + is_root.size());
  std::size_t next_id = 0;
  idx_t root_index = forest.num_local_edges();
  for (auto const& v : forest.vertices()) {
    std::size_t const degree = forest.degree(v);
    auto neighbors = forest.neighbors(v);
    bool const is_v_root = is_root.contains(v);
    for (std::size_t i = 0; i < degree; ++i) {
      const auto u_prev = neighbors[(i + degree - 1) % degree];
      const auto u = neighbors[i];
      edge_to_index[std::make_pair(v, u)] = next_id;
      index_to_edge[next_id] = std::make_pair(v, u);
      // succ[std::make_pair(u_prev, v)] = std::make_pair(v, u);
      // edge (v,u) has ID next_id
      dist_to_root[next_id] = assign_weight(v, u);
      is_upward_edge[next_id] = (dist_to_root[next_id] > 0);
      int owner_u_prev = forest.get_rank(u_prev);
      if (is_v_root && i == 0) {
        // break up list for each tree
        send_bufs[owner_u_prev].emplace_back(
            u_prev, v, euler_tour_dist.get_global_idx(root_index, comm.rank()));
        dist_to_root[root_index] = 0;
        is_upward_edge[root_index] = false;
        edge_to_index[std::make_pair(0, 0)] = root_index;
        index_to_edge[root_index] = std::make_pair(0, 0);
        ++root_index;
      } else {
        send_bufs[owner_u_prev].emplace_back(
            u_prev, v, euler_tour_dist.get_global_idx(next_id, comm.rank()));
      }
      next_id++;
    }
  }

  auto [send_buf, send_counts, send_displs] = kamping::flatten(send_bufs, comm.size());
  auto recv_buf = comm.alltoallv(kmp::send_buf(send_buf), kmp::send_counts(send_counts),
                                 kmp::send_displs(send_displs));
  std::vector<idx_t> succ_array(forest.num_local_edges() + is_root.size());
  std::ranges::iota(succ_array, euler_tour_dist.get_exclusive_prefix(comm.rank()));
  for (const auto& [u, v, succ] : recv_buf) {
    auto it = edge_to_index.find(std::make_pair(u, v));
    KASSERT(it != edge_to_index.end());
    succ_array[it->second] = succ;
  }
  return EulerTour{.succ_array = std::move(succ_array),
                   .is_upward_edge = std::move(is_upward_edge),
                   .rank_array = std::move(dist_to_root),
                   .edge_to_index = std::move(edge_to_index),
                   .index_to_edge = std::move(index_to_edge),
                   .distribution = std::move(euler_tour_dist)};
}

auto format_as(std::pair<EulerTour const&, kamping::Communicator<> const&> obj)
    -> std::string {
  namespace kmp = kamping::params;
  using Edge = EulerTour::Edge;

  auto const& [euler_tour, comm] = obj;

  // 1. Build per-rank request buffers for successor ids
  absl::flat_hash_map<int, std::vector<idx_t>> rank_to_requests;

  for (auto const succ : euler_tour.succ_array) {
    auto const owner = euler_tour.distribution.get_owner(succ);
    rank_to_requests[static_cast<int>(owner)].push_back(succ);
  }

  // exchange requests
  auto [send_buf, send_counts, send_displs] =
      kamping::flatten(rank_to_requests, comm.size());
  auto [recv_buf, recv_counts] =
      comm.alltoallv(kmp::send_buf(send_buf), kmp::send_counts(send_counts),
                     kmp::send_displs(send_displs), kmp::recv_counts_out());

  std::vector<std::pair<Edge, idx_t>> replies =
      recv_buf | std::views::transform([&](idx_t request) {
        auto const local_id = euler_tour.distribution.get_local_idx(request, comm.rank());
        KASSERT(local_id < euler_tour.index_to_edge.size());
        return std::pair<Edge, idx_t>{euler_tour.index_to_edge[local_id], request};
      }) |
      std::ranges::to<std::vector>();

  // 3. Send replies back
  auto recv_replies =
      comm.alltoallv(kmp::send_buf(replies), kmp::send_counts(recv_counts));

  absl::flat_hash_map<idx_t, Edge> reply_map;
  reply_map.reserve(recv_replies.size());
  for (auto const& [edge, id] : recv_replies) {
    reply_map.emplace(id, edge);
  }

  // 4. Build a lazy range view instead of a concrete output container
  auto triplet_view = std::views::iota(std::size_t{0}, euler_tour.succ_array.size()) |
                      std::views::transform([&](std::size_t i) {
                        auto const& elem = euler_tour.index_to_edge[i];

                        auto const it = reply_map.find(euler_tour.succ_array[i]);
                        KASSERT(it != reply_map.end());
                        auto const& succ = it->second;

                        auto const dist = euler_tour.rank_array[i];
                        return std::tuple<Edge, Edge, std::int64_t>{elem, succ, dist};
                      });

  auto global_id_edge_view =
      std::views::iota(std::size_t{0}, euler_tour.succ_array.size()) |
      std::views::transform([&](std::size_t i) {
        auto const& elem = euler_tour.index_to_edge[i];
        return std::pair<idx_t, Edge>{
            euler_tour.distribution.get_global_idx(i, comm.rank()), elem};
      });

  return fmt::format("{},\nmap: {}", triplet_view, global_id_edge_view);
}
namespace {
auto request_original_succesors(const EulerTour& euler_tour,
                                kamping::Communicator<> const& comm) {
  namespace kmp = kamping::params;
  KASSERT(euler_tour.is_upward_edge.size() == euler_tour.succ_array.size());
  absl::flat_hash_set<idx_t> requests;
  for (std::size_t i = 0; i < euler_tour.succ_array.size(); ++i) {
    if (euler_tour.is_upward_edge[i]) {
      auto succ = euler_tour.succ_array[i];
      requests.emplace(succ);
    }
  }
  absl::flat_hash_map<int, std::vector<idx_t>> send_bufs;
  for (const auto request : requests) {
    int const owner = euler_tour.distribution.get_owner_signed(request);
    send_bufs[owner].push_back(request);
  }
  // exchange requests
  auto [send_buf, send_counts, send_displs] = kamping::flatten(send_bufs, comm.size());
  auto [recv_buf, recv_counts] =
      comm.alltoallv(kmp::send_buf(send_buf), kmp::send_counts(send_counts),
                     kmp::send_displs(send_displs), kmp::recv_counts_out());

  std::vector<std::pair<idx_t, idx_t>> replies =
      recv_buf | std::views::transform([&](idx_t request) {
        auto const local_id = euler_tour.distribution.get_local_idx(request, comm.rank());
        KASSERT(local_id < euler_tour.index_to_edge.size());
        return std::pair<idx_t, idx_t>{request,
                                       euler_tour.index_to_edge[local_id].second};
      }) |
      std::ranges::to<std::vector>();
  auto recv_replies =
      comm.alltoallv(kmp::send_buf(replies), kmp::send_counts(recv_counts));

  absl::flat_hash_map<idx_t, idx_t> reply_map;
  reply_map.reserve(recv_replies.size());
  for (auto const& [succ, vertex] : recv_replies) {
    reply_map.emplace(succ, vertex);
  }
  return reply_map;
}

}  // namespace
void map_euler_tour_back(EulerTour const& euler_tour,
                         std::span<idx_t> root_array,
                         std::span<rank_t> rank_array,
                         kamping::Communicator<> const& comm) {
  namespace kmp = kamping::params;
  KASSERT(root_array.size() == rank_array.size());

  Distribution org_distribution(root_array.size(), comm);
  auto org_successors = request_original_succesors(euler_tour, comm);
  absl::flat_hash_map<int, std::vector<std::tuple<idx_t, idx_t, rank_t>>> send_bufs;
  for (std::size_t i = 0; i < euler_tour.succ_array.size(); ++i) {
    if (euler_tour.is_upward_edge[i]) {
      auto succ = euler_tour.succ_array[i];
      auto it = org_successors.find(succ);
      KASSERT(it != org_successors.end());
      auto root = it->second;
      auto rank = euler_tour.rank_array[i];
      auto v = euler_tour.index_to_edge[i].first;
      int owner = org_distribution.get_owner_signed(v);
      send_bufs[owner].emplace_back(v, root, rank);
    }
  }

  auto [send_buf, send_counts, send_displs] = kamping::flatten(send_bufs, comm.size());
  auto recv_buf = comm.alltoallv(kmp::send_buf(send_buf), kmp::send_counts(send_counts),
                                 kmp::send_displs(send_displs));

  for (auto [v, root, rank] : recv_buf) {
    auto local_v = org_distribution.get_local_idx(v, comm.rank());
    KASSERT(local_v < org_distribution.get_local_size(comm.rank()));
    root_array[local_v] = root;
    rank_array[local_v] = rank;
  }
}

namespace {
void rank_via_euler_tour_select_algorithm(EulerTourConfig const& config,
                                          EulerTour& euler_tour,
                                          kamping::Communicator<> const& comm) {
  switch (config.algorithm) {
    case Algorithm::GatherChase: {
      rank_on_root(euler_tour.succ_array, euler_tour.rank_array, euler_tour.distribution,
                   comm);
      break;
    }
    case Algorithm::PointerDoubling: {
      kascade::pointer_doubling(std::any_cast<PointerDoublingConfig>(config.algo_config),
                                euler_tour.succ_array, euler_tour.rank_array,
                                euler_tour.distribution, comm);
      break;
    }
    case Algorithm::AsyncPointerDoubling: {
      kascade::async_pointer_doubling(
          std::any_cast<AsyncPointerChasingConfig>(config.algo_config),
          euler_tour.succ_array, euler_tour.rank_array, euler_tour.distribution, comm);
      break;
    }
    case Algorithm::SparseRulingSet: {
      kascade::sparse_ruling_set(std::any_cast<SparseRulingSetConfig>(config.algo_config),
                                 euler_tour.succ_array, euler_tour.rank_array,
                                 euler_tour.distribution, comm);
      break;
    }
    case Algorithm::RMAPointerDoubling: {
      kascade::rma_pointer_doubling(
          std::any_cast<RMAPointerChasingConfig>(config.algo_config),
          euler_tour.succ_array, euler_tour.rank_array, euler_tour.distribution, comm);
      break;
    }
    default: {
      throw std::runtime_error("No suitable algorithm selected");
    }
  }
}

}  // namespace

void rank_via_euler_tour(EulerTourConfig const& config,
                         std::span<idx_t> succ_array,
                         std::span<rank_t> rank_array,
                         Distribution const& dist,
                         kamping::Communicator<> const& comm) {
  auto tree = reverse_rooted_tree(succ_array, rank_array, dist, comm, true);
  auto euler_tour = compute_euler_tour(tree, succ_array, comm);
  rank_via_euler_tour_select_algorithm(config, euler_tour, comm);
  map_euler_tour_back(euler_tour, succ_array, rank_array, comm);
}

}  // namespace kascade
