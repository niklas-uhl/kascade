#include "kascade/successor_utils.hpp"

#include <ranges>

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <kamping/collectives/allgather.hpp>
#include <kamping/collectives/allreduce.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/collectives/exscan.hpp>
#include <kamping/measurements/timer.hpp>
#include <kamping/types/unsafe/tuple.hpp>
#include <kamping/types/unsafe/utility.hpp>
#include <kamping/utils/flatten.hpp>
#include <spdlog/spdlog.h>

#include "grid_alltoall.hpp"
#include "kascade/assertion_levels.hpp"
#include "kascade/distribution.hpp"
#include "types.hpp"

namespace kascade {
auto is_list(std::span<const idx_t> succ_array, kamping::Communicator<> const& comm)
    -> bool {
  Distribution dist{succ_array.size(), comm};
  return is_list(succ_array, dist, comm);
}

/// Check if the given successor array represents a list (no node has more than one
/// predecessor)
auto is_list(std::span<const idx_t> succ_array,
             Distribution const& dist,
             kamping::Communicator<> const& comm) -> bool {
  absl::flat_hash_map<int, std::vector<idx_t>> requests;
  auto indices = std::views::iota(idx_t{0}, static_cast<idx_t>(succ_array.size())) |
                 std::views::transform([&](auto local_idx) {
                   return dist.get_global_idx(local_idx, comm.rank());
                 });
  for (auto [idx, succ] : std::views::zip(indices, succ_array)) {
    if (succ == idx) {
      continue;
    }
    auto owner = dist.get_owner_signed(succ);
    requests[owner].push_back(succ);
  }
  auto [send_buf, send_counts, send_displs] = kamping::flatten(requests, comm.size());
  requests.clear();
  auto recv_buf =
      comm.alltoallv(kamping::send_buf(send_buf), kamping::send_counts(send_counts),
                     kamping::send_displs(send_displs));
  // if we receive a duplicate request, it's not a list
  absl::flat_hash_set<idx_t> received;
  bool duplicate_found = false;
  for (auto succ : recv_buf) {
    if (!received.insert(succ).second) {
      duplicate_found = true;
      break;
    }
  }
  bool is_list = !duplicate_found;
  comm.allreduce(kamping::send_recv_buf(is_list), kamping::op(std::logical_and{}));
  return is_list;
}

auto is_root(std::size_t local_idx,
             std::span<const idx_t> succ_array,
             Distribution const& dist,
             kamping::Communicator<> const& comm) -> bool {
  return succ_array[local_idx] == dist.get_global_idx(local_idx, comm.rank());
}

LeafInfo::LeafInfo(std::span<const idx_t> succ_array,
                   Distribution const& dist_ref,
                   kamping::Communicator<> const& comm_ref)
    : has_pred_(succ_array.size(), false),
      dist_(&dist_ref),
      comm_(&comm_ref),
      num_local_leaves_(succ_array.size()) {
  absl::flat_hash_map<int, std::vector<idx_t>> requests;
  for (auto [global_idx, succ] :
       std::views::zip(dist_ref.global_indices(comm_ref.rank()), succ_array)) {
    if (succ == global_idx) {
      continue;
    }
    auto owner = dist_->get_owner(succ);
    if (owner == comm_->rank()) {
      auto succ_local_idx = dist_->get_local_idx(succ, comm_->rank());
      if (!has_pred_[succ_local_idx]) {
        num_local_leaves_--;
      }
      has_pred_[succ_local_idx] = true;
      continue;
    }
    requests[static_cast<int>(owner)].push_back(succ);
  }
  // de-duplicate
  for (auto& [dst, buf] : requests) {
    std::ranges::sort(buf);
    auto result = std::ranges::unique(buf);
    buf.erase(result.begin(), result.end());
  }
  auto preds =
      kamping::with_flattened(requests, comm_->size()).call([&](auto... flattened) {
        return comm_->alltoallv(std::move(flattened)...);
      });
  for (auto& pred : preds) {
    KASSERT(dist_->get_owner(pred) == comm_->rank());
    auto local_idx = dist_->get_local_idx(pred, comm_->rank());
    if (!has_pred_[local_idx]) {
      num_local_leaves_--;
    }
    has_pred_[local_idx] = true;
  }
};

/// a leaf that is also a root is not a leaf
auto LeafInfo::is_leaf(idx_t local_idx) const -> bool {
  return !has_pred_[local_idx];
};

auto LeafInfo::num_local_leaves() const -> std::size_t {
  return num_local_leaves_;
};

auto leaves(std::span<const idx_t> succ_array,
            Distribution const& dist,
            kamping::Communicator<> const& comm) -> std::vector<idx_t> {
  LeafInfo info{succ_array, dist, comm};
  return info.leaves() | std::ranges::to<std::vector>();
}

auto roots(std::span<const idx_t> succ_array,
           Distribution const& dist,
           kamping::Communicator<> const& comm) -> std::vector<idx_t> {
  auto local_indices = std::views::iota(idx_t{0}, static_cast<idx_t>(succ_array.size()));
  return local_indices | std::views::filter([&](auto local_idx) {
           return is_root(local_idx, succ_array, dist, comm);
         }) |
         std::ranges::to<std::vector>();
}

/// @return the original roots of the tree (which are now leaves)
auto reverse_list(std::span<const idx_t> succ_array,
                  std::span<const rank_t> dist_to_succ,
                  std::span<idx_t> pred_array,
                  std::span<rank_t> dist_to_pred,
                  Distribution const& dist,
                  kamping::Communicator<> const& comm,
                  bool use_grid_comm /* = false */
                  ) -> std::vector<idx_t> {
  KASSERT(is_list(succ_array, dist, comm), kascade::assert::with_communication);
  struct message_type {
    idx_t pred;
    idx_t succ;
    rank_t dist_pred_succ;
  };
  kamping::measurements::timer().start("build_messages");
  absl::flat_hash_map<int, std::vector<message_type>> requests;
  auto roots = dist.local_indices(comm.rank()) | std::views::filter([&](idx_t local_idx) {
                 return is_root(local_idx, succ_array, dist, comm);
               }) |
               std::ranges::to<std::vector>();
  auto request_range =
      dist.local_indices(comm.rank()) | std::views::filter([&](idx_t local_idx) {
        // filter non-roots
        return !is_root(local_idx, succ_array, dist, comm);
      }) |
      std::views::transform([&](idx_t local_idx) {
        auto global_idx = dist.get_global_idx(local_idx, comm.rank());
        auto succ = succ_array[local_idx];
        auto weight = dist_to_succ[local_idx];
        auto owner = dist.get_owner(succ);
        return std::pair{
            owner,
            message_type{.pred = global_idx, .succ = succ, .dist_pred_succ = weight}};
      });
  SPDLOG_DEBUG("[reverse_list] Using grid_communication={} for alltoall", use_grid_comm);
  AlltoallDispatcher<message_type> dispatcher(use_grid_comm, comm);
  auto recv_buf = dispatcher.alltoallv(request_range);
  // initially, every node is its own predecessor
  std::ranges::copy(dist.global_indices(comm.rank()), pred_array.begin());
  std::ranges::fill(dist_to_pred, 0);
  for (auto const& msg : recv_buf) {
    auto local_idx = dist.get_local_idx(msg.succ, comm.rank());
    pred_array[local_idx] = msg.pred;
    dist_to_pred[local_idx] = msg.dist_pred_succ;
  }
  kamping::measurements::timer().stop();
  KASSERT(is_list(pred_array, dist, comm), kascade::assert::with_communication);
  return roots;
}

namespace {

auto make_graph(std::ranges::forward_range auto&& recv_edges,
                Distribution const& dist,
                kamping::Communicator<> const& comm) -> graph::DistributedCSRGraph {
  kagen::Graph kagen_graph;
  kagen_graph.vertex_range.first = dist.index_range_begin(comm.rank());
  kagen_graph.vertex_range.second = dist.index_range_end(comm.rank());
  kagen_graph.xadj.clear();
  kagen_graph.xadj.resize(dist.get_local_size(comm.rank()) + 1, 0);  // edge offsets
  kagen_graph.adjncy.clear();                                        // edges
  kagen_graph.adjncy.resize(recv_edges.size());                      // edges
  kagen_graph.edge_weights.resize(recv_edges.size());                // edges

  for (const auto& [src, dst, w] : recv_edges) {
    ++kagen_graph.xadj[dist.get_local_idx(src, comm.rank())];
  }
  std::inclusive_scan(kagen_graph.xadj.begin(), kagen_graph.xadj.end(),
                      kagen_graph.xadj.begin());
  for (const auto& [src, dst, w] : recv_edges) {
    auto local_src = dist.get_local_idx(src, comm.rank());
    auto insert_pos = --kagen_graph.xadj[local_src];
    kagen_graph.adjncy[insert_pos] = dst;
    kagen_graph.edge_weights[insert_pos] = w;
  }
  return graph::DistributedCSRGraph(std::move(kagen_graph), comm);
}

}  // namespace

/// @returns the reversed tree and the original roots of the tree (which are now leaves)
auto reverse_rooted_tree(std::span<idx_t const> succ_array,
                         std::span<rank_t const> dist_to_succ,
                         Distribution const& dist,
                         kamping::Communicator<> const& comm,
                         bool add_back_edge) -> ReversedTree {
  namespace kmp = kamping::params;
  struct Edge {
    idx_t src;
    idx_t dst;
    rank_t weight;
  };
  std::vector<idx_t> roots;
  absl::flat_hash_map<int, std::vector<Edge>> send_bufs;
  if (add_back_edge) {
    send_bufs[comm.rank_signed()].reserve(succ_array.size());
  }
  for (idx_t i = 0; i < succ_array.size(); ++i) {
    if (is_root(i, succ_array, dist, comm)) {
      roots.push_back(i);
      continue;
    }
    auto src = dist.get_global_idx(i, comm.rank());
    auto dst = succ_array[i];
    int target_rank = dist.get_owner_signed(dst);
    if (add_back_edge) {
      send_bufs[comm.rank_signed()].emplace_back(src, dst, dist_to_succ[i]);
    }
    send_bufs[target_rank].emplace_back(dst, src, dist_to_succ[i]);
  }
  auto [send_buf, send_counts, send_displs] = kamping::flatten(send_bufs, comm.size());
  auto recv_edges = comm.alltoallv(kmp::send_buf(send_buf), kmp::send_counts(send_counts),
                                   kmp::send_displs(send_displs));
  return ReversedTree{.num_proxy_vertices = 0,
                      .parent_array = succ_array,
                      .tree = make_graph(std::move(recv_edges), dist, comm)};
}

auto reverse_rooted_tree(std::span<const idx_t> succ_array,
                         std::span<const rank_t> dist_to_succ,
                         Distribution const& dist,
                         kamping::Communicator<> const& comm,
                         bool add_back_edge,
                         resolve_high_degree_tag /* tag */) -> ReversedTree {
  namespace kmp = kamping::params;
  struct Edge {
    idx_t src;
    idx_t dst;
    rank_t weight;
  };

  auto encode_local_proxy_id = [](idx_t local_proxy_id) {
    return -(static_cast<std::int64_t>(local_proxy_id) + 1);
  };
  auto decode_local_proxy_id = [](std::int64_t encoded_id) {
    KASSERT(encoded_id < 0);
    return static_cast<idx_t>(-(encoded_id + 1));
  };

  // count remote parents
  absl::flat_hash_map<idx_t, std::int64_t> remote_info;
  for (idx_t succ : succ_array) {
    if (dist.get_owner(succ) != comm.rank()) {
      ++remote_info[succ];
    }
  }

  // compute proxy vertices
  std::size_t const threshold = 1;
  std::size_t next_proxy_id = 0;
  for (auto& kv : remote_info) {
    if (kv.second > static_cast<std::int64_t>(threshold)) {
      kv.second = encode_local_proxy_id(next_proxy_id);
      ++next_proxy_id;
    }
  }
  std::size_t num_proxies = next_proxy_id;
  std::vector<idx_t> parent_array(succ_array.size() + num_proxies);

  Distribution new_distribution(succ_array.size() + num_proxies, comm);

  auto to_new_global = [&](idx_t global_id, std::size_t owner) {
    const auto local = dist.get_local_idx(global_id, owner);
    return new_distribution.get_global_idx(local, owner);
  };

  auto proxy_to_global = [&](std::size_t local_proxy_id) {
    return new_distribution.get_global_idx(
        dist.get_local_size(comm.rank()) + static_cast<idx_t>(local_proxy_id),
        comm.rank());
  };

  // vertex -> {parent or proxy}
  absl::flat_hash_map<int, std::vector<Edge>> send_bufs;
  if (add_back_edge) {
    send_bufs[comm.rank_signed()].reserve(succ_array.size() + (2 * num_proxies));
  }
  for (idx_t i = 0; i < succ_array.size(); ++i) {
    if (is_root(i, succ_array, dist, comm)) {
      parent_array[i] = to_new_global(succ_array[i], comm.rank());
      continue;
    }
    const idx_t v = new_distribution.get_global_idx(i, comm.rank());
    const idx_t parent = succ_array[i];
    const rank_t weight = dist_to_succ[i];

    auto it = remote_info.find(parent);
    if (it != remote_info.end() && it->second < 0) {
      // add edge: v -> local proxy
      auto local_proxy_id = static_cast<idx_t>(-(it->second + 1));
      auto proxy_id = proxy_to_global(local_proxy_id);
      send_bufs[comm.rank_signed()].emplace_back(proxy_id, v, weight);
      parent_array[i] = proxy_id;
      if (add_back_edge) {
        send_bufs[comm.rank_signed()].emplace_back(v, proxy_id, weight);
      }
    } else {
      // add edge: v -> parent
      auto parent_owner = dist.get_owner(parent);
      auto new_parent_name = to_new_global(parent, parent_owner);
      parent_array[i] = new_parent_name;
      send_bufs[static_cast<int>(parent_owner)].emplace_back(new_parent_name, v, weight);
      if (add_back_edge) {
        send_bufs[comm.rank_signed()].emplace_back(v, new_parent_name, weight);
      }
    }
  }

  // add edges: proxy -> real parent
  // these edges have weight 0, as they connect a newly inserted proxy vertex to the
  // actual parent vertex
  for (auto const& [parent, proxy_info] : remote_info) {
    if (proxy_info < 0) {
      const auto local_proxy_id = decode_local_proxy_id(proxy_info);
      const auto owner = dist.get_owner(parent);
      auto new_parent_name = to_new_global(parent, owner);
      parent_array[succ_array.size() + local_proxy_id] = new_parent_name;
      send_bufs[static_cast<int>(owner)].emplace_back(new_parent_name,
                                                      proxy_to_global(local_proxy_id), 0);

      if (add_back_edge) {
        send_bufs[comm.rank_signed()].emplace_back(proxy_to_global(local_proxy_id),
                                                   to_new_global(parent, owner), 0);
      }
    }
  }

  auto [send_buf, send_counts, send_displs] = kamping::flatten(send_bufs, comm.size());
  auto recv_edges = comm.alltoallv(kmp::send_buf(send_buf), kmp::send_counts(send_counts),
                                   kmp::send_displs(send_displs));

  return ReversedTree{.num_proxy_vertices = num_proxies,
                      .parent_array = std::move(parent_array),
                      .tree = make_graph(std::move(recv_edges), new_distribution, comm)};
}

auto trace_successor_list(std::span<const idx_t> root_array,
                          std::span<const rank_t> rank_array,
                          const kamping::Communicator<>& comm) -> std::string {
  namespace kmp = kamping::params;

  const auto global_root_array = comm.allgatherv(kmp::send_buf(root_array));
  const auto global_rank_array = comm.allgatherv(kmp::send_buf(rank_array));

  const auto size = std::size(global_root_array);

  std::vector<bool> is_leaf(static_cast<std::size_t>(size), true);
  for (idx_t idx = 0; idx < size; ++idx) {
    const auto parent = global_root_array[static_cast<std::size_t>(idx)];
    KASSERT(parent < size);
    is_leaf[static_cast<std::size_t>(parent)] = false;
  }

  std::ostringstream out;
  for (idx_t idx = 0; idx < size; ++idx) {
    if (!is_leaf[static_cast<std::size_t>(idx)]) {
      continue;
    }
    auto cur = idx;
    for (;;) {
      out << '(' << cur << ", r: " << global_rank_array[static_cast<std::size_t>(cur)]
          << ")\n";

      const auto parent = global_root_array[static_cast<std::size_t>(cur)];
      if (parent == cur) {
        break;
      }
      cur = parent;
    }
  }
  return out.str();
}
}  // namespace kascade
