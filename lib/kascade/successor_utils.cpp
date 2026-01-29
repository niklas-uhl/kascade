#include "kascade/successor_utils.hpp"

#include <ranges>

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <kamping/collectives/allreduce.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/utils/flatten.hpp>

#include "kascade/distribution.hpp"

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

auto leaves(std::span<const idx_t> succ_array,
            Distribution const& dist,
            kamping::Communicator<> const& comm) -> std::vector<idx_t> {
  absl::flat_hash_map<int, std::vector<idx_t>> requests;
  std::vector<bool> has_pred(succ_array.size(), false);
  for (idx_t i = 0; i < succ_array.size(); i++) {
    if (is_root(i, succ_array, dist, comm)) {
      continue;
    }
    auto succ = succ_array[i];
    auto owner = dist.get_owner(succ);
    if (owner == comm.rank()) {
      has_pred[dist.get_local_idx(succ, comm.rank())] = true;
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
      kamping::with_flattened(requests, comm.size()).call([&](auto... flattened) {
        return comm.alltoallv(std::move(flattened)...);
      });
  for (auto& pred : preds) {
    KASSERT(dist.get_owner(pred) == comm.rank());
    auto local_idx = dist.get_local_idx(pred, comm.rank());
    has_pred[local_idx] = true;
  }
  std::vector<idx_t> leaf_indices;
  for (idx_t i = 0; i < succ_array.size(); i++) {
    if (!has_pred[i] && !is_root(i, succ_array, dist, comm)) {
      leaf_indices.push_back(i);
    }
  }
  return leaf_indices;
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
}  // namespace kascade
