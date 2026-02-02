#include "kascade/successor_utils.hpp"

#include <ranges>

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <kamping/collectives/allreduce.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/utils/flatten.hpp>

#include "kascade/assertion_levels.hpp"
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

auto invert_list(std::span<const idx_t> succ_array,
                 std::span<const idx_t> dist_to_succ,
                 std::span<idx_t> pred_array,
                 std::span<idx_t> dist_to_pred,
                 Distribution const& dist,
                 kamping::Communicator<> const& comm) -> void {
  KASSERT(is_list(succ_array, dist, comm), kascade::assert::with_communication);
  struct message_type {
    idx_t pred;
    idx_t succ;
    idx_t dist_pred_succ;
  };
  absl::flat_hash_map<int, std::vector<message_type>> requests;
  for (auto [global_idx, succ, weight] :
       std::views::zip(dist.global_indices(comm.rank()), succ_array, dist_to_succ)) {
    if (succ == global_idx) {
      continue;
    }
    auto owner = dist.get_owner(succ);
    requests[static_cast<int>(owner)].push_back(
        message_type{.pred = global_idx, .succ = succ, .dist_pred_succ = weight});
  }
  auto [send_buf, send_counts, send_displs] = kamping::flatten(requests, comm.size());
  requests.clear();
  auto recv_buf =
      comm.alltoallv(kamping::send_buf(send_buf), kamping::send_counts(send_counts),
                     kamping::send_displs(send_displs));
  // initially, every node is its own predecessor
  std::ranges::copy(dist.global_indices(comm.rank()), pred_array.begin());
  std::ranges::fill(dist_to_pred, 0);
  for (auto const& msg : recv_buf) {
    auto local_idx = dist.get_local_idx(msg.succ, comm.rank());
    pred_array[local_idx] = msg.pred;
    dist_to_pred[local_idx] = msg.dist_pred_succ;
  }
  KASSERT(is_list(pred_array, dist, comm), kascade::assert::with_communication);
};

}  // namespace kascade
