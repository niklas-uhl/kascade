#pragma once

#include <concepts>
#include <ranges>
#include <span>
#include <vector>

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/communicator.hpp>
#include <kamping/utils/flatten.hpp>

#include "kascade/distribution.hpp"
#include "kascade/types.hpp"

namespace kascade {

auto is_list(std::span<const idx_t> succ_array, kamping::Communicator<> const& comm)
    -> bool;

auto is_list(std::span<const idx_t> succ_array,
             Distribution const& dist,
             kamping::Communicator<> const& comm) -> bool;

auto invert_list(std::span<const idx_t> succ_array,
                 std::span<const idx_t> dist_to_succ,
                 std::span<idx_t> pred_array,
                 std::span<idx_t> dist_to_pred,
                 Distribution const& dist,
                 kamping::Communicator<> const& comm) -> void;

auto is_root(std::size_t local_idx,
             std::span<const idx_t> succ_array,
             Distribution const& dist,
             kamping::Communicator<> const& comm) -> bool;

struct LeafInfo {
private:
  std::vector<bool> has_pred_;  // per-local index
  absl::flat_hash_set<idx_t> has_pred_set_;
  Distribution const* dist_;
  kamping::Communicator<> const* comm_;
  std::size_t num_local_leaves_;

public:
  LeafInfo(std::span<const idx_t> succ_array,
           Distribution const& dist_ref,
           kamping::Communicator<> const& comm_ref);

  LeafInfo(std::span<const idx_t> succ_array,
           Distribution const& dist_ref,
           IndexRange auto const& active_local_indices,
           std::predicate<idx_t> auto&& follow_successor,
           kamping::Communicator<> const& comm_ref)
      : has_pred_(succ_array.size(), false),
        dist_(&dist_ref),
        comm_(&comm_ref),
        num_local_leaves_(std::ranges::size(active_local_indices)) {
    absl::flat_hash_map<int, std::vector<idx_t>> requests;
    for (idx_t local_index : active_local_indices) {
      auto global_idx = dist_->get_global_idx(local_index, comm_->rank());
      auto succ = succ_array[local_index];
      if (succ == global_idx || !follow_successor(succ)) {
        continue;
      }
      if (dist_->is_local(succ, comm_->rank())) {
        auto succ_local_idx = dist_->get_local_idx(succ, comm_->rank());
        if (!has_pred_[succ_local_idx]) {
          num_local_leaves_--;
        }
        has_pred_[succ_local_idx] = true;
        continue;
      }
      auto owner = dist_->get_owner(succ);
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
      KASSERT(dist_->is_local(pred, comm_->rank()));
      auto local_idx = dist_->get_local_idx(pred, comm_->rank());
      if (!has_pred_[local_idx]) {
        num_local_leaves_--;
      }
      has_pred_[local_idx] = true;
    }
  }

  [[nodiscard]] auto is_leaf(idx_t local_idx) const -> bool;

  [[nodiscard]] auto num_local_leaves() const -> std::size_t;
  [[nodiscard]] auto leaves() const {
    auto indices = std::views::iota(idx_t{0}, static_cast<idx_t>(has_pred_.size()));

    return indices |
           std::views::filter([&](auto local_idx) { return this->is_leaf(local_idx); });
  }
};

auto roots(std::span<const idx_t> succ_array,
           Distribution const& dist,
           kamping::Communicator<> const& comm) -> std::vector<idx_t>;

auto leaves(std::span<const idx_t> succ_array,
            Distribution const& dist,
            kamping::Communicator<> const& comm) -> std::vector<idx_t>;
}  // namespace kascade
