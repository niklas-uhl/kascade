#pragma once

#include <ranges>
#include <span>
#include <vector>

#include <kamping/communicator.hpp>

#include "kascade/distribution.hpp"
#include "kascade/types.hpp"

namespace kascade {

auto is_list(std::span<const idx_t> succ_array, kamping::Communicator<> const& comm)
    -> bool;

auto is_list(std::span<const idx_t> succ_array,
             Distribution const& dist,
             kamping::Communicator<> const& comm) -> bool;

auto invert_list(std::span<const idx_t> succ_array,
                 std::span<const rank_t> dist_to_succ,
                 std::span<idx_t> pred_array,
                 std::span<rank_t> dist_to_pred,
                 Distribution const& dist,
                 kamping::Communicator<> const& comm) -> void;

auto is_root(std::size_t local_idx,
             std::span<const idx_t> succ_array,
             Distribution const& dist,
             kamping::Communicator<> const& comm) -> bool;

struct LeafInfo {
private:
  std::vector<bool> has_pred_;  // per-local index
  Distribution const* dist_;
  kamping::Communicator<> const* comm_;
  std::size_t num_local_leaves_;

public:
  LeafInfo(std::span<const idx_t> succ_array,
           Distribution const& dist,
           kamping::Communicator<> const& comm);

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
