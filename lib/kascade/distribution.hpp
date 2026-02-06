#pragma once

#include <algorithm>
#include <cstddef>
#include <ranges>
#include <vector>

#include <kamping/collectives/allgather.hpp>
#include <kamping/communicator.hpp>
#include <kassert/kassert.hpp>

#include "kascade/types.hpp"

namespace kascade {
class Distribution {
public:
  Distribution(std::size_t local_size, kamping::Communicator<> const& comm)
      : counts_(comm.size()), inclusive_prefix_sum_(comm.size()) {
    namespace kmp = kamping::params;
    comm.allgather(kmp::send_buf(local_size), kmp::recv_buf(counts_));
    std::inclusive_scan(counts_.begin(), counts_.end(), inclusive_prefix_sum_.begin());
  }

  [[nodiscard]] auto get_owner(idx_t idx) const -> std::size_t {
    auto it =
        std::ranges::upper_bound(inclusive_prefix_sum_, static_cast<std::size_t>(idx));
    KASSERT(it != inclusive_prefix_sum_.end());
    return static_cast<std::size_t>(std::distance(inclusive_prefix_sum_.begin(), it));
  }

  [[nodiscard]] auto is_local(idx_t idx, std::size_t rank) const -> bool {
    auto begin = inclusive_prefix_sum_[rank] - counts_[rank];
    auto end = inclusive_prefix_sum_[rank];
    return static_cast<std::size_t>(idx) >= begin && static_cast<std::size_t>(idx) < end;
  }

  [[nodiscard]] auto get_owner_signed(idx_t idx) const -> int {
    return static_cast<int>(get_owner(idx));
  }

  [[nodiscard]] auto counts() const -> std::vector<std::size_t> const& { return counts_; }

  [[nodiscard]] auto get_count(std::size_t rank) const -> std::size_t {
    return counts_[rank];
  }

  [[nodiscard]] auto get_exclusive_prefix(std::size_t rank) const -> std::size_t {
    return inclusive_prefix_sum_[rank] - counts_[rank];
  }

  [[nodiscard]] auto get_local_idx(idx_t idx, std::size_t rank) const -> idx_t {
    return idx - static_cast<idx_t>(inclusive_prefix_sum_[rank] - counts_[rank]);
  }

  [[nodiscard]] auto get_global_idx(idx_t idx, std::size_t rank) const -> idx_t {
    return idx + static_cast<idx_t>(inclusive_prefix_sum_[rank] - counts_[rank]);
  }

  [[nodiscard]] auto get_global_size() const -> std::size_t {
    return inclusive_prefix_sum_.back();
  }

  [[nodiscard]] auto get_local_size(std::size_t rank) const -> std::size_t {
    return counts_[rank];
  }

  [[nodiscard]] auto local_indices(std::size_t rank) const
      -> std::ranges::iota_view<idx_t, idx_t> {
    return std::views::iota(idx_t{0}, idx_t{get_local_size(rank)});
  }

  [[nodiscard]] auto global_indices(std::size_t rank) const
      -> std::ranges::iota_view<idx_t, idx_t> {
    return std::views::iota(
        static_cast<idx_t>(inclusive_prefix_sum_[rank] - counts_[rank]),
        static_cast<idx_t>(inclusive_prefix_sum_[rank]));
  }

private:
  std::vector<std::size_t> counts_;
  std::vector<std::size_t> inclusive_prefix_sum_;
};

}  // namespace kascade
