#pragma once

#include <algorithm>
#include <cstddef>
#include <numeric>
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
      : offset_(comm.size() + 1) {
    namespace kmp = kamping::params;
    offset_[comm.rank()] = local_size;
    comm.allgather(kmp::send_recv_buf(std::span{offset_}.subspan(0, comm.size())));
    std::exclusive_scan(offset_.begin(), offset_.end(), offset_.begin(), std::size_t{0});
  }

  template <std::ranges::forward_range Range>
    requires std::ranges::sized_range<Range> &&
             std::convertible_to<std::ranges::range_value_t<Range>, std::size_t>
  Distribution(Range&& offsets, kamping::Communicator<> const& comm)
      : offset_(offsets.begin(), offsets.end()) {
    KASSERT(offset_.size() == comm.size() + 1, "Offsets size must be comm.size() + 1");
    KASSERT(std::ranges::is_sorted(offset_), "Offsets must be monotonically increasing");
  }

  [[nodiscard]] auto get_owner(idx_t idx) const -> std::size_t {
    auto it = std::ranges::upper_bound(offset_, static_cast<std::size_t>(idx));
    KASSERT(it != offset_.begin() && it != offset_.end(), "Index out of bounds");
    return static_cast<std::size_t>(std::distance(offset_.begin(), it)) - 1;
  }

  [[nodiscard]] auto get_owner_signed(idx_t idx) const -> int {
    return static_cast<int>(get_owner(idx));
  }

  [[nodiscard]] auto counts() const {
    return offset_ |
           std::views::pairwise_transform([](auto const& begin, auto const& end) {
             return static_cast<std::size_t>(end - begin);
           });
  }

  [[nodiscard]] auto index_range_begin(std::size_t rank) const -> std::size_t {
    return offset_[rank];
  }

  [[nodiscard]] auto index_range_end(std::size_t rank) const -> std::size_t {
    return offset_[rank + 1];
  }

  [[nodiscard]] auto is_local(idx_t idx, std::size_t rank) const -> bool {
    return idx >= index_range_begin(rank) && idx < index_range_end(rank);
  }

  [[nodiscard]] auto get_local_idx(idx_t idx, std::size_t rank) const -> idx_t {
    KASSERT(is_local(idx, rank));
    return idx - index_range_begin(rank);
  }

  [[nodiscard]] auto get_global_idx(idx_t idx, std::size_t rank) const -> idx_t {
    return idx + index_range_begin(rank);
  }

  [[nodiscard]] auto get_global_size() const -> std::size_t { return offset_.back(); }

  [[nodiscard]] auto get_local_size(std::size_t rank) const -> std::size_t {
    return index_range_end(rank) - index_range_begin(rank);
  }

  [[nodiscard]] auto local_indices(std::size_t rank) const
      -> std::ranges::iota_view<idx_t, idx_t> {
    return std::views::iota(idx_t{0}, idx_t{get_local_size(rank)});
  }

  [[nodiscard]] auto global_indices(std::size_t rank) const
      -> std::ranges::iota_view<idx_t, idx_t> {
    return std::views::iota(static_cast<idx_t>(index_range_begin(rank)),
                            static_cast<idx_t>(index_range_end(rank)));
  }

private:
  std::vector<std::size_t> offset_;
};

class PartitionedDistribution {
public:
  PartitionedDistribution(std::size_t local_size_a,
                          std::size_t local_size_b,
                          kamping::Communicator<> const& comm)
      : dist_(local_size_a + local_size_b, comm) {
    namespace kmp = kamping::params;
    counts_first_partition_ = comm.allgather(kmp::send_buf(local_size_a));
  }
  auto get_distribution() const -> auto const& { return dist_; }
  [[nodiscard]] auto is_in_first_partition(idx_t idx) const -> bool {
    auto owner = dist_.get_owner(idx);
    auto local_idx = dist_.get_local_idx(idx, owner);
    return local_idx < counts_first_partition_[owner];
  }

private:
  Distribution dist_;
  std::vector<std::size_t> counts_first_partition_;
};

}  // namespace kascade
