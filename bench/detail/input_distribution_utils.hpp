#pragma once

#include <cstddef>

#include "kascade/types.hpp"

namespace kascade::bench {
class Distribution {
public:
  Distribution(std::size_t size, std::size_t comm_size)
      : size_(size),
        comm_size_(comm_size),
        base_(size_ / comm_size),
        remainder_(size_ % comm_size),
        threshold_(remainder_ * (base_ + 1)) {}

  [[nodiscard]] auto get_owner(kascade::idx_t idx) const -> std::size_t {
    if (base_ == 0) {
      return static_cast<std::size_t>(idx);
    }
    if (idx < threshold_) {
      return static_cast<std::size_t>(idx) / (base_ + 1U);
    }
    return remainder_ + ((static_cast<std::size_t>(idx) - threshold_) / base_);
  }

  [[nodiscard]] auto get_count(std::size_t rank) const -> std::size_t {
    return static_cast<std::size_t>(base_ + ((rank < remainder_) ? 1U : 0U));
  }

  [[nodiscard]] auto get_prefix(std::size_t rank) const -> std::size_t {
    if (rank < remainder_) {
      return rank * (base_ + 1U);
    }
    return (remainder_ * (base_ + 1)) + ((rank - remainder_) * base_);
  }

private:
  std::size_t size_;
  std::size_t comm_size_;
  std::size_t base_;
  std::size_t remainder_;
  std::size_t threshold_;
};
}  // namespace kascade::bench
