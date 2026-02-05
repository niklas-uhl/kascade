#pragma once

#include <limits>

#include "kascade/types.hpp"

namespace kascade::bits {

constexpr idx_t root_flag_mask = idx_t(1) << (std::numeric_limits<idx_t>::digits - 1);

[[nodiscard]] constexpr auto set_root_flag(idx_t value) noexcept -> idx_t {
  return value | root_flag_mask;
}

[[nodiscard]] constexpr auto clear_root_flag(idx_t value) noexcept -> idx_t {
  return value & ~root_flag_mask;
}

[[nodiscard]] constexpr auto has_root_flag(idx_t value) noexcept -> bool {
  return (value & root_flag_mask) != 0;
}
}  // namespace kascade::bits
