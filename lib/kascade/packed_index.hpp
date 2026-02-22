#pragma once

#include <cstdint>
#include <ostream>

#include <fmt/base.h>
#include <fmt/ostream.h>

#include "types.hpp"

namespace kascade {
struct packed_index {
  idx_t value;
  constexpr packed_index() = default;
  constexpr explicit packed_index(idx_t v) : value(v) {}

  constexpr void set_pe_rank(std::uint32_t rank) noexcept {
    rank &= pe_mask32;  // truncate to 19 bits
    value = (value & ~pe_mask) | (static_cast<idx_t>(rank) << index_bits);
  }
  [[nodiscard]] constexpr auto get_pe_rank() const noexcept -> std::uint32_t {
    return static_cast<std::uint32_t>((value >> index_bits) & pe_mask32);
  }

  [[nodiscard]] constexpr auto get_index() const noexcept -> idx_t {
    return value & index_mask;
  }
  constexpr void set_index(idx_t low) noexcept {
    value = (value & ~index_mask) | (low & index_mask);
  }

  [[nodiscard]] constexpr auto is_msb_set() const noexcept -> bool {
    return (value & msb_mask) != 0;
  }
  constexpr void set_msb() noexcept { value |= msb_mask; }
  constexpr void clear_msb() noexcept { value &= ~msb_mask; }

  // masks and bit widths
  static constexpr unsigned pe_bits = 19U;
  static constexpr unsigned index_bits = 44U;
  static constexpr idx_t index_mask = (idx_t(1) << index_bits) - 1;  // bits 0..43
  static constexpr idx_t pe_mask = ((idx_t(1) << pe_bits) - 1)
                                   << index_bits;  // bits 44..62
  static constexpr std::uint32_t pe_mask32 =
      (1U << pe_bits) - 1;  // 19-bit mask as 32-bit
  static constexpr idx_t msb_mask = idx_t(1) << 63;
};

inline auto operator<<(std::ostream& out, packed_index v) -> std::ostream& {
  return out << v.value;
}

}  // namespace kascade
template <>
struct fmt::formatter<kascade::packed_index> : fmt::ostream_formatter {};
