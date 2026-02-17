#pragma once

#include <cstddef>
#include <cstdint>
#include <ranges>

namespace kascade {
using idx_t = std::size_t;
using rank_t = std::int64_t;

template <typename R>
concept IndexRange =
    std::ranges::forward_range<R> && std::same_as<std::ranges::range_value_t<R>, idx_t>;
}  // namespace kascade
