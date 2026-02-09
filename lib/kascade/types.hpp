#pragma once

#include <cstddef>
#include <ranges>

namespace kascade {
using idx_t = std::size_t;

template <typename R>
concept IndexRange = std::ranges::forward_range<R> && std::ranges::sized_range<R> &&
                     std::same_as<std::ranges::range_value_t<R>, idx_t>;
}  // namespace kascade
