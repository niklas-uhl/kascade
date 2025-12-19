#pragma once

#include <vector>

#include "allocators.hpp"

namespace karam::utils {
template <typename T>
using default_init_vector = std::vector<T, default_init_allocator<T>>;

///@brief Ensure that the storage allocated to the vector passed to this function is really freed.
///
///@param vector_to_dump Vector whose storage should be cleared and deallocated.
template <typename... Args>
void dump(std::vector<Args...>&& vector_to_dump) {
    std::vector<Args...> tmp;
    std::swap(tmp, vector_to_dump);
    vector_to_dump.clear();
}
} // namespace karam::utils
