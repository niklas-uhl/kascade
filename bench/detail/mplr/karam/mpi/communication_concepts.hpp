#pragma once

#include "datatype.hpp"

namespace karam::mpi {
template <typename T, typename Datatype>
concept TargetBufferPair = requires(T kv) {
                               { kv.first } -> std::common_with<int>;
                               { kv.second } -> MPIBuffer<Datatype>;
                           };

template <typename Container, typename Datatype>
concept SendBufferList =
    std::ranges::input_range<Container> && TargetBufferPair<std::ranges::range_value_t<Container>, Datatype>;

template <typename Container>
using send_buffer_list_datatype =
    std::ranges::range_value_t<typename std::ranges::range_value_t<Container>::second_type>;

} // namespace karam::mpi
