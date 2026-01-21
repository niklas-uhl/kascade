#pragma once

#include <span>

#include <kamping/communicator.hpp>

#include "kascade/types.hpp"

namespace kascade {
void pointer_doubling(std::span<const idx_t> succ_array,
                      std::span<idx_t> rank_array,
                      std::span<idx_t> root_array,
                      kamping::Communicator<> const& comm);

void async_pointer_doubling(std::span<const idx_t> succ_array,
                      std::span<idx_t> rank_array,
                      std::span<idx_t> root_array,
                      kamping::Communicator<> const& comm);

}  // namespace kascade
