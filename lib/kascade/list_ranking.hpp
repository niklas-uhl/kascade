#pragma once

#include <span>

#include <mpi.h>

#include <kascade/types.hpp>

#include <kamping/communicator.hpp>

namespace kascade {
void rank(std::span<const idx_t> succ_array,
          std::span<idx_t> rank_array,
          std::span<idx_t> root_array,
          kamping::Communicator<> const& comm);

void rank_on_root(std::span<const idx_t> succ_array,
                  std::span<idx_t> rank_array,
                  std::span<idx_t> root_array,
                  kamping::Communicator<> const& comm);

}  // namespace kascade
