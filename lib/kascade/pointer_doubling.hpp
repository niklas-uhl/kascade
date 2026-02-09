#pragma once

#include <span>

#include <kamping/communicator.hpp>

#include "kascade/configuration.hpp"
#include "kascade/distribution.hpp"
#include "kascade/types.hpp"

namespace kascade {

/// @param[inout] succ_array Array of successor indices; upon completion contains root
/// indices.
/// @param[inout] rank_array Array of ranks; initialize with distance to successor (0 for
/// roots, > 1 otherwise). Upon completion, contains distance to the root.
void pointer_doubling(PointerDoublingConfig config,
                      std::span<idx_t> succ_array,
                      std::span<rank_t> rank_array,
                      Distribution const& dist,
                      kamping::Communicator<> const& comm);

/// @param[inout] succ_array Array of successor indices; upon completion contains root
/// indices.
/// @param[inout] rank_array Array of ranks; initialize with distance to successor (0 for
/// roots, > 1 otherwise). Upon completion, contains distance to the root.
void async_pointer_doubling(AsyncPointerChasingConfig const& config,
                            std::span<idx_t> succ_array,
                            std::span<rank_t> rank_array,
                            Distribution const& dist,
                            kamping::Communicator<> const& comm);

/// @param[inout] succ_array Array of successor indices; upon completion contains root
/// indices.
/// @param[inout] rank_array Array of ranks; initialize with distance to successor (0 for
/// roots, > 1 otherwise). Upon completion, contains distance to the root.
void rma_pointer_doubling(RMAPointerChasingConfig const& config,
                          std::span<idx_t> succ_array,
                          std::span<rank_t> rank_array,
                          Distribution const& dist,
                          kamping::Communicator<> const& comm);

}  // namespace kascade
