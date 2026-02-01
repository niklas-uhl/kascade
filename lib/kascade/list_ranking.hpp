#pragma once

#include <span>

#include <mpi.h>

#include <kascade/types.hpp>

#include <kamping/communicator.hpp>

#include "kascade/distribution.hpp"

namespace kascade {
void rank(std::span<const idx_t> succ_array,
          std::span<idx_t> rank_array,
          std::span<idx_t> root_array,
          kamping::Communicator<> const& comm);

void rank_on_root(std::span<idx_t> succ_array,
                  std::span<idx_t> rank_array,
                  Distribution const& dist,
                  kamping::Communicator<> const& comm);

void local_pointer_chasing(std::span<idx_t> succ_array, std::span<idx_t> rank_array);
void local_pointer_chasing(std::span<idx_t> succ_array,
                           std::span<idx_t> rank_array,
                           std::size_t rank,
                           Distribution const& dist);

/// Initialize root and rank arrays for a distributed successor list.
/// - Copies each local successor into the corresponding root entry.
/// - Sets all ranks to 1, then sets ranks of global roots (indices equal to their
/// successor) to 0.
/// - Returns the distribution mapping between global and local indices.
/// @param succ_array Local span of successors; succ_array[i] is the global index of the
/// successor of local node i.
/// @param root_array Local span to be filled with initial roots (copied from succ_array).
/// @param rank_array Local span to be filled with initial ranks (1 for non-roots, 0 for
/// roots).
/// @param comm The communicator.
/// @return Distribution object describing the global-to-local index mapping across ranks.
auto set_initial_ranking_state(std::span<const idx_t> succ_array,
                               std::span<idx_t> root_array,
                               std::span<idx_t> rank_array,
                               kamping::Communicator<> const& comm) -> Distribution;

}  // namespace kascade
