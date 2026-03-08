#pragma once

#include <span>

#include <kamping/communicator.hpp>

#include "configuration.hpp"
#include "grid_communicator.hpp"
#include "kascade/distribution.hpp"
#include "kascade/types.hpp"

namespace kascade {
void sparse_ruling_set(SparseRulingSetConfig const& config,
                       std::span<idx_t> succ_array,
                       std::span<rank_t> rank_array,
                       Distribution const& dist,
                       kamping::Communicator<> const& comm,
                       std::optional<TopologyAwareGridCommunicator> const& grid_comm);
}
