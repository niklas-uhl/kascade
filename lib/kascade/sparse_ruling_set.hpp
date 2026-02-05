#pragma once

#include <span>

#include <kamping/communicator.hpp>

#include "configuration.hpp"
#include "kascade/distribution.hpp"
#include "kascade/types.hpp"

namespace kascade {
void sparse_ruling_set(SparseRulingSetConfig const& config,
                       std::span<idx_t> succ_array,
                       std::span<idx_t> rank_array,
                       Distribution const& dist,
                       kamping::Communicator<> const& comm);
}
