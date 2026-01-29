#pragma once

#include <span>

#include <kamping/communicator.hpp>

#include "kascade/types.hpp"
struct Stats {
  Stats(std::span<kascade::idx_t const> root,
        std::span<kascade::idx_t const> rank,
        kamping::Communicator<> const& comm);
  std::size_t num_roots;
  std::size_t max_rank;
  double avg_rank;
};
