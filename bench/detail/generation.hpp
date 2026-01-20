#pragma once
#include <vector>

#include <kamping/communicator.hpp>

#include "benchmark_config.hpp"
#include "kascade/types.hpp"

auto generate_input(Config const& config, kamping::Communicator<> const& comm)
    -> std::vector<kascade::idx_t>;
