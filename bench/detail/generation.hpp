#pragma once
#include <vector>
#include <kamping/communicator.hpp>
#include "benchmark_config.hpp"

auto generate_input(Config const& config, kamping::Communicator<> const& comm)
    -> std::vector<std::size_t>;
