#pragma once

#include <memory>

#include <kamping/communicator.hpp>

#include "./algorithm.hpp"
#include "./benchmark_config.hpp"

auto get_algorithm(const Config& config,
                   kamping::Communicator<> const& comm,
                   std::size_t ranks_per_compute_node)
    -> std::unique_ptr<AbstractAlgorithm>;
