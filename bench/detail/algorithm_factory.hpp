#pragma once

#include <kamping/communicator.hpp>
#include <memory>
#include "./algorithm.hpp"
#include "./benchmark_config.hpp"

auto get_algorithm(const Config& config, kamping::Communicator<> const& comm)
    -> std::unique_ptr<AbstractAlgorithm>;
