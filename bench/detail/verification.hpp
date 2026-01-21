#pragma once

#include <cstddef>
#include <span>

#include <kamping/communicator.hpp>

#include "detail/algorithm.hpp"
#include "kascade/types.hpp"

auto verify(std::span<const kascade::idx_t> succ,
            AbstractAlgorithm& reference,
            AbstractAlgorithm& algorithm,
            std::size_t verify_level,
            bool continue_on_mismatch,
            kamping::Communicator<> const& comm) -> void;
