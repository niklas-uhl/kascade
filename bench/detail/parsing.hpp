#pragma once
#include <span>
#include "benchmark_config.hpp"

auto parse_args(std::span<char*> args) -> Config;
