#pragma once

#include <nlohmann/detail/macro_scope.hpp>
#include <nlohmann/json.hpp>

#include "benchmark_config.hpp"

NLOHMANN_JSON_SERIALIZE_ENUM(Algorithm,
                             {{Algorithm::invalid, nullptr},
                              {Algorithm::GatherChase, "GatherChase"},
                              {Algorithm::PointerDoubling, "PointerDoubling"}})

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Config,
                                   num_ranks,
                                   git_tag,
                                   output_path,
                                   iterations,
                                   kagen_option_string,
                                   algorithm,
                                   verify_level,
                                   verify_continue_on_mismatch)
