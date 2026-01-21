#pragma once

#include <nlohmann/detail/macro_scope.hpp>
#include <nlohmann/json.hpp>

#include "benchmark_config.hpp"
#include "input/generation.hpp"

NLOHMANN_JSON_SERIALIZE_ENUM(Algorithm,
                             {{Algorithm::invalid, nullptr},
                              {Algorithm::GatherChase, "GatherChase"},
                              {Algorithm::PointerDoubling, "PointerDoubling"},
                              {Algorithm::AsyncPointerDoubling, "AsyncPointerDoubling"}})

namespace kascade::input {
NLOHMANN_JSON_SERIALIZE_ENUM(InputProcessing,
                             {{InputProcessing::invalid, nullptr},
                              {InputProcessing::bfs, "bfs"},
                              {InputProcessing::none, "none"}})

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Config, kagen_option_string, input_processing);

}  // namespace kascade::input

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Config,
                                   num_ranks,
                                   git_tag,
                                   output_path,
                                   iterations,
                                   input,
                                   algorithm,
                                   verify_level,
                                   verify_continue_on_mismatch)
