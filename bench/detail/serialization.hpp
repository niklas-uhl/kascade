#pragma once

#include <nlohmann/json.hpp>

#include "benchmark_config.hpp"
#include "input/generation.hpp"
#include "kascade/configuration.hpp"

NLOHMANN_JSON_SERIALIZE_ENUM(Algorithm,
                             {{Algorithm::invalid, nullptr},
                              {Algorithm::GatherChase, "GatherChase"},
                              {Algorithm::PointerDoubling, "PointerDoubling"},
                              {Algorithm::AsyncPointerDoubling, "AsyncPointerDoubling"},
                              {Algorithm::RMAPointerDoubling, "RMAPointerDoubling"}})

namespace kascade::input {
NLOHMANN_JSON_SERIALIZE_ENUM(InputProcessing,
                             {{InputProcessing::invalid, nullptr},
                              {InputProcessing::bfs, "bfs"},
                              {InputProcessing::none, "none"}})

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Config, kagen_option_string, input_processing);

}  // namespace kascade::input

namespace kascade {
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(AsyncPointerChasingConfig, use_caching);
}

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Config,
                                   num_ranks,
                                   git_tag,
                                   output_path,
                                   iterations,
                                   input,
                                   algorithm,
                                   async_pointer_chasing,
                                   verify_level,
                                   verify_continue_on_mismatch)
