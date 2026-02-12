#include "parsing.hpp"

#include <charconv>
#include <cstdlib>

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>

#include "./benchmark_config.hpp"
#include "./serialization.hpp"  // IWYU pragma: keep
#include "kascade/configuration.hpp"

auto lexical_cast(const std::string& input, StatsLevel& stats_level) -> bool {
  nlohmann::json input_json = input;
  stats_level = input_json.template get<StatsLevel>();
  return stats_level != StatsLevel::invalid;
}

namespace kascade {

auto lexical_cast(const std::string& input, Algorithm& algo) -> bool {
  nlohmann::json input_json = input;
  algo = input_json.template get<Algorithm>();
  return algo != Algorithm::invalid;
}

auto lexical_cast(const std::string& input, AggregationLevel& aggregation_level) -> bool {
  nlohmann::json input_json = input;
  aggregation_level = input_json.template get<kascade::AggregationLevel>();
  return aggregation_level != kascade::AggregationLevel::invalid;
}
auto lexical_cast(const std::string& input, RMASyncMode& sync_mode) -> bool {
  nlohmann::json input_json = input;
  sync_mode = input_json.template get<kascade::RMASyncMode>();
  return sync_mode != kascade::RMASyncMode::invalid;
}
auto lexical_cast(const std::string& input,
                  RulerSelectionStrategy& ruler_selection_strategy) -> bool {
  nlohmann::json input_json = input;
  ruler_selection_strategy = input_json.template get<kascade::RulerSelectionStrategy>();
  return ruler_selection_strategy != kascade::RulerSelectionStrategy::invalid;
}

namespace input {
auto lexical_cast(const std::string& input, InputProcessing& processing) -> bool {
  nlohmann::json input_json = input;
  processing = input_json.template get<kascade::input::InputProcessing>();
  return processing != kascade::input::InputProcessing::invalid;
}
}  // namespace input
}  // namespace kascade

namespace {
namespace {
template <std::integral T>
auto parse_integral_or_inf(std::string_view input, T& out) -> bool {
  if (input == "inf" || input == "INF") {
    out = std::numeric_limits<T>::max();
    return true;
  }

  auto [ptr, ec] = std::from_chars(input.begin(), input.end(), out);
  return ec == std::errc{} && ptr == input.end();
}
}  // namespace
}  // namespace

auto parse_args(std::span<char*> args) -> Config {
  Config config;
  CLI::App app;
  app.option_defaults()->always_capture_default();
  app.add_option("--kagen_option_string", config.input.kagen_option_string)->required();
  app.add_option("--input-processing", config.input.input_processing)->required();
  app.add_option("--iterations", config.iterations);
  app.add_option("--output-file", config.output_path);
  app.add_option("--algorithm", config.algorithm)->required();
  app.add_option("--verify-level", config.verify_level);
  app.add_flag("--verify-continue-on-mismatch", config.verify_continue_on_mismatch);
  app.add_option("--statistics-level", config.statistics_level);

  // async pointer doubling
  app.add_flag("--async-pointer-chasing-use-caching",
               config.async_pointer_chasing.use_caching);
  // RMA poiner doubling
  app.add_option("--rma-pointer-chasing-sync-mode", config.rma_pointer_chasing.sync_mode);
  app.add_option_function<std::string>(
      "--rma-pointer-chasing-batch-size", [&](const std::string& value) {
        std::size_t parsed{};
        if (!parse_integral_or_inf(value, parsed)) {
          throw CLI::ValidationError("--rma-pointer-chasing-batch-size",
                                     "Expected a non-negative integer or 'inf'");
        }
        config.rma_pointer_chasing.batch_size = parsed;
      });
  app.add_flag("--pointer-doubling-use-local-preprocessing",
               config.pointer_doubling.use_local_preprocessing);
  app.add_option("--pointer-doubling-aggregation-level",
                 config.pointer_doubling.aggregation_level);

  app.add_option("--sparse-ruling-set-ruler-selection",
                 config.sparse_ruling_set.ruler_selection)
      ->group("Sparse Ruling Set");
  app.add_option("--sparse-ruling-set-dehne-factor",
                 config.sparse_ruling_set.dehne_factor)
      ->group("Sparse Ruling Set");
  app.add_option("--sparse-ruling-set-heuristic-factor",
                 config.sparse_ruling_set.heuristic_factor)
      ->group("Sparse Ruling Set");
  app.add_flag("--sparse-ruling-set-sync", config.sparse_ruling_set.sync)
      ->group("Sparse Ruling Set");
  app.add_flag("--sparse-ruling-set-sync-locality-aware",
               config.sparse_ruling_set.sync_locality_aware)
      ->group("Sparse Ruling Set");
  app.add_flag("--sparse-ruling-set-spawn", config.sparse_ruling_set.spawn)
      ->group("Sparse Ruling Set");

  app.add_option("--eulertour-algorithm", config.euler_tour.algorithm);
  app.add_flag("--eulertour-use-high-degree-handling",
               config.euler_tour.use_high_degree_handling);

  try {
    app.parse(static_cast<int>(args.size()), args.data());
  } catch (const CLI ::ParseError& e) {
    int errcode = app.exit(e);
    std::exit(errcode);
  };
  return config;
}
