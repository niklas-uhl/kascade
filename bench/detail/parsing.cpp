#include "parsing.hpp"

#include <cstdlib>

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>

#include "./benchmark_config.hpp"
#include "./serialization.hpp"  // IWYU pragma: keep

auto lexical_cast(const std::string& input, Algorithm& algo) -> bool {
  nlohmann::json input_json = input;
  algo = input_json.template get<Algorithm>();
  return algo != Algorithm::invalid;
}

namespace kascade::input {
auto lexical_cast(const std::string& input, InputProcessing& processing) -> bool {
  nlohmann::json input_json = input;
  processing = input_json.template get<kascade::input::InputProcessing>();
  return processing != kascade::input::InputProcessing::invalid;
}
}  // namespace kascade::input

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

  try {
    app.parse(static_cast<int>(args.size()), args.data());
  } catch (const CLI ::ParseError& e) {
    int errcode = app.exit(e);
    std::exit(errcode);
  };
  return config;
}
