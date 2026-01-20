#include "reporting.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <kamping/communicator.hpp>
#include <kamping/measurements/timer.hpp>
#include <kamping/nlohmann_json_adapter/printer.hpp>
#include <memory>
#include <ostream>

auto make_output_stream(std::string const& output_file) -> std::unique_ptr<std::ostream> {
  std::ostream out(std::cout.rdbuf());
  if (output_file == "stdout") {
    return std::make_unique<std::ostream>(std::cout.rdbuf());
  }
  if (output_file == "stderr") {
    return std::make_unique<std::ostream>(std::cerr.rdbuf());
  }
  auto path = std::filesystem::path(output_file);
  if (path.extension() != ".json") {
    path.replace_extension(".json");
  }
  auto file_stream = std::make_unique<std::ofstream>(path);
  if (!file_stream->is_open()) {
    throw std::runtime_error("Failed to open output file: " + path.string());
  }
  return file_stream;
}

void Report::step_iteration() {
  kamping::measurements::NLohmannJsonPrinter printer;
  kamping::measurements::timer().aggregate_and_print(printer);
  if (kamping::comm_world().is_root()) {
    times_.emplace_back(printer.json());
  }
  kamping::measurements::timer().clear();
};

void Report::print(std::ostream& out) {
  if (kamping::comm_world().is_root()) {
    nlohmann::json json;
    json["config"] = config_;
    json["stats"] = stats_;
    // json["stats"]["components"] = component_stats;
    json["timer"] = times_;
    out << json.dump(2) << '\n';
  }
};
