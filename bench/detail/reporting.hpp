#pragma once
#include <concepts>
#include <ostream>
#include <vector>

#include <nlohmann/json.hpp>

auto make_output_stream(std::string const& output_file) -> std::unique_ptr<std::ostream>;

class Report {
public:
  void step_iteration();
  void print(std::ostream& out);

  void push_stats(std::string const& key, std::invocable auto&& compute_stats) {
    if (stats_.contains(key)) {
      return;
    }
    stats_[key] = compute_stats();
  }

  template <typename T>
  void push_stats(std::string const& key, T const& stats) {
    push_stats(key, [&]() { return stats; });
  }

  template <typename T>
  void push_config(T const& config) {
    config_ = config;
  }

private:
  nlohmann::json config_;
  nlohmann::json stats_;
  std::vector<nlohmann::json> times_;
};
