#include "detail/algorithm_factory.hpp"

#include "detail/algorithm.hpp"
#include "detail/algorithm_impl.hpp"
#include "detail/benchmark_config.hpp"

auto get_algorithm(const Config& config, kamping::Communicator<> const& comm)
    -> std::unique_ptr<AbstractAlgorithm> {
  switch (config.algorithm) {
    case Algorithm::GatherChase:
      return std::make_unique<GatherRank>(comm);
    case Algorithm::PointerDoubling:
      return std::make_unique<PointerDoubling>(comm);
    default:
    case Algorithm::invalid:
      throw std::runtime_error("Invalid algorithm selected.");
  }
};
