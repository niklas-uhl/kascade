#include "detail/algorithm_factory.hpp"

#include <memory>
#include <utility>

#include "detail/algorithm.hpp"
#include "detail/algorithm_impl.hpp"
#include "detail/benchmark_config.hpp"

auto get_algorithm(const Config& config, kamping::Communicator<> const& comm)
    -> std::unique_ptr<AbstractAlgorithm> {
  switch (config.algorithm) {
    case Algorithm::GatherChase:
      return std::make_unique<GatherRank>(comm);
    case Algorithm::PointerDoubling:
      return std::make_unique<PointerDoubling>(config.pointer_doubling, comm);
    case Algorithm::AsyncPointerDoubling:
      return std::make_unique<AsyncPointerDoubling>(config.async_pointer_chasing, comm);
    case Algorithm::RMAPointerDoubling:
      return std::make_unique<RMAPointerDoubling>(config.rma_pointer_chasing, comm);
    case Algorithm::SparseRulingSet:
      return std::make_unique<SparseRulingSet>(config.sparse_ruling_set, comm);
    case Algorithm::invalid:
      throw std::runtime_error("Invalid algorithm selected.");
  }
  std::unreachable();
};
