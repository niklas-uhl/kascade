#include "detail/algorithm_factory.hpp"

#include <memory>
#include <utility>

#include "detail/algorithm.hpp"
#include "detail/algorithm_impl.hpp"
#include "detail/benchmark_config.hpp"

auto get_algorithm(const Config& config, kamping::Communicator<> const& comm)
    -> std::unique_ptr<AbstractAlgorithm> {
  switch (config.algorithm) {
    case kascade::Algorithm::GatherChase:
      return std::make_unique<GatherRank>(comm);
    case kascade::Algorithm::PointerDoubling:
      return std::make_unique<PointerDoubling>(config.pointer_doubling, comm);
    case kascade::Algorithm::AsyncPointerDoubling:
      return std::make_unique<AsyncPointerDoubling>(config.async_pointer_chasing, comm);
    case kascade::Algorithm::RMAPointerDoubling:
      return std::make_unique<RMAPointerDoubling>(config.rma_pointer_chasing, comm);
    case kascade::Algorithm::SparseRulingSet:
      return std::make_unique<SparseRulingSet>(config, comm);
    case kascade::Algorithm::EulerTour:
      return std::make_unique<EulerTour>(config, comm);
    case kascade::Algorithm::MPLR:
      return std::make_unique<MPLR>(config.mplr, comm);
    case kascade::Algorithm::invalid:
      throw std::runtime_error("Invalid algorithm selected.");
  }
  std::unreachable();
};
