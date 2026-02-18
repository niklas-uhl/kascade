#pragma once
#include <utility>

#include <kamping/communicator.hpp>

#include "./algorithm.hpp"
#include "detail/benchmark_config.hpp"
#include "detail/mplr/mplr.hpp"
#include "kascade/configuration.hpp"
#include "kascade/eulertour.hpp"
#include "kascade/list_ranking.hpp"
#include "kascade/pointer_doubling.hpp"
#include "kascade/sparse_ruling_set.hpp"
#include "kascade/types.hpp"

class AlgorithmBase : public AbstractAlgorithm {
public:
  AlgorithmBase(kamping::Communicator<> const& comm) : comm_(&comm) {}
  void ingest(std::span<const kascade::idx_t> succ_array) override {
    succ_array_ = succ_array;
    rank_array_.resize(succ_array_.size());
    root_array_.resize(succ_array_.size());
  };
  auto get_rank_array() -> std::vector<kascade::rank_t> const& override {
    return rank_array_;
  }
  auto get_root_array() -> std::vector<kascade::idx_t> const& override {
    return root_array_;
  }

protected:
  // NOLINTBEGIN(*-non-private-member-variables-in-classes)
  std::span<const kascade::idx_t> succ_array_;
  std::vector<kascade::rank_t> rank_array_;
  std::vector<kascade::idx_t> root_array_;
  kamping::Communicator<> const* comm_;
  // NOLINTEND(*-non-private-member-variables-in-classes)
};

class GatherRank : public AlgorithmBase {
public:
  GatherRank(kamping::Communicator<> const& comm) : AlgorithmBase(comm) {}
  void run() override {
    auto dist =
        kascade::set_initial_ranking_state(succ_array_, root_array_, rank_array_, *comm_);
    kascade::rank_on_root(root_array_, rank_array_, dist, *comm_);
  }
};

class PointerDoubling : public AlgorithmBase {
public:
  PointerDoubling(kascade::PointerDoublingConfig const& config,
                  kamping::Communicator<> const& comm)
      : AlgorithmBase(comm), config_{config} {}
  void run() override {
    auto dist =
        kascade::set_initial_ranking_state(succ_array_, root_array_, rank_array_, *comm_);
    kascade::pointer_doubling(config_, root_array_, rank_array_, dist, *comm_);
  }

private:
  kascade::PointerDoublingConfig config_;
};

class AsyncPointerDoubling : public AlgorithmBase {
public:
  AsyncPointerDoubling(kascade::AsyncPointerChasingConfig const& config,
                       kamping::Communicator<> const& comm)
      : AlgorithmBase(comm), config_{config} {}
  void run() override {
    auto dist =
        kascade::set_initial_ranking_state(succ_array_, root_array_, rank_array_, *comm_);
    kascade::async_pointer_doubling(config_, root_array_, rank_array_, dist, *comm_);
  }

private:
  kascade::AsyncPointerChasingConfig config_;
};

class RMAPointerDoubling : public AlgorithmBase {
public:
  RMAPointerDoubling(kascade::RMAPointerChasingConfig const& config,
                     kamping::Communicator<> const& comm)
      : AlgorithmBase(comm), config_(config) {}
  void run() override {
    auto dist =
        kascade::set_initial_ranking_state(succ_array_, root_array_, rank_array_, *comm_);
    kascade::rma_pointer_doubling(config_, root_array_, rank_array_, dist, *comm_);
  }

private:
  kascade::RMAPointerChasingConfig config_;
};

class SparseRulingSet : public AlgorithmBase {
public:
  explicit SparseRulingSet(Config config, kamping::Communicator<> const& comm)
      : AlgorithmBase(comm), config_(std::move(config)) {}
  void run() override {
    auto dist =
        kascade::set_initial_ranking_state(succ_array_, root_array_, rank_array_, *comm_);
    switch (config_.sparse_ruling_set.base_algorithm) {
      case kascade::Algorithm::GatherChase:
        config_.sparse_ruling_set.base_algorithm_config = nullptr;
        break;
      case kascade::Algorithm::PointerDoubling:
        config_.sparse_ruling_set.base_algorithm_config = config_.pointer_doubling;
        break;
      case kascade::Algorithm::AsyncPointerDoubling:
        config_.sparse_ruling_set.base_algorithm_config = config_.async_pointer_chasing;
        break;
      case kascade::Algorithm::RMAPointerDoubling:
        config_.sparse_ruling_set.base_algorithm_config = config_.rma_pointer_chasing;
        break;
      case kascade::Algorithm::SparseRulingSet:
        config_.sparse_ruling_set.base_algorithm_config = config_.sparse_ruling_set;
        break;
      default:
        throw std::runtime_error("Invalid base algorithm selected for sparse ruling set");
    }
    kascade::sparse_ruling_set(config_.sparse_ruling_set, root_array_, rank_array_, dist,
                               *comm_);
  }

private:
  Config config_;
};

class EulerTour : public AlgorithmBase {
public:
  explicit EulerTour(Config config, kamping::Communicator<> const& comm)
      : AlgorithmBase(comm), config_(std::move(config)) {}
  void run() override {
    auto dist =
        kascade::set_initial_ranking_state(succ_array_, root_array_, rank_array_, *comm_);
    switch (config_.euler_tour.algorithm) {
      case kascade::Algorithm::GatherChase:
        config_.euler_tour.algo_config = nullptr;
        break;
      case kascade::Algorithm::PointerDoubling:
        config_.euler_tour.algo_config = config_.pointer_doubling;
        break;
      case kascade::Algorithm::AsyncPointerDoubling:
        config_.euler_tour.algo_config = config_.async_pointer_chasing;
        break;
      case kascade::Algorithm::RMAPointerDoubling:
        config_.euler_tour.algo_config = config_.rma_pointer_chasing;
        break;
      case kascade::Algorithm::SparseRulingSet:
        config_.euler_tour.algo_config = config_.sparse_ruling_set;
        break;
      case kascade::Algorithm::EulerTour:
        throw std::runtime_error("Cannot apply Euler Tour recursively.");
        break;
      case kascade::Algorithm::MPLR:
        throw std::runtime_error("Invalid algorithm selected.");
        break;
      case kascade::Algorithm::invalid:
        throw std::runtime_error("Invalid algorithm selected.");
        break;
    }
    kascade::rank_via_euler_tour(config_.euler_tour, root_array_, rank_array_, dist,
                                 *comm_);
  }

private:
  Config config_;
};

class MPLR : public AbstractAlgorithm {
public:
  explicit MPLR(mplr::Configuration config, kamping::Communicator<> const& comm)
      : config_(config), comm_(comm.mpi_communicator()) {}
  void ingest(std::span<const kascade::idx_t> succ_array) override {
    succ_array_.resize(succ_array.size());
    std::ranges::copy(succ_array, succ_array_.begin());
  }
  void run() override {
    root_array_.resize(succ_array_.size());
    std::ranges::copy(succ_array_, root_array_.begin());
    switch (config_.algorithm) {
      case mplr::Algorithm::ForestRulingSet:
        std::tie(root_array_, rank_array_) =
            mplr::forest_ruling_set(config_, root_array_, comm_);
        break;
      case mplr::Algorithm::PointerDoubling:
        std::tie(root_array_, rank_array_) =
            mplr::forest_pointer_doubling(config_, root_array_, comm_);
        break;
      case mplr::Algorithm::invalid:
        throw std::runtime_error("invalid algorithm selection");
    }
  }
  auto get_rank_array() -> std::vector<kascade::rank_t> const& override {
    return rank_array_;
  }

  auto get_root_array() -> std::vector<kascade::idx_t> const& override {
    if (!converted_root_array_) {
      converted_root_array_.emplace();
      converted_root_array_->resize(root_array_.size());
      std::ranges::copy(root_array_, converted_root_array_->begin());
    }
    return *converted_root_array_;
  }

private:
  mplr::Configuration config_;
  kamping::Communicator<> comm_;
  std::vector<std::int64_t> succ_array_;
  std::vector<std::int64_t> rank_array_;
  std::vector<std::uint64_t> root_array_;
  std::optional<std::vector<kascade::idx_t>> converted_root_array_;
};
