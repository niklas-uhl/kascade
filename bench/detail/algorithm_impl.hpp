#pragma once
#include <kamping/communicator.hpp>

#include "./algorithm.hpp"
#include "kascade/configuration.hpp"
#include "kascade/list_ranking.hpp"
#include "kascade/pointer_doubling.hpp"

class AlgorithmBase : public AbstractAlgorithm {
public:
  AlgorithmBase(kamping::Communicator<> const& comm) : comm_(&comm) {}
  void ingest(std::span<const kascade::idx_t> succ_array) override {
    succ_array_ = succ_array;
    rank_array_.resize(succ_array_.size());
    root_array_.resize(succ_array_.size());
  };
  auto get_rank_array() -> std::vector<kascade::idx_t> const& override {
    return rank_array_;
  }
  auto get_root_array() -> std::vector<kascade::idx_t> const& override {
    return root_array_;
  }

protected:
  // NOLINTBEGIN(*-non-private-member-variables-in-classes)
  std::span<const kascade::idx_t> succ_array_;
  std::vector<kascade::idx_t> rank_array_;
  std::vector<kascade::idx_t> root_array_;
  kamping::Communicator<> const* comm_;
  // NOLINTEND(*-non-private-member-variables-in-classes)
};

class GatherRank : public AlgorithmBase {
public:
  GatherRank(kamping::Communicator<> const& comm) : AlgorithmBase(comm) {}
  void run() override {
    kascade::rank_on_root(succ_array_, rank_array_, root_array_, *comm_);
  }
};

class PointerDoubling : public AlgorithmBase {
public:
  PointerDoubling(kamping::Communicator<> const& comm) : AlgorithmBase(comm) {}
  void run() override {
    auto dist =
        kascade::set_initial_ranking_state(succ_array_, root_array_, rank_array_, *comm_);
    kascade::pointer_doubling(root_array_, rank_array_, dist, *comm_);
  }
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
