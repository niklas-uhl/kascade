#pragma once

#include <span>
#include <vector>
#include "kascade/types.hpp"

class AbstractAlgorithm {
public:
  AbstractAlgorithm() = default;
  AbstractAlgorithm(const AbstractAlgorithm&) = default;
  AbstractAlgorithm(AbstractAlgorithm&&) = delete;
  auto operator=(const AbstractAlgorithm&) -> AbstractAlgorithm& = default;
  auto operator=(AbstractAlgorithm&&) -> AbstractAlgorithm& = delete;

  virtual void ingest(std::span<const kascade::idx_t> succ_array) = 0;
  virtual void run() = 0;
  virtual auto get_rank_array() -> std::vector<kascade::idx_t> const& = 0;
  virtual auto get_root_array() -> std::vector<kascade::idx_t> const& = 0;
  virtual ~AbstractAlgorithm() = default;
};
