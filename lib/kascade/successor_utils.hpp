#pragma once

#include <span>
#include <vector>

#include <kamping/communicator.hpp>

#include "kascade/distribution.hpp"
#include "kascade/types.hpp"

namespace kascade {

auto is_list(std::span<const idx_t> succ_array, kamping::Communicator<> const& comm)
    -> bool;

auto is_list(std::span<const idx_t> succ_array,
             Distribution const& dist,
             kamping::Communicator<> const& comm) -> bool;

auto is_root(std::size_t local_idx,
             std::span<const idx_t> succ_array,
             Distribution const& dist,
             kamping::Communicator<> const& comm) -> bool;

struct leaf_info {
private:
  std::vector<bool> has_pred;  // per-local index
  Distribution const* dist;
  kamping::Communicator<> const* comm;

public:
  leaf_info(std::span<const idx_t> succ_array,
            Distribution const& dist,
            kamping::Communicator<> const& comm);

  [[nodiscard]] auto is_leaf(idx_t local_idx) const -> bool;
};

auto roots(std::span<const idx_t> succ_array,
           Distribution const& dist,
           kamping::Communicator<> const& comm) -> std::vector<idx_t>;

auto leaves(std::span<const idx_t> succ_array,
            Distribution const& dist,
            kamping::Communicator<> const& comm) -> std::vector<idx_t>;
}  // namespace kascade
