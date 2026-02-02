#include <ranges>

#include <kascade/pointer_doubling.hpp>

#include "kascade/pointer_doubling_generic.hpp"

void kascade::pointer_doubling(kascade::PointerDoublingConfig config,
                               std::span<idx_t> succ_array,
                               std::span<idx_t> rank_array,
                               Distribution const& dist,
                               kamping::Communicator<> const& comm) {
  auto all_local_indices = std::views::iota(idx_t{0}, idx_t{succ_array.size()});
  pointer_doubling_generic(config, succ_array, rank_array, dist, all_local_indices, comm);
}
