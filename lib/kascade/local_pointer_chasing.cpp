#include <span>

#include <kassert/kassert.hpp>

#include "kascade/distribution.hpp"
#include "kascade/list_ranking.hpp"
#include "kascade/local_pointer_chasing_generic.hpp"
#include "kascade/types.hpp"

namespace kascade {

void local_pointer_chasing(std::span<idx_t> succ_array,
                           std::span<idx_t> rank_array,
                           std::size_t rank,
                           Distribution const& dist) {
  auto is_local = [&](idx_t idx) {
    auto begin = dist.get_exclusive_prefix(rank);
    return begin <= idx && idx < begin + dist.get_count(rank);
  };
  auto to_global = [&](idx_t idx) { return dist.get_global_idx(idx, rank); };
  auto to_local = [&](idx_t idx) { return dist.get_local_idx(idx, rank); };
  local_pointer_chasing_generic(succ_array, rank_array, to_local, to_global, is_local);
}
void local_pointer_chasing(std::span<idx_t> succ_array, std::span<idx_t> rank_array) {
  auto is_local = [](idx_t) { return true; };
  auto identity = std::identity{};
  local_pointer_chasing_generic(succ_array, rank_array, identity, identity, is_local);
}
}  // namespace kascade
