#include "kascade/sparse_ruling_set.hpp"

#include <algorithm>
#include <queue>
#include <random>
#include <ranges>

#include <absl/container/flat_hash_map.h>
#include <briefkasten/queue_builder.hpp>
#include <fmt/ranges.h>
#include <kamping/utils/flatten.hpp>
#include <kassert/kassert.hpp>
#include <spdlog/spdlog.h>

#include "kascade/assertion_levels.hpp"
#include "kascade/configuration.hpp"
#include "kascade/distribution.hpp"
#include "kascade/pointer_doubling.hpp"
#include "kascade/pointer_doubling_generic.hpp"
#include "kascade/successor_utils.hpp"
#include "kascade/types.hpp"

namespace kascade {
namespace {
auto pick_rulers(std::span<const idx_t> succ_array,
                 std::size_t local_num_rulers,
                 std::mt19937::result_type seed,
                 std::predicate<idx_t> auto const& idx_predicate) -> std::vector<idx_t> {
  std::vector<idx_t> rulers(local_num_rulers);
  std::mt19937 rng{seed};
  auto indices = std::views::iota(idx_t{0}, static_cast<idx_t>(succ_array.size())) |
                 std::views::filter(idx_predicate);
  auto it = std::ranges::sample(
      indices, rulers.begin(),
      static_cast<std::ranges::range_difference_t<decltype(indices)>>(local_num_rulers),
      rng);
  rulers.erase(it, rulers.end());
  return rulers;
}
}  // namespace

void sparse_ruling_set(std::span<idx_t> succ_array,
                       std::span<idx_t> rank_array,
                       Distribution const& dist,
                       kamping::Communicator<> const& comm) {
  KASSERT(is_list(succ_array, dist, comm), kascade::assert::with_communication);
  SPDLOG_TRACE("succ={}, ranks={}", succ_array, rank_array);
  invert_list(succ_array, rank_array, succ_array, rank_array, dist, comm);
  SPDLOG_TRACE("inverted: succ={}, ranks={}", succ_array, rank_array);
  LeafInfo leaf_info{succ_array, dist, comm};

  std::vector<bool> is_ruler(succ_array.size(), false);
  std::size_t local_num_rulers = 2;  // FIXME
  auto rulers =
      pick_rulers(succ_array, local_num_rulers, 42 + comm.rank(), [&](idx_t local_idx) {
        return !is_root(local_idx, succ_array, dist, comm) &&
               !leaf_info.is_leaf(local_idx);
      });
  for (auto const& ruler : rulers) {
    is_ruler[ruler] = true;
  }
  SPDLOG_TRACE("rulers={}, roots={}, leaves={}", rulers, roots(succ_array, dist, comm),
               leaves(succ_array, dist, comm));

  std::vector<idx_t> ruler_array(succ_array.size(), -1);
  std::vector<idx_t> dist_from_ruler_array(succ_array.size());

  std::queue<idx_t> local_queue;

  for (std::size_t local_idx : dist.local_indices(comm.rank())) {
    if (is_ruler[local_idx] || leaf_info.is_leaf(local_idx)) {
      auto global_idx = dist.get_global_idx(local_idx, comm.rank());
      local_queue.push(global_idx);
      ruler_array[local_idx] = global_idx;
      dist_from_ruler_array[local_idx] = 0;
    }
  }
  // SPDLOG_TRACE("ruler_array={}", ruler_array);
  using message_type = struct {
    idx_t target_idx;
    idx_t ruler;
    idx_t ruler_dist;
  };
  auto queue =
      briefkasten::BufferedMessageQueueBuilder<message_type>(comm.mpi_communicator())
          .build();
  auto on_message = [&](auto env) {
    for (message_type const& msg : env.message) {
      auto target_idx = msg.target_idx;
      KASSERT(dist.get_owner(target_idx) == comm.rank());
      auto target_idx_local = dist.get_local_idx(target_idx, comm.rank());
      ruler_array[target_idx_local] = msg.ruler;
      dist_from_ruler_array[target_idx_local] = msg.ruler_dist;
      if (is_ruler[target_idx_local] ||
          is_root(target_idx_local, succ_array, dist, comm)) {
        continue;
      }
      local_queue.push(target_idx);
    }
  };
  do {  // NOLINT(*-avoid-do-while)
    while (!local_queue.empty()) {
      queue.poll_throttled(on_message);
      auto idx = local_queue.front();
      local_queue.pop();
      KASSERT(dist.get_owner(idx) == comm.rank());
      auto idx_local = dist.get_local_idx(idx, comm.rank());
      // if (is_ruler[idx_local] || is_root(idx, succ_array, dist, comm)) {
      //   continue;
      // }
      auto succ = succ_array[idx_local];
      auto ruler = ruler_array[idx_local];
      auto dist_from_ruler = dist_from_ruler_array[idx_local];
      auto dist_to_succ = rank_array[idx_local];
      auto succ_rank = dist.get_owner_signed(succ);
      if (succ_rank == comm.rank_signed()) {
        auto succ_local = dist.get_local_idx(succ, comm.rank());
        ruler_array[succ_local] = ruler;
        dist_from_ruler_array[succ_local] = dist_from_ruler + dist_to_succ;
        if (is_ruler[succ_local] || is_root(succ_local, succ_array, dist, comm)) {
          continue;
        }
        local_queue.push(succ);
        continue;
      }
      queue.post_message_blocking({.target_idx = succ,
                                   .ruler = ruler,
                                   .ruler_dist = dist_from_ruler + dist_to_succ},
                                  succ_rank, on_message);
    }
  } while (!queue.terminate(on_message));
  SPDLOG_TRACE("rulers={}, dist_from_ruler={}", ruler_array, dist_from_ruler_array);
  PointerDoublingConfig conf;
  for (auto leaf : leaf_info.leaves()) {
    rulers.push_back(leaf);
  }
  pointer_doubling_generic(conf, ruler_array, dist_from_ruler_array, dist, rulers, comm);
  SPDLOG_TRACE("after recursion: rulers={}, dist_from_ruler={}", ruler_array,
               dist_from_ruler_array);
  struct ruler_request {
    idx_t requester;
    idx_t ruler;
  };
  absl::flat_hash_map<int, std::vector<ruler_request>> requests;
  for (auto local_idx : dist.local_indices(comm.rank())) {
    if (!(is_ruler[local_idx] || leaf_info.is_leaf(local_idx))) {
      auto ruler = ruler_array[local_idx];
      if (dist.get_owner(ruler) == comm.rank()) {
        auto ruler_local = dist.get_local_idx(ruler, comm.rank());
        ruler_array[local_idx] = ruler_array[ruler_local];
        dist_from_ruler_array[local_idx] =
            dist_from_ruler_array[ruler_local] + dist_from_ruler_array[local_idx];
        continue;
      }
      requests[dist.get_owner_signed(ruler)].push_back(ruler_request{
          .requester = dist.get_global_idx(local_idx, comm.rank()), .ruler = ruler});
    }
  }
  auto requests_received =
      kamping::with_flattened(requests, comm.size()).call([&](auto... flattened) {
        requests.clear();
        return comm.alltoallv(std::move(flattened)...);
      });
  struct ruler_reply {
    idx_t requester;
    idx_t ruler;
    idx_t dist;
  };
  absl::flat_hash_map<idx_t, std::vector<ruler_reply>> replies;
  for (auto const& msg : requests_received) {
    KASSERT(dist.get_owner(msg.ruler) == comm.rank());
    auto ruler_local = dist.get_local_idx(msg.ruler, comm.rank());
    replies[dist.get_owner_signed(msg.requester)].push_back(
        ruler_reply{.requester = msg.requester,
                    .ruler = ruler_array[ruler_local],
                    .dist = dist_from_ruler_array[ruler_local]});
  }
  requests_received.clear();
  auto replies_received =
      kamping::with_flattened(replies, comm.size()).call([&](auto... flattened) {
        replies.clear();
        return comm.alltoallv(std::move(flattened)...);
      });
  for (auto const& msg : replies_received) {
    KASSERT(dist.get_owner(msg.requester) == comm.rank());
    auto requester_local = dist.get_local_idx(msg.requester, comm.rank());
    ruler_array[requester_local] = msg.ruler;
    dist_from_ruler_array[requester_local] += msg.dist;
  }

  std::ranges::copy(ruler_array, succ_array.begin());
  std::ranges::copy(dist_from_ruler_array, rank_array.begin());

  // Distribution sub_dist{sub_succ.size(), comm};
  // pointer_doubling(conf, sub_succ, sub_rank, sub_dist, comm);
  // std::ranges::copy(sub_arrays_zipped, subproblem.begin());
  // SPDLOG_TRACE("final: succ={}, ranks={}", succ_array, rank_array);

  // pointer_doubling(, std::span<idx_t> succ_array, std::span<idx_t> rank_array, const
  // Distribution &dist, const kamping::Communicator<> &comm) SPDLOG_TRACE("sub_succ={},
  // sub_rank={}", sub_succ, sub_rank);
}
}  // namespace kascade
