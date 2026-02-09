#include "kascade/sparse_ruling_set.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <queue>
#include <random>
#include <ranges>
#include <utility>

#include <absl/container/flat_hash_map.h>
#include <briefkasten/queue_builder.hpp>
#include <fmt/ranges.h>
#include <kamping/collectives/allreduce.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/communicator.hpp>
#include <kamping/data_buffer.hpp>
#include <kamping/measurements/counter.hpp>
#include <kamping/measurements/measurement_aggregation_definitions.hpp>
#include <kamping/measurements/timer.hpp>
#include <kamping/mpi_ops.hpp>
#include <kamping/named_parameters.hpp>
#include <kamping/utils/flatten.hpp>
#include <kamping/types/tuple.hpp>
#include <kassert/kassert.hpp>
#include <spdlog/spdlog.h>

#include "kascade/assertion_levels.hpp"
#include "kascade/bits.hpp"
#include "kascade/configuration.hpp"
#include "kascade/distribution.hpp"
#include "kascade/pointer_doubling_generic.hpp"
#include "kascade/successor_utils.hpp"
#include "kascade/types.hpp"

namespace kascade {
namespace {
auto compute_local_num_rulers(SparseRulingSetConfig const& config,
                              Distribution const& dist,
                              kamping::Communicator<> const& comm) -> std::size_t {
  switch (config.ruler_selection) {
    case RulerSelectionStrategy::dehne:
      // pick O(n/p) rulers in total
      return static_cast<std::size_t>(config.dehne_factor *
                                      (static_cast<double>(dist.get_global_size()) /
                                       static_cast<double>(comm.size()))) /
             comm.size();
    case RulerSelectionStrategy::heuristic:
      // pick heuristic_factor * local_num_leaves per PE
      return static_cast<std::size_t>(
          config.heuristic_factor *
          static_cast<double>(dist.get_local_size(comm.rank())));
    case RulerSelectionStrategy::invalid:
      throw std::runtime_error("Invalid ruler selection strategy");
      break;
  }
  std::unreachable();
}
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

struct RulerMessage {
  idx_t target_idx;
  idx_t ruler;
  rank_t dist_from_ruler;
};

namespace ruler_chasing {
struct sync_tag {};
struct async_tag {};
constexpr sync_tag sync{};
constexpr async_tag async{};
}  // namespace ruler_chasing

auto handle_messages(auto&& initialize,
                     auto&& work_on_item,
                     Distribution const& dist,
                     kamping::Communicator<> const& comm,
                     ruler_chasing::async_tag /* tag */) {
  auto queue =
      briefkasten::BufferedMessageQueueBuilder<RulerMessage>(comm.mpi_communicator())
          .build();
  std::queue<RulerMessage> local_queue;
  auto on_message = [&](auto env) {
    for (RulerMessage const& msg : env.message) {
      KASSERT(dist.get_owner(msg.target_idx) == comm.rank());
      local_queue.push(msg);
    }
  };
  auto enqueue_locally = [&](RulerMessage const& msg) { local_queue.push(msg); };
  auto send_to = [&](RulerMessage const& msg, int target) {
    queue.post_message_blocking(msg, target, on_message);
  };
  initialize(enqueue_locally);
  do {  // NOLINT(*-avoid-do-while)
    while (!local_queue.empty()) {
      queue.poll_throttled(on_message);
      auto msg = local_queue.front();
      local_queue.pop();
      work_on_item(msg, enqueue_locally, send_to);
    }
  } while (!queue.terminate(on_message));
}

auto handle_messages(auto&& initialize,
                     auto&& work_on_item,
                     Distribution const& /* dist */,
                     kamping::Communicator<> const& comm,
                     ruler_chasing::sync_tag /* tag */) {
  auto queue =
      briefkasten::BufferedMessageQueueBuilder<RulerMessage>(comm.mpi_communicator())
          .build();
  std::vector<RulerMessage> local_work;
  absl::flat_hash_map<int, std::vector<RulerMessage>> messages;

  auto enqueue_locally_init = [&](RulerMessage const& msg) { local_work.push_back(msg); };
  auto send_to = [&](RulerMessage const& msg, int target) {
    messages[target].push_back(msg);
  };
  auto enqueue_locally = [&](RulerMessage const& msg) {
    messages[comm.rank_signed()].push_back(msg);
  };
  initialize(enqueue_locally_init);
  namespace kmp = kamping::params;
  std::int64_t rounds = 0;
  while (!comm.allreduce_single(kmp::send_buf(local_work.empty()),
                                kmp::op(std::logical_and<>{}))) {
    for (auto const& msg : local_work) {
      work_on_item(msg, enqueue_locally, send_to);
    }
    auto [send_buf, send_counts, send_displs] = kamping::flatten(messages, comm.size());
    messages.clear();
    kamping::measurements::timer().start("alltoall");
    comm.alltoallv(kmp::recv_buf<kamping::BufferResizePolicy::resize_to_fit>(local_work),
                   kmp::send_buf(send_buf), kmp::send_counts(send_counts),
                   kmp::send_displs(send_displs));
    kamping::measurements::timer().stop_and_append();
    rounds++;
  }
  kamping::measurements::counter().add(
      "ruler_chasing_rounds", rounds,
      {kamping::measurements::GlobalAggregationMode::max,
       kamping::measurements::GlobalAggregationMode::min});
}
}  // namespace

void sparse_ruling_set(SparseRulingSetConfig const& config,
                       std::span<idx_t> succ_array,
                       std::span<rank_t> rank_array,
                       Distribution const& dist,
                       kamping::Communicator<> const& comm) {
  KASSERT(is_list(succ_array, dist, comm), kascade::assert::with_communication);
  kamping::measurements::timer().synchronize_and_start("invert_list");
  invert_list(succ_array, rank_array, succ_array, rank_array, dist, comm);
  kamping::measurements::timer().stop();

  kamping::measurements::timer().synchronize_and_start("find_rulers");
  LeafInfo leaf_info{succ_array, dist, comm};

  std::int64_t local_num_rulers =
      static_cast<std::int64_t>(compute_local_num_rulers(config, dist, comm)) -
      leaf_info.num_local_leaves();
  local_num_rulers = std::max(local_num_rulers, std::int64_t{0});
  kamping::measurements::counter().add(
      "local_num_ruler", local_num_rulers,
      {kamping::measurements::GlobalAggregationMode::max,
       kamping::measurements::GlobalAggregationMode::min});
  kamping::measurements::counter().add(
      "local_num_leaves", static_cast<std::int64_t>(leaf_info.num_local_leaves()),
      {kamping::measurements::GlobalAggregationMode::max,
       kamping::measurements::GlobalAggregationMode::min});
  SPDLOG_DEBUG("picking {} rulers", local_num_rulers);

  auto rulers =
      pick_rulers(succ_array, local_num_rulers, 42 + comm.rank(), [&](idx_t local_idx) {
        return !is_root(local_idx, succ_array, dist, comm) &&
               !leaf_info.is_leaf(local_idx);
      });
  kamping::measurements::timer().stop();

  //////////////////////
  // Ruler chasing    //
  //////////////////////
  kamping::measurements::timer().synchronize_and_start("chase_ruler");

  // initialization
  for (auto leaf : leaf_info.leaves()) {
    rulers.push_back(leaf);
  }
  // chasing loop
  auto init = [&](auto&& enqueue_locally) {
    for (auto const& ruler_local : rulers) {
      auto ruler = dist.get_global_idx(ruler_local, comm.rank());
      auto target_idx = ruler;
      if (leaf_info.is_leaf(ruler_local)) {
        // this is a leaf, so set the msb to distinguish it from normal rulers, which
        // will help to avoid them requesting their ruler's information in the end
        ruler = bits::set_root_flag(ruler);
      }
      enqueue_locally(
          RulerMessage{.target_idx = target_idx, .ruler = ruler, .dist_from_ruler = 0});
    }
  };
  absl::flat_hash_map<idx_t, idx_t> ruler_list_length;
  auto work_on_item = [&](RulerMessage const& msg, auto&& enqueue_locally,
                          auto&& send_to) {
    const auto& [idx, ruler, dist_from_ruler] = msg;
    KASSERT(dist.get_owner(idx) == comm.rank());
    auto idx_local = dist.get_local_idx(idx, comm.rank());
    auto succ = succ_array[idx_local];
    auto dist_to_succ = rank_array[idx_local];
    succ_array[idx_local] = ruler;
    rank_array[idx_local] = dist_from_ruler;
    if (succ == idx) {
      // this is either a root, or a ruler, which points to itself after
      // initialization
      ruler_list_length[ruler] = dist_from_ruler;
      return;
    }
    auto succ_rank = dist.get_owner_signed(succ);
    if (succ_rank == comm.rank_signed()) {
      enqueue_locally({.target_idx = succ,
                       .ruler = ruler,
                       .dist_from_ruler = dist_from_ruler + dist_to_succ});
      return;
    }
    send_to({.target_idx = succ,
             .ruler = ruler,
             .dist_from_ruler = dist_from_ruler + dist_to_succ},
            succ_rank);
  };
  if (config.sync) {
    handle_messages(init, work_on_item, dist, comm, ruler_chasing::sync);
  } else {
    handle_messages(init, work_on_item, dist, comm, ruler_chasing::async);
  }
  kamping::measurements::timer().stop();

  // resetting the leafs' msb
  for (auto local_idx : rulers) {
    succ_array[local_idx] = bits::clear_root_flag(succ_array[local_idx]);
  }

  kamping::measurements::timer().synchronize_and_start("base_case");
  PointerDoublingConfig conf;
  pointer_doubling_generic(conf, succ_array, rank_array, dist, rulers, comm);
  kamping::measurements::timer().stop();

  // set msb for all rulers to avoid them requesting their ruler's information in the next
  // step
  for (auto local_idx : rulers) {
    succ_array[local_idx] = bits::set_root_flag(succ_array[local_idx]);
  }

  ///////////////////////////////////////
  // Request rank and root from rulers //
  ///////////////////////////////////////
  kamping::measurements::timer().synchronize_and_start("ruler_propagation");

  struct ruler_request {
    idx_t requester;
    idx_t ruler;
  };
  absl::flat_hash_map<int, std::vector<ruler_request>> requests;
  for (auto local_idx : dist.local_indices(comm.rank())) {
    if (bits::has_root_flag(succ_array[local_idx])) {
      // this is a ruler, so skip
      continue;
    }
    auto ruler = succ_array[local_idx];
    if (dist.get_owner(ruler) == comm.rank()) {
      auto ruler_local = dist.get_local_idx(ruler, comm.rank());
      succ_array[local_idx] = bits::clear_root_flag(succ_array[ruler_local]);
      rank_array[local_idx] = rank_array[ruler_local] + rank_array[local_idx];
      continue;
    }
    requests[dist.get_owner_signed(ruler)].push_back(ruler_request{
        .requester = dist.get_global_idx(local_idx, comm.rank()), .ruler = ruler});
  }
  for (auto local_idx : dist.local_indices(comm.rank())) {
    succ_array[local_idx] = bits::clear_root_flag(succ_array[local_idx]);
  }

  auto requests_received =
      kamping::with_flattened(requests, comm.size()).call([&](auto... flattened) {
        requests.clear();
        return comm.alltoallv(std::move(flattened)...);
      });

  struct ruler_reply {
    idx_t requester;
    idx_t ruler;
    rank_t dist;
  };
  absl::flat_hash_map<idx_t, std::vector<ruler_reply>> replies;
  for (auto const& msg : requests_received) {
    KASSERT(dist.get_owner(msg.ruler) == comm.rank());
    auto ruler_local = dist.get_local_idx(msg.ruler, comm.rank());
    KASSERT(!leaf_info.is_leaf(ruler_local));
    replies[dist.get_owner_signed(msg.requester)].push_back(
        ruler_reply{.requester = msg.requester,
                    .ruler = succ_array[ruler_local],
                    .dist = rank_array[ruler_local]});
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
    succ_array[requester_local] = msg.ruler;
    rank_array[requester_local] += msg.dist;
  }
  kamping::measurements::timer().stop();
  kamping::measurements::timer().start("stats");
  idx_t min_length = std::numeric_limits<idx_t>::max();
  idx_t max_length = std::numeric_limits<idx_t>::min();
  idx_t length_sum = 0;
  std::size_t num_rulers = ruler_list_length.size();
  for (auto& [_, length] : ruler_list_length) {
    min_length = std::min(min_length, length);
    max_length = std::max(max_length, length);
    length_sum += length;
  }
  auto data = std::make_tuple(min_length, max_length, length_sum, num_rulers);
  auto op = [](auto const& lhs, auto const& rhs) {
    return std::make_tuple(std::min(std::get<0>(lhs), std::get<0>(rhs)),
                           std::max(std::get<1>(lhs), std::get<1>(rhs)),
                           std::plus<>{}(std::get<2>(lhs), std::get<2>(rhs)),
                           std::plus<>{}(std::get<3>(lhs), std::get<3>(rhs)));
  };
  comm.allreduce(kamping::send_buf(data), kamping::op(op, kamping::ops::commutative));
  kamping::measurements::counter().add(
      "ruler_list_length_min", std::get<0>(data),
      {
          kamping::measurements::GlobalAggregationMode::min,
      });
  kamping::measurements::counter().add(
      "ruler_list_length_max", std::get<1>(data),
      {
          kamping::measurements::GlobalAggregationMode::max,
      });
  kamping::measurements::counter().add("ruler_list_length_avg",
      static_cast<double>(std::get<2>(data)) / static_cast<double>(std::get<3>(data)),
      {
          kamping::measurements::GlobalAggregationMode::max,
      });
  kamping::measurements::timer().stop();
}
}  // namespace kascade
