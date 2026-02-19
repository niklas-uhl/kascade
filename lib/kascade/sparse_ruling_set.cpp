#include "kascade/sparse_ruling_set.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <queue>
#include <random>
#include <ranges>
#include <utility>

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <briefkasten/buffered_queue.hpp>
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
#include <kassert/kassert.hpp>
#include <spdlog/spdlog.h>

#include "kascade/assertion_levels.hpp"
#include "kascade/bits.hpp"
#include "kascade/configuration.hpp"
#include "kascade/distribution.hpp"
#include "kascade/grid_alltoall.hpp"
#include "kascade/list_ranking.hpp"
#include "kascade/pack.hpp"
#include "kascade/pointer_doubling.hpp"
#include "kascade/successor_utils.hpp"
#include "kascade/types.hpp"

namespace kascade {
namespace {
auto compute_local_num_rulers(SparseRulingSetConfig const& config,
                              Distribution const& dist,
                              kamping::Communicator<> const& comm) -> std::size_t {
  // NOLINTBEGIN(readability-identifier-length)
  auto n = dist.get_global_size();
  auto p = comm.size();
  auto rel_local_size =
      static_cast<double>(dist.get_local_size(comm.rank())) / static_cast<double>(n);
  // NOLINTEND(readability-identifier-length)

  switch (config.ruler_selection) {
    case RulerSelectionStrategy::dehne:
      // pick O(n/p) rulers in total
      return static_cast<std::size_t>(config.dehne_factor *
                                      (static_cast<double>(n) / static_cast<double>(p))) /
             p;
    case RulerSelectionStrategy::heuristic:
      // pick heuristic_factor * local_num_leaves per PE
      return static_cast<std::size_t>(
          config.heuristic_factor *
          static_cast<double>(dist.get_local_size(comm.rank())));
    case kascade::RulerSelectionStrategy::sanders:
      return static_cast<std::size_t>(config.sanders_factor * std::sqrt(n) *
                                      static_cast<double>(p) / std::log(n) *
                                      rel_local_size);
    case RulerSelectionStrategy::invalid:
      throw std::runtime_error("Invalid ruler selection strategy");
      break;
  }
  std::unreachable();
}
auto pick_rulers(std::span<const idx_t> succ_array,
                 std::size_t local_num_rulers,
                 auto& rng,
                 std::predicate<idx_t> auto const& idx_predicate) -> std::vector<idx_t> {
  std::vector<idx_t> rulers(local_num_rulers);
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

auto handle_messages(SparseRulingSetConfig const& config,
                     auto&& initialize,
                     auto&& work_on_item,
                     Distribution const& dist,
                     kamping::Communicator<> const& comm,
                     ruler_chasing::async_tag /* tag */) {
  briefkasten::Config briefkasten_config;
  briefkasten_config.local_threshold_bytes = config.briefkasten.local_threshold;
  auto queue = briefkasten::BufferedMessageQueueBuilder<RulerMessage>(
                   briefkasten_config, comm.mpi_communicator())
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
  initialize(enqueue_locally, send_to);
  do {  // NOLINT(*-avoid-do-while)
    while (!local_queue.empty()) {
      queue.poll_throttled(on_message, config.briefkasten.poll_skip_threshold);
      auto msg = local_queue.front();
      local_queue.pop();
      work_on_item(msg, enqueue_locally, send_to);
    }
  } while (!queue.terminate(on_message));
}

auto handle_messages(SparseRulingSetConfig const& config,
                     auto&& initialize,
                     auto&& work_on_item,
                     Distribution const& /* dist */,
                     kamping::Communicator<> const& comm,
                     ruler_chasing::sync_tag /* tag */) -> std::size_t {
  auto queue =
      briefkasten::BufferedMessageQueueBuilder<RulerMessage>(comm.mpi_communicator())
          .build();
  std::vector<RulerMessage> local_work;
  std::vector<std::pair<int, RulerMessage>> messages;

  auto send_to = [&](RulerMessage const& msg, int target) {
    messages.emplace_back(target, msg);
  };
  auto enqueue_locally_to_message_buffer = [&](RulerMessage const& msg) {
    messages.emplace_back(comm.rank_signed(), msg);
  };
  auto enqueue_locally = [&](RulerMessage const& msg) { local_work.push_back(msg); };
  initialize(enqueue_locally_to_message_buffer, send_to);

  AlltoallDispatcher<RulerMessage> dispatcher(config.use_grid_communication, comm);

  std::size_t rounds = 0;
  namespace kmp = kamping::params;
  while (!comm.allreduce_single(kmp::send_buf(messages.empty()),
                                kmp::op(std::logical_and<>{}))) {
    kamping::measurements::timer().start("alltoall");
    dispatcher.alltoallv(messages, local_work);
    messages.clear();
    kamping::measurements::timer().stop_and_append();

    for (std::size_t i = 0; i < local_work.size(); ++i) {
      // we copy the message here, since enqueue_locally might append new messages to
      // local_work, which might invalidate references
      auto const msg = local_work[i];
      if (config.sync_locality_aware) {
        work_on_item(msg, enqueue_locally, send_to);
      } else {
        work_on_item(msg, enqueue_locally_to_message_buffer, send_to);
      }
    }
    rounds++;
  }
  return rounds;
}

enum class NodeType : std::uint8_t { root, leaf, ruler, unreached, reached };

struct RulerTrace {
  absl::flat_hash_map<idx_t, idx_t> ruler_list_length;
  std::size_t local_subproblem_size_{};
  RulerTrace(const RulerTrace&) = default;
  RulerTrace(RulerTrace&&) = delete;
  auto operator=(const RulerTrace&) -> RulerTrace& = default;
  auto operator=(RulerTrace&&) -> RulerTrace& = delete;
  std::size_t local_num_rulers_;
  std::size_t local_num_leaves_;
  RulerTrace(std::size_t local_num_rulers, std::size_t local_num_leaves)
      : local_num_rulers_(local_num_rulers), local_num_leaves_(local_num_leaves) {}
  ~RulerTrace() {
    kamping::measurements::timer().start("aggregate_ruler_stats");
    idx_t min_length = std::numeric_limits<idx_t>::max();
    idx_t max_length = std::numeric_limits<idx_t>::min();
    idx_t length_sum = 0;
    std::size_t num_rulers = ruler_list_length.size();
    for (auto& [_, length] : ruler_list_length) {
      min_length = std::min(min_length, length);
      max_length = std::max(max_length, length);
      length_sum += length;
    }
    struct ruler_stats {
      idx_t min_length;
      idx_t max_length;
      idx_t length_sum;
      std::size_t num_rulers;
    };
    ruler_stats stats{.min_length = min_length,
                      .max_length = max_length,
                      .length_sum = length_sum,
                      .num_rulers = num_rulers};
    auto agg = [](auto const& lhs, auto const& rhs) {
      return ruler_stats{.min_length = std::min(lhs.min_length, rhs.min_length),
                         .max_length = std::max(lhs.max_length, rhs.max_length),
                         .length_sum = std::plus<>{}(lhs.length_sum, rhs.length_sum),
                         .num_rulers = std::plus<>{}(lhs.num_rulers, rhs.num_rulers)};
    };
    kamping::comm_world().allreduce(kamping::send_recv_buf(stats),
                                    kamping::op(agg, kamping::ops::commutative));
    kamping::measurements::counter().append(
        "ruler_list_length_min", static_cast<std::int64_t>(stats.min_length),
        {
            kamping::measurements::GlobalAggregationMode::min,
        });
    kamping::measurements::counter().append(
        "ruler_list_length_max", static_cast<std::int64_t>(stats.max_length),
        {
            kamping::measurements::GlobalAggregationMode::max,
        });
    kamping::measurements::counter().append(
        "ruler_list_length_avg",
        static_cast<std::int64_t>(static_cast<double>(stats.length_sum) /
                                  static_cast<double>(stats.num_rulers)),
        {
            kamping::measurements::GlobalAggregationMode::max,
        });
    kamping::measurements::counter().append(
        "local_num_ruler", static_cast<std::int64_t>(local_num_rulers_),
        {kamping::measurements::GlobalAggregationMode::max,
         kamping::measurements::GlobalAggregationMode::min});
    kamping::measurements::counter().append(
        "local_num_leaves", static_cast<std::int64_t>(local_num_leaves_),
        {kamping::measurements::GlobalAggregationMode::max,
         kamping::measurements::GlobalAggregationMode::min});
    kamping::measurements::counter().append(
        "local_subproblem_size", static_cast<std::int64_t>(local_subproblem_size_),
        {kamping::measurements::GlobalAggregationMode::sum,
         kamping::measurements::GlobalAggregationMode::max,
         kamping::measurements::GlobalAggregationMode::min});
    kamping::measurements::counter().append(
        "num_spawned_rulers", static_cast<std::int64_t>(num_spawned_rulers_),
        {kamping::measurements::GlobalAggregationMode::sum,
         kamping::measurements::GlobalAggregationMode::max,
         kamping::measurements::GlobalAggregationMode::min});
    kamping::measurements::counter().append(
        "ruler_chasing_rounds", static_cast<std::int64_t>(rounds_),
        {kamping::measurements::GlobalAggregationMode::max,
         kamping::measurements::GlobalAggregationMode::min});
    kamping::measurements::timer().stop();
  }
  void track_chain_end(idx_t ruler, rank_t dist_from_ruler) {
    ruler_list_length[ruler] = dist_from_ruler;
  }
  std::size_t num_spawned_rulers_ = 0;
  void track_spawn() { num_spawned_rulers_++; }
  void track_base_case(std::size_t local_subproblem_size) {
    local_subproblem_size_ = local_subproblem_size;
  }
  std::size_t rounds_;
  void track_ruler_chasing_rounds(std::size_t ruler_chasing_rounds) {
    rounds_ = ruler_chasing_rounds;
  }
};

namespace propagation_mode {
struct pull_tag {};
struct push_tag {};
constexpr pull_tag pull{};
constexpr push_tag push{};
}  // namespace propagation_mode

auto ruler_propagation(SparseRulingSetConfig const& config,
                       std::span<idx_t> succ_array,
                       std::span<rank_t> rank_array,
                       std::vector<NodeType> const& node_type,
                       Distribution const& dist,
                       kamping::Communicator<> const& comm,
                       propagation_mode::pull_tag /* tag */ = {}) {
  auto needs_to_request_ruler = [&](idx_t local_idx) {
    // if the msb is set, this node was reached from a leaf, so root and rank are already
    // correct
    // rulers and leafs also have the correct result already from the base algorithm
    return !bits::has_root_flag(succ_array[local_idx]) &&
           node_type[local_idx] != NodeType::ruler &&
           node_type[local_idx] != NodeType::leaf;
  };
  kamping::measurements::timer().start("collect_requests");
  std::size_t deduped_requests = 0;
  absl::flat_hash_set<idx_t> requested_rulers;
  for (auto local_idx : dist.local_indices(comm.rank())) {
    if (!needs_to_request_ruler(local_idx)) {
      continue;
    }
    auto ruler = succ_array[local_idx];
    if (dist.is_local(ruler, comm.rank())) {
      continue;
    }
    bool inserted = requested_rulers.insert(ruler).second;
    if (inserted) {
      SPDLOG_TRACE("Requesting info for ruler {} from PE {}", ruler,
                   dist.get_owner(ruler));
    } else {
      deduped_requests++;
    }
  }
  kamping::measurements::timer().stop();
  SPDLOG_DEBUG("[ruler_propagation] removed {} duplicate requests", deduped_requests);

  struct ruler_request {
    int requesting_rank;
    idx_t requested_ruler;
  };

  AlltoallDispatcher<ruler_request> request_dispatcher(config.use_grid_communication,
                                                       comm);
  kamping::measurements::timer().start("pack_requests");
  auto requests =
      requested_rulers | std::views::transform([&](auto const& requested_ruler) {
        return std::make_pair(dist.get_owner(requested_ruler),
                              ruler_request{comm.rank_signed(), requested_ruler});
      }) |
      std::ranges::to<std::vector>();
  kamping::measurements::timer().stop();
  kamping::measurements::timer().start("exchange_requests");
  auto requests_received = request_dispatcher.alltoallv(requests);
  kamping::measurements::timer().stop();

  // map requests to replies
  kamping::measurements::timer().start("build_replies");
  struct ruler_reply {
    idx_t ruler;
    idx_t root;
    rank_t dist_to_root;
  };
  AlltoallDispatcher<ruler_reply> reply_dispatcher(config.use_grid_communication, comm);
  auto replies = requests_received | std::views::transform([&](auto const& request) {
                   auto const& [requesting_rank, requested_ruler] = request;
                   KASSERT(dist.get_owner(requested_ruler) == comm.rank());
                   auto local_idx = dist.get_local_idx(requested_ruler, comm.rank());
                   return std::pair{requesting_rank,
                                    ruler_reply{.ruler = requested_ruler,
                                                .root = succ_array[local_idx],
                                                .dist_to_root = rank_array[local_idx]}};
                 }) |
                 std::ranges::to<std::vector>();
  kamping::measurements::timer().stop();
  kamping::measurements::timer().start("exchange_replies");
  auto replies_received = reply_dispatcher.alltoallv(replies);
  kamping::measurements::timer().stop();

  // store replies
  kamping::measurements::timer().start("process_replies");
  absl::flat_hash_map<idx_t, ruler_reply> ruler_info;
  for (auto const& reply : replies_received) {
    ruler_info[reply.ruler] = reply;
  }
  for (auto local_idx : dist.local_indices(comm.rank())) {
    if (!needs_to_request_ruler(local_idx)) {
      // this node might have been reached by a leaf, so its msb might be still be set,
      // fix that
      succ_array[local_idx] = bits::clear_root_flag(succ_array[local_idx]);
      continue;
    }
    auto ruler = succ_array[local_idx];
    if (dist.is_local(ruler, comm.rank())) {
      auto ruler_local = dist.get_local_idx(ruler, comm.rank());
      succ_array[local_idx] = bits::clear_root_flag(succ_array[ruler_local]);
      rank_array[local_idx] = rank_array[ruler_local] + rank_array[local_idx];
      continue;
    }
    auto info_it = ruler_info.find(ruler);
    KASSERT(info_it != ruler_info.end(),
            fmt::format("Did not receive info for ruler {}. This should not happen, "
                        "since we requested it.",
                        ruler));
    succ_array[local_idx] = info_it->second.root;
    rank_array[local_idx] += info_it->second.dist_to_root;
  }
  kamping::measurements::timer().stop();
}

auto ruler_propagation(SparseRulingSetConfig const& config,
                       std::span<idx_t> succ_array,
                       std::span<rank_t> rank_array,
                       std::span<idx_t> initial_succ_array,
                       std::vector<NodeType> const& node_type,
                       std::span<idx_t> rulers,
                       Distribution const& dist,
                       kamping::Communicator<> const& comm,
                       propagation_mode::push_tag /* tag */) {
  auto init = [&](auto&& enqueue_locally, auto&& send_to) {
    for (auto const& ruler_local : rulers) {
      if (node_type[ruler_local] == NodeType::leaf) {
        // nodes reached from leafs already have the correct root and rank, so we can just
        // skip them
        continue;
      }
      auto succ = initial_succ_array[ruler_local];
      auto ruler_root = succ_array[ruler_local];
      auto ruler_rank = rank_array[ruler_local];
      RulerMessage msg{
          .target_idx = succ, .ruler = ruler_root, .dist_from_ruler = ruler_rank};
      if (dist.is_local(succ, comm.rank())) {
        enqueue_locally(msg);
        continue;
      }
      send_to(msg, dist.get_owner(succ));
    }
  };
  auto work_on_item = [&](RulerMessage const& msg, auto&& enqueue_locally,
                          auto&& send_to) {
    const auto& [idx, ruler_root, ruler_rank] = msg;
    KASSERT(dist.is_local(idx, comm.rank()));
    auto idx_local = dist.get_local_idx(idx, comm.rank());
    if (node_type[idx_local] == NodeType::ruler) {
      return;
    }
    succ_array[idx_local] = ruler_root;
    rank_array[idx_local] += ruler_rank;
    if (node_type[idx_local] == NodeType::root) {
      return;
    }
    KASSERT(node_type[idx_local] == NodeType::reached);
    auto succ = initial_succ_array[idx_local];
    RulerMessage forward_msg = msg;
    forward_msg.target_idx = succ;
    if (dist.is_local(succ, comm.rank())) {
      enqueue_locally(forward_msg);
      return;
    }
    send_to(forward_msg, dist.get_owner(succ));
  };
  if (config.sync) {
    handle_messages(config, init, work_on_item, dist, comm, ruler_chasing::sync);
  } else {
    handle_messages(config, init, work_on_item, dist, comm, ruler_chasing::async);
  }
}
}  // namespace

using BaseAlgorithm = std::function<void(std::span<idx_t>,
                                         std::span<rank_t>,
                                         Distribution const&,
                                         kamping::Communicator<> const&)>;
namespace {

void sparse_ruling_set(SparseRulingSetConfig const& config,
                       std::span<idx_t> succ_array,
                       std::span<rank_t> rank_array,
                       Distribution const& dist,
                       BaseAlgorithm const& base_algorithm,
                       kamping::Communicator<> const& comm) {
  KASSERT(is_list(succ_array, dist, comm), kascade::assert::with_communication);
  kamping::measurements::timer().synchronize_and_start("invert_list");
  auto leaves = reverse_list(succ_array, rank_array, succ_array, rank_array, dist, comm,
                             config.use_grid_communication);
  kamping::measurements::timer().stop();

  kamping::measurements::timer().start("cache_owners");
  std::optional<std::vector<std::size_t>> succ_owner;
  if (config.cache_owners) {
    succ_owner.emplace(succ_array.size());
    std::vector<std::size_t> succ_owner(succ_array.size());
    for (std::size_t i = 0; i < succ_owner.size(); i++) {
      auto succ = succ_array[i];
      if (dist.is_local(succ, comm.rank())) {
        succ_owner[i] = comm.rank();
      } else {
        succ_owner[i] = dist.get_owner(succ);
      }
    }
  }
  kamping::measurements::timer().stop();
  auto get_succ_owner = [&](idx_t local_idx, idx_t succ_global) {
    if (!config.cache_owners) {
      return dist.get_owner(succ_global);
    }
    return (*succ_owner)[local_idx];
  };

  kamping::measurements::timer().start("init_node_type");
  std::vector<NodeType> node_type(succ_array.size(), NodeType::unreached);
  std::size_t num_unreached = 0;
  for (auto local_idx : dist.local_indices(comm.rank())) {
    if (is_root(local_idx, succ_array, dist, comm)) {
      node_type[local_idx] = NodeType::root;
    } else {
      num_unreached++;
    }
  }
  std::size_t num_real_leaves = 0;
  for (auto leaf_local : leaves) {
    if (node_type[leaf_local] == NodeType::root) {
      continue;
    }
    node_type[leaf_local] = NodeType::leaf;
    num_real_leaves++;
    num_unreached--;
  }
  kamping::measurements::timer().stop();
  kamping::measurements::timer().start("find_rulers");

  std::int64_t local_num_rulers =
      // NOLINTNEXTLINE(*-narrowing-conversions)
      static_cast<std::int64_t>(compute_local_num_rulers(config, dist, comm)) -
      num_real_leaves;
  local_num_rulers = std::max(local_num_rulers, std::int64_t{0});

  RulerTrace trace{static_cast<size_t>(local_num_rulers), num_real_leaves};

  SPDLOG_DEBUG("picking {} rulers", local_num_rulers);
  std::mt19937 rng{static_cast<std::mt19937::result_type>(42 + comm.rank_signed())};

  auto rulers = pick_rulers(succ_array, local_num_rulers, rng, [&](idx_t local_idx) {
    return node_type[local_idx] == NodeType::unreached;
  });
  num_unreached -= rulers.size();
  kamping::measurements::timer().stop();

  //////////////////////
  // Ruler chasing    //
  //////////////////////
  kamping::measurements::timer().synchronize_and_start("backup_succ_array");
  std::optional<std::vector<idx_t>> inital_succ_array;
  if (config.ruler_propagation_mode == RulerPropagationMode::push) {
    inital_succ_array = std::vector(succ_array.begin(), succ_array.end());
  }
  kamping::measurements::timer().stop();
  kamping::measurements::timer().synchronize_and_start("chase_ruler");

  // initialization
  for (auto leaf_local : leaves) {
    // only use "real" leaves as rulers
    if (node_type[leaf_local] == NodeType::leaf) {
      rulers.push_back(leaf_local);
    }
  }
  // chasing loop
  auto init = [&](auto&& enqueue_locally, auto&& send_to) {
    for (auto const& ruler_local : rulers) {
      auto ruler = dist.get_global_idx(ruler_local, comm.rank());
      auto succ = succ_array[ruler_local];
      auto dist_to_succ = rank_array[ruler_local];
      if (node_type[ruler_local] == NodeType::leaf) {
        // this is a leaf, so set the msb to distinguish it from normal rulers, which
        // will help to avoid them requesting their ruler's information in the end
        if (config.ruler_propagation_mode == RulerPropagationMode::pull) {
          ruler = bits::set_root_flag(ruler);
        }
        succ_array[ruler_local] = ruler;
        rank_array[ruler_local] = 0;
      } else {
        node_type[ruler_local] = NodeType::ruler;
      }
      if (dist.is_local(succ, comm.rank())) {
        enqueue_locally(
            {.target_idx = succ, .ruler = ruler, .dist_from_ruler = dist_to_succ});
        continue;
      }
      send_to({.target_idx = succ, .ruler = ruler, .dist_from_ruler = dist_to_succ},
              get_succ_owner(ruler_local, succ));
      // dist.get_owner(succ));
    }
  };

  auto spawn_new_ruler = [&](auto&& enqueue_locally, auto&& send_to) {
    if (num_unreached == 0) {
      return;
    }
    std::uniform_int_distribution<std::size_t> distribution(0, succ_array.size() - 1);
    std::size_t ruler_local{};
    do {
      ruler_local = distribution(rng);
      if (node_type[ruler_local] == NodeType::unreached) {
        break;
      }
    } while (true);
    rulers.push_back(ruler_local);
    trace.track_spawn();
    auto ruler = dist.get_global_idx(ruler_local, comm.rank());
    SPDLOG_TRACE("spawning new ruler {}", ruler);
    node_type[ruler_local] = NodeType::ruler;
    num_unreached--;
    auto succ = succ_array[ruler_local];
    auto dist_to_succ = rank_array[ruler_local];
    if (dist.is_local(succ, comm.rank())) {
      enqueue_locally(RulerMessage{
          .target_idx = succ, .ruler = ruler, .dist_from_ruler = dist_to_succ});
      return;
    }
    send_to(
        RulerMessage{.target_idx = succ, .ruler = ruler, .dist_from_ruler = dist_to_succ},
        get_succ_owner(ruler_local, succ));
    // dist.get_owner(succ));
  };

  auto work_on_item = [&](RulerMessage const& msg, auto&& enqueue_locally,
                          auto&& send_to) {
    const auto& [idx, ruler, dist_from_ruler] = msg;
    KASSERT(dist.get_owner(idx) == comm.rank());
    auto idx_local = dist.get_local_idx(idx, comm.rank());
    auto succ = succ_array[idx_local];
    auto dist_to_succ = rank_array[idx_local];
    succ_array[idx_local] = ruler;
    rank_array[idx_local] = dist_from_ruler;
    if (node_type[idx_local] == NodeType::ruler ||
        node_type[idx_local] == NodeType::root) {
      // we stop chasing here
      trace.track_chain_end(ruler, dist_from_ruler);
      // select new unreached vertex as ruler
      if (config.spawn) {
        spawn_new_ruler(enqueue_locally, send_to);
      }
      return;
    }
    KASSERT(node_type[idx_local] == NodeType::unreached);
    node_type[idx_local] = NodeType::reached;
    num_unreached--;
    if (dist.is_local(succ, comm.rank())) {
      enqueue_locally({.target_idx = succ,
                       .ruler = ruler,
                       .dist_from_ruler = dist_from_ruler + dist_to_succ});
      return;
    }
    send_to({.target_idx = succ,
             .ruler = ruler,
             .dist_from_ruler = dist_from_ruler + dist_to_succ},
            get_succ_owner(idx_local, succ));
    // succ_rank);
  };
  if (config.sync) {
    auto rounds =
        handle_messages(config, init, work_on_item, dist, comm, ruler_chasing::sync);
    trace.track_ruler_chasing_rounds(rounds);
  } else {
    handle_messages(config, init, work_on_item, dist, comm, ruler_chasing::async);
  }
  kamping::measurements::timer().stop();
  KASSERT(std::ranges::all_of(node_type,
                              [&](NodeType type) {
                                return type == NodeType::root ||
                                       type == NodeType::ruler ||
                                       type == NodeType::leaf ||
                                       type == NodeType::reached;
                              }),
          "Not all nodes were reached during ruler chasing!");

  // resetting the leafs' msb
  for (auto local_idx : rulers) {
    succ_array[local_idx] = bits::clear_root_flag(succ_array[local_idx]);
  }

  {
    kamping::measurements::timer().start("pack_base_case");
    std::vector<idx_t> succ_array_base(rulers.size());
    std::vector<rank_t> rank_array_base(rulers.size());
    auto [dist_base, unpack] = pack(succ_array, rank_array, dist, rulers, succ_array_base,
                                    rank_array_base, comm);
    kamping::measurements::timer().stop();

    kamping::measurements::timer().synchronize_and_start("base_case");
    trace.track_base_case(succ_array_base.size());
    base_algorithm(succ_array_base, rank_array_base, dist_base, comm);
    kamping::measurements::timer().stop();

    kamping::measurements::timer().start("unpack_base_case");
    unpack(succ_array_base, rank_array_base, dist_base, succ_array, rank_array, dist,
           rulers, comm);
    kamping::measurements::timer().stop();
  }

  ///////////////////////////////////////
  // Request rank and root from rulers //
  ///////////////////////////////////////
  kamping::measurements::timer().synchronize_and_start("ruler_propagation");
  switch (config.ruler_propagation_mode) {
    case RulerPropagationMode::pull:
      ruler_propagation(config, succ_array, rank_array, node_type, dist, comm,
                        propagation_mode::pull);
      break;
    case RulerPropagationMode::push:
      ruler_propagation(config, succ_array, rank_array, inital_succ_array.value(),
                        node_type, rulers, dist, comm, propagation_mode::push);
      break;
    case RulerPropagationMode::invalid:
      throw std::runtime_error("Invalid ruler propagation mode");
      break;
  }
  kamping::measurements::timer().stop();
}
}  // namespace

void sparse_ruling_set(SparseRulingSetConfig const& config,
                       std::span<idx_t> succ_array,
                       std::span<rank_t> rank_array,
                       Distribution const& dist,
                       kamping::Communicator<> const& comm) {
  BaseAlgorithm base_algorithm;
  switch (config.base_algorithm) {
    case kascade::Algorithm::PointerDoubling:
      base_algorithm = [&](auto&&... args) {
        kascade::pointer_doubling(
            std::any_cast<PointerDoublingConfig>(config.base_algorithm_config), args...);
      };
      break;
    case Algorithm::GatherChase:
      base_algorithm = [&](auto&&... args) { kascade::rank_on_root(args...); };
      break;
    case Algorithm::AsyncPointerDoubling:
      base_algorithm = [&](auto&&... args) {
        kascade::async_pointer_doubling(
            std::any_cast<AsyncPointerChasingConfig>(config.base_algorithm_config),
            args...);
      };
      break;
    case Algorithm::RMAPointerDoubling:
      base_algorithm = [&](auto&&... args) {
        kascade::rma_pointer_doubling(
            std::any_cast<RMAPointerChasingConfig>(config.base_algorithm_config),
            args...);
      };
      break;
    case Algorithm::SparseRulingSet:
      base_algorithm = [&](auto&&... args) {
        auto nested_config =
            std::any_cast<SparseRulingSetConfig>(config.base_algorithm_config);
        // avoid infinite recursion by setting the base algorithm of the nested config to
        // a non-recursive one
        nested_config.base_algorithm = kascade::Algorithm::PointerDoubling;
        nested_config.base_algorithm_config = PointerDoublingConfig{};
        sparse_ruling_set(nested_config, args...);
      };
      break;
    default:
      throw std::runtime_error("Invalid base algorithm selected for sparse ruling set");
  }
  sparse_ruling_set(config, succ_array, rank_array, dist, base_algorithm, comm);
}
}  // namespace kascade
