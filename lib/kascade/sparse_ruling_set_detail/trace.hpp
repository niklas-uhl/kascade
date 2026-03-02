#pragma once

#include <chrono>
#include <limits>

#include <absl/container/flat_hash_map.h>
#include <kamping/collectives/allreduce.hpp>
#include <kamping/measurements/counter.hpp>
#include <kamping/measurements/timer.hpp>

#include "kascade/types.hpp"

namespace kascade::sparse_ruling_set_detail {
struct RulerTrace {
  struct chain_stats {
    rank_t min_length = std::numeric_limits<rank_t>::max();
    rank_t max_length = std::numeric_limits<rank_t>::min();
    rank_t length_sum = 0;
    std::size_t num_rulers = 0;
  };

  std::vector<std::pair<idx_t, idx_t>> ruler_list_length;
  std::size_t local_subproblem_size_{};
  RulerTrace() = default;
  RulerTrace(const RulerTrace&) = default;
  RulerTrace(RulerTrace&&) = delete;
  auto operator=(const RulerTrace&) -> RulerTrace& = default;
  auto operator=(RulerTrace&&) -> RulerTrace& = delete;
  std::size_t local_num_rulers_;
  std::size_t local_num_leaves_;
  ~RulerTrace() {
    kamping::measurements::timer().start("aggregate_ruler_stats");
    auto agg = [](auto const& lhs, auto const& rhs) {
      return chain_stats{.min_length = std::min(lhs.min_length, rhs.min_length),
                         .max_length = std::max(lhs.max_length, rhs.max_length),
                         .length_sum = std::plus<>{}(lhs.length_sum, rhs.length_sum),
                         .num_rulers = std::plus<>{}(lhs.num_rulers, rhs.num_rulers)};
    };
    std::array<chain_stats, 2> send_buf{local_ruler_stats_, local_contraction_stats_};
    kamping::comm_world().allreduce(kamping::send_recv_buf(send_buf),
                                    kamping::op(agg, kamping::ops::commutative));
    local_ruler_stats_ = send_buf[0];
    local_contraction_stats_ = send_buf[1];
    kamping::measurements::counter().append(
        "ruler_list_length_min", static_cast<std::int64_t>(local_ruler_stats_.min_length),
        {
            kamping::measurements::GlobalAggregationMode::min,
        });
    kamping::measurements::counter().append(
        "ruler_list_length_max", static_cast<std::int64_t>(local_ruler_stats_.max_length),
        {
            kamping::measurements::GlobalAggregationMode::max,
        });
    kamping::measurements::counter().append(
        "ruler_list_length_avg",
        static_cast<std::int64_t>(static_cast<double>(local_ruler_stats_.length_sum) /
                                  static_cast<double>(local_ruler_stats_.num_rulers)),
        {
            kamping::measurements::GlobalAggregationMode::max,
        });
    kamping::measurements::counter().append(
        "contraction_chain_length_min",
        static_cast<std::int64_t>(local_contraction_stats_.min_length),
        {
            kamping::measurements::GlobalAggregationMode::min,
        });
    kamping::measurements::counter().append(
        "contraction_chain_length_max",
        static_cast<std::int64_t>(local_contraction_stats_.max_length),
        {
            kamping::measurements::GlobalAggregationMode::max,
        });
    kamping::measurements::counter().append(
        "contraction_chain_length_avg",
        static_cast<std::int64_t>(
            static_cast<double>(local_contraction_stats_.length_sum) /
            static_cast<double>(local_contraction_stats_.num_rulers)),
        {
            kamping::measurements::GlobalAggregationMode::max,
        });
    kamping::measurements::counter().append(
        "num_locally_contracted", static_cast<std::int64_t>(num_locally_contracted_),
        {kamping::measurements::GlobalAggregationMode::sum,
         kamping::measurements::GlobalAggregationMode::max,
         kamping::measurements::GlobalAggregationMode::min});
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
    kamping::measurements::counter().append(
        "num_unreached", static_cast<std::int64_t>(num_unreached_),
        {kamping::measurements::GlobalAggregationMode::sum,
         kamping::measurements::GlobalAggregationMode::max,
         kamping::measurements::GlobalAggregationMode::min});
    kamping::measurements::counter().append(
        "total_spawn_time_millis",
        static_cast<std::int64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(total_spawn_time_)
                .count()),
        {kamping::measurements::GlobalAggregationMode::max,
         kamping::measurements::GlobalAggregationMode::min});
    kamping::measurements::timer().stop();
  }
  chain_stats local_ruler_stats_{};
  void track_chain_end(idx_t /* ruler */, rank_t dist_from_ruler) {
    local_ruler_stats_.min_length =
        std::min(local_ruler_stats_.min_length, dist_from_ruler);
    local_ruler_stats_.max_length =
        std::max(local_ruler_stats_.max_length, dist_from_ruler);
    local_ruler_stats_.length_sum += dist_from_ruler;
    local_ruler_stats_.num_rulers++;
  }

  std::size_t num_spawned_rulers_ = 0;
  std::chrono::duration<double> total_spawn_time_ = std::chrono::seconds(0);
  void track_spawn(std::chrono::duration<double> spawn_time = std::chrono::seconds(0)) {
    num_spawned_rulers_++;
    total_spawn_time_ += spawn_time;
  }
  chain_stats local_contraction_stats_;
  void track_local_contraction(rank_t chain_length) {
    local_contraction_stats_.min_length =
        std::min(local_contraction_stats_.min_length, chain_length);
    local_contraction_stats_.max_length =
        std::max(local_contraction_stats_.max_length, chain_length);
    local_contraction_stats_.length_sum += chain_length;
    local_contraction_stats_.num_rulers++;
  }
  std::size_t num_locally_contracted_ = 0;
  void track_num_locally_contracted(std::size_t num_locally_contracted) {
    num_locally_contracted_ = num_locally_contracted;
  }
  void track_base_case(std::size_t local_subproblem_size) {
    local_subproblem_size_ = local_subproblem_size;
  }
  std::size_t rounds_;
  void track_ruler_chasing_rounds(std::size_t ruler_chasing_rounds) {
    rounds_ = ruler_chasing_rounds;
  }
  std::size_t num_unreached_;
  void track_unreached(std::size_t num_unreached) { num_unreached_ = num_unreached; }
  void track_local_num_rulers(std::size_t local_num_rulers) {
    local_num_rulers_ = local_num_rulers;
  }
  void track_local_num_leaves(std::size_t local_num_leaves) {
    local_num_leaves_ = local_num_leaves;
  }
};
}  // namespace kascade::sparse_ruling_set_detail
