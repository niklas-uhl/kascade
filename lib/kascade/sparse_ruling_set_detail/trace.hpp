#pragma once

#include <chrono>

#include <absl/container/flat_hash_map.h>
#include <kamping/collectives/allreduce.hpp>
#include <kamping/measurements/counter.hpp>
#include <kamping/measurements/timer.hpp>

#include "kascade/types.hpp"

namespace kascade::sparse_ruling_set_detail {
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
    kamping::measurements::counter().append(
        "total_spawn_time_millis",
        static_cast<std::int64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(total_spawn_time_)
                .count()),
        {kamping::measurements::GlobalAggregationMode::max,
         kamping::measurements::GlobalAggregationMode::min});
    kamping::measurements::timer().stop();
  }
  void track_chain_end(idx_t ruler, rank_t dist_from_ruler) {
    ruler_list_length[ruler] = dist_from_ruler;
  }
  std::size_t num_spawned_rulers_ = 0;
  std::chrono::duration<double> total_spawn_time_ = std::chrono::seconds(0);
  void track_spawn(std::chrono::duration<double> spawn_time = std::chrono::seconds(0)) {
    num_spawned_rulers_++;
    total_spawn_time_ += spawn_time;
  }
  void track_base_case(std::size_t local_subproblem_size) {
    local_subproblem_size_ = local_subproblem_size;
  }
  std::size_t rounds_;
  void track_ruler_chasing_rounds(std::size_t ruler_chasing_rounds) {
    rounds_ = ruler_chasing_rounds;
  }
};
}  // namespace kascade::sparse_ruling_set_detail
