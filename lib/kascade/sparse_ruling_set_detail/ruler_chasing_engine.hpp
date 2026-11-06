#pragma once

#include <cstddef>
#include <cstdint>
#include <queue>
#include <string>

#include <briefkasten/buffered_queue.hpp>
#include <briefkasten/grid_indirection.hpp>
#include <briefkasten/indirection.hpp>
#include <briefkasten/queue_builder.hpp>
#include <kamping/collectives/allreduce.hpp>
#include <kamping/measurements/counter.hpp>
#include <kamping/types/unsafe/tuple.hpp>

#include "kascade/configuration.hpp"
#include "kascade/distribution.hpp"
#include "kascade/grid_alltoall.hpp"
#include "kascade/sparse_ruling_set_detail/types.hpp"

namespace kascade {
namespace ruler_chasing {
struct sync_tag {};
struct async_tag {};
constexpr sync_tag sync{};
constexpr async_tag async{};
}  // namespace ruler_chasing

using WireRulerMessage = std::tuple<idx_t, idx_t, std::uint64_t>;

constexpr auto to_wire(const kascade::RulerMessage& msg) -> WireRulerMessage {
  return {msg.target_idx, msg.ruler, std::bit_cast<std::uint64_t>(msg.dist_from_ruler)};
}

constexpr auto from_wire(const WireRulerMessage& msg) -> kascade::RulerMessage {
  return {.target_idx = std::get<0>(msg),
          .ruler = std::get<1>(msg),
          .dist_from_ruler = std::bit_cast<rank_t>(std::get<2>(msg))};
}

// One entry per engine invocation, like the other sparse ruling set counters.
inline void record_queue_counters(auto const& queue) {
  auto append = [](char const* name, std::size_t value) {
    kamping::measurements::counter().append(
        name, static_cast<std::int64_t>(value),
        {kamping::measurements::GlobalAggregationMode::sum,
         kamping::measurements::GlobalAggregationMode::max});
  };
  append("bk_terminate_calls", queue.num_terminate_calls());
  append("bk_termination_rounds", queue.num_termination_rounds());
  append("bk_overflow_flushes", queue.num_overflows());
  append("bk_forced_flushes", queue.num_forced_flushes());
  append("bk_forced_flush_elements", queue.num_forced_flush_elements());
  append("bk_idle_flushes", queue.num_idle_flushes());
  append("bk_idle_flush_elements", queue.num_idle_flush_elements());
  append("bk_idle_flush_skips", queue.num_idle_flush_skips());
  append("bk_idle_flush_deferred", queue.num_idle_flush_deferred());
  append("bk_idle_flush_stops", queue.num_idle_flush_stops());
  append("bk_peak_send_backlog", queue.peak_send_backlog());
  append("bk_elements_flushed", queue.num_elements_flushed());
  // stalls and flow control: where a sender waits for a send slot or receiver credit
  append("bk_buffer_stalls", queue.num_buffer_stalls());
  append("bk_termination_drains", queue.num_termination_drains());
  append("bk_drain_capacity_waits", queue.num_drain_capacity_waits());
  append("bk_overflow_capacity_waits", queue.num_overflow_capacity_waits());
  append("bk_parked_for_capacity", queue.num_parked_for_capacity());
  append("bk_parked_for_credit", queue.num_parked_for_credit());
  append("bk_grants_sent", queue.num_grants_sent());
  append("bk_grants_withheld", queue.num_grants_withheld());
  append("bk_oversize_passes", queue.num_oversize_passes());
  // indirection only (zero for the flat queue)
  append("bk_redirect_buffer_stalls", queue.num_redirect_buffer_stalls());
  append("bk_redirect_overdraft", queue.redirect_overdraft());
  // progress
  append("bk_polls", queue.num_polls());
  append("bk_unproductive_polls", queue.num_unproductive_polls());
}

auto ruler_chasing_engine(SparseRulingSetConfig const& config,
                          auto&& initialize,
                          auto&& work_on_item,
                          Distribution const& dist,
                          kamping::Communicator<> const& comm,
                          ruler_chasing::async_tag /* tag */) {
  briefkasten::Config briefkasten_config;
  briefkasten_config.local_threshold_bytes = config.briefkasten.local_threshold;
  switch (config.briefkasten.idle_flush) {
    case IdleFlush::off:
      briefkasten_config.idle_flush = briefkasten::IdleFlush::off;
      break;
    case IdleFlush::all:
      briefkasten_config.idle_flush = briefkasten::IdleFlush::all;
      break;
    case IdleFlush::stale:
      briefkasten_config.idle_flush = briefkasten::IdleFlush::stale;
      break;
    case IdleFlush::library_default:
    case IdleFlush::invalid:
      break;
  }
  briefkasten_config.idle_flush_round_robin = !config.briefkasten.idle_flush_fixed_start;
  if (auto const& backlog = config.briefkasten.send_backlog_capacity; backlog == "comm-size") {
    briefkasten_config.send_backlog_capacity = comm.size();
  } else if (backlog != "default") {
    briefkasten_config.send_backlog_capacity = std::stoull(backlog);  // validated by the parser
  }
  auto with_queue = [&](auto&& func) {
    if (config.briefkasten.use_indirection(config.use_grid_communication)) {
      briefkasten::IndirectionAdapter queue{
          briefkasten::BufferedMessageQueueBuilder<WireRulerMessage>(
              briefkasten_config, comm.mpi_communicator())
              .with_buffer_type<std::uint64_t>()
              .with_merger(briefkasten::aggregation::TupleMerger{})
              .with_splitter(briefkasten::aggregation::TupleSplitter<WireRulerMessage>{})
              .build(),
          briefkasten::GridIndirectionScheme{comm.mpi_communicator()}};
      func(queue);
    } else {
      auto queue = briefkasten::BufferedMessageQueueBuilder<WireRulerMessage>(
                       briefkasten_config, comm.mpi_communicator())
                       .build();
      func(queue);
    }
  };
  with_queue([&](auto& queue) {
    std::queue<RulerMessage> local_queue;
    auto on_message = [&](auto env) {
      for (WireRulerMessage const& msg : env.message) {
        KASSERT(dist.get_owner(from_wire(msg).target_idx) == comm.rank());
        local_queue.push(from_wire(msg));
      }
    };
    auto enqueue_locally = [&](RulerMessage const& msg) { local_queue.push(msg); };
    auto send_to = [&](RulerMessage const& msg, int target) {
      queue.post_message_blocking(to_wire(msg), target, on_message);
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
    record_queue_counters([&]() -> auto& {
      if constexpr (requires { queue.queue(); }) {
        return queue.queue();  // IndirectionAdapter
      } else {
        return queue;
      }
    }());
  });
}

auto ruler_chasing_engine(SparseRulingSetConfig const& config,
                          auto&& initialize,
                          auto&& work_on_item,
                          Distribution const& /* dist */,
                          kamping::Communicator<> const& comm,
                          std::optional<TopologyAwareGridCommunicator> const& grid_comm,
                          ruler_chasing::sync_tag /* tag */) -> std::size_t {
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

  AlltoallDispatcher<RulerMessage> dispatcher(config.use_grid_communication, comm,
                                              grid_comm, config.alltoall_algorithm);

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
}  // namespace kascade
