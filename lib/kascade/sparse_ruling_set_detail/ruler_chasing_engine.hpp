#pragma once

#include <cstddef>
#include <cstdint>
#include <queue>

#include <briefkasten/buffered_queue.hpp>
#include <briefkasten/grid_indirection.hpp>
#include <briefkasten/indirection.hpp>
#include <briefkasten/queue_builder.hpp>
#include <kamping/collectives/allreduce.hpp>
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

auto ruler_chasing_engine(SparseRulingSetConfig const& config,
                          auto&& initialize,
                          auto&& work_on_item,
                          Distribution const& dist,
                          kamping::Communicator<> const& comm,
                          ruler_chasing::async_tag /* tag */) {
  briefkasten::Config briefkasten_config;
  briefkasten_config.local_threshold_bytes = config.briefkasten.local_threshold;
  // briefkasten_config.max_num_send_buffers = 2 * comm.size();
  // briefkasten_config.out_buffer_capacity = comm.size();
  auto with_queue = [&](auto&& func) {
    if (config.use_grid_communication) {
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
