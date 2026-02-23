#pragma once

#include <queue>

#include <briefkasten/buffered_queue.hpp>
#include <briefkasten/queue_builder.hpp>
#include <kamping/collectives/allreduce.hpp>

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

auto ruler_chasing_engine(SparseRulingSetConfig const& config,
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

auto ruler_chasing_engine(SparseRulingSetConfig const& config,
                     auto&& initialize,
                     auto&& work_on_item,
                     Distribution const& /* dist */,
                     kamping::Communicator<> const& comm,
                     std::optional<TopologyAwareGridCommunicator> const& grid_comm,
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

  AlltoallDispatcher<RulerMessage> dispatcher(config.use_grid_communication, comm,
                                              grid_comm);

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
