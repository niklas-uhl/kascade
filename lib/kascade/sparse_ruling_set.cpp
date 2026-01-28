#include "kascade/sparse_ruling_set.hpp"

#include <queue>

#include <briefkasten/queue_builder.hpp>
#include <kassert/kassert.hpp>

#include "kascade/assertion_levels.hpp"
#include "kascade/distribution.hpp"
#include "kascade/successor_utils.hpp"

namespace kascade {
namespace {
auto pick_rulers(std::span<const idx_t> succ_array, std::size_t local_num_rulers)
    -> std::vector<idx_t> {
  // TODO implement
  return {};
}
}  // namespace

void sparse_ruling_set(std::span<idx_t> succ_array,
                       std::span<idx_t> rank_array,
                       Distribution const& dist,
                       kamping::Communicator<> const& comm) {
  KASSERT(is_list(succ_array, dist, comm), kascade::assert::with_communication);
  // std::vector<bool> is_ruler(succ_array.size(), false);
  // std::size_t local_num_rulers = 10;  // FIXME
  // auto rulers = pick_rulers(succ_array, local_num_rulers);
  // for (auto ruler : rulers) {
  //   is_ruler[ruler] = true;
  // }
  // using message_type = struct {
  //   idx_t node;
  //   idx_t distance;
  // };
  // auto queue =
  //     briefkasten::BufferedMessageQueueBuilder<message_type>(comm.mpi_communicator())
  //         .build();
  // std::queue<idx_t> local_queue;
  // for (auto ruler : rulers) {
  //   rank_array[ruler] = 0;
  //   root_array[ruler] = ruler;
  //   local_queue.push(ruler);
  // }
  // auto on_message = [&](auto env) {
  //   for (auto const& msg : env.message) {
  //   }
  // };
  // do {
  //   while (!local_queue.empty()) {
  //     auto node = local_queue.front();
  //     local_queue.pop();
  //     auto succ = succ_array[node];
  //     if (is_ruler[succ]) {
  //       continue;
  //     }
  //     if (rank_array[succ] == static_cast<idx_t>(succ_array.size())) {
  //       rank_array[succ] = rank_array[node] + 1;
  //       root_array[succ] = root_array[node];
  //       local_queue.push(succ);
  //     }
  //   }

  // } while (!queue.terminate(on_message));
}
}  // namespace kascade
