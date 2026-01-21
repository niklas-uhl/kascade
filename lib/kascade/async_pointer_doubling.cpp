#include <queue>

#include <kascade/pointer_doubling.hpp>
#include <kascade/types.hpp>
#include <fmt/ostream.h>

#include "briefkasten/buffered_queue.hpp"
#include "briefkasten/queue_builder.hpp"
#include "kascade/distribution.hpp"
#include "spdlog/spdlog.h"

namespace {

constexpr kascade::idx_t msb_flag_mask =
    kascade::idx_t(1) << (std::numeric_limits<kascade::idx_t>::digits - 1);

[[nodiscard]] constexpr auto set_msb(kascade::idx_t value) noexcept -> kascade::idx_t {
  return value | msb_flag_mask;
}

[[nodiscard]] constexpr auto clear_msb(kascade::idx_t value) noexcept -> kascade::idx_t {
  return value & ~msb_flag_mask;
}

[[nodiscard]] constexpr auto has_msb_set(kascade::idx_t value) noexcept -> bool {
  return (value & msb_flag_mask) != 0;
}

[[nodiscard]] constexpr auto set_root_flag(kascade::idx_t value) noexcept
    -> kascade::idx_t {
  return set_msb(value);
}

[[nodiscard]] constexpr auto clear_root_flag(kascade::idx_t value) noexcept
    -> kascade::idx_t {
  return clear_msb(value);
}

[[nodiscard]] constexpr auto has_root_flag(kascade::idx_t value) noexcept -> bool {
  return has_msb_set(value);
}

[[nodiscard]] constexpr auto set_pe_rank_flag(kascade::idx_t value) noexcept
    -> kascade::idx_t {
  return set_msb(value);
}

[[nodiscard]] constexpr auto clear_pe_rank_flag(kascade::idx_t value) noexcept
    -> kascade::idx_t {
  return clear_msb(value);
}

[[nodiscard]] constexpr auto has_pe_rank_flag(kascade::idx_t value) noexcept -> bool {
  return has_msb_set(value);
}

struct Msg {
  kascade::idx_t write_back_idx;
  kascade::idx_t succ;
  kascade::idx_t rank;  // List ranking rank in reply or PE rank in request
  friend auto operator<<(std::ostream& out, Msg const& msg) -> std::ostream& {
    return out << "(" << msg.write_back_idx << ", " << msg.succ << ", " << msg.rank
               << ")";
  }
};

struct SendEvent {
  [[nodiscard]] auto is_sending_request_event() const -> bool { return !has_pe_rank_flag(msg.rank); }
  [[nodiscard]] auto is_sending_reply_event() const -> bool { return has_pe_rank_flag(msg.rank); }
  Msg msg;
};

}  // namespace
//

template <>
struct fmt::formatter<Msg> : ostream_formatter {};
namespace kascade {
void async_pointer_doubling(std::span<const idx_t> succ_array,
                            std::span<idx_t> rank_array,
                            std::span<idx_t> root_array,
                            kamping::Communicator<> const& comm) {
  auto queue =
      briefkasten::BufferedMessageQueueBuilder<Msg>(comm.mpi_communicator()).build();
  Distribution dist(succ_array.size(), comm);

  // initialize rank and root array
  std::ranges::copy(succ_array, root_array.begin());
  std::ranges::fill(rank_array, 1);
  std::queue<SendEvent> event_queue;
  // update_requests.reserve(succ_array.size());
  for (std::size_t i = 0; i < succ_array.size(); ++i) {
    idx_t global_idx = dist.get_global_idx(i, comm.rank());
    if (succ_array[i] == global_idx) {
      root_array[i] = set_root_flag(global_idx);
      rank_array[i] = 0;
    } else {
      event_queue.emplace(
          SendEvent{.msg = Msg{.write_back_idx = i, .succ = root_array[i], .rank = 0}});
    }
  }

  auto on_message = [&](auto env) {
    for (auto msg : env.message) {
      bool has_recv_request = has_pe_rank_flag(msg.rank);
      if (has_recv_request) {
        event_queue.emplace(msg);
      } else {
        // has_recv_reply
        root_array[msg.write_back_idx] = msg.succ;
        rank_array[msg.write_back_idx] += msg.rank;
        if (!has_root_flag(msg.succ)) {
          msg.rank = 0;
          event_queue.emplace(msg);
        }
      }
    }
  };

  do {
    while (!event_queue.empty()) {
      auto event = event_queue.front();
      event_queue.pop();
      if (event.is_sending_request_event()) {
        KASSERT(!has_root_flag(event.msg.succ),
                "Do not continue on already finised elements.");

        KASSERT(!has_pe_rank_flag(event.msg.rank), "Set pe rank flag at send time.");
        event.msg.rank = set_pe_rank_flag(comm.rank());
        int owner = dist.get_owner_signed(event.msg.succ);
        queue.post_message_blocking(event.msg, owner, on_message);
      } else {
        // is sending reply event
        auto local_idx = dist.get_local_idx(event.msg.succ, comm.rank());
        event.msg.succ = root_array[local_idx];
        std::size_t owner = clear_pe_rank_flag(event.msg.rank);
        event.msg.rank = rank_array[local_idx];
        queue.post_message_blocking(event.msg, static_cast<int>(owner), on_message);
      }
      queue.poll_throttled(on_message);
    }
  } while (!queue.terminate(on_message));
  std::ranges::transform(root_array, root_array.begin(),
                         [](idx_t elem) { return clear_root_flag(elem); });
}
}  // namespace kascade
