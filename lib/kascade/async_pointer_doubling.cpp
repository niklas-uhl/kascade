#include <queue>

#include <kascade/pointer_doubling.hpp>
#include <kascade/types.hpp>

#include <absl/container/flat_hash_map.h>  // for flat_hash_map
#include <fmt/ostream.h>

#include "briefkasten/buffered_queue.hpp"
#include "briefkasten/queue_builder.hpp"
#include "kascade/configuration.hpp"
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
  [[nodiscard]] auto is_sending_request_event() const -> bool {
    return !has_pe_rank_flag(msg.rank);
  }
  [[nodiscard]] auto is_sending_reply_event() const -> bool {
    return has_pe_rank_flag(msg.rank);
  }
  Msg msg;
};

}  // namespace
//

template <>
struct fmt::formatter<Msg> : ostream_formatter {};
namespace kascade {
void async_pointer_doubling(AsyncPointerChasingConfig const& config,
                            std::span<idx_t> succ_array,
                            std::span<idx_t> rank_array,
                            Distribution const& dist,
                            kamping::Communicator<> const& comm) {
  auto queue =
      briefkasten::BufferedMessageQueueBuilder<Msg>(comm.mpi_communicator()).build();
  absl::flat_hash_map<idx_t, std::pair<idx_t, idx_t>> cache;

  std::queue<SendEvent> event_queue;
  // update_requests.reserve(succ_array.size());
  for (std::size_t i = 0; i < succ_array.size(); ++i) {
    idx_t global_idx = dist.get_global_idx(i, comm.rank());
    if (succ_array[i] == global_idx) {
      KASSERT(rank_array[i] == 0);
      succ_array[i] = set_root_flag(global_idx);
    } else {
      event_queue.emplace(
          SendEvent{.msg = Msg{.write_back_idx = i, .succ = succ_array[i], .rank = 0}});
    }
  }

  auto process_recv_reply = [&](auto& msg) {
    succ_array[msg.write_back_idx] = msg.succ;
    rank_array[msg.write_back_idx] += msg.rank;
    if (!has_root_flag(msg.succ)) {
      msg.rank = 0;
      event_queue.emplace(msg);
    }
  };

  auto on_message = [&](auto env) {
    for (auto msg : env.message) {
      bool has_recv_request = has_pe_rank_flag(msg.rank);
      if (has_recv_request) {
        event_queue.emplace(msg);
      } else {
        // has_recv_reply
        if (config.use_caching) {
          cache[succ_array[msg.write_back_idx]] = std::make_pair(msg.succ, msg.rank);
        }
        process_recv_reply(msg);
      }
    }
  };

  auto do_cache_lookup = [&](auto& msg) -> bool {
    if (!config.use_caching) {
      return false;
    }
    auto it = cache.find(msg.succ);
    if (it == cache.end()) {
      return false;
    }
    // treat as if this was a remote recv reply
    msg.succ = it->second.first;
    msg.rank = it->second.second;
    process_recv_reply(msg);
    return true;
  };

  do {
    while (!event_queue.empty()) {
      auto event = event_queue.front();
      event_queue.pop();
      if (event.is_sending_request_event()) {
        KASSERT(!has_root_flag(event.msg.succ),
                "Do not continue on already finished elements.");
        KASSERT(!has_pe_rank_flag(event.msg.rank), "Set pe rank flag at send time.");
        auto successful_cache_lookup = do_cache_lookup(event.msg);
        if (!successful_cache_lookup) {
          event.msg.rank = set_pe_rank_flag(comm.rank());
          int owner = dist.get_owner_signed(event.msg.succ);
          queue.post_message_blocking(event.msg, owner, on_message);
        }
      } else {
        // is sending reply event
        auto local_idx = dist.get_local_idx(event.msg.succ, comm.rank());
        event.msg.succ = succ_array[local_idx];
        std::size_t owner = clear_pe_rank_flag(event.msg.rank);
        event.msg.rank = rank_array[local_idx];
        queue.post_message_blocking(event.msg, static_cast<int>(owner), on_message);
      }
      queue.poll_throttled(on_message);
    }
  } while (!queue.terminate(on_message));
  std::ranges::transform(succ_array, succ_array.begin(),
                         [](idx_t elem) { return clear_root_flag(elem); });
}
}  // namespace kascade
