#include <kascade/pointer_doubling.hpp>

#include <kamping/collectives/allreduce.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/communicator.hpp>
#include <kamping/measurements/timer.hpp>
#include <kamping/utils/flatten.hpp>

#include "kascade/distribution.hpp"
#include "kascade/types.hpp"

namespace {
using std::size_t;

constexpr kascade::idx_t root_flag_mask =
    kascade::idx_t(1) << (std::numeric_limits<kascade::idx_t>::digits - 1);

[[nodiscard]] constexpr auto set_root_flag(kascade::idx_t value) noexcept
    -> kascade::idx_t {
  return value | root_flag_mask;
}

[[nodiscard]] constexpr auto clear_root_flag(kascade::idx_t value) noexcept
    -> kascade::idx_t {
  return value & ~root_flag_mask;
}

[[nodiscard]] constexpr auto has_root_flag(kascade::idx_t value) noexcept -> bool {
  return (value & root_flag_mask) != 0;
}

auto is_finished(std::size_t unfinished_elems, kamping::Communicator<> const& comm)
    -> bool {
  namespace kmp = kamping::params;
  size_t const global_unfinished_elems =
      comm.allreduce_single(kmp::send_buf(unfinished_elems), kmp::op(std::plus<>{}));
  return global_unfinished_elems == 0U;
}

struct Request {
  kascade::idx_t write_back_idx;
  kascade::idx_t succ;
};

struct Reply {
  kascade::idx_t write_back_idx;
  kascade::idx_t succ;
  kascade::idx_t rank;
};

auto do_doubling_step(std::span<kascade::idx_t> rank_array,
                      std::span<kascade::idx_t> root_array,
                      std::span<kascade::idx_t> local_request_array,
                      kascade::Distribution const& dist,
                      kamping::Communicator<> const& comm) -> std::span<kascade::idx_t> {
  namespace kmp = kamping::params;
  std::unordered_map<int, std::vector<Request>> send_bufs;
  for (auto& local_elem_idx : local_request_array) {
    auto succ = root_array[local_elem_idx];
    KASSERT(!has_root_flag(succ), "Do not continue on already finised elements.");
    int owner = dist.get_owner_signed(succ);
    send_bufs[owner].emplace_back(local_elem_idx, succ);
  }
  auto [send_buf, send_counts, send_displs] = kamping::flatten(send_bufs, comm.size());
  auto [recv_requests, recv_counts] =
      comm.alltoallv(kmp::send_buf(send_buf), kmp::send_counts(send_counts),
                     kmp::send_displs(send_displs), kmp::recv_counts_out());

  std::vector<Reply> replies;
  replies.reserve(recv_requests.size());
  for (auto [write_back_idx, succ] : recv_requests) {
    auto local_idx = dist.get_local_idx(succ, comm.rank());
    succ = root_array[local_idx];
    replies.emplace_back(write_back_idx, succ, rank_array[local_idx]);
  }

  auto recv_replies =
      comm.alltoallv(kmp::send_buf(replies), kmp::send_counts(recv_counts));
  std::size_t unfinished_elems = 0;
  for (const auto [write_back_idx, succ, rank] : recv_replies) {
    // local index
    root_array[write_back_idx] = succ;
    rank_array[write_back_idx] += rank;
    if (!has_root_flag(succ)) {
      local_request_array[unfinished_elems++] = write_back_idx;
    }
  }
  return local_request_array.first(unfinished_elems);
}
}  // namespace

void kascade::pointer_doubling(std::span<idx_t> succ_array,
                               std::span<idx_t> rank_array,
                               Distribution const& dist,
                               kamping::Communicator<> const& comm) {
  kamping::measurements::timer().synchronize_and_start("pointer_doubling_alltoall");

  std::vector<idx_t> local_req_storage;
  local_req_storage.reserve(succ_array.size());
  for (std::size_t i = 0; i < succ_array.size(); ++i) {
    idx_t global_idx = dist.get_global_idx(i, comm.rank());
    if (succ_array[i] == global_idx) {
      KASSERT(rank_array[i] == 0);
      succ_array[i] = set_root_flag(global_idx);
    } else {
      local_req_storage.emplace_back(i);
    }
  }

  // do pointer doubling
  std::span<idx_t> local_req_idxs = local_req_storage;
  while (!is_finished(local_req_idxs.size(), comm)) {
    local_req_idxs = do_doubling_step(rank_array, succ_array, local_req_idxs, dist, comm);
  }

  // clear result
  std::ranges::transform(succ_array, succ_array.begin(),
                         [](idx_t elem) { return clear_root_flag(elem); });
  kamping::measurements::timer().stop();
}
