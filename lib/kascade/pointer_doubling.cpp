#include <kascade/pointer_doubling.hpp>

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <kamping/collectives/allreduce.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/communicator.hpp>
#include <kamping/measurements/timer.hpp>
#include <kamping/utils/flatten.hpp>

#include "kascade/distribution.hpp"
#include "kascade/list_ranking.hpp"
#include "kascade/types.hpp"

namespace kascade {
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

template <typename MakeSendBufsFn, typename MakeRepliesFn, typename UpdateFn>
auto do_doubling_step_framework(std::span<kascade::idx_t> rank_array,
                                std::span<kascade::idx_t> root_array,
                                std::span<kascade::idx_t> local_request_array,
                                kascade::Distribution const& dist,
                                kamping::Communicator<> const& comm,
                                MakeSendBufsFn const& make_send_bufs,
                                MakeRepliesFn const& make_replies,
                                UpdateFn const& update) {
  namespace kmp = kamping::params;
  auto send_bufs = make_send_bufs(root_array, local_request_array, dist);
  auto [send_buf, send_counts, send_displs] = kamping::flatten(send_bufs, comm.size());
  auto [recv_requests, recv_counts] =
      comm.alltoallv(kmp::send_buf(send_buf), kmp::send_counts(send_counts),
                     kmp::send_displs(send_displs), kmp::recv_counts_out());

  auto replies = make_replies(root_array, rank_array, recv_requests, comm.rank(), dist);
  auto recv_replies =
      comm.alltoallv(kmp::send_buf(replies), kmp::send_counts(recv_counts));

  std::size_t const num_remaining_elems =
      update(recv_replies, rank_array, root_array, local_request_array);
  return local_request_array.first(num_remaining_elems);
}

namespace naive {

struct Request {
  kascade::idx_t write_back_idx;
  kascade::idx_t succ;
};

struct Reply {
  kascade::idx_t write_back_idx;
  kascade::idx_t succ;
  kascade::idx_t rank;
};
auto make_send_bufs(std::span<kascade::idx_t> root_array,
                    std::span<kascade::idx_t> local_request_array,
                    kascade::Distribution const& dist)
    -> absl::flat_hash_map<int, std::vector<Request>> {
  absl::flat_hash_map<int, std::vector<Request>> send_bufs;
  for (auto& local_elem_idx : local_request_array) {
    auto succ = root_array[local_elem_idx];
    KASSERT(!has_root_flag(succ), "Do not continue on already finised elements.");
  }
  for (auto& local_elem_idx : local_request_array) {
    auto succ = root_array[local_elem_idx];
    KASSERT(!has_root_flag(succ), "Do not continue on already finised elements.");
    int owner = dist.get_owner_signed(succ);
    send_bufs[owner].emplace_back(local_elem_idx, succ);
  }
  return send_bufs;
}

auto make_replies(std::span<kascade::idx_t> root_array,
                  std::span<kascade::idx_t> rank_array,
                  std::span<Request> recv_requests,
                  std::size_t rank,
                  kascade::Distribution const& dist) -> std::vector<Reply> {
  std::vector<Reply> replies;
  replies.reserve(recv_requests.size());
  for (auto [write_back_idx, succ] : recv_requests) {
    auto local_idx = dist.get_local_idx(succ, rank);
    succ = root_array[local_idx];
    replies.emplace_back(write_back_idx, succ, rank_array[local_idx]);
  }
  return replies;
}

auto update(std::span<Reply const> recv_replies,
            std::span<kascade::idx_t> rank_array,
            std::span<kascade::idx_t> root_array,
            std::span<kascade::idx_t> local_request_array) -> std::size_t {
  std::size_t unfinished_elems = 0;
  for (const auto [write_back_idx, succ, rank] : recv_replies) {
    // local index
    root_array[write_back_idx] = succ;
    rank_array[write_back_idx] += rank;
    if (!has_root_flag(succ)) {
      local_request_array[unfinished_elems++] = write_back_idx;
    }
  }
  return unfinished_elems;
}
auto do_doubling_step(std::span<kascade::idx_t> rank_array,
                      std::span<kascade::idx_t> root_array,
                      std::span<kascade::idx_t> local_request_array,
                      kascade::Distribution const& dist,
                      kamping::Communicator<> const& comm) -> std::span<kascade::idx_t> {
  auto make_send_bufs = [](std::span<idx_t> root_array,
                           std::span<idx_t> local_request_array, auto const& dist) {
    return naive::make_send_bufs(root_array, local_request_array, dist);
  };
  auto make_replies = [](std::span<idx_t> root_array, std::span<idx_t> rank_array,
                         std::span<Request> recv_requests, std::size_t rank,
                         auto const& dist) {
    return naive::make_replies(root_array, rank_array, recv_requests, rank, dist);
  };
  auto update = [](std::span<Reply> recv_replies, std::span<kascade::idx_t> rank_array,
                   std::span<kascade::idx_t> root_array,
                   std::span<kascade::idx_t> local_request_array) {
    return naive::update(recv_replies, rank_array, root_array, local_request_array);
  };

  return do_doubling_step_framework(rank_array, root_array, local_request_array, dist,
                                    comm, make_send_bufs, make_replies, update);
}
}  // namespace naive

namespace deduplication {

struct Reply {
  kascade::idx_t req_succ;
  kascade::idx_t succ;
  kascade::idx_t rank;
};
auto make_send_bufs(std::span<kascade::idx_t> root_array,
                    std::span<kascade::idx_t> local_request_array,
                    kascade::Distribution const& dist)
    -> absl::flat_hash_map<int, std::vector<idx_t>> {
  absl::flat_hash_set<idx_t> request_buffer;
  for (auto& local_elem_idx : local_request_array) {
    auto succ = root_array[local_elem_idx];
    KASSERT(!has_root_flag(succ), "Do not continue on already finised elements.");
    request_buffer.emplace(succ);
  }
  absl::flat_hash_map<int, std::vector<idx_t>> send_bufs;
  for (const auto& succ : request_buffer) {
    int owner = dist.get_owner_signed(succ);
    send_bufs[owner].emplace_back(succ);
  }
  return send_bufs;
}

auto make_replies(std::span<kascade::idx_t> root_array,
                  std::span<kascade::idx_t> rank_array,
                  std::span<kascade::idx_t> recv_requests,
                  std::size_t rank,
                  kascade::Distribution const& dist) -> std::vector<Reply> {
  std::vector<Reply> replies;
  replies.reserve(recv_requests.size());
  for (auto req_succ : recv_requests) {
    auto local_idx = dist.get_local_idx(req_succ, rank);
    auto succ = root_array[local_idx];
    replies.emplace_back(req_succ, succ, rank_array[local_idx]);
  }
  return replies;
}

auto update(std::span<Reply const> recv_replies,
            std::span<kascade::idx_t> rank_array,
            std::span<kascade::idx_t> root_array,
            std::span<kascade::idx_t> local_request_array) -> std::size_t {
  // prepare lookup table
  absl::flat_hash_map<idx_t, std::pair<idx_t, idx_t>> lookup_table;
  for (const auto [req_succ, succ, rank] : recv_replies) {
    lookup_table.emplace(req_succ, std::make_pair(succ, rank));
  }
  std::size_t unfinished_elems = 0;
  for (auto& local_elem_idx : local_request_array) {
    // local index
    auto cur_succ = root_array[local_elem_idx];
    auto it = lookup_table.find(cur_succ);
    KASSERT(it != lookup_table.end());
    auto const& [succ, rank] = it->second;
    root_array[local_elem_idx] = succ;
    rank_array[local_elem_idx] += rank;
    if (!has_root_flag(succ)) {
      local_request_array[unfinished_elems++] = local_elem_idx;
    }
  }
  return unfinished_elems;
}
auto do_doubling_step(std::span<kascade::idx_t> rank_array,
                      std::span<kascade::idx_t> root_array,
                      std::span<kascade::idx_t> local_request_array,
                      kascade::Distribution const& dist,
                      kamping::Communicator<> const& comm) -> std::span<kascade::idx_t> {
  auto make_send_bufs = [](std::span<idx_t> root_array,
                           std::span<idx_t> local_request_array, auto const& dist) {
    return deduplication::make_send_bufs(root_array, local_request_array, dist);
  };
  auto make_replies = [](std::span<idx_t> root_array, std::span<idx_t> rank_array,
                         std::span<idx_t> recv_requests, std::size_t rank,
                         auto const& dist) {
    return deduplication::make_replies(root_array, rank_array, recv_requests, rank, dist);
  };
  auto update = [](std::span<Reply> recv_replies, std::span<kascade::idx_t> rank_array,
                   std::span<kascade::idx_t> root_array,
                   std::span<kascade::idx_t> local_request_array) {
    return deduplication::update(recv_replies, rank_array, root_array,
                                 local_request_array);
  };

  return do_doubling_step_framework(rank_array, root_array, local_request_array, dist,
                                    comm, make_send_bufs, make_replies, update);
}
}  // namespace deduplication

auto do_doubling_step(kascade::PointerDoublingConfig const& config,
                      std::span<kascade::idx_t> rank_array,
                      std::span<kascade::idx_t> root_array,
                      std::span<kascade::idx_t> local_request_array,
                      kascade::Distribution const& dist,
                      kamping::Communicator<> const& comm) -> std::span<kascade::idx_t> {
  if (config.use_local_aggregation) {
    return deduplication::do_doubling_step(rank_array, root_array, local_request_array,
                                           dist, comm);
  }
  return naive::do_doubling_step(rank_array, root_array, local_request_array, dist, comm);
}
}  // namespace
}  // namespace kascade

void kascade::pointer_doubling(kascade::PointerDoublingConfig config,
                               std::span<idx_t> succ_array,
                               std::span<idx_t> rank_array,
                               Distribution const& dist,
                               kamping::Communicator<> const& comm) {
  if (config.use_local_preprocessing) {
    kamping::measurements::timer().synchronize_and_start("local_preprocessing");
    local_pointer_chasing(succ_array, rank_array, comm.rank(), dist);
    kamping::measurements::timer().stop();
  }

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
    local_req_idxs =
        do_doubling_step(config, rank_array, succ_array, local_req_idxs, dist, comm);
  }

  // clear result
  std::ranges::transform(succ_array, succ_array.begin(),
                         [](idx_t elem) { return clear_root_flag(elem); });
  kamping::measurements::timer().stop();
}
