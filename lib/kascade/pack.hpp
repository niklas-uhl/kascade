#pragma once

#include <ranges>
#include <span>

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <kamping/communicator.hpp>
#include <kamping/measurements/timer.hpp>
#include <kamping/types/unsafe/utility.hpp>
#include <kassert/kassert.hpp>

#include "kascade/distribution.hpp"
#include "kascade/grid_communicator.hpp"
#include "kascade/request_aggregation_scheme.hpp"
#include "kascade/types.hpp"

namespace kascade {
/// @brief Packs the given arrays according to the given distribution and active indices.
auto pack(std::span<const idx_t> succ_array,
          std::span<const rank_t> rank_array,
          Distribution const& dist,
          IndexRange auto const& active_indices,
          std::span<idx_t> succ_array_packed,
          std::span<rank_t> rank_array_packed,
          kamping::Communicator<> const& comm,
          bool use_grid_communication = false) {
  std::optional<TopologyAwareGridCommunicator> grid_comm;
  if (use_grid_communication) {
    grid_comm.emplace(comm);
  }
  auto local_size_packed = std::ranges::size(active_indices);
  KASSERT(succ_array_packed.size() >= local_size_packed);
  KASSERT(rank_array_packed.size() >= local_size_packed);
  std::vector<idx_t> idx_local_packed_to_unpacked(local_size_packed);
  absl::flat_hash_map<idx_t, idx_t> idx_local_unpacked_to_packed;
  for (auto [idx_local_packed, idx_local_unpacked] :
       std::views::enumerate(active_indices)) {
    idx_local_packed_to_unpacked[idx_local_packed] = idx_local_unpacked;
    idx_local_unpacked_to_packed[idx_local_unpacked] = idx_local_packed;
    succ_array_packed[idx_local_packed] = succ_array[idx_local_unpacked];
    rank_array_packed[idx_local_packed] = rank_array[idx_local_unpacked];
  }
  Distribution packed_dist{local_size_packed, comm};
  auto packed_idx_requests =
      succ_array_packed.first(local_size_packed) |
      std::views::filter([&](auto succ_unpacked) {
        return !dist.is_local(succ_unpacked, comm.rank());
      }) |
      std::views::transform([&](auto succ_unpacked) {
        return std::pair{dist.get_owner(succ_unpacked), succ_unpacked};
      }) |
      std::ranges::to<std::vector>();
  auto make_reply = [&](idx_t succ_unpacked) {
    KASSERT(dist.is_local(succ_unpacked, comm.rank()));
    auto succ_unpacked_idx_local = dist.get_local_idx(succ_unpacked, comm.rank());
    auto succ_packed_idx_local = idx_local_unpacked_to_packed[succ_unpacked_idx_local];
    auto succ_packed = packed_dist.get_global_idx(succ_packed_idx_local, comm.rank());
    return std::pair{succ_unpacked, succ_packed};
  };
  std::vector<std::pair<idx_t, idx_t>> replies;
  if (use_grid_communication) {
    replies = request_reply_without_remote_aggregation(packed_idx_requests, make_reply,
                                                       *grid_comm);
  } else {
    replies =
        request_reply_without_remote_aggregation(packed_idx_requests, make_reply, comm);
  }
  absl::flat_hash_map<idx_t, idx_t> succ_unpacked_to_packed{replies.begin(),
                                                            replies.end()};
  for (auto& succ_unpacked : succ_array_packed.first(local_size_packed)) {
    if (dist.is_local(succ_unpacked, comm.rank())) {
      auto succ_unpacked_idx_local = dist.get_local_idx(succ_unpacked, comm.rank());
      auto succ_packed_idx_local = idx_local_unpacked_to_packed[succ_unpacked_idx_local];
      auto succ_packed = packed_dist.get_global_idx(succ_packed_idx_local, comm.rank());
      succ_unpacked = succ_packed;
      continue;
    }
    succ_unpacked = succ_unpacked_to_packed[succ_unpacked];
  }
  auto unpack = [idx_local_packed_to_unpacked_ = std::move(idx_local_packed_to_unpacked)](
                    std::span<const idx_t> succ_array_packed,
                    std::span<const rank_t> rank_array_packed,
                    Distribution const& dist_packed, std::span<idx_t> succ_array,
                    std::span<rank_t> rank_array, Distribution const& dist_unpacked,
                    IndexRange auto const& active_indices,
                    kamping::Communicator<> const& comm,
                    bool use_grid_communication = false) {
    std::optional<TopologyAwareGridCommunicator> grid_comm;
    if (use_grid_communication) {
      grid_comm.emplace(comm);
    }
    auto local_packed_size = std::ranges::size(active_indices);
    auto unpacked_idx_requests = succ_array_packed.first(local_packed_size) |
                                 std::views::filter([&](auto succ_packed) {
                                   return !dist_packed.is_local(succ_packed, comm.rank());
                                 });
    kamping::measurements::timer().synchronize_and_start("build_request_set");
    absl::flat_hash_set<idx_t> unpacked_idx_requests_dedup;
    std::size_t duplicates = 0;
    for (auto const& succ_packed : unpacked_idx_requests) {
      auto [_, inserted] = unpacked_idx_requests_dedup.insert(succ_packed);
      if (!inserted) {
        duplicates++;
      }
    }
    SPDLOG_DEBUG("[unpack] Removed {} duplicate requests", duplicates);

    kamping::measurements::timer().stop();
    kamping::measurements::timer().synchronize_and_start("request_reply");
    auto requests = unpacked_idx_requests_dedup |
                    std::views::transform([&](auto succ_packed) {
                      return std::pair{dist_packed.get_owner(succ_packed), succ_packed};
                    }) |
                    std::ranges::to<std::vector>();
    std::vector<std::pair<idx_t, idx_t>> replies;
    if (use_grid_communication) {
      replies = request_reply_without_remote_aggregation(
          requests,
          [&](idx_t succ_packed) {
            KASSERT(dist_packed.is_local(succ_packed, comm.rank()));
            auto succ_packed_idx_local =
                dist_packed.get_local_idx(succ_packed, comm.rank());
            auto succ_unpacked_idx_local =
                idx_local_packed_to_unpacked_[succ_packed_idx_local];
            auto succ_unpacked =
                dist_unpacked.get_global_idx(succ_unpacked_idx_local, comm.rank());
            return std::pair{succ_packed, succ_unpacked};
          },
          *grid_comm);
    } else {
      replies = request_reply_without_remote_aggregation(
          requests,
          [&](idx_t succ_packed) {
            KASSERT(dist_packed.is_local(succ_packed, comm.rank()));
            auto succ_packed_idx_local =
                dist_packed.get_local_idx(succ_packed, comm.rank());
            auto succ_unpacked_idx_local =
                idx_local_packed_to_unpacked_[succ_packed_idx_local];
            auto succ_unpacked =
                dist_unpacked.get_global_idx(succ_unpacked_idx_local, comm.rank());
            return std::pair{succ_packed, succ_unpacked};
          },
          comm);
    }
    kamping::measurements::timer().stop();
    kamping::measurements::timer().synchronize_and_start("build_unpack_map");
    absl::flat_hash_map<idx_t, idx_t> succ_packed_to_unpacked{replies.begin(),
                                                              replies.end()};
    kamping::measurements::timer().stop();
    kamping::measurements::timer().synchronize_and_start("write_unpacked");
    for (std::size_t index_local_packed = 0; index_local_packed < local_packed_size;
         ++index_local_packed) {
      auto succ_packed = succ_array_packed[index_local_packed];
      auto succ_unpacked = succ_packed;
      if (dist_packed.is_local(succ_packed, comm.rank())) {
        auto succ_packed_idx_local = dist_packed.get_local_idx(succ_packed, comm.rank());
        auto succ_unpacked_idx_local =
            idx_local_packed_to_unpacked_[succ_packed_idx_local];
        succ_unpacked =
            dist_unpacked.get_global_idx(succ_unpacked_idx_local, comm.rank());
      } else {
        succ_unpacked = succ_packed_to_unpacked[succ_packed];
      }
      succ_array[idx_local_packed_to_unpacked_[index_local_packed]] = succ_unpacked;
      rank_array[idx_local_packed_to_unpacked_[index_local_packed]] =
          rank_array_packed[index_local_packed];
    }
    kamping::measurements::timer().stop();
  };
  return std::make_pair(std::move(packed_dist), std::move(unpack));
}
}  // namespace kascade
