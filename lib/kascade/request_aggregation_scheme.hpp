#pragma once

#include <span>

#include <absl/container/flat_hash_map.h>
#include <kamping/measurements/timer.hpp>
#include <kamping/utils/flatten.hpp>

#include "kascade/grid_communicator.hpp"

namespace kascade {

namespace aggregation::internal {
template <typename K, std::ranges::forward_range R, typename KeyFn>
auto build_map_by_key(R const& items, KeyFn const& key_fn) {
  using T = std::ranges::range_value_t<R>;
  absl::flat_hash_map<K, T> map;
  map.reserve(items.size());
  for (auto const& item : items) {
    map.emplace(key_fn(item), item);
  }
  return map;
}

template <std::ranges::forward_range Requests, typename GetNextTargetRankFn>
auto do_exchange(Requests const& requests,
                 GetNextTargetRankFn const& get_next_target_rank,
                 kamping::Communicator<> const& comm) {
  namespace kmp = kamping::params;
  using Request = std::ranges::range_value_t<Requests>;
  // compute first target in inter-node-comm
  absl::flat_hash_map<int, std::vector<Request>> send_bufs;
  for (auto const& request : requests) {
    size_t target_rank = get_next_target_rank(request);
    send_bufs[target_rank].emplace_back(request);
  }
  auto [send_buf, send_counts, send_displs] = kamping::flatten(send_bufs, comm.size());

  // exchange on inter-node comm
  return comm.alltoallv(kmp::send_buf(send_buf), kmp::send_counts(send_counts),
                        kmp::send_displs(send_displs), kmp::recv_counts_out());
}
}  // namespace aggregation::internal

template <typename RequestKey,
          typename Reply,
          std::ranges::forward_range Requests,
          typename GetTargetRankFn,
          typename MakeReplyFn,
          typename GetRequestKeyFn,
          typename GetReplyKeyFn>
auto request_with_remote_aggreation(Requests const& requests,
                                    GetTargetRankFn const& get_target_rank,
                                    GetRequestKeyFn const& get_request_key,
                                    GetReplyKeyFn const& get_reply_key,
                                    MakeReplyFn const& make_reply,
                                    TopologyAwareGridCommunicator const& grid_comm)
    -> std::vector<Reply> {
  namespace kmp = kamping::params;
  using namespace aggregation::internal;
  using Request = std::ranges::range_value_t<Requests>;

  // exchange on internode communciator
  auto [recv_requests_inter, recv_counts_inter] = do_exchange(
      requests,
      [&](auto const& req) { return grid_comm.inter_node_rank(get_target_rank(req)); },
      grid_comm.inter_node_comm());

  // do aggregation
  auto agg_requests = build_map_by_key<RequestKey>(recv_requests_inter, get_request_key);

  // exchange on internode communciator
  auto [recv_requests_intra, recv_counts_intra] = do_exchange(
      agg_requests | std::views::transform(
                         [](auto const& kv) -> const Request& { return kv.second; }),
      [&](auto const& req) { return grid_comm.intra_node_rank(get_target_rank(req)); },
      grid_comm.intra_node_comm());

  // create replies
  auto replies_intra =
      recv_requests_intra |
      std::views::transform([&](auto const& request) { return make_reply(request); }) |
      std::ranges::to<std::vector>();

  // exchange replies on intra-node comm
  auto recv_replies_intra = grid_comm.intra_node_comm().alltoallv(
      kmp::send_buf(replies_intra), kmp::send_counts(recv_counts_intra));

  auto lookup_table = build_map_by_key<RequestKey>(recv_replies_intra, get_reply_key);

  // do deaggregation
  auto replies_inter = recv_requests_inter |
                       std::views::transform([&](auto const& request) {
                         auto key = get_request_key(request);
                         auto it = lookup_table.find(key);
                         KASSERT(it != lookup_table.end());
                         return it->second;
                       }) |
                       std::ranges::to<std::vector>();

  // exchange disaggreated replies on inter-node comm
  auto recv_replies_inter = grid_comm.inter_node_comm().alltoallv(
      kmp::send_buf(replies_inter), kmp::send_counts(recv_counts_inter));
  return recv_replies_inter;
}

template <typename RequestKey,
          typename Reply,
          std::ranges::forward_range Requests,
          typename GetTargetRankFn,
          typename MakeReplyFn,
          typename GetRequestKeyFn,
          typename GetReplyKeyFn>
auto request_without_remote_aggregation(Requests const& requests,
                                 GetTargetRankFn const& get_target_rank,
                                 GetRequestKeyFn const& /*get_request_key*/,
                                 GetReplyKeyFn const& /*get_reply_key*/,
                                 MakeReplyFn const& make_reply,
                                 kamping::Communicator<> const& comm)
    -> std::vector<Reply> {
  namespace kmp = kamping::params;
  using Request = std::ranges::range_value_t<Requests>;
  absl::flat_hash_map<int, std::vector<Request>> send_bufs;
  kamping::measurements::timer().synchronize_and_start("route_requests");
  for (auto const& req : requests) {
    int owner = get_target_rank(req);
    send_bufs[owner].emplace_back(req);
  }
  kamping::measurements::timer().stop_and_append();
  kamping::measurements::timer().synchronize_and_start("pack_requests");
  auto [send_buf, send_counts, send_displs] = kamping::flatten(send_bufs, comm.size());
  kamping::measurements::timer().stop_and_append();
  kamping::measurements::timer().synchronize_and_start("send_requests");
  auto [recv_requests, recv_counts] =
      comm.alltoallv(kmp::send_buf(send_buf), kmp::send_counts(send_counts),
                     kmp::send_displs(send_displs), kmp::recv_counts_out());
  kamping::measurements::timer().stop_and_append();
  kamping::measurements::timer().synchronize_and_start("pack_replies");
  auto replies =
      recv_requests |
      std::views::transform([&](auto const& request) { return make_reply(request); }) |
      std::ranges::to<std::vector>();
  kamping::measurements::timer().stop_and_append();
  kamping::measurements::timer().synchronize_and_start("send_replies");
  auto result = comm.alltoallv(kmp::send_buf(replies), kmp::send_counts(recv_counts));
  kamping::measurements::timer().stop_and_append();
  return result;
}

}  // namespace kascade
