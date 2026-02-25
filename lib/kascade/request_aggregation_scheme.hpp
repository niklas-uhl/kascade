#pragma once

#include <absl/container/flat_hash_map.h>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/measurements/timer.hpp>
#include <kamping/utils/flatten.hpp>

#include "alltoall_utils.hpp"
#include "grid_alltoall.hpp"
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
auto request_without_remote_aggregation(Requests&& requests,
                                        GetTargetRankFn const& get_target_rank,
                                        GetRequestKeyFn const& /*get_request_key*/,
                                        GetReplyKeyFn const& /*get_reply_key*/,
                                        MakeReplyFn const& make_reply,
                                        kamping::Communicator<> const& comm)
    -> std::vector<Reply> {
  namespace kmp = kamping::params;
  using Request = std::ranges::range_value_t<Requests>;
  kamping::measurements::timer().start("collect");
  absl::flat_hash_map<int, std::vector<Request>> send_bufs;

  for (auto const& req : requests) {
    int owner = get_target_rank(req);
    send_bufs[owner].emplace_back(req);
  }
  kamping::measurements::timer().stop();
  kamping::measurements::timer().start("flatten");
  auto [send_buf, send_counts, send_displs] = kamping::flatten(send_bufs, comm.size());
  kamping::measurements::timer().stop();
  kamping::measurements::timer().start("request_alltoall");
  auto [recv_requests, recv_counts] =
      comm.alltoallv(kmp::send_buf(send_buf), kmp::send_counts(send_counts),
                     kmp::send_displs(send_displs), kmp::recv_counts_out());
  kamping::measurements::timer().stop();
  kamping::measurements::timer().start("build_replies");
  auto replies =
      recv_requests |
      std::views::transform([&](auto const& request) { return make_reply(request); }) |
      std::ranges::to<std::vector>();
  kamping::measurements::timer().stop();
  kamping::measurements::timer().start("reply_alltoall");
  auto result = comm.alltoallv(kmp::send_buf(replies), kmp::send_counts(recv_counts));
  kamping::measurements::timer().stop();
  return result;
}

template <EnvelopedMsgRange Requests,
          typename Request,
          typename Reply,
          typename MakeReplyFn>
auto request_reply_without_remote_aggregation(Requests const& requests,
                                              MakeReplyFn const& make_reply,
                                              MPIBuffer<Request>& req_sbuffer,
                                              MPIBuffer<Request>& req_rbuffer,
                                              std::vector<Reply>& reply_sbuffer,
                                              std::vector<Reply>& reply_rbuffer,
                                              kamping::Communicator<> const& comm) {
  namespace kmp = kamping::params;
  prepare_send_buf_inplace(requests, req_sbuffer, comm.size());

  comm.alltoallv(kmp::send_buf(req_sbuffer.data), kmp::send_counts(req_sbuffer.counts),
                 kmp::send_displs(req_sbuffer.displs),
                 kmp::recv_buf_out<kamping::resize_to_fit>(req_rbuffer.data),
                 kmp::recv_counts_out<kamping::resize_to_fit>(req_rbuffer.counts),
                 kmp::recv_displs_out<kamping::resize_to_fit>(req_rbuffer.displs));

  reply_sbuffer.clear();
  reply_sbuffer.resize(req_rbuffer.data.size());
  std::ranges::transform(req_rbuffer.data, reply_sbuffer.begin(),
                         [&](auto const& request) { return make_reply(request); });
  comm.alltoallv(kmp::send_buf(reply_sbuffer), kmp::send_counts(req_rbuffer.counts),
                 kmp::send_displs(req_rbuffer.displs),
                 kmp::recv_buf_out<kamping::resize_to_fit>(reply_rbuffer));
}

template <EnvelopedMsgRange Requests, typename MakeReplyFn>
auto request_reply_without_remote_aggregation(Requests const& requests,
                                              MakeReplyFn const& make_reply,
                                              kamping::Communicator<> const& comm) {
  using Request = MsgTypeOf<std::ranges::range_value_t<Requests>>;
  using Reply = std::remove_cvref_t<std::invoke_result_t<MakeReplyFn, Request>>;
  MPIBuffer<Request> req_sbuffer;
  MPIBuffer<Request> req_rbuffer;
  std::vector<Reply> reply_sbuffer;
  std::vector<Reply> reply_rbuffer;
  request_reply_without_remote_aggregation(requests, make_reply, req_sbuffer, req_rbuffer,
                                           reply_sbuffer, reply_rbuffer, comm);
  return reply_rbuffer;
}

namespace request_reply_mode {
struct reorder_output_tag {};
constexpr reorder_output_tag reorder_output;
}  // namespace request_reply_mode

template <EnvelopedMsgRange Requests, typename MakeReplyFn>
auto request_reply_without_remote_aggregation(
    Requests const& requests,
    MakeReplyFn const& make_reply,
    kamping::Communicator<> const& comm,
    request_reply_mode::reorder_output_tag /*tag*/) {
  using Request = MsgTypeOf<std::ranges::range_value_t<Requests>>;
  using Reply = std::remove_cvref_t<std::invoke_result_t<MakeReplyFn, Request>>;
  MPIBuffer<Request> req_sbuffer;
  MPIBuffer<Request> req_rbuffer;
  std::vector<Reply> reply_sbuffer;
  std::vector<Reply> reply_rbuffer;
  request_reply_without_remote_aggregation(requests, make_reply, req_sbuffer, req_rbuffer,
                                           reply_sbuffer, reply_rbuffer, comm);
  return std::pair(std::move(reply_rbuffer), std::move(req_sbuffer.displs));
}

template <EnvelopedMsgRange Requests, typename MakeReplyFn>
auto request_reply_without_remote_aggregation(
    Requests const& requests,
    MakeReplyFn const& make_reply,
    TopologyAwareGridCommunicator const& grid_comm) {
  namespace kmp = kamping::params;
  using msg_t = MsgTypeOf<std::ranges::range_value_t<Requests>>;

  auto recv_buf_inter = [&]() {
    //*************************
    // inter-node-comm exchange
    //*************************
    auto packed_env =
        requests | std::views::transform([&grid_comm](auto const& envelope) {
          return SourcedEnvelope<msg_t>{
              .source_rank = grid_comm.global_comm().rank_signed(),
              .target_rank = get_target_rank(envelope),
              .msg = get_message(envelope)};
        });
    auto inter_node_comm_targets =
        requests | std::views::transform([&grid_comm](auto const& envelope) {
          return static_cast<int>(grid_comm.inter_node_rank(get_target_rank(envelope)));
        });

    auto [send_buf_inter, send_counts_inter, send_displs_inter] =
        prepare_send_buf(std::views::zip(inter_node_comm_targets, packed_env),
                         grid_comm.inter_node_comm().size());

    return grid_comm.inter_node_comm().alltoallv(kmp::send_buf(send_buf_inter),
                                                 kmp::send_counts(send_counts_inter),
                                                 kmp::send_displs(send_displs_inter));
  }();

  auto [recv_buf_intra, recv_counts_intra] = [&]() {
    //*************************
    // intra-node-comm exchange
    //*************************
    auto intra_node_comm_targets =
        recv_buf_inter | std::views::transform([&grid_comm](auto const& envelope) {
          return static_cast<int>(grid_comm.intra_node_rank(get_target_rank(envelope)));
        });

    auto [send_buf_intra, send_counts_intra, send_displs_intra] =
        prepare_send_buf(std::views::zip(intra_node_comm_targets, recv_buf_inter),
                         grid_comm.intra_node_comm().size());
    recv_buf_inter.clear();
    recv_buf_inter.shrink_to_fit();

    return grid_comm.intra_node_comm().alltoallv(
        kmp::send_buf(send_buf_intra), kmp::send_counts(send_counts_intra),
        kmp::send_displs(send_displs_intra), kmp::recv_counts_out());
  }();

  auto replies_intra = recv_buf_intra | std::views::transform([&](auto const& request) {
                         return Envelope{.target_rank = get_source_rank(request),
                                         .msg = make_reply(get_message(request))};
                       }) |
                       std::ranges::to<std::vector>();

  //*************************
  // Communicate Replies back to origin
  //*************************

  auto recv_replies_intra = grid_comm.intra_node_comm().alltoallv(
      kmp::send_buf(replies_intra), kmp::send_counts(recv_counts_intra));

  auto unpacked_replies_intra =
      recv_replies_intra |
      std::views::transform([](auto const& envelope) { return get_message(envelope); });

  auto reply_targets_inter =
      recv_replies_intra | std::views::transform([&](auto const& envelope) {
        return static_cast<int>(grid_comm.inter_node_rank(get_target_rank(envelope)));
      });

  auto [send_buf_replies_inter, send_counts_replies_inter, send_displs_replies_inter] =
      prepare_send_buf(std::views::zip(reply_targets_inter, unpacked_replies_intra),
                       grid_comm.inter_node_comm().size());
  recv_replies_intra.clear();
  recv_replies_intra.shrink_to_fit();

  auto recv_replies_inter = grid_comm.inter_node_comm().alltoallv(
      kmp::send_buf(send_buf_replies_inter), kmp::send_counts(send_counts_replies_inter));

  return recv_replies_inter;
}

template <EnvelopedMsgRange Requests, typename MakeReplyFn>
auto request_reply_without_remote_aggregation(
    Requests const& requests,
    MakeReplyFn const& make_reply,
    TopologyAwareGridCommunicator const& grid_comm,
    request_reply_mode::reorder_output_tag /*tag*/) {
  namespace kmp = kamping::params;
  using Request = MsgTypeOf<std::ranges::range_value_t<Requests>>;
  using Reply = std::remove_cvref_t<std::invoke_result_t<MakeReplyFn, Request>>;

  std::vector<int> global_send_displs(grid_comm.global_comm().size(), 0);
  auto recv_buf_inter = [&]() {
    //*************************
    // inter-node-comm exchange
    //*************************

    // do packing explicit, otherwise third pass over requests for displs would be
    // required
    std::vector<std::pair<int, SourcedEnvelope<Request>>> packed_env;
    packed_env.reserve(requests.size());
    for (const auto& request_env : requests) {
      int global_target = get_target_rank(request_env);
      int inter_node_target =
          grid_comm.inter_node_rank_signed(get_target_rank(request_env));
      packed_env.emplace_back(
          inter_node_target,
          SourcedEnvelope<Request>{grid_comm.global_comm().rank_signed(), global_target,
                                   get_message(request_env)});
      ++global_send_displs[static_cast<std::size_t>(global_target)];
    }
    auto [send_buf_inter, send_counts_inter, send_displs_inter] =
        prepare_send_buf(packed_env, grid_comm.inter_node_comm().size());

    return grid_comm.inter_node_comm().alltoallv(kmp::send_buf(send_buf_inter),
                                                 kmp::send_counts(send_counts_inter),
                                                 kmp::send_displs(send_displs_inter));
  }();
  std::exclusive_scan(global_send_displs.begin(), global_send_displs.end(),
                      global_send_displs.begin(), 0);

  auto [recv_buf_intra, recv_counts_intra] = [&]() {
    //*************************
    // intra-node-comm exchange
    //*************************
    auto intra_node_comm_targets =
        recv_buf_inter | std::views::transform([&grid_comm](auto const& envelope) {
          return static_cast<int>(grid_comm.intra_node_rank(get_target_rank(envelope)));
        });

    auto [send_buf_intra, send_counts_intra, send_displs_intra] =
        prepare_send_buf(std::views::zip(intra_node_comm_targets, recv_buf_inter),
                         grid_comm.intra_node_comm().size());
    recv_buf_inter.clear();
    recv_buf_inter.shrink_to_fit();

    return grid_comm.intra_node_comm().alltoallv(
        kmp::send_buf(send_buf_intra), kmp::send_counts(send_counts_intra),
        kmp::send_displs(send_displs_intra), kmp::recv_counts_out());
  }();

  auto replies_intra = recv_buf_intra | std::views::transform([&](auto const& request) {
                         return SourcedEnvelope<Reply>{
                             .source_rank = grid_comm.global_comm().rank_signed(),
                             .target_rank = get_source_rank(request),
                             .msg = make_reply(get_message(request))};
                       }) |
                       std::ranges::to<std::vector>();

  //*************************
  // Communicate Replies back to origin
  //*************************

  auto recv_replies_intra = grid_comm.intra_node_comm().alltoallv(
      kmp::send_buf(replies_intra), kmp::send_counts(recv_counts_intra));

  auto reply_targets_inter =
      recv_replies_intra | std::views::transform([&](auto const& envelope) {
        return static_cast<int>(grid_comm.inter_node_rank(get_target_rank(envelope)));
      }) |
      std::ranges::to<std::vector>();

  auto [send_buf_replies_inter, send_counts_replies_inter, send_displs_replies_inter] =
      prepare_send_buf(std::views::zip(reply_targets_inter, recv_replies_intra),
                       grid_comm.inter_node_comm().size());

  reply_targets_inter.clear();
  reply_targets_inter.shrink_to_fit();
  recv_replies_intra.clear();
  recv_replies_intra.shrink_to_fit();

  auto recv_replies_inter = grid_comm.inter_node_comm().alltoallv(
      kmp::send_buf(send_buf_replies_inter), kmp::send_counts(send_counts_replies_inter));

  KASSERT(recv_replies_inter.size() == requests.size());
  auto write_back_indices = global_send_displs;
  std::vector<Reply> reordered_replies(recv_replies_inter.size());
  for (std::size_t i = 0; i < recv_replies_inter.size(); ++i) {
    auto initial_target = get_source_rank(recv_replies_inter[i]);
    KASSERT(0 <= initial_target &&
            initial_target < grid_comm.global_comm().size_signed());
    auto insert_pos = static_cast<std::size_t>(write_back_indices[initial_target]++);
    reordered_replies[insert_pos] = get_message(recv_replies_inter[i]);
  }
  return std::pair(std::move(reordered_replies), std::move(global_send_displs));
}

}  // namespace kascade
