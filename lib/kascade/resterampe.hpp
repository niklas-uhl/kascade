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
