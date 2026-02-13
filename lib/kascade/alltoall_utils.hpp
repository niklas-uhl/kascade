#pragma once

#include <concepts>
#include <numeric>
#include <ranges>
#include <vector>

#include <kassert/kassert.hpp>

namespace kascade {

template <typename T, typename Msg>
auto get_target_rank(std::pair<T, Msg> const& envelope) -> int {
  return static_cast<int>(std::get<0>(envelope));
}

template <typename T, typename Msg>
auto get_message(std::pair<T, Msg> const& envelope) -> Msg const& {
  return std::get<1>(envelope);
}

template <typename T, typename Msg>
auto get_message(std::pair<T, Msg>& envelope) -> Msg& {
  return std::get<1>(envelope);
}

template <typename T, typename Msg>
auto get_target_rank(std::tuple<T, Msg> const& envelope) -> int {
  return static_cast<int>(std::get<0>(envelope));
}

template <typename T, typename Msg>
auto get_message(std::tuple<T, Msg> const& envelope) -> Msg const& {
  return std::get<1>(envelope);
}

template <typename T, typename Msg>
auto get_message(std::tuple<T, Msg>& envelope) -> Msg& {
  return std::get<1>(envelope);
}

template <typename M>
concept EnvelopedMsg = requires(M m) {
  requires std::same_as<int, std::remove_cvref_t<decltype(get_target_rank(
                                 m))>>;  // first part is target rank
  { get_message(m) };                    // second part is actual msg
  { true };
};

template <typename R>
concept EnvelopedMsgRange =
    std::ranges::forward_range<R> && EnvelopedMsg<std::ranges::range_reference_t<R>>;

template <EnvelopedMsg M>
using MsgTypeOf = std::remove_cvref_t<decltype(std::get<1>(std::declval<M>()))>;

template <EnvelopedMsgRange R>
auto prepare_send_buf(R const& messages,
                      std::vector<MsgTypeOf<std::ranges::range_value_t<R>>> send_buf,
                      std::vector<int> send_counts,
                      std::vector<int> send_displs,
                      std::size_t comm_size) {
  send_counts.clear();
  send_displs.clear();
  send_counts.resize(comm_size, 0);
  send_displs.resize(comm_size);
  for (auto&& msg : messages) {
    int target_rank = get_target_rank(msg);
    KASSERT(0 <= target_rank && target_rank < static_cast<int>(comm_size));
    ++send_counts[target_rank];
  }
  std::exclusive_scan(send_counts.begin(), send_counts.end(), send_displs.begin(), 0);
  send_buf.resize(send_displs.back() + send_counts.back());
  for (auto&& msg : messages) {
    int target_rank = get_target_rank(msg);
    int pos = send_displs[target_rank]++;
    send_buf[static_cast<std::size_t>(pos)] = get_message(msg);
  }
  std::exclusive_scan(send_counts.begin(), send_counts.end(), send_displs.begin(), 0);
  return std::make_tuple(std::move(send_buf), std::move(send_counts),
                         std::move(send_displs));
}

template <EnvelopedMsgRange R>
auto prepare_send_buf(R const& messages, std::size_t comm_size) {
  using Msg = MsgTypeOf<std::ranges::range_value_t<R>>;
  std::vector<int> send_counts;
  send_counts.reserve(comm_size);
  std::vector<int> send_displs;
  send_displs.reserve(comm_size);
  std::vector<Msg> send_buf;
  return prepare_send_buf(messages, std::move(send_buf), std::move(send_counts),
                          std::move(send_displs), comm_size);
}
}  // namespace kascade
