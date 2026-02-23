#pragma once

#include <type_traits>

#include <kamping/measurements/timer.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/mpi_datatype.hpp>
#include <spdlog/spdlog.h>
#include <fmt/ranges.h>

#include "kascade/alltoall_utils.hpp"
#include "kascade/grid_communicator.hpp"

namespace kascade {

template <typename Msg>
struct Envelope {
  int target_rank;
  Msg msg;
};
}  // namespace kascade

namespace kamping {
template <typename Msg>
struct mpi_type_traits<
    kascade::Envelope<Msg>,
    std::enable_if_t<std::is_same_v<decltype(type_dispatcher<kascade::Envelope<Msg>>()),
                                    no_matching_type>>>
    // : kamping::struct_type<kascade::Envelope<Msg>> {};
    : kamping::byte_serialized<kascade::Envelope<Msg>> {};
}  // namespace kamping

namespace kascade {
template <typename Msg>
auto get_target_rank(Envelope<Msg> const& envelope) -> int {
  return envelope.target_rank;
}

template <typename Msg>
auto get_message(Envelope<Msg> const& envelope) -> auto const& {
  return envelope.msg;
}

template <typename Msg>
auto get_message(Envelope<Msg>& envelope) -> auto& {
  return envelope.msg;
}

template <typename Msg>
struct SourcedEnvelope {
  int source_rank;
  int target_rank;
  Msg msg;
};
template <typename Msg>
auto get_target_rank(SourcedEnvelope<Msg> const& envelope) -> int {
  return envelope.target_rank;
}

template <typename Msg>
auto get_source_rank(SourcedEnvelope<Msg> const& envelope) -> int {
  return envelope.source_rank;
}

template <typename Msg>
auto get_message(SourcedEnvelope<Msg> const& envelope) -> auto const& {
  return envelope.msg;
}

template <typename Msg>
auto get_message(SourcedEnvelope<Msg>& envelope) -> auto& {
  return envelope.msg;
}

template <EnvelopedMsgRange R>
auto grid_alltoallv(R&& messages, TopologyAwareGridCommunicator const& grid_comm) {
  namespace kmp = kamping::params;
  using msg_t = MsgTypeOf<std::ranges::range_value_t<R>>;

  //*************************
  // inter-node-comm exchange
  //*************************
  // repack message to ensure that global target rank is stored in MPI sendable envelope
  // TODO maybe unnecessary as we could send std::pair/std::tuple anyway?
  auto packed_env = messages | std::views::transform([](auto const& envelope) {
                      return Envelope<msg_t>{.target_rank = get_target_rank(envelope),
                                             .msg = get_message(envelope)};
                    });
  auto inter_node_comm_targets =
      messages | std::views::transform([&grid_comm](auto const& envelope) {
        return static_cast<int>(grid_comm.inter_node_rank(get_target_rank(envelope)));
      });

  auto [send_buf_inter, send_counts_inter, send_displs_inter] =
      prepare_send_buf(std::views::zip(inter_node_comm_targets, packed_env),
                       grid_comm.inter_node_comm().size());

  SPDLOG_LOGGER_TRACE(spdlog::get("gather"), "counts {}, displs {}, size {}",
                      send_counts_inter, send_displs_inter,
                      grid_comm.inter_node_comm().size());

  auto recv_buf_inter = grid_comm.inter_node_comm().alltoallv(
      kmp::send_buf(send_buf_inter), kmp::send_counts(send_counts_inter),
      kmp::send_displs(send_displs_inter));

  //*************************
  // intra-node-comm exchange
  //*************************
  auto unpacked_messages =
      recv_buf_inter |
      std::views::transform([](auto const& envelope) { return get_message(envelope); });

  auto intra_node_comm_targets =
      recv_buf_inter | std::views::transform([&grid_comm](auto const& envelope) {
        return static_cast<int>(grid_comm.intra_node_rank(get_target_rank(envelope)));
      });

  auto [send_buf_intra, send_counts_intra, send_displs_intra] =
      prepare_send_buf(std::views::zip(intra_node_comm_targets, unpacked_messages),
                       grid_comm.intra_node_comm().size());

  SPDLOG_LOGGER_TRACE(spdlog::get("gather"), "counts {}, displs {}, size {}",
                      send_counts_intra, send_displs_intra,
                      grid_comm.intra_node_comm().size());
  return grid_comm.intra_node_comm().alltoallv(kmp::send_buf(send_buf_intra),
                                               kmp::send_counts(send_counts_intra),
                                               kmp::send_displs(send_displs_intra));
}

template <typename T>
class AlltoallDispatcher {
public:
  AlltoallDispatcher(bool use_grid_alltoall,
                     kamping::Communicator<> const& comm,
                     std::optional<TopologyAwareGridCommunicator> const& grid_comm)
      : use_grid_alltoall_{use_grid_alltoall}, comm_{&comm} {
    if (use_grid_alltoall) {
      KASSERT(grid_comm.has_value());
      grid_comm_ = &*grid_comm;
    }
  }

  // TODO make recv_buf a general range if necessary
  template <EnvelopedMsgRange R>
    requires std::is_same_v<MsgTypeOf<std::ranges::range_value_t<R>>, T>
  auto alltoallv(R&& messages, std::vector<T>& recv_buf) {
    namespace kmp = kamping::params;
    if (use_grid_alltoall_) {
      KASSERT(grid_comm_ != nullptr);
      recv_buf = kascade::grid_alltoallv(std::forward<R>(messages), *grid_comm_);
    } else {
      kamping::measurements::timer().start("prepare_send_buf");
      prepare_send_buf_inplace(std::forward<R>(messages), send_buf, send_counts,
                               send_displs, comm_->size());
      kamping::measurements::timer().stop();
      kamping::measurements::timer().start("alltoallv");
      comm_->alltoallv(
          kmp::send_buf(send_buf), kmp::send_counts(send_counts),
          kmp::send_displs(send_displs),
          kmp::recv_buf<kamping::BufferResizePolicy::resize_to_fit>(recv_buf));
      kamping::measurements::timer().stop();
    }
  }

  template <EnvelopedMsgRange R>
    requires std::is_same_v<MsgTypeOf<std::ranges::range_value_t<R>>, T>
  auto alltoallv(R&& messages) {
    std::vector<T> recv_buf;
    alltoallv(std::forward<R>(messages), recv_buf);
    return recv_buf;
  }

private:
  bool use_grid_alltoall_;
  kamping::Communicator<> const* comm_;
  TopologyAwareGridCommunicator const* grid_comm_ = nullptr;
  std::vector<T> send_buf;
  std::vector<int> send_counts;
  std::vector<int> send_displs;
};

}  // namespace kascade
