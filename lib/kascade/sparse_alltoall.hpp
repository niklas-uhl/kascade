#pragma once

#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <kamping/communicator.hpp>
#include <kamping/mpi_datatype.hpp>
#include <kamping/plugin/alltoall_sparse.hpp>
#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kascade/configuration.hpp"

namespace kascade {

using SparseAlltoallCommunicator =
    kamping::Communicator<std::vector, kamping::plugin::SparseAlltoall>;

/// @brief Exchanges an alltoallv-style send buffer using kamping's sparse alltoall plugin
/// (NBX), i.e. only ranks with a non-empty message are contacted.
///
/// For each received message, `get_recv_span(source, count)` is called and has to return a
/// span of exactly `count` elements the message is received into.
template <typename T, typename GetRecvSpanFn>
void sparse_alltoallv(std::span<T const> send_buf,
                      std::span<int const> send_counts,
                      std::span<int const> send_displs,
                      GetRecvSpanFn&& get_recv_span,
                      kamping::Communicator<> const& comm) {
  namespace kmp = kamping::params;
  namespace plugin = kamping::plugin::sparse_alltoall;
  KASSERT(send_counts.size() == comm.size());
  KASSERT(send_displs.size() == comm.size());
  std::vector<std::pair<int, std::span<T const>>> sparse_send_buf;
  for (std::size_t rank = 0; rank < send_counts.size(); ++rank) {
    if (send_counts[rank] > 0) {
      sparse_send_buf.emplace_back(
          static_cast<int>(rank),
          send_buf.subspan(static_cast<std::size_t>(send_displs[rank]),
                           static_cast<std::size_t>(send_counts[rank])));
    }
  }
  SparseAlltoallCommunicator sparse_comm{comm.mpi_communicator()};
  sparse_comm.alltoallv_sparse(
      plugin::sparse_send_buf(sparse_send_buf),
      plugin::on_message([&](auto& probed_msg) {
        auto const count = probed_msg.recv_count();
        std::span<T> recv_span = get_recv_span(probed_msg.source_signed(), count);
        KASSERT(recv_span.size() == count);
        probed_msg.recv(kmp::recv_buf(recv_span));
      }));
}

/// Tag of the point-to-point messages of `pcx_alltoallv`.
inline constexpr int pcx_tag = 20563;

/// @brief Exchanges an alltoallv-style send buffer using the personalized census protocol
/// PCX (Hoefler et al., "Scalable communication protocols for dynamic sparse data
/// exchange", 2010): a reduce-scatter tells each rank how many messages it receives, then
/// only non-empty messages are sent and the expected number of messages is probed for.
///
/// For each received message, `get_recv_span(source, count)` is called and has to return a
/// span of exactly `count` elements the message is received into.
///
/// A fixed tag suffices: a rank only sends after the reduce-scatter of its exchange has
/// completed, which requires every rank to have finished receiving in the previous one.
template <typename T, typename GetRecvSpanFn>
void pcx_alltoallv(std::span<T const> send_buf,
                   std::span<int const> send_counts,
                   std::span<int const> send_displs,
                   GetRecvSpanFn&& get_recv_span,
                   kamping::Communicator<> const& comm) {
  KASSERT(send_counts.size() == comm.size());
  KASSERT(send_displs.size() == comm.size());
  MPI_Comm const mpi_comm = comm.mpi_communicator();
  MPI_Datatype const type = kamping::mpi_datatype<T>();
  auto const rank = comm.rank();

  // personalized census: the number of other ranks that send a message to this rank
  std::vector<int> has_message(comm.size(), 0);
  for (std::size_t target = 0; target < send_counts.size(); ++target) {
    has_message[target] = (target != rank && send_counts[target] > 0) ? 1 : 0;
  }
  int num_incoming = 0;
  MPI_Reduce_scatter_block(has_message.data(), &num_incoming, 1, MPI_INT, MPI_SUM,
                           mpi_comm);

  std::vector<MPI_Request> requests;
  for (std::size_t target = 0; target < send_counts.size(); ++target) {
    if (target == rank || send_counts[target] == 0) {
      continue;
    }
    MPI_Isend(send_buf.data() + send_displs[target], send_counts[target], type,
              static_cast<int>(target), pcx_tag, mpi_comm, &requests.emplace_back());
  }
  // the message to this rank itself is copied directly
  if (send_counts[rank] > 0) {
    auto const own = send_buf.subspan(static_cast<std::size_t>(send_displs[rank]),
                                      static_cast<std::size_t>(send_counts[rank]));
    std::span<T> recv_span = get_recv_span(comm.rank_signed(), own.size());
    KASSERT(recv_span.size() == own.size());
    std::ranges::copy(own, recv_span.begin());
  }
  for (int i = 0; i < num_incoming; ++i) {
    MPI_Status status;
    MPI_Probe(MPI_ANY_SOURCE, pcx_tag, mpi_comm, &status);
    int count = 0;
    MPI_Get_count(&status, type, &count);
    std::span<T> recv_span =
        get_recv_span(status.MPI_SOURCE, static_cast<std::size_t>(count));
    KASSERT(recv_span.size() == static_cast<std::size_t>(count));
    MPI_Recv(recv_span.data(), count, type, status.MPI_SOURCE, pcx_tag, mpi_comm,
             MPI_STATUS_IGNORE);
  }
  MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
}

namespace internal {
/// Receive callback that appends each message to `recv_buf` and records its layout.
template <typename T>
auto append_messages(std::vector<T>& recv_buf,
                     std::vector<int>& recv_counts,
                     std::vector<int>& recv_displs,
                     std::size_t comm_size) {
  recv_buf.clear();
  recv_counts.assign(comm_size, 0);
  recv_displs.assign(comm_size, 0);
  return [&](int source, std::size_t count) {
    auto const offset = recv_buf.size();
    recv_counts[static_cast<std::size_t>(source)] = static_cast<int>(count);
    recv_displs[static_cast<std::size_t>(source)] = static_cast<int>(offset);
    recv_buf.resize(offset + count);
    return std::span<T>(recv_buf).subspan(offset, count);
  };
}
}  // namespace internal

/// @brief Runs the sparse exchange selected by `algorithm` (NBX or PCX) with a receive
/// callback, see `sparse_alltoallv` and `pcx_alltoallv`.
template <typename T, typename GetRecvSpanFn>
void sparse_exchange(AlltoallAlgorithm algorithm,
                     std::span<T const> send_buf,
                     std::span<int const> send_counts,
                     std::span<int const> send_displs,
                     GetRecvSpanFn&& get_recv_span,
                     kamping::Communicator<> const& comm) {
  switch (algorithm) {
    case AlltoallAlgorithm::nbx:
      sparse_alltoallv(send_buf, send_counts, send_displs,
                       std::forward<GetRecvSpanFn>(get_recv_span), comm);
      return;
    case AlltoallAlgorithm::pcx:
      pcx_alltoallv(send_buf, send_counts, send_displs,
                    std::forward<GetRecvSpanFn>(get_recv_span), comm);
      return;
    default:
      throw std::invalid_argument("not a sparse alltoall algorithm");
  }
}

/// @brief Sparse variant of alltoallv. Messages are appended to `recv_buf` in arrival
/// order; `recv_counts` and `recv_displs` describe the received layout per source rank (so
/// the displacements are not necessarily increasing).
template <typename T>
void sparse_exchange(AlltoallAlgorithm algorithm,
                     std::span<T const> send_buf,
                     std::span<int const> send_counts,
                     std::span<int const> send_displs,
                     std::vector<T>& recv_buf,
                     std::vector<int>& recv_counts,
                     std::vector<int>& recv_displs,
                     kamping::Communicator<> const& comm) {
  sparse_exchange(algorithm, send_buf, send_counts, send_displs,
                  internal::append_messages(recv_buf, recv_counts, recv_displs, comm.size()),
                  comm);
}
}  // namespace kascade
