#pragma once

#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include <kamping/communicator.hpp>
#include <kamping/plugin/alltoall_sparse.hpp>
#include <kassert/kassert.hpp>

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

/// @brief Sparse variant of alltoallv. Messages are appended to `recv_buf` in arrival order;
/// `recv_counts` and `recv_displs` describe the received layout per source rank (so the
/// displacements are not necessarily increasing).
template <typename T>
void sparse_alltoallv(std::span<T const> send_buf,
                      std::span<int const> send_counts,
                      std::span<int const> send_displs,
                      std::vector<T>& recv_buf,
                      std::vector<int>& recv_counts,
                      std::vector<int>& recv_displs,
                      kamping::Communicator<> const& comm) {
  recv_buf.clear();
  recv_counts.assign(comm.size(), 0);
  recv_displs.assign(comm.size(), 0);
  auto append = [&](int source, std::size_t count) {
    auto const offset = recv_buf.size();
    recv_counts[static_cast<std::size_t>(source)] = static_cast<int>(count);
    recv_displs[static_cast<std::size_t>(source)] = static_cast<int>(offset);
    recv_buf.resize(offset + count);
    return std::span<T>(recv_buf).subspan(offset, count);
  };
  sparse_alltoallv(send_buf, send_counts, send_displs, append, comm);
}
}  // namespace kascade
