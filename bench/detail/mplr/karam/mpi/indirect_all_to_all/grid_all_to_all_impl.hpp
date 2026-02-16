#pragma once

#include <algorithm>
#include <numeric>

#include <kamping/collectives/alltoall.hpp>
#include <kamping/collectives/barrier.hpp>
#include <kamping/communicator.hpp>

#include "../communication_concepts.hpp"
#include "grid_communication_helpers.hpp"
#include "../../utils/containers.hpp"


namespace karam::mpi {
template <
  typename InputRange,
  typename DestinationHandler,
  typename T = send_buffer_list_datatype<InputRange>>
  requires SendBufferList<InputRange, T>
auto compute_send_counts(
  InputRange const& send_buf, DestinationHandler&& get_destination, std::size_t comm_size
) {
  std::vector<int> send_counts(comm_size, 0);
  for (auto const& kv: send_buf) {
    std::size_t const target = get_destination(kv.first);
    send_counts[target] += static_cast<int>(std::ssize(kv.second));
  }
  return send_counts;
}

template <typename Requests, typename DestinationHandler>
auto compute_send_counts_from_simple_buffer(
  Requests const& requests, DestinationHandler&& get_destination, std::size_t comm_size
) {
  std::vector<int> send_counts(comm_size, 0);
  for (auto const& elem: requests) {
    std::size_t const destination_pe = get_destination(elem);
    ++send_counts[destination_pe];
  }
  return send_counts;
}

template <typename InputRange, typename T = send_buffer_list_datatype<InputRange>>
  requires SendBufferList<InputRange, T>
auto rowwise_exchange(InputRange const& send_buf, GridCommunicator const& grid_comm) {
  kamping::Communicator<> const& comm                   = kamping::comm_world();
  auto                           get_destination_in_row = [&](int final_destination) {
    return grid_comm.proxy_col_index(static_cast<std::size_t>(final_destination));
  };
  auto const send_counts =
    compute_send_counts(send_buf, get_destination_in_row, grid_comm.row_comm().size());
  auto send_displacements = send_counts;
  std::exclusive_scan(send_counts.begin(), send_counts.end(), send_displacements.begin(), 0ull);
  auto const total_send_count =
    static_cast<std::size_t>(send_displacements.back() + send_counts.back());

  utils::default_init_vector<IndirectMessage<T>> contiguous_send_buf(total_send_count);
  for (auto const& kv: send_buf) {
    std::size_t const target = get_destination_in_row(kv.first);
    auto const&       data   = kv.second;
    std::transform(
      data.begin(),
      data.end(),
      contiguous_send_buf.begin() + send_displacements[target],
      [&](auto const& elem) {
        return IndirectMessage<T>(
          static_cast<std::uint32_t>(comm.rank()),
          static_cast<std::uint32_t>(kv.first),
          elem
        );
      }
    );
    send_displacements[target] += static_cast<int>(std::size(data));
  }

  return grid_comm.row_comm().alltoallv(
    kamping::send_buf(contiguous_send_buf),
    kamping::send_counts(send_counts)
  );
}

template <bool use_indirect_wrapper = true, typename SendBuffer, typename DestinationHandler>
auto rowwise_exchange(
  SendBuffer const&       send_buf,
  DestinationHandler&&    get_final_destination,
  GridCommunicator const& grid_comm
) {
  using T                                               = typename SendBuffer::value_type;
  kamping::Communicator<> const& comm                   = kamping::comm_world();
  auto                           get_destination_in_row = [&](auto const& elem) {
    auto final_destination = get_final_destination(elem);
    return grid_comm.proxy_col_index(static_cast<std::size_t>(final_destination));
  };
  auto const send_counts = compute_send_counts_from_simple_buffer(
    send_buf,
    get_destination_in_row,
    grid_comm.row_comm().size()
  );
  auto send_displacements = send_counts;
  std::exclusive_scan(send_counts.begin(), send_counts.end(), send_displacements.begin(), 0ull);
  auto       index_displacements = send_displacements;
  auto const total_send_count =
    static_cast<std::size_t>(send_displacements.back() + send_counts.back());

  using MsgType = std::conditional_t<use_indirect_wrapper, IndirectMessage<T>, T>;
  utils::default_init_vector<MsgType> contiguous_send_buf(total_send_count);
  for (auto const& elem: send_buf) {
    auto const final_destination = get_final_destination(elem);
    auto const destination_in_row =
      grid_comm.proxy_col_index(static_cast<std::size_t>(final_destination));
    auto const idx = index_displacements[destination_in_row]++;
    if constexpr (use_indirect_wrapper) {
      contiguous_send_buf[static_cast<std::size_t>(idx)] = MsgType(
        static_cast<std::uint32_t>(comm.rank()),
        static_cast<std::uint32_t>(final_destination),
        elem
      );
    } else {
      contiguous_send_buf[static_cast<std::size_t>(idx)] = elem;
    }
  }
	

	
  return grid_comm.row_comm().alltoallv(
    kamping::send_buf(contiguous_send_buf),
    kamping::send_counts(send_counts)
  );
}



template <typename IntermediateRecvBuffer>
auto columnwise_exchange(IntermediateRecvBuffer&& recv_buf, GridCommunicator const& grid_comm) {
  auto get_destination_in_column = [&](int final_destination) {
    return grid_comm.proxy_row_index(static_cast<std::size_t>(final_destination));
  };
  using ElementType = typename std::remove_reference_t<IntermediateRecvBuffer>::value_type;
  utils::default_init_vector<ElementType> intermediate_send_buf(recv_buf.size());
  std::vector<int>                        send_counts(grid_comm.col_comm().size(), 0);
  for (auto const& elem: recv_buf) {
    ++send_counts[get_destination_in_column(static_cast<int>(elem.get_destination()))];
  }
  auto send_offsets = send_counts;
  std::exclusive_scan(send_counts.begin(), send_counts.end(), send_offsets.begin(), 0ull);
  auto send_displacements = send_offsets;

  for (auto const& elem: recv_buf) {
    auto const target = get_destination_in_column(static_cast<int>(elem.get_destination()));
    intermediate_send_buf[static_cast<std::size_t>(send_offsets[target]++)] = elem;
  }
  // deallocate recv_buf as it is not needed anymore
  utils::dump(std::move(recv_buf));
  return grid_comm.col_comm().alltoallv(
    kamping::send_buf(intermediate_send_buf),
    kamping::send_counts(send_counts),
    kamping::send_displs(send_displacements)
  );
}

} // namespace karam::mpi
