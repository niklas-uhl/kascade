#pragma once

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <vector>

#include <absl/container/flat_hash_set.h>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/collectives/barrier.hpp>
#include <kamping/communicator.hpp>

#include "karam/distributed_array_helpers/datatypes.hpp"
#include "karam/mpi/indirect_all_to_all/grid_all_to_all_impl.hpp"
#include "karam/mpi/indirect_all_to_all/grid_communication_helpers.hpp"

namespace karam {

template <typename Container, typename Predicate>
auto extract_unique_elements_from_sorted_container(
  Container const& container, Predicate&& is_equal
) {
  if (container.empty()) {
    return container;
  }
  std::size_t num_unique_elem = 1ull;
  for (std::size_t i = 1; i < container.size(); ++i) {
    num_unique_elem += !is_equal(container[i - 1], container[i]);
  }
  Container unique_elems(num_unique_elem);
  std::unique_copy(container.begin(), container.end(), unique_elems.begin(), is_equal);
  return unique_elems;
}

template <typename Container, typename GetIndex>
auto extract_unique_indices(Container const& container, GetIndex&& get_index) {
  using IndexType = decltype(get_index(typename Container::value_type{}));
  if (container.empty()) {
    return utils::default_init_vector<IndexType>{};
    ;
  }
  absl::flat_hash_set<IndexType> set;
  for (auto const& elem: container) {
    set.insert(get_index(elem));
  }
  utils::default_init_vector<IndexType> unique_indices;
  unique_indices.reserve(set.size());
  for (auto& elem: set) {
    unique_indices.push_back(elem);
  }
  // for(const auto& index : set
  return unique_indices;
}

template <typename T>
class TD;

// template <typename Requests, typename GetIndex, typename DestinationHandler>
// auto rowwise_exchange(
//   Requests const&              requests,
//   mpi::GridCommunicator const& grid_comm,
//   GetIndex&&                   get_index,
//   DestinationHandler&&         get_destination
//) {
//   auto get_destination_in_row = [&](auto const& elem) {
//     const auto& index             = get_index(elem);
//     auto const& final_destination = get_destination(index);
//     return grid_comm.proxy_col_index(static_cast<std::size_t>(final_destination));
//   };
//
//   auto const send_counts = mpi::compute_send_counts_from_simple_buffer(
//     requests,
//     get_destination_in_row,
//     grid_comm.row_comm().size()
//   );
//   auto send_displacements = send_counts;
//   std::exclusive_scan(send_counts.begin(), send_counts.end(), send_displacements.begin(), 0ull);
//   auto     index_displacements = send_displacements;
//   Requests contiguous_send_buf(requests.size());
//   for (auto const& request: requests) {
//     auto const destination_in_row                      = get_destination_in_row(request);
//     auto const idx                                     =
//     index_displacements[destination_in_row]++; contiguous_send_buf[static_cast<std::size_t>(idx)]
//     = request;
//   }
//
//   return grid_comm.row_comm().alltoallv(
//     kamping::send_buf(contiguous_send_buf),
//     kamping::send_counts(send_counts),
//     kamping::send_displs(send_displacements)
//   );
// }

template <typename ContiguousBuffer, typename GetIndex, typename DestinationHandler>
auto columnwise_exchange(
  ContiguousBuffer&            recv_buf,
  mpi::GridCommunicator const& grid_comm,
  GetIndex&&                   get_index,
  DestinationHandler&&         get_destination
) {
  auto get_destination_in_column = [&](auto const& elem) {
    const auto& index             = get_index(elem);
    auto const& final_destination = get_destination(index);
    return grid_comm.proxy_row_index(static_cast<std::size_t>(final_destination));
  };
  // using ElementType = typename std::remove_reference_t<ContiguousBuffer>::value_type;
  ContiguousBuffer send_buffer(recv_buf.size());
  std::vector<int> send_counts(grid_comm.col_comm().size(), 0);
  for (auto const& elem: recv_buf) {
    ++send_counts[get_destination_in_column(elem)];
  }
  auto send_offsets = send_counts;
  std::exclusive_scan(send_counts.begin(), send_counts.end(), send_offsets.begin(), 0ull);
  auto send_displacements = send_offsets;

  for (auto const& elem: recv_buf) {
    auto const destination_in_column = get_destination_in_column(elem);
    send_buffer[static_cast<std::size_t>(send_offsets[destination_in_column]++)] = elem;
  }
  // deallocate recv_buf as it is not needed anymore
  // utils::dump(std::move(recv_buf));
  return grid_comm.col_comm().alltoallv(
    kamping::send_buf(send_buffer),
    kamping::send_counts(send_counts),
    kamping::send_displs(send_displacements)
  );
}

void is_reached_by_all(std::string const& msg) {
  kamping::comm_world().barrier();
  if (kamping::comm_world().rank() == 0) {
    std::cout << "Is reached: " << msg << std::endl;
  }
}

template <
  typename RequestedType,
  typename IndexType,
  typename Requests,
  typename GetIndex,
  typename GetDestinationOfIndex,
  typename CreateReply>
auto handle_read_requests_in_grid_with_filter(
  Requests const&              requests,
  mpi::GridCommunicator const& grid_comm,
  GetIndex&&                   get_index,
  GetDestinationOfIndex&&      get_destination_of_index,
  CreateReply&&                create_reply
) {
  using IndexDataMap               = std::unordered_map<IndexType, RequestedType>;
  auto mpi_result_requests_rowwise = rowwise_exchange<false>(
    requests,
    [&](auto const request) { return get_destination_of_index(get_index(request)); },
    grid_comm
  );
  // rowwise_exchange<false>(requests, grid_comm, get_index, get_destination_of_index);

  auto rowwise_requests_recv_buf    = mpi_result_requests_rowwise.extract_recv_buffer();
  auto rowwise_requests_recv_counts = mpi_result_requests_rowwise.extract_recv_counts();
  auto rowwise_requests_recv_displs = mpi_result_requests_rowwise.extract_recv_displs();

  auto unique_elements = extract_unique_indices(rowwise_requests_recv_buf, get_index);
  auto mpi_result_requests_columnwise = columnwise_exchange(
    unique_elements,
    grid_comm,
    [](auto& elem) { return elem; },
    get_destination_of_index
  );

  auto columnwise_requests_recv_buf    = mpi_result_requests_columnwise.extract_recv_buffer();
  auto columnwise_requests_recv_counts = mpi_result_requests_columnwise.extract_recv_counts();
  auto columnwise_requests_recv_displs = mpi_result_requests_columnwise.extract_recv_displs();

  utils::default_init_vector<IndexData<RequestedType>> columnwise_replies_buffer(
    columnwise_requests_recv_buf.size()
  );
  std::transform(
    columnwise_requests_recv_buf.begin(),
    columnwise_requests_recv_buf.end(),
    columnwise_replies_buffer.begin(),
    create_reply
  );

  auto mpi_result_replies_columnwise = grid_comm.col_comm().alltoallv(
    kamping::send_buf(columnwise_replies_buffer),
    kamping::send_counts(columnwise_requests_recv_counts),
    kamping::send_displs(columnwise_requests_recv_displs)
  );
  auto columnwise_replies_recv_buffer = mpi_result_replies_columnwise.extract_recv_buffer();

  IndexDataMap index_data;
  for (auto const& elem: columnwise_replies_recv_buffer) {
    index_data.emplace(elem.index, elem.data);
  }

  utils::default_init_vector<AddressData<RequestedType>> rowwise_replies_buffer(
    rowwise_requests_recv_buf.size()
  );

  std::transform(
    rowwise_requests_recv_buf.begin(),
    rowwise_requests_recv_buf.end(),
    rowwise_replies_buffer.begin(),
    [&](IndexAddress const& index_address) {
      auto it = index_data.find(index_address.index);
      return AddressData<RequestedType>{index_address.address, it->second};
    }
  );

  return grid_comm.row_comm().alltoallv(
    kamping::send_buf(rowwise_replies_buffer),
    kamping::send_counts(rowwise_requests_recv_counts),
    kamping::send_displs(rowwise_requests_recv_displs)
  );
}

template <
  typename RequestedType,
  typename IndexType,
  typename Requests,
  typename GetIndex,
  typename GetDestinationOfIndex,
  typename CreateReply>
auto handle_read_requests_in_grid(
  Requests const&              requests,
  mpi::GridCommunicator const& grid_comm,
  GetIndex&&                   get_index,
  GetDestinationOfIndex&&      get_destination_of_index,
  CreateReply&&                create_reply
) {
  auto get_final_destination_request = [&](auto const& request) {
    return get_destination_of_index(get_index(request));
  };

  auto recv_buffer = mpi::grid_mpi_all_to_all(requests, get_final_destination_request, grid_comm)
                       .extract_recv_buffer();

  std::unordered_map<int, std::vector<AddressData<RequestedType>>> replies;
  for (auto const& recv_elem: recv_buffer) {
    auto const& elem   = recv_elem.payload();
    auto const& source = recv_elem.get_source();
    replies[static_cast<int>(source)].emplace_back(create_reply(elem));
  }
  return mpi::grid_mpi_all_to_all(replies, grid_comm);
}

} // namespace karam
