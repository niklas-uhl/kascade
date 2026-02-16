#include <algorithm>
#include <numeric>

#include <kamping/collectives/alltoall.hpp>
#include <kamping/collectives/barrier.hpp>
#include <kamping/communicator.hpp>

#include "indirect_all_to_all/grid_all_to_all_impl.hpp"

namespace karam::mpi {

template <typename SendBuffer, typename DestinationHandler>
auto grid_mpi_all_to_all(
  SendBuffer const&       send_buf,
  DestinationHandler&&    get_final_destination,
  karam::mpi::GridCommunicator const& grid_comm
) {
  auto mpi_result_rowwise  = rowwise_exchange<true>(send_buf, get_final_destination, grid_comm);
  auto rowwise_recv_buf    = mpi_result_rowwise.extract_recv_buffer();
  auto rowwise_recv_counts = mpi_result_rowwise.extract_recv_counts();
  auto rowwise_recv_displs = mpi_result_rowwise.extract_recv_displs();

  return columnwise_exchange(rowwise_recv_buf, grid_comm);
}
//from here on my code



template <typename SendBuffer>
auto my_grid_all_to_all(
  SendBuffer const&       send_buf,
  std::vector<std::int32_t>    send_counts,
  karam::mpi::GridCommunicator const& grid_comm,
  kamping::Communicator<>& comm
) {
	using T                                               = typename SendBuffer::value_type;
	 
	std::vector<std::int32_t> send_counts_row = std::vector<std::int32_t>(grid_comm.row_comm().size(),0);
	for (std::int32_t p = 0; p < comm.size(); p++)
	{
		std::int32_t targetPE = grid_comm.proxy_col_index(static_cast<std::size_t>(p));
		send_counts_row[targetPE] += send_counts[p];
	}
	
	auto send_displacements = send_counts_row;
	std::exclusive_scan(send_counts_row.begin(), send_counts_row.end(), send_displacements.begin(), 0ull);
	auto       index_displacements = send_displacements;
	
	utils::default_init_vector<IndirectMessage<T>> contiguous_send_buf(send_buf.size()); 
	
	std::uint64_t index = 0;
	for (std::int32_t p = 0; p < comm.size(); p++)
	{

		for (std::uint64_t i = 0; i < send_counts[p]; i++)
		{
			auto const final_destination = p;
			auto const destination_in_row = grid_comm.proxy_col_index(static_cast<std::size_t>(final_destination));
			auto const idx = index_displacements[destination_in_row]++;

			contiguous_send_buf[static_cast<std::size_t>(idx)] = IndirectMessage<T>(
				static_cast<std::uint32_t>(comm.rank()),
				static_cast<std::uint32_t>(final_destination),
				send_buf[index++]
			  );
			
			
		}
		
	}

	auto mpi_result_rowwise = grid_comm.row_comm().alltoallv(
		kamping::send_buf(contiguous_send_buf),
		kamping::send_counts(send_counts_row)
	);

  //auto mpi_result_rowwise  = rowwise_exchange<true>(send_buf, get_final_destination, grid_comm);
	//auto rowwise_recv_buf    = mpi_result_rowwise.extract_recv_buffer();


	return columnwise_exchange(mpi_result_rowwise, grid_comm);
}


}



