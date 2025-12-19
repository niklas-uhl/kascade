#pragma once


class list_ranking
{
	virtual void start(std::vector<std::uint64_t>& successors, kamping::Communicator<>& comm, karam::mpi::GridCommunicator& grid_comm) = 0;
	virtual std::vector<std::int64_t> get_ranks() = 0;
};


template <typename T> 
class communicator
{
	public:
	
	virtual std::vector<std::int32_t> extract_recv_counts() = 0;
	virtual std::vector<T> extract_recv_buffer() = 0;
	
	
};
