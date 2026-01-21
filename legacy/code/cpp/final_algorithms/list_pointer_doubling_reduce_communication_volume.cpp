#pragma once

#include "../helper_functions.cpp"
class list_pointer_doubling_reduce_communication_volume
{
	
	public:
	
	list_pointer_doubling_reduce_communication_volume(bool grid)
	{
		this->grid = grid;
	}
	
	std::vector<std::int64_t> start(std::vector<std::uint64_t>& s, kamping::Communicator<>& comm, karam::mpi::GridCommunicator& grid_comm)
	{
		std::uint32_t size = comm.size();
		std::uint32_t rank = comm.rank();
		std::uint64_t num_local_vertices = s.size();
		std::uint64_t num_global_vertices = num_local_vertices * size;
		std::uint64_t node_offset = num_local_vertices * rank;
		
		std::vector<std::string> categories = {"local_work", "communication"};
		timer timer("start", categories, "local_work", "pointer_doubling_reduce_comm_volume");
		
		timer.add_info(std::string("num_local_vertices"), std::to_string(num_local_vertices));
		timer.add_info("grid", std::to_string(grid));
				
		//initialize variables
		std::vector<std::uint64_t> q = s;
		//1. msb will indicate if node is passive, 2. msb will indicate if node just got passive
		
		std::vector<std::int64_t> r(num_local_vertices, 1);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			if (s[i] == i + node_offset)
			{
				r[i] = 0;	
				q[i] = mark(q[i],0);//mark as passive
			}
		}
		std::vector<std::int32_t> num_packets_per_PE(size,0);
		std::vector<std::int32_t> send_displacements(size + 1,0);
		std::vector<std::uint64_t> requests(num_local_vertices); //has at most one entry to big for first iteration
		std::uint64_t max_iteration = std::ceil(std::log2(num_global_vertices-1))+1;
		
		//start doubling
		for (std::uint64_t iteration = 0; iteration < max_iteration; iteration++)
		{
			std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
			for (std::uint64_t local_index = 0; local_index < num_local_vertices;local_index++)
			{
				if (!is_marked(q[local_index],0) || is_marked(q[local_index],1))
				{
					std::int32_t targetPE = unmask(q[local_index]) / num_local_vertices;
					num_packets_per_PE[targetPE]++;
				}
			}
			calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
			requests.resize(send_displacements[size]);
			
			for (std::uint64_t local_index = 0; local_index < num_local_vertices; local_index++)
			{
				if (!is_marked(q[local_index],0) || is_marked(q[local_index],1))
				{
					std::int32_t targetPE = unmask(q[local_index]) / num_local_vertices;
					std::uint64_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
					
					if (is_marked(q[local_index],1))
						requests[packet_index] = mark(unmask(q[local_index]),0);
					else // here !is_marked(q[local_index],0))
						requests[packet_index] = unmask(q[local_index]);
				}
			}
			
			
			std::function<std::uint64_t(const std::uint64_t)> lambda = [&] (std::uint64_t request) { //here msb indicates if requested node is active
				if (is_marked(request,0))
					return (std::uint64_t) r[unmask(request) - node_offset];
				std::int32_t local_index = request - node_offset;
				if (!is_marked(q[local_index],0))
					return mark(unmask(q[local_index]),0);
				else
					return unmask(q[local_index]);
			};
			
			std::vector<std::uint64_t> recv_answers = request_reply(timer, requests, num_packets_per_PE, lambda, comm, grid_comm, grid);
		
			std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
			for (std::uint64_t local_index = 0; local_index < num_local_vertices; local_index++)
			{
				if (!is_marked(q[local_index],0) || is_marked(q[local_index],1))
				{
					std::int32_t targetPE = unmask(q[local_index]) / num_local_vertices;
					std::uint64_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
					
					if (is_marked(q[local_index],1))
					{
						r[local_index] += recv_answers[packet_index];
						q[local_index] = unmark(q[local_index],1);
					}
					else // here !is_marked(q[local_index],0))
					{
						if (is_marked(recv_answers[packet_index],0))//iff q[local_index] is active
						{
							q[local_index] = unmask(recv_answers[packet_index]);	
							r[local_index] *= 2;
						}
						else
						{
							q[local_index] = mark(q[local_index],0); //this node becomes passive
							q[local_index] = mark(q[local_index],1); //this node just became passive
						}
					}
				}	
			}
		}
		timer.finalize(comm, "pointer_doubling_reduce_comm_volume");
		
		return r;
		
	}
	
	void calculate_send_displacements_and_reset_num_packets_per_PE(std::vector<std::int32_t>& send_displacements, std::vector<std::int32_t>& num_packets_per_PE)
	{
		send_displacements[0]=0;
		for (std::int32_t i = 1; i < num_packets_per_PE.size() + 1; i++)
			send_displacements[i] = send_displacements[i-1] + num_packets_per_PE[i-1];
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
	}
	

	
	
	bool grid;

};