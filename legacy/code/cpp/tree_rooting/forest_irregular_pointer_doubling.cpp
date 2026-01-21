#pragma once

class forest_irregular_pointer_doubling
{
	struct node_request {
		std::uint64_t mst;
	};

	struct answer {
		std::int64_t r_of_mst;
		std::uint64_t mst_of_mst;
		std::uint32_t targetPE_of_mst;
		bool passive_of_mst;
		std::uint64_t non_recursive_index_of_mst;
	};
	

	
	public:
	
	forest_irregular_pointer_doubling(std::vector<std::uint64_t>& s, std::vector<std::int64_t>& r, std::vector<std::uint32_t>& targetPEs, std::vector<std::uint64_t>& prefix_sum_num_vertices_per_PE, std::vector<std::uint64_t>& local_rulers, bool grid, bool aggregate)
	{		
		std::cout << "DEPRECATED" << std::endl;
		this->s = s;
		this->r = r;
		this->targetPEs = targetPEs;
		num_global_vertices = prefix_sum_num_vertices_per_PE[size];
		node_offset = prefix_sum_num_vertices_per_PE[rank];
		this->local_rulers = local_rulers;
		this->grid = grid;
		this->aggregate = aggregate;
	}
	
	forest_irregular_pointer_doubling(std::vector<std::uint64_t>& s, std::vector<std::int64_t>& r, std::vector<std::uint32_t>& targetPEs, std::uint64_t node_offset, std::uint64_t num_global_vertices, std::vector<std::uint64_t>& local_rulers, bool grid, bool aggregate)
	{
		
		this->s = s;
		this->r = r;
		this->targetPEs = targetPEs;
		this->num_global_vertices = num_global_vertices;
		this->node_offset = node_offset;
		this->local_rulers = local_rulers;
		this->grid = grid;
		this->aggregate = aggregate;
	}
	
	
	void start(kamping::Communicator<>& comm, karam::mpi::GridCommunicator& grid_comm)
	{
		rank = comm.rank();
		size = comm.size();
		num_local_vertices = s.size();
		std::vector<std::string> categories = {"local_work", "communication", "other"};
		timer timer("graph_umdrehen", categories, "local_work", "forest_irregular_pointer_doubling");
		timer.add_info(std::string("grid"), std::to_string(grid));
		timer.add_info(std::string("aggregate"), std::to_string(aggregate));

		//timer.add_info(std::string("num_local_vertices"), std::to_string(num_local_vertices), true);
		timer.add_info(std::string("average_num_local_vertices"), std::to_string(num_global_vertices / size));
		
		q = s;
		std::vector<bool> passive(num_local_vertices, false);
		
		std::uint64_t active_nodes = num_local_vertices;
		for (std::int32_t local_index = 0; local_index < num_local_vertices; local_index++)
			if (q[local_index] == local_index + node_offset)
			{
				passive[local_index] = true;
				active_nodes--;
			}
		
		
	
		
		std::vector<std::int32_t> num_packets_per_PE(size,0);
		std::vector<std::int32_t> send_displacements(size + 1,0);
		
		std::vector<node_request> requests(num_local_vertices);
		std::vector<answer> answers(num_local_vertices);

		
		
		std::int32_t max_iteration = std::log2(num_global_vertices) + 2;
		
		
		//while (any_PE_has_work(comm, active_nodes > 0))
		for (std::uint32_t iteration = 0; iteration < max_iteration; iteration++)
		{
			
			

			std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
			for (std::int32_t local_index = 0; local_index < num_local_vertices;local_index++)
			{
				if (!passive[local_index])
				{
					std::int32_t targetPE = targetPEs[local_index];
					num_packets_per_PE[targetPE]++;
				}
			}
			calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
			requests.resize(send_displacements[size]);
			
			//dann requests gefülllt
			for (std::int32_t local_index = 0; local_index < num_local_vertices;local_index++)
			{
				if (!passive[local_index])
				{
					std::int32_t targetPE = targetPEs[local_index];
					std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
					//requests[packet_index].node = local_index + node_offset;
					requests[packet_index].mst = q[local_index];
					
				}
				
			}
			
			/*
			auto recv = comm.alltoallv(kamping::send_buf(requests), kamping::send_counts(num_packets_per_PE));
			std::vector<node_request> recv_requests = recv.extract_recv_buffer();
			num_packets_per_PE = recv.extract_recv_counts();
			answers.resize(recv_requests.size());
			
		
			
			for (std::int32_t i = 0; i < recv_requests.size(); i++)
			{
				
				std::int32_t local_index = recv_requests[i].mst - node_offset;
				answers[i].r_of_mst = r[local_index];
				answers[i].mst_of_mst = q[local_index];
				answers[i].targetPE_of_mst = targetPEs[local_index];
				answers[i].passive_of_mst = passive[local_index];
				answers[i].non_recursive_index_of_mst = local_rulers[local_index];
			}
		
			std::vector<answer> recv_answers = comm.alltoallv(kamping::send_buf(answers), kamping::send_counts(num_packets_per_PE)).extract_recv_buffer();*/
			//dann answers eingetragen
			std::function<answer(const node_request)> lambda = [&] (node_request request) { 
				std::uint64_t local_index = request.mst - node_offset;
				answer answer;
				answer.r_of_mst = r[local_index];
				answer.mst_of_mst = q[local_index];
				answer.targetPE_of_mst = targetPEs[local_index];
				answer.passive_of_mst = passive[local_index];
				answer.non_recursive_index_of_mst = local_rulers[local_index];
				return answer;
			};
			//std::vector<answer> recv_answers = request_reply(timer, requests, num_packets_per_PE, lambda, comm, grid_comm, grid);
			std::function<std::uint64_t(const node_request)> request_assignment =  [](node_request request) {return request.mst;};

			std::vector<answer> recv_answers = request_reply(timer, requests, num_packets_per_PE, lambda, comm, grid_comm, grid, aggregate, request_assignment);

		
			
			std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
			for (std::int32_t local_index = 0; local_index < num_local_vertices;local_index++)
			{
				if (!passive[local_index])
				{
					std::int32_t targetPE = targetPEs[local_index];
					std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
					
					targetPEs[local_index] = recv_answers[packet_index].targetPE_of_mst;
					q[local_index] = recv_answers[packet_index].mst_of_mst;
					r[local_index] = r[local_index] + recv_answers[packet_index].r_of_mst;
					passive[local_index] = recv_answers[packet_index].passive_of_mst;
					active_nodes -= passive[local_index];
					local_rulers[local_index] = recv_answers[packet_index].non_recursive_index_of_mst;
				}
				
			}
			
		}
		
		/*
		std::cout << "recursion on PE " << rank << ":"; 
		for (int i = 0; i< num_local_vertices; i++)
			std::cout << i + node_offset << "," << q[i] << "," << r[i] << "," << local_rulers[i] << std::endl;
		*/
	
		//timer.finalize(comm, "forest_irregular_pointer_doubling");
		
	}
	
	
		
	
	void calculate_send_displacements_and_reset_num_packets_per_PE(std::vector<std::int32_t>& send_displacements, std::vector<std::int32_t>& num_packets_per_PE)
	{
		send_displacements[0]=0;
		for (std::int32_t i = 1; i < size + 1; i++)
			send_displacements[i] = send_displacements[i-1] + num_packets_per_PE[i-1];
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
	}
	
	bool any_PE_has_work(kamping::Communicator<>& comm, bool this_PE_has_work)
	{
		std::int32_t work = this_PE_has_work;
		std::vector<std::int32_t> send(1,work);
		std::vector<std::int32_t> recv;
		comm.allgather(kamping::send_buf(send), kamping::recv_buf<kamping::resize_to_fit>(recv));
		
		for (std::int32_t i = 0; i < size; i++)
			work += recv[i];
		return work > 0;
	}
	
	public:
	bool grid;
	bool aggregate;
	std::uint64_t node_offset;
	std::uint64_t num_local_vertices;
	std::uint64_t num_global_vertices;
	std::uint64_t rank, size;
	std::vector<std::uint64_t> s;
	std::vector<std::uint64_t> q;
	std::vector<std::int64_t> r;
	std::vector<std::uint32_t> targetPEs;
	std::vector<std::uint64_t> prefix_sum_num_vertices_per_PE;
	
	std::vector<std::uint64_t> local_rulers;
};