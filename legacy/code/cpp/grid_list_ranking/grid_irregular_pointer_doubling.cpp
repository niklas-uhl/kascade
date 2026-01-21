
class grid_irregular_pointer_doubling
{
	struct node_request {
		std::uint64_t node;
		std::uint64_t mst;
	};

	struct answer {
		std::uint64_t node;
		std::int64_t r_of_mst;
		std::uint64_t mst_of_mst;
		std::uint32_t targetPE_of_mst;
		bool passive_of_mst;
	};
	

	
	public:
	
	grid_irregular_pointer_doubling(std::vector<std::uint64_t>& s, std::vector<std::int64_t> r, std::vector<std::uint32_t> targetPEs, std::vector<std::uint64_t> prefix_sum_num_vertices_per_PE, int communication_mode)
	{
		this->s = s;
		this->r = r;
		this->targetPEs = targetPEs;
		this->prefix_sum_num_vertices_per_PE = prefix_sum_num_vertices_per_PE;
		this->communication_mode = communication_mode;
	}
	
	
	std::vector<std::int64_t> start(kamping::Communicator<>& comm, karam::mpi::GridCommunicator grid_comm)
	{
		std::vector<std::string> categories = {"local_work", "communication", "other"};
		timer timer("ruler_pakete_senden", categories, "local_work", "grid_irregular_pointer_doubling");
		
		timer.add_info(std::string("num_local_vertices"), std::to_string(num_local_vertices), true);
		
		
		
		rank = comm.rank();
		size = comm.size();
		num_local_vertices = s.size();
		num_global_vertices = prefix_sum_num_vertices_per_PE[size];
		node_offset = prefix_sum_num_vertices_per_PE[rank];
		
		std::vector<std::uint64_t> q = s;
		std::vector<bool> passive(num_local_vertices, false);
		
		for (std::int32_t local_index = 0; local_index < num_local_vertices; local_index++)
			if (q[local_index] == local_index + node_offset)
			{
				passive[local_index] = true;
			}
		
		
	
		
		std::vector<std::int32_t> num_packets_per_PE(size,0);
		std::vector<std::int32_t> send_displacements(size + 1,0);
		
		std::vector<node_request> requests(num_local_vertices);
		std::vector<answer> answers(num_local_vertices);

		
		std::int32_t max_iteration = std::log2(num_global_vertices) + 2;
		
		for (std::int32_t iteration = 0; iteration < max_iteration; iteration++)
		{
			
			
			

			//zuerst request packets gezählt
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
					requests[packet_index].node = local_index + node_offset;
					requests[packet_index].mst = q[local_index];
					
				}
				
			}

			std::function<answer(const node_request)> lambda = [&] (node_request request) { 
				std::uint64_t local_index = request.mst - node_offset;
				answer answer;
				answer.node = request.node;
				answer.r_of_mst = r[local_index];
				answer.mst_of_mst = q[local_index];
				answer.targetPE_of_mst = targetPEs[local_index];
				answer.passive_of_mst = passive[local_index];
				return answer;
			};
			std::vector<answer> recv_answers = request_reply<node_request,answer>(timer, requests, num_packets_per_PE, lambda, comm, grid_comm, communication_mode);
		
			
			
			for (std::int32_t i = 0; i < recv_answers.size(); i++)
			{				
				std::int32_t local_index = recv_answers[i].node - node_offset;
				
				targetPEs[local_index] = recv_answers[i].targetPE_of_mst;
				q[local_index] = recv_answers[i].mst_of_mst;
				r[local_index] = r[local_index] + recv_answers[i].r_of_mst;
				passive[local_index] = recv_answers[i].passive_of_mst;
			}
			
		}
		
		return r;
		
	}
	
	void calculate_send_displacements_and_reset_num_packets_per_PE(std::vector<std::int32_t>& send_displacements, std::vector<std::int32_t>& num_packets_per_PE)
	{
		send_displacements[0]=0;
		for (std::int32_t i = 1; i < size + 1; i++)
			send_displacements[i] = send_displacements[i-1] + num_packets_per_PE[i-1];
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
	}
	
	
	private:
	int communication_mode;
	
	std::uint64_t node_offset;
	std::uint64_t num_local_vertices;
	std::uint64_t num_global_vertices;
	std::uint64_t rank, size;
	std::vector<std::uint64_t> s;
	std::vector<std::int64_t> r;
	std::vector<std::uint32_t> targetPEs;
	std::vector<std::uint64_t> prefix_sum_num_vertices_per_PE;
};