class shuffle_load_balance 
{
	
	
	public:
	
	shuffle_load_balance()
	{

	}
	

	std::vector<std::int64_t>  start(std::vector<std::uint64_t>& s, kamping::Communicator<>& comm, karam::mpi::GridCommunicator grid_comm, std::uint32_t comm_rounds)
	{
		size = comm.size();
		rank = comm.rank();
		num_local_vertices = s.size();
		node_offset = rank * num_local_vertices;
		
		
		std::vector<std::uint64_t> shuffle = generator::generate_regular_successor_vector(num_local_vertices, comm);
		//da shuffle liste und kein kreis ist, muss ende und anfang gefunden werden
		
		struct edge{
			std::uint64_t source;
			std::uint64_t destination;
		};
		std::vector<std::int32_t> num_packets_per_PE(size,0);
		std::vector<std::int32_t> send_displacements(size + 1,0);
		std::int64_t final_node = -1;
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			if (i + node_offset == shuffle[i])
			{
				final_node = i + node_offset;
				continue;
			}
			
			std::int32_t targetPE = shuffle[i] / num_local_vertices;
			num_packets_per_PE[targetPE]++;		
		}
		calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE, comm);
		std::vector<edge> edges(send_displacements[size]);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			if (i + node_offset == shuffle[i])
				continue;
			
			std::int32_t targetPE = shuffle[i] / num_local_vertices;
			std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			
			edges[packet_index].source = i + node_offset;
			edges[packet_index].destination = shuffle[i];
		}
		
		auto recv = comm.alltoallv(kamping::send_buf(edges), kamping::send_counts(num_packets_per_PE)).extract_recv_buffer();
		std::vector<std::uint64_t> shuffle_rev(num_local_vertices);
		std::vector<bool> reached(num_local_vertices, false);
		for (std::uint64_t i = 0; i <  recv.size(); i++)
		{
			reached[recv[i].destination - node_offset] = true;
			shuffle_rev[recv[i].destination - node_offset] = recv[i].source;
		}
		
		std::int64_t start_node = -1;
		for (int i = 0; i < recv.size(); i++)
			if (!reached[i])
				start_node = i + node_offset;
		std::vector<std::uint64_t> send_start_node(0);
		if (start_node != -1)
		{
			send_start_node.resize(1);
			send_start_node[0] = start_node;
		}
		std::vector<std::uint64_t> send_final_node(0);
		if (final_node != -1)
		{
			send_final_node.resize(1);
			send_final_node[0] = final_node;
		}
		std::vector<std::uint64_t> recv_start_node(1);
		std::vector<std::uint64_t> recv_final_node(1);
		comm.allgatherv(kamping::send_buf(send_start_node), kamping::recv_buf<kamping::resize_to_fit>(recv_start_node));
		comm.allgatherv(kamping::send_buf(send_final_node), kamping::recv_buf<kamping::resize_to_fit>(recv_final_node));

		if (start_node != -1)
			shuffle_rev[start_node - node_offset] = recv_final_node[0];
		if (final_node != -1)
			shuffle[final_node - node_offset] = recv_start_node[0];

		//now we can transform our instance to shuffled instance
		//node (i,s[i]) now will be node (shuffle[i],shuffle[s[i]])
		//so we need to find out what shuffle[s[i]] is		
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
		
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::uint32_t targetPE = s[i] / num_local_vertices;
			num_packets_per_PE[targetPE]++;
		}
		calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE, comm);
		std::vector<std::uint64_t> request(send_displacements[size]);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::uint32_t targetPE = s[i] / num_local_vertices;
			std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			
			request[packet_index] = s[i];
		}
		
		
		auto recv_request = comm.alltoallv(kamping::send_buf(request), kamping::send_counts(num_packets_per_PE));
		num_packets_per_PE = recv_request.extract_recv_counts();
		std::vector<std::uint64_t> recv_request_buffer = recv_request.extract_recv_buffer(); 
		for (std::uint64_t i = 0; i< recv_request_buffer.size(); i++)
			recv_request_buffer[i] = shuffle[recv_request_buffer[i] - node_offset];
		
		std::vector<std::uint64_t> recv_answer = comm.alltoallv(kamping::send_buf(recv_request_buffer), kamping::send_counts(num_packets_per_PE)).extract_recv_buffer();
		
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
		edges.resize(num_local_vertices);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::uint32_t targetPE = s[i] / num_local_vertices;
			std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			//recv_answer[packet_index] müsste shuffle[s[i]] sein
			edges[i] = {shuffle[i], recv_answer[packet_index] };
		}
		
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::uint32_t targetPE = shuffle[i] / num_local_vertices;
			num_packets_per_PE[targetPE]++;
		}
		calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE, comm);
		std::vector<edge> final_edges(num_local_vertices);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::uint32_t targetPE = shuffle[i] / num_local_vertices;
			std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			
			final_edges[packet_index] = edges[i];
		}

		edges = comm.alltoallv(kamping::send_buf(final_edges), kamping::send_counts(num_packets_per_PE)).extract_recv_buffer();
		std::vector<std::uint64_t> s_shuffle(num_local_vertices,-1);
		for (std::uint64_t i = 0; i < edges.size(); i++)
			s_shuffle[edges[i].source - node_offset] = edges[i].destination;
		
		//regular_pointer_doubling algorithm(s_shuffle, comm);
		//std::vector<std::int64_t> ranks = algorithm.start(comm, grid_comm);
		//std::vector<std::int64_t> ranks =forest_regular_ruling_set2(s_shuffle, comm_rounds, comm,1).result_dist;
		std::vector<std::int64_t> ranks = tree_euler_tour(comm, s_shuffle, comm_rounds).start(comm, s_shuffle);
		struct result {
			std::uint64_t node;
			std::int64_t rank;
		};
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
					
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::uint32_t targetPE = shuffle_rev[i] / num_local_vertices;
			num_packets_per_PE[targetPE]++;
		}
		calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE, comm);
		std::vector<result> results(send_displacements[size]);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::uint32_t targetPE = shuffle_rev[i] / num_local_vertices;
			std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			results[packet_index] = {shuffle_rev[i] , ranks[i]};
		}		
		std::vector<result> recv_results = comm.alltoallv(kamping::send_buf(results), kamping::send_counts(num_packets_per_PE)).extract_recv_buffer();
		std::vector<std::int64_t> final_results(num_local_vertices);
		for (std::uint64_t i = 0; i < recv_results.size(); i++)
		{
			final_results[recv_results[i].node - node_offset] = recv_results[i].rank;
		}
		
		return final_results;
		/*
		std::cout << rank << " with s arr: ";
		for (int i = 0; i < num_local_vertices; i++)
			std::cout << s[i] << " ";
		std::cout << "\n und with s_shuffle arr: ";
		for (int i = 0; i < num_local_vertices; i++)
			std::cout << s_shuffle[i] << " ";
		std::cout << "\n und with final_results arr: ";
		for (int i = 0; i < num_local_vertices; i++)
			std::cout << final_results[i] << " ";
		std::cout << std::endl;*/
	}
	
	private:
	
	std::uint64_t node_offset;
	std::uint64_t num_local_vertices;
	std::int32_t rank, size;
};