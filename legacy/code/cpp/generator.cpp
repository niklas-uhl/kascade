#include "helper_functions.cpp"

class generator
{
	public:
	

	static std::vector<std::uint64_t> generate_regular_star_vector(std::uint64_t num_local_vertices, kamping::Communicator<>& comm)
	{
		std::uint64_t mpi_rank = comm.rank();
		std::uint64_t mpi_size = comm.size();
		
		std::vector<std::uint64_t> s(num_local_vertices,0);

		return s;
		
		
	}
	
	static std::vector<std::uint64_t> generate_tailed_star(std::uint64_t num_local_vertices, kamping::Communicator<>& comm)
	{
		std::uint64_t rank = comm.rank();
		std::uint64_t size = comm.size();
		std::uint64_t node_offset = rank*num_local_vertices;
		
		std::vector<std::uint64_t> s(num_local_vertices,0);
		
		if (rank > size/2)
		{
			s[0] = node_offset-1;
			for (int i = 1; i < num_local_vertices; i++)
			{
				s[i] = node_offset + i-1;
			}
		}
		return s;
	}
	
	static std::vector<std::uint64_t> generate_regular_caterpillar_vector(std::uint64_t num_local_vertices, std::uint64_t high_degree, kamping::Communicator<>& comm)
	{
		std::uint64_t rank = comm.rank();
		std::uint64_t size = comm.size();
		std::uint64_t node_offset = rank*num_local_vertices;
		//eigentlich haben wir einen einfachen pfat mit s[i]= i-1 (s[0]=0), aber wir sagen, auf jedem Pe zeigt die erste hälfte der nodes auf den ersten node des selben PEs
		
		std::vector<std::uint64_t> s(num_local_vertices);
		for (int i = 0; i < num_local_vertices; i++)
			if ((i + node_offset) % (2*high_degree) <= high_degree)
				s[i] = i + node_offset-1;
			else
				s[i] = 2*high_degree *((i + node_offset) / (2*high_degree)) + high_degree;
		if (rank == 0)
			s[0] = 0;
	
		return s;
	}
	
	static std::vector<std::uint64_t> generate_regular_caterpillar_vector(std::uint64_t num_local_vertices, kamping::Communicator<>& comm)
	{
		std::uint64_t rank = comm.rank();
		std::uint64_t size = comm.size();
		std::uint64_t node_offset = rank*num_local_vertices;
		//eigentlich haben wir einen einfachen pfat mit s[i]= i-1 (s[0]=0), aber wir sagen, auf jedem Pe zeigt die erste hälfte der nodes auf den ersten node des selben PEs
		
		std::vector<std::uint64_t> s(num_local_vertices);
		if (rank == 0)
		{
			for (int i = 0; i < num_local_vertices; i++)
			{
				if (i < num_local_vertices /2)
					s[i] = 1;
				else
					s[i] = i-1;
			}
			s[1] = 0;
			s[0] = 0;
		}
		else
		{
			s[0] = node_offset - 1;
			for (int i = 1; i < num_local_vertices; i++)
			{
				if (i < num_local_vertices /2)
					s[i] = node_offset;
				else
					s[i] = node_offset + i-1;
			}
		}
		return s;
		
		
		
	}
	
	//we have num_local_vertices stars where on PE 0 every node is root of one star
	static std::vector<std::uint64_t> generate_regular_stars_vector(std::uint64_t num_local_vertices, kamping::Communicator<>& comm)
	{
		std::uint64_t mpi_rank = comm.rank();
		std::uint64_t mpi_size = comm.size();
		
		std::vector<std::uint64_t> s(num_local_vertices,0);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
			s[i] = i;

		return s;
		
		
	}

	static std::vector<std::uint64_t> shuffle_instance(std::vector<std::uint64_t> s, kamping::Communicator<>& comm)
	{
		std::uint32_t size = comm.size();
		std::uint32_t rank = comm.rank();
		std::uint64_t num_local_vertices = s.size();
		std::uint64_t node_offset = rank * num_local_vertices;
		
		
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
		return s_shuffle;
	}


	static std::vector<std::uint64_t> generate_regular_tree_vector(std::uint64_t num_local_vertices, kamping::Communicator<>& comm)
	{
		std::uint64_t mpi_rank = comm.rank();
		std::uint64_t mpi_size = comm.size();
		
		std::vector<std::uint64_t> s(num_local_vertices);
		std::uint64_t node_offset = num_local_vertices * comm.rank();
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			if (i == 0 && comm.rank() == 0)
				s[i] = 0;
			else
				s[i] = hash64(i + node_offset) % (i + node_offset);
		}
		return s;
		
		
	}

	static std::vector<std::uint64_t> generate_regular_wood_vector(std::uint64_t num_local_vertices, kamping::Communicator<>& comm)
	{
		std::uint64_t mpi_rank = comm.rank();
		std::uint64_t mpi_size = comm.size();
		
		
		std::int32_t max_edge_weight = 0; 
		std::uint64_t node_offset = mpi_rank * num_local_vertices;
		kagen::KaGen gen(MPI_COMM_WORLD);
		
		double prob = 20 / ((double) num_local_vertices * mpi_size);
		prob = prob > 1 ? 1 : prob;
		auto graph = gen.GenerateUndirectedGNP(num_local_vertices * mpi_size, prob, false);
		
		/*
		std::uint64_t m = 20 * num_local_vertices * mpi_size;
		m = std::min(num_local_vertices,m);
		auto graph = gen.GenerateUndirectedGNM(num_local_vertices * mpi_size, m, false);
*/
		//now every edge finds edge with lowest edge weight
		std::vector<std::uint64_t> s(num_local_vertices,-1); //s[i] will be the node j that minimizes c(i,j) with
		//std::iota(s.begin(), s.end(), node_offset);
		std::vector<std::int32_t> w(num_local_vertices, max_edge_weight + 1);
		
		
		
		for (auto const& [src, dst]: graph.edges)
		{
			
			std::string edge_string = src < dst ? std::to_string(src) + "," + std::to_string(dst): std::to_string(dst) + "," + std::to_string(src);
			std::uint64_t edge_weight = std::hash<std::string>{}(edge_string)  % (max_edge_weight + 1);
			
			edge_weight = hash64(hash64(src) + hash64(dst)) % (max_edge_weight +1);
			
			if (edge_weight < w[src - node_offset] && dst < s[src - node_offset])
			{
				s[src - node_offset] = dst;
				w[src - node_offset] = edge_weight;
			}
			
		}
		
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			if (s[i] == -1)
				s[i] = i+node_offset;
		}
		

		//for each node u on this PE we calculate the lightest edge (u,v) and send this information to v
		struct lightest_edge {
			std::uint64_t source;
			std::uint64_t destination;
		};
		
		std::vector<std::int32_t> num_packets_per_PE(mpi_size,0);
		std::vector<std::int32_t> send_displacements(mpi_size + 1,0);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::int32_t targetPE = s[i] / num_local_vertices;
			num_packets_per_PE[targetPE]++;
		}
		std::vector<lightest_edge> send(num_local_vertices);
		for (std::int32_t i = 1; i < mpi_size + 1; i++)
			send_displacements[i] = send_displacements[i-1] + num_packets_per_PE[i-1];
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::int32_t targetPE = s[i] / num_local_vertices;
			std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			send[packet_index].source = i + node_offset;
			send[packet_index].destination = s[i];
		}

		auto recv = comm.alltoallv(kamping::send_buf(send), kamping::send_counts(num_packets_per_PE)).extract_recv_buffer();
		for (std::uint64_t i = 0; i < recv.size(); i++)
		{
			lightest_edge e = recv[i];
			if (s[e.destination - node_offset] == e.source && e.destination < e.source) // der node mit kleinerer id wird root
				s[e.destination - node_offset] = e.destination;
		
			//if (s[e.destination - node_offset] == e.source && hash64(e.destination) < hash64(e.source))
				//s[e.destination - node_offset] = e.destination;
		
		}

		return s;
	}
	
	static std::vector<std::uint64_t> generate_regular_wood_vector_from_irregular_graph(std::uint64_t num_local_vertices, kamping::Communicator<>& comm)
	{
		return generate_regular_wood_vector_from_irregular_graph(num_local_vertices, 1, comm);
	}

	static std::vector<std::uint64_t> generate_regular_wood_vector_from_irregular_graph(std::uint64_t num_local_vertices, std::uint64_t max_edge_weight, kamping::Communicator<>& comm)
	{
		std::uint64_t mpi_rank = comm.rank();
		std::uint64_t mpi_size = comm.size();
 
		std::uint64_t node_offset = mpi_rank * num_local_vertices;
		kagen::KaGen gen(MPI_COMM_WORLD);

		auto graph = gen.GenerateRGG2D_NM(num_local_vertices * mpi_size, num_local_vertices * mpi_size*10);
		//graph.vertex_range.first
		std::vector<std::int32_t> num_packets_per_PE(mpi_size,0);
		std::vector<std::int32_t> send_displacements(mpi_size + 1,0);

		for (auto const& [src, dst]: graph.edges)
		{
			std::int32_t targetPE = src / num_local_vertices;
			num_packets_per_PE[targetPE]++;
		}
		std::vector<std::pair<std::uint64_t,std::uint64_t>> send_edges(graph.edges.size());
		for (std::int32_t i = 1; i < mpi_size + 1; i++)
			send_displacements[i] = send_displacements[i-1] + num_packets_per_PE[i-1];
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
		for (auto const& [src, dst]: graph.edges)
		{
			std::int32_t targetPE = src / num_local_vertices;
			std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			send_edges[packet_index] = {src, dst};
		}

		auto edges = comm.alltoallv(kamping::send_buf(send_edges), kamping::send_counts(num_packets_per_PE)).extract_recv_buffer();
		
		std::vector<std::uint64_t> s(num_local_vertices); //s[i] will be the node j that minimizes c(i,j) with
		std::iota(s.begin(), s.end(), node_offset);
		std::vector<std::int32_t> w(num_local_vertices, max_edge_weight + 1);
		

		for (auto const& [src, dst]: edges)
		{
			std::uint64_t edge_weight = hash64(hash64(src) + hash64(dst)) % (max_edge_weight +1);
			
			if (edge_weight < w[src - node_offset])
			{
				s[src - node_offset] = dst;
				w[src - node_offset] = edge_weight;
			}
			
		}

		//for each node u on this PE we calculate the lightest edge (u,v) and send this information to v
		struct lightest_edge {
			std::uint64_t source;
			std::uint64_t destination;
		};
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
		std::fill(send_displacements.begin(), send_displacements.end(), 0);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::int32_t targetPE = s[i] / num_local_vertices;
			num_packets_per_PE[targetPE]++;
		}
		std::vector<lightest_edge> send(num_local_vertices);
		for (std::int32_t i = 1; i < mpi_size + 1; i++)
			send_displacements[i] = send_displacements[i-1] + num_packets_per_PE[i-1];
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::int32_t targetPE = s[i] / num_local_vertices;
			std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			send[packet_index].source = i + node_offset;
			send[packet_index].destination = s[i];
		}
		auto recv = comm.alltoallv(kamping::send_buf(send), kamping::send_counts(num_packets_per_PE)).extract_recv_buffer();
		for (std::uint64_t i = 0; i < recv.size(); i++)
		{
			lightest_edge e = recv[i];
			if (s[e.destination - node_offset] == e.source && e.destination < e.source) // der node mit kleinerer id wird root
				s[e.destination - node_offset] = e.destination;
		}

		return s;
	}

	
	static std::vector<std::uint64_t> generate_regular_successor_vector(std::uint64_t num_local_vertices, kamping::Communicator<>& comm)
	{
		
		kagen::KaGen gen(MPI_COMM_WORLD);
		
		
		
		std::vector<std::uint64_t> s(num_local_vertices);
		std::uint64_t num_global_vertices = comm.size() * num_local_vertices;
		auto path = gen.GenerateDirectedPath(num_global_vertices, true); //true means the values are shuffled
		
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
			s[i] = i + comm.rank() * num_local_vertices;
		
		for (auto const& [src, dst]: path.edges)
			s[src - comm.rank() * num_local_vertices] = dst;
		

		return s;
	}
	
	private:
	
	static std::uint64_t hash64(std::uint64_t x) {
		x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
		x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
		x = x ^ (x >> 31);
		return x;
	}
	
	std::uint64_t rank;
	std::uint64_t size;
	
	
};