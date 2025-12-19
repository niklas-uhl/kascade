class forest_load_balance_regular_ruling_set2 
{
	
	
	public:
	
	forest_load_balance_regular_ruling_set2(std::uint64_t comm_rounds)
	{
		this-> comm_rounds = comm_rounds;
	}
	
	//start klappt einigermaßen
	//diese Methode ist nachprogrammierung von start mit mehr effizienz und mehr objektorientierung
	void start2(std::vector<std::uint64_t>& s, kamping::Communicator<>& comm)
	{
		
		std::vector<std::string> categories = {"calc, other"};
		timer timer("indegrees", categories, "calc", "all_indegrees_test");
		
		size = comm.size();
		rank = comm.rank();
		num_local_vertices = s.size();
		node_offset = rank * num_local_vertices;
		
		std::vector<std::int32_t> num_packets_per_PE(size,0);
		std::vector<std::int32_t> send_displacements(size + 1,0);
		
		std::unordered_map<std::uint64_t, std::int64_t> local_node_indegrees;

		std::vector<std::uint64_t> num_edges_per_PE(size,0);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{	
			if (local_node_indegrees.contains(s[i]))
				local_node_indegrees[s[i]] = local_node_indegrees[s[i]] + 1;
			else
				local_node_indegrees[s[i]] = 1;
		}
		
		
		for (const auto& [key, value] : local_node_indegrees)
		{
			std::int32_t targetPE = key / num_local_vertices;
			num_packets_per_PE[targetPE]++;
		}
		
		calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
		struct packet {
			std::uint64_t node;
			std::uint64_t indegree;
		};
		std::vector<packet> send_packets(send_displacements[size]);
		for (const auto& [key, value] : local_node_indegrees)
		{
			std::int32_t targetPE = key / num_local_vertices;
			std::int64_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			send_packets[packet_index].node = key;
			send_packets[packet_index].indegree = value;
		}
		auto recv = comm.alltoallv(kamping::send_buf(send_packets), kamping::send_counts(num_packets_per_PE));
		std::vector<packet> recv_packets = recv.extract_recv_buffer();
		std::vector<std::uint64_t> indegrees(num_local_vertices,0);

		for (int i = 0; i < recv_packets.size(); i++)
		{
			indegrees[recv_packets[i].node - node_offset] += recv_packets[i].indegree;
		}
		
		timer.finalize(comm, "all_indegrees_test");
	}

	
	//das hier ist erste unoptimierte implementierung
	void start(std::vector<std::uint64_t>& s, kamping::Communicator<>& comm)
	{
		size = comm.size();
		rank = comm.rank();
		num_local_vertices = s.size();
		node_offset = rank * num_local_vertices;
		
		/*
		std::cout << rank << " with s array:";
		for (int i = 0; i < s.size(); i++)
			std::cout << s[i] << " ";
		std::cout << std::endl;*/
		
		
		//als allererstes kanten umdrehen, aber so load balancen, sodass jeder PE gleich viele kanten hat
		//dabei können auch knoten auf verschiedene PEs aufgeteilt werden
	
		//targetPE wird struct mit targetPE_lower_bound <= targetPE < targetPE_upper_bound
		
		
		
		std::vector<std::int32_t> num_packets_per_PE(size,0);
		std::vector<std::int32_t> send_displacements(size + 1,0);
		
		std::unordered_map<std::uint64_t, std::int64_t> local_node_indegrees;

		std::vector<std::uint64_t> num_edges_per_PE(size,0);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{	
			if (local_node_indegrees.contains(s[i]))
				local_node_indegrees[s[i]] = local_node_indegrees[s[i]] + 1;
			else
				local_node_indegrees[s[i]] = 1;
		}
		
		
		for (const auto& [key, value] : local_node_indegrees)
		{
			std::int32_t targetPE = key / num_local_vertices;
			num_packets_per_PE[targetPE]++;
		}
		
		calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
		struct packet {
			std::uint64_t node;
			std::uint64_t indegree;
		};
		std::vector<packet> send_packets(send_displacements[size]);
		for (const auto& [key, value] : local_node_indegrees)
		{
			std::int32_t targetPE = key / num_local_vertices;
			std::int64_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			send_packets[packet_index].node = key;
			send_packets[packet_index].indegree = value;
		}
		
		auto recv = comm.alltoallv(kamping::send_buf(send_packets), kamping::send_counts(num_packets_per_PE));
		std::vector<packet> recv_packets = recv.extract_recv_buffer();
		std::vector<std::int32_t> recv_counts = recv.extract_recv_counts();
		std::vector<std::int32_t> recv_displs = recv.extract_recv_displs();
			
		std::vector<std::uint64_t> indegrees(num_local_vertices,0);
		std::uint64_t weight = num_local_vertices;
		for (int i = 0; i < recv_packets.size(); i++)
		{
			weight += recv_packets[i].indegree;
			indegrees[recv_packets[i].node - node_offset] += recv_packets[i].indegree;
		}
		
		std::vector<std::uint64_t> all_weights;
		comm.allgather(kamping::send_buf(weight), kamping::recv_buf<kamping::resize_to_fit>(all_weights));


		


		std::vector<std::uint64_t> prefix_sum_all_weights(size+1,0);
		for (std::uint32_t i = 1; i < size + 1; i++)
		{
			prefix_sum_all_weights[i] = prefix_sum_all_weights[i-1] + all_weights[i-1];
		}
		
	

	
		struct nodes_to_PE_assignment {
			std::uint32_t targetPE;
			std::int64_t start_part_node;
			std::uint64_t start_part_node_start_degree;
			std::uint64_t node_start_index;
			std::uint64_t node_end_index;
			std::int64_t end_part_node;
			std::uint64_t end_part_node_end_degree;
		};
		
	
		
		std::vector<nodes_to_PE_assignment> nodes_to_PE_assignments(1);
		nodes_to_PE_assignments[0].start_part_node = -1;
		nodes_to_PE_assignments[0].node_start_index = node_offset;
		
		struct cut_node {
			std::uint64_t local_index;
			std::uint64_t cut_degree;
		};
		std::vector<cut_node> cut_nodes(0);
		std::uint64_t dynamic_start_weight = prefix_sum_all_weights[rank];
		std::int64_t next_weight_cut = ((dynamic_start_weight + 2*num_local_vertices) / (2*num_local_vertices)) * (2*num_local_vertices) - 2*num_local_vertices;
		
		std::uint64_t dynamic_start_node = 0;
		std::uint64_t dynamic_start_degree = 0;
		
		
		
		while (dynamic_start_weight < prefix_sum_all_weights[rank+1])
		{			
			next_weight_cut+= 2*num_local_vertices;
						
			nodes_to_PE_assignment& nodes_to_PE_assignment = nodes_to_PE_assignments.back();
			nodes_to_PE_assignment.targetPE = (next_weight_cut - 1) / (2*num_local_vertices);
			
			if (next_weight_cut >= prefix_sum_all_weights[rank+1])
			{
				nodes_to_PE_assignment.node_end_index = node_offset + num_local_vertices;
				nodes_to_PE_assignment.end_part_node = -1;
				
				//iff there is no weight cut
				
				break;
			}
			else
			{
				std::uint64_t start_node = nodes_to_PE_assignments.back().node_start_index - node_offset;
				while (dynamic_start_weight < next_weight_cut)
					dynamic_start_weight += indegrees[dynamic_start_node++] +1;
				
				if (dynamic_start_weight == next_weight_cut)
				{
					nodes_to_PE_assignment.node_end_index = dynamic_start_node + node_offset;
					nodes_to_PE_assignment.end_part_node = -1;
					nodes_to_PE_assignments.resize(nodes_to_PE_assignments.size() + 1);
					nodes_to_PE_assignments.back().node_start_index = dynamic_start_node + node_offset;
					nodes_to_PE_assignments.back().start_part_node = -1;
					
					
					//iff there is a clean weight cut between following nodes
					

				}
				else
				{
					std::uint64_t cut_node = dynamic_start_node - 1 + node_offset;
					std::uint64_t cut_degree = next_weight_cut - (dynamic_start_weight - indegrees[dynamic_start_node-1]);
					
					
					if (cut_node == nodes_to_PE_assignments.back().start_part_node)
						nodes_to_PE_assignments.back().node_start_index = cut_node;
					
					nodes_to_PE_assignment.node_end_index = cut_node;
					nodes_to_PE_assignment.end_part_node = dynamic_start_node - 1 + node_offset;
					nodes_to_PE_assignment.end_part_node_end_degree = cut_degree;
					nodes_to_PE_assignments.resize(nodes_to_PE_assignments.size() + 1);
					nodes_to_PE_assignments.back().start_part_node = cut_node;
					nodes_to_PE_assignments.back().start_part_node_start_degree = cut_degree;
					nodes_to_PE_assignments.back().node_start_index = dynamic_start_node + node_offset;
					
					//iff there is is a cut between edges of the same node

					cut_nodes.push_back({cut_node - node_offset, cut_degree});
					
					dynamic_start_node--;
					dynamic_start_weight -= indegrees[dynamic_start_node] +1;
				}
				
			}
			
			
			
		}
		
		
	
		/*
		for (int i = 0; i < nodes_to_PE_assignments.size(); i++)
		{
			nodes_to_PE_assignment info = nodes_to_PE_assignments[i];
			std::cout << rank << " hat info für PE " << info.targetPE << ": (" << info.start_part_node << "," << info.start_part_node_start_degree << "),(" <<info.node_start_index << "," << info.node_end_index << "),(" << info.end_part_node << "," << info.end_part_node_end_degree << ")" << std::endl;
		}*/
		
		
		
		struct cut_information{
			std::uint64_t cut_node;
			std::uint64_t global_cut_degree;
			std::uint32_t cut_PE;
			std::uint64_t local_cut_degree;
		};
		
		
		
		std::vector<cut_information> cut_informations(cut_nodes.size());
		
		
		
		for (int k = 0; k < cut_nodes.size(); k++)
		{
			cut_node node = cut_nodes[k];
			
			if (node.cut_degree >= 0)
			{
				
				std::uint64_t local_index = node.local_index;
				std::uint64_t current_degree = 0;
				for (std::uint32_t p = 0; p < size; p++)
				{
					std::uint64_t current_degree_from_PE_with_lower_index = current_degree;
					for (std::uint64_t i = 0; i < recv_counts[p]; i++)
					{
						if (recv_packets[recv_displs[p]+i].node == local_index + node_offset)
						{
							current_degree += recv_packets[recv_displs[p]+i].indegree;
							if (current_degree >= node.cut_degree)
							{
								std::cout << "cut node (" << node.local_index + node_offset << "," << node.cut_degree << ") has to be cut at PE " << p << " at " << current_degree - current_degree_from_PE_with_lower_index << "th index" <<  std::endl;
								
								cut_informations[k].global_cut_degree = node.cut_degree;
								cut_informations[k].cut_node = node.local_index + node_offset;
								cut_informations[k].cut_PE = p;
								cut_informations[k].local_cut_degree = current_degree - current_degree_from_PE_with_lower_index;
								
								i = recv_counts[p];
								p=size;
							}
						}
					}
				}
			}
		}
		
		std::vector<cut_information> recv_cut_information;
		comm.allgatherv(kamping::send_buf(cut_informations), kamping::recv_buf<kamping::resize_to_fit>(recv_cut_information));
		
		std::vector<nodes_to_PE_assignment> recv_nodes_to_PE_assignments;
		comm.allgatherv(kamping::send_buf(nodes_to_PE_assignments), kamping::recv_buf<kamping::resize_to_fit>(recv_nodes_to_PE_assignments));
		
		
		//compress nodes_to_PE_assignment so that every PE has only one assignemnt
		std::vector<nodes_to_PE_assignment> compressed_nodes_to_PE_assignments(size);
		std::int64_t i = -1;
		for (std::uint32_t p = 0; p < size; p++)
		{
			i++;
			compressed_nodes_to_PE_assignments[p].targetPE = p;
			
			compressed_nodes_to_PE_assignments[p].start_part_node = recv_nodes_to_PE_assignments[i].start_part_node;
			compressed_nodes_to_PE_assignments[p].start_part_node_start_degree = recv_nodes_to_PE_assignments[i].start_part_node_start_degree;
			compressed_nodes_to_PE_assignments[p].node_start_index = recv_nodes_to_PE_assignments[i].node_start_index;
			
			while (recv_nodes_to_PE_assignments[i].targetPE == p) i++;
			i--;
			
			compressed_nodes_to_PE_assignments[p].end_part_node = recv_nodes_to_PE_assignments[i].end_part_node;
			compressed_nodes_to_PE_assignments[p].end_part_node_end_degree = recv_nodes_to_PE_assignments[i].end_part_node_end_degree;
			compressed_nodes_to_PE_assignments[p].node_end_index = recv_nodes_to_PE_assignments[i].node_end_index;
		}
		
		
		
		
		//now nodes_to_PE_assignment must be translated to with local cut information
		//for instance iff rank = 0, nodes_to_PE_assignment = [(-1,0),(0,2),(2,7)] but cut_information for cut_node 2 has cut_PE 1 ==> nodes_to_PE_assignment = [(-1,0),(0,3),(-1,0)]
		
		
		struct cut_local_info {
			std::uint32_t cut_PE;
			std::uint64_t local_cut_degree;
		};
		
		struct
		{
			std::string operator()(int a, int b) const { return "(" + std::to_string(a) + "," + std::to_string(b) + ")"; }
		} cut_to_string;
		std::unordered_map<std::uint64_t,std::uint64_t> cut_node_local_send_degree; //das hier sagt für cut_node (4,2) wenn cut_node_local_send_degree[4]=1, dass 1 kante zu cut_node 4 verschickt wurde				
		std::unordered_map<std::string,cut_local_info> local_cut_info_map; 
		//wenn wir cut_node (global_index, cut_degree)=(4,2) haben, dann greifen wir auf die lokale info zu mit: local_cut_info_map["(4,2)"]

		
		for (std::uint64_t i = 0; i < recv_cut_information.size(); i++)
		{
			cut_information info = recv_cut_information[i];
			cut_node_local_send_degree[info.cut_node] = 1;
			local_cut_info_map[cut_to_string(info.cut_node, info.global_cut_degree)] = {info.cut_PE, info.local_cut_degree};		
		}
		std::vector<nodes_to_PE_assignment> compressed_nodes_to_PE_assignments_local_info(size);
		for (std::uint64_t i = 0; i < compressed_nodes_to_PE_assignments.size(); i++)
		{
			nodes_to_PE_assignment assign = compressed_nodes_to_PE_assignments[i];
			
			if (assign.start_part_node != -1)
			{
				std::string cut_string = cut_to_string(assign.start_part_node, assign.start_part_node_start_degree);
				if (local_cut_info_map[cut_string].cut_PE == rank)
				{
					assign.start_part_node_start_degree = local_cut_info_map[cut_string].local_cut_degree;
				}
				else if (local_cut_info_map[cut_string].cut_PE > rank)
				{
					assign.start_part_node = -1;
					assign.start_part_node_start_degree = 0; //für lesbarkeit
				}
				else
				{
					assign.node_start_index--;
					assign.start_part_node = -1;
					assign.start_part_node_start_degree = 0; //für lesbarkeit

				}
			}
			
			if (assign.end_part_node != -1)
			{
				std::string cut_string = cut_to_string(assign.end_part_node, assign.end_part_node_end_degree);
				if (local_cut_info_map[cut_string].cut_PE == rank)
				{
					assign.end_part_node_end_degree = local_cut_info_map[cut_string].local_cut_degree;
				}
				else if (local_cut_info_map[cut_string].cut_PE < rank)
				{
					assign.end_part_node = -1;
					assign.end_part_node_end_degree = 0; //für lesbarkeit

				}
				else
				{
					assign.node_end_index++;
					assign.end_part_node = -1;
					assign.end_part_node_end_degree = 0; //für lesbarkeit

				}
			}
			
			
			compressed_nodes_to_PE_assignments_local_info[i] = assign;
			
			
		}
		/*
		if (rank == 1)
		{
			std::cout << "compressed nodes_to_PE_assignments\n";
			for (int i = 0; i < compressed_nodes_to_PE_assignments.size(); i++)
			{
				nodes_to_PE_assignment info = compressed_nodes_to_PE_assignments[i];
				std::cout << "info für PE " << info.targetPE << ": (" << info.start_part_node << "," << info.start_part_node_start_degree << "),(" <<info.node_start_index << "," << info.node_end_index << "),(" << info.end_part_node << "," << info.end_part_node_end_degree << ")" << std::endl;
			}
			std::cout << "compressed local nodes_to_PE_assignments\n";
			for (int i = 0; i < compressed_nodes_to_PE_assignments_local_info.size(); i++)
			{
				nodes_to_PE_assignment info = compressed_nodes_to_PE_assignments_local_info[i];
				std::cout << "info für PE " << info.targetPE << ": (" << info.start_part_node << "," << info.start_part_node_start_degree << "),(" <<info.node_start_index << "," << info.node_end_index << "),(" << info.end_part_node << "," << info.end_part_node_end_degree << ")" << std::endl;
			}
		}*/
		
		
		struct node {
			std::uint64_t source;
			std::uint64_t destination;
			std::uint32_t targetPE;
		};
		std::vector<node> send_nodes;
		
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::uint64_t dynamic_lower_bound = 0; //inclusive
			std::uint64_t dynamic_upper_bound = size; //exclusive
			while (dynamic_upper_bound - dynamic_lower_bound > 1)
			{
				std::uint64_t middle = (dynamic_lower_bound + dynamic_upper_bound) / 2;
				nodes_to_PE_assignment& assign = compressed_nodes_to_PE_assignments_local_info[middle];

				
				if (assign.start_part_node == s[i] && assign.end_part_node == s[i])
				{
					if (assign.start_part_node_start_degree <= cut_node_local_send_degree[s[i]] && assign.end_part_node_end_degree < cut_node_local_send_degree[s[i]])
					{
						dynamic_lower_bound = middle;
						//cut_node_local_send_degree[s[i]]++;
						break;
					}
					else if (assign.start_part_node_start_degree <= cut_node_local_send_degree[s[i]])
					{
						dynamic_lower_bound = middle +1;
					}
					else if (assign.end_part_node_end_degree < cut_node_local_send_degree[s[i]])
					{
						dynamic_upper_bound = middle;
					}
					else
					{
						if (rank == 0) std::cout << "ERROR" << std::endl;
					}
				}
				else if (assign.start_part_node == s[i])
				{
					if (assign.start_part_node_start_degree <= cut_node_local_send_degree[s[i]])
					{
						dynamic_lower_bound = middle;
						//cut_node_local_send_degree[s[i]]++;
						break;
					}
					else
					{
						dynamic_upper_bound = middle;
					}
				}
				else if (assign.end_part_node == s[i])
				{
					if (assign.end_part_node_end_degree > cut_node_local_send_degree[s[i]])
					{
						dynamic_lower_bound = middle;
						//cut_node_local_send_degree[s[i]]++;
						break;
					}
					else
					{
						dynamic_lower_bound = middle + 1;
					}
				}
				else if (assign.node_start_index <= s[i] && s[i] < assign.node_end_index)
				{
					dynamic_lower_bound = middle;
					break;
				}
				else if (s[i] < assign.node_start_index)
				{
					dynamic_upper_bound = middle;
				}
				else 
				{
					dynamic_lower_bound = middle + 1;
				}
			}
			nodes_to_PE_assignment info = compressed_nodes_to_PE_assignments_local_info[dynamic_lower_bound];
			if (s[i] == info.start_part_node || s[i] == info.end_part_node)
				cut_node_local_send_degree[s[i]]++;
			
			std::uint32_t targetPE = info.targetPE;
			send_nodes.push_back({i + node_offset, s[i], targetPE});
			
			//if (rank == 1) std::cout << "s[i]=" << s[i] << " ab zu PE " << targetPE << std::endl;
		}
		
		
		
		struct final_node {
			std::uint64_t source;
			std::uint64_t destination;
		};
		std::vector<final_node> send_final_nodes(send_nodes.size());
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
		for (std::uint64_t i = 0; i < send_nodes.size(); i++)
		{
			std::int32_t targetPE = send_nodes[i].targetPE;
			num_packets_per_PE[targetPE]++;
		}
		calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
		
		for (std::uint64_t i = 0; i < send_nodes.size(); i++)
		{
			std::int32_t targetPE = send_nodes[i].targetPE;
			std::int64_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			send_final_nodes[packet_index] = {send_nodes[i].source, send_nodes[i].destination};
		}
		
		auto recv_final_nodes = comm.alltoallv(kamping::send_buf(send_final_nodes), kamping::send_counts(num_packets_per_PE)).extract_recv_buffer();
		
		if (rank == 0) {
			std::cout << "compressed nodes_to_PE_assignments\n";
			for (int i = 0; i < compressed_nodes_to_PE_assignments.size(); i++)
			{
				nodes_to_PE_assignment info = compressed_nodes_to_PE_assignments[i];
				std::cout << "info für PE " << info.targetPE << ": (" << info.start_part_node << "," << info.start_part_node_start_degree << "),(" <<info.node_start_index << "," << info.node_end_index << "),(" << info.end_part_node << "," << info.end_part_node_end_degree << ")" << std::endl;
			}
		}
		
		nodes_to_PE_assignment info = compressed_nodes_to_PE_assignments[rank];
		std::uint64_t lb_num_local_vertices = info.node_end_index - info.node_start_index;
		std::uint64_t lb_lowest_node_index = info.node_start_index;
		if (info.start_part_node != -1)
		{
			lb_lowest_node_index = info.start_part_node;
			lb_num_local_vertices++;
		}
		if (info.end_part_node != -1 && info.end_part_node != info.start_part_node)
			lb_num_local_vertices++;
				
		std::vector<std::uint64_t> num_edges_per_node(lb_num_local_vertices);
		
		std::vector<std::int64_t> real_nodes(lb_num_local_vertices); //real_nodes[i] == -1 iff i is pseudo node
		if (info.start_part_node != -1)
		{
			real_nodes[0] = -1;
			for (std::uint64_t i = 0; i < info.node_end_index - info.node_start_index; i++)
				real_nodes[i+1] = i + info.node_start_index;
			
			if (info.end_part_node != -1 && info.end_part_node != info.start_part_node)
				real_nodes[lb_num_local_vertices-1] = -1;
		}

		if (info.start_part_node == -1)
		{
			for (std::uint64_t i = 0; i < info.node_end_index - info.node_start_index; i++)
				real_nodes[i] = i + info.node_start_index;
			
			if (info.end_part_node != -1)
				real_nodes[lb_num_local_vertices-1] = info.end_part_node;
		}
		
		std::cout << rank << " with real nodes:\n";
		for (int i = 0; i < real_nodes.size(); i++)
			std::cout << real_nodes[i] << " ";
		std::cout << std::endl;
		
		//std::cout << rank << " with lb_lowest_node_index "<< lb_lowest_node_index << std::endl; 
		
		for (std::uint64_t i = 0; i < recv_final_nodes.size(); i++)
		{
			num_edges_per_node[recv_final_nodes[i].destination - lb_lowest_node_index]++;
		}
		
		
		
		std::vector<std::uint64_t> lb_num_local_vertices_per_PE;
		comm.allgather(kamping::send_buf(lb_num_local_vertices), kamping::recv_buf<kamping::resize_to_fit>(lb_num_local_vertices_per_PE));
		std::vector<std::uint64_t> lb_prefix_sum_num_local_vertices_per_PE(size+1,0);
		for (std::uint32_t i = 1; i <size+1 ; i++)
			lb_prefix_sum_num_local_vertices_per_PE[i] = lb_prefix_sum_num_local_vertices_per_PE[i-1] + lb_num_local_vertices_per_PE[i-1];
		
		if (info.end_part_node != -1 && info.end_part_node != info.start_part_node)
		{
			//node which was splittet starts on this rank and has to add pseudo edges
			std::uint64_t upper_rank = rank +1;
			while (compressed_nodes_to_PE_assignments[upper_rank].end_part_node == info.end_part_node)
				upper_rank++;
			
			num_edges_per_node[lb_num_local_vertices-1] += upper_rank - rank;
		}
		
		
		
		std::vector<std::uint64_t> bounds(lb_num_local_vertices+1,0);
		for (std::uint64_t i = 1; i < lb_num_local_vertices+1; i++)
			bounds[i] = bounds[i-1] + num_edges_per_node[i-1];
		
		std::vector<std::uint64_t> all_edges(bounds[lb_num_local_vertices]);
		
		std::fill(num_edges_per_node.begin(), num_edges_per_node.end(), 0);
		
		for (std::uint64_t i = 0; i < recv_final_nodes.size(); i++)
		{
			std::uint64_t target_node = recv_final_nodes[i].destination - lb_lowest_node_index;
			std::uint64_t packet_index = bounds[target_node] + num_edges_per_node[target_node]++;

			all_edges[packet_index] = recv_final_nodes[i].source;
		}
		
		std::cout << rank << " with all_edges :\n";
		for (int i = 0; i < all_edges.size(); i++)
			std::cout << all_edges[i] << " ";
		std::cout << std::endl; 
		
	}
	
	
	void calculate_send_displacements_and_reset_num_packets_per_PE(std::vector<std::int32_t>& send_displacements, std::vector<std::int32_t>& num_packets_per_PE)
	{
		send_displacements[0]=0;
		for (std::int32_t i = 1; i < size + 1; i++)
			send_displacements[i] = send_displacements[i-1] + num_packets_per_PE[i-1];
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
	}
	
	
	private:
	std::uint64_t comm_rounds;
	std::uint64_t node_offset;
	std::uint64_t num_local_vertices;
	std::uint64_t rank, size;
};