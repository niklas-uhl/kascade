class real_load_balance 
{
	
	
	public:
	
	real_load_balance(std::uint64_t comm_rounds)
	{
		this-> comm_rounds = comm_rounds;
	}
	

	void start(std::vector<std::uint64_t>& s, kamping::Communicator<>& comm, karam::mpi::GridCommunicator grid_comm)
	{
		std::vector<std::string> categories = {"calc, other"};
		timer timer("indegrees", categories, "calc", "real_load_balance");
		
		size = comm.size();
		rank = comm.rank();
		num_local_vertices = s.size();
		node_offset = rank * num_local_vertices;
		
		
		std::vector<std::uint64_t> indegrees = calculate_indegrees(s, comm, grid_comm);
		
		timer.add_checkpoint("nodes_to_PE_assignments");

		calculate_weight_cuts(s, indegrees, comm, timer);
		
	}
	
	void calculate_weight_cuts(std::vector<std::uint64_t>& s, std::vector<std::uint64_t>& indegrees, kamping::Communicator<>& comm, timer timer)
	{
		std::uint64_t weight = num_local_vertices; //because every node has weight 1, the additional 1 for every edge in counted in the loop below
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
			weight += indegrees[i];
		
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
			std::uint64_t global_index;
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
					
					if (cut_degree == 0)
						cut_degree = 1;
					
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

					cut_nodes.push_back({cut_node, cut_degree});
					
					dynamic_start_node--;
					dynamic_start_weight -= indegrees[dynamic_start_node] +1;
				}	
			}
		}
		
		timer.add_checkpoint("cut_informations");

		/*
		for (int i = 0; i < nodes_to_PE_assignments.size(); i++)
		{
			nodes_to_PE_assignment info = nodes_to_PE_assignments[i];
			std::cout << rank << " hat info für PE " << info.targetPE << ": (" << info.start_part_node << "," << info.start_part_node_start_degree << "),(" <<info.node_start_index << "," << info.node_end_index << "),(" << info.end_part_node << "," << info.end_part_node_end_degree << ")" << std::endl;
		}*/
		
		struct request_cut_degree {
			std::uint64_t cut_node;
			std::int32_t source_PE;
		};
		std::vector<request_cut_degree> request_cut_degrees;
		for (std::uint64_t i = 0; i < cut_nodes.size(); i++)
		{
			cut_node node = cut_nodes[i];
			request_cut_degrees.push_back({node.global_index, rank});
			while (i + 1 < cut_nodes.size() && cut_nodes[i+1].global_index == cut_nodes[i].global_index) i++;
		}
		
		std::vector<request_cut_degree> recv_requst_cut_degrees;
		comm.allgatherv(kamping::send_buf(request_cut_degrees), kamping::recv_buf<kamping::resize_to_fit>(recv_requst_cut_degrees));
		
		std::unordered_map<std::uint64_t,std::uint64_t> index_cut_degrees;
		std::vector<std::int32_t> num_packets_per_PE(size,0);
		std::vector<std::int32_t> send_displacements(size + 1,0);
		struct answer_cut_degree {
			std::uint64_t cut_node;
			std::int32_t requestPE;
			std::int32_t answerPE;
			std::uint64_t degree;
		};
		std::vector<answer_cut_degree> answer_cut_degrees(recv_requst_cut_degrees.size());
		for (std::uint64_t i = 0; i < recv_requst_cut_degrees.size(); i++)
		{
			std::uint32_t targetPE = recv_requst_cut_degrees[i].source_PE;
			num_packets_per_PE[targetPE]++;
		}
		calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
		for (std::uint64_t i = 0; i < recv_requst_cut_degrees.size(); i++)
		{
			std::uint32_t targetPE = recv_requst_cut_degrees[i].source_PE;
			std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			
			answer_cut_degrees[packet_index].cut_node = recv_requst_cut_degrees[i].cut_node;
			answer_cut_degrees[packet_index].requestPE = recv_requst_cut_degrees[i].source_PE;
			answer_cut_degrees[packet_index].answerPE = rank;
			answer_cut_degrees[packet_index].degree = 0;
			
			index_cut_degrees[recv_requst_cut_degrees[i].cut_node] = packet_index;
		}
		
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			if (index_cut_degrees.contains(s[i]))
				answer_cut_degrees[index_cut_degrees[s[i]]].degree++;
		}
		
		std::vector<answer_cut_degree> recv_answer_cut_degrees =  comm.alltoallv(kamping::send_buf(answer_cut_degrees), kamping::send_counts(num_packets_per_PE)).extract_recv_buffer();

		std::sort(recv_answer_cut_degrees.begin(), recv_answer_cut_degrees.end(), [](answer_cut_degree a, answer_cut_degree b) { return a.cut_node < b.cut_node;});
		
		struct cut_information{ //this tells us which PE has to 
			std::uint64_t cut_node;
			std::uint64_t global_cut_degree;
			std::uint32_t cut_PE;
		};
		std::unordered_map<std::uint64_t,std::uint64_t> cut_node_map;
		for (std::uint64_t i = 0; i < recv_answer_cut_degrees.size(); i++)
		{
			answer_cut_degree answer = recv_answer_cut_degrees[i];
			std::uint64_t start_index = i;
			cut_node_map[answer.cut_node] = start_index;
			while (i + 1 < recv_answer_cut_degrees.size() && recv_answer_cut_degrees[i+1].cut_node == recv_answer_cut_degrees[i].cut_node) i++;
			std::uint64_t end_index = i+1;
			std::sort(&recv_answer_cut_degrees[start_index], &recv_answer_cut_degrees[end_index], [](answer_cut_degree a, answer_cut_degree b) { return a.answerPE < b.answerPE;});
		}
		/*
		for (int i = 0; i < recv_answer_cut_degrees.size(); i++)
			std::cout << rank << " hat answer erhalten von node " << recv_answer_cut_degrees[i].cut_node << " und PE " << recv_answer_cut_degrees[i].answerPE << " hat " << recv_answer_cut_degrees[i].degree << " edges" << std::endl;
		*/
		std::vector<cut_information> cut_informations(cut_nodes.size());
		for (int k = 0; k < cut_nodes.size(); k++)
		{
			cut_node node = cut_nodes[k];
			
			std::uint64_t start_index = cut_node_map[node.global_index];
			std::uint64_t dynamic_degree = 0;
			std::uint64_t cut_degree = node.cut_degree;
			
			for (std::uint32_t i = start_index; i < start_index + size; i++)
			{
				dynamic_degree += recv_answer_cut_degrees[i].degree;
				
				if (dynamic_degree >= cut_degree)
				{
					std::uint32_t cut_PE = i - start_index;
					cut_informations[k] = {node.global_index, cut_degree, cut_PE};
					//std::cout << "(" << node.global_index << "," << cut_degree << ") hat cut_PE " << cut_PE << std::endl;
					break;
				}
			}
			
		}
		timer.add_checkpoint("gather_informations");

		std::vector<cut_information> recv_cut_information;
		comm.allgatherv(kamping::send_buf(cut_informations), kamping::recv_buf<kamping::resize_to_fit>(recv_cut_information));
		
		std::vector<nodes_to_PE_assignment> recv_nodes_to_PE_assignments;
		comm.allgatherv(kamping::send_buf(nodes_to_PE_assignments), kamping::recv_buf<kamping::resize_to_fit>(recv_nodes_to_PE_assignments));
		//nodes_to_PE_assignment compressen
		//aufpassen: node_ranges [5,5] + [6,6] != [5,6]
		
		timer.add_checkpoint("compress_node_assignments");

		
		std::vector<nodes_to_PE_assignment> compressed_nodes_to_PE_assignments(size);
		std::int64_t i = -1;
		for (std::uint32_t p = 0; p < size; p++)
		{
			i++;
			compressed_nodes_to_PE_assignments[p].targetPE = p;
			
			compressed_nodes_to_PE_assignments[p].start_part_node = recv_nodes_to_PE_assignments[i].start_part_node;
			compressed_nodes_to_PE_assignments[p].start_part_node_start_degree = recv_nodes_to_PE_assignments[i].start_part_node_start_degree;
			
			std::int64_t node_start_index = -1;
			std::int64_t node_end_index = -1;
			
			do
			{
				if ((node_start_index == -1) && (recv_nodes_to_PE_assignments[i].node_start_index < recv_nodes_to_PE_assignments[i].node_end_index))
					node_start_index = recv_nodes_to_PE_assignments[i].node_start_index;
				if (recv_nodes_to_PE_assignments[i].node_start_index < recv_nodes_to_PE_assignments[i].node_end_index)
					node_end_index = recv_nodes_to_PE_assignments[i].node_end_index;
				i++;
			} while ((i < recv_nodes_to_PE_assignments.size()) && (recv_nodes_to_PE_assignments[i].targetPE == p));
			i--;
			
			if (node_start_index == -1) //then node_end_index == -1 too
			{
				node_start_index = recv_nodes_to_PE_assignments[i].node_end_index;
				node_end_index = recv_nodes_to_PE_assignments[i].node_end_index;
			}
				
			compressed_nodes_to_PE_assignments[p].node_start_index = node_start_index;
			compressed_nodes_to_PE_assignments[p].node_end_index = node_end_index;
			
			
			compressed_nodes_to_PE_assignments[p].end_part_node = recv_nodes_to_PE_assignments[i].end_part_node;
			compressed_nodes_to_PE_assignments[p].end_part_node_end_degree = recv_nodes_to_PE_assignments[i].end_part_node_end_degree;
		}
		/*
		if (rank == 0){
			std::cout << "compressed assignments" << std::endl;
			for (int i = 0; i < compressed_nodes_to_PE_assignments.size(); i++)
			{
				nodes_to_PE_assignment info = compressed_nodes_to_PE_assignments[i];
				std::cout << "PE " << info.targetPE << ": (" << info.start_part_node << "," << info.start_part_node_start_degree << "),(" <<info.node_start_index << "," << info.node_end_index << "),(" << info.end_part_node << "," << info.end_part_node_end_degree << ")" << std::endl;
			}
		}*/
		
		timer.add_checkpoint("node_ranges");

		
		//the cut_map tells us efficiently the cut_information for a cut
		//usage: our cut (cut_node,cut_degree) = (0,15) has cut_PE = 10 then cut_map["(0,15)"] = 10
		std::unordered_map<std::string,std::uint32_t> cut_map; 
		std::unordered_map<std::uint64_t,std::uint64_t> aggregated_cut_map; //this only conteins aggregated_cut_map[0] = *irgendetwas*
		auto get_cut_string = [](std::uint64_t cut_node, std::uint64_t cut_degree) { return "(" + std::to_string(cut_node) + "," + std::to_string(cut_degree) + ")"; };
		
		for (std::uint64_t i = 0; i < recv_cut_information.size(); i++)
		{
			cut_information info = recv_cut_information[i];
			cut_map[get_cut_string(info.cut_node,info.global_cut_degree)] = info.cut_PE;
			
			aggregated_cut_map[info.cut_node] = 0;
		}
		
		
		struct node_range {
			std::uint64_t node_start_index;
			std::uint64_t node_end_index;
		};
		
		std::vector<node_range> node_ranges(size); //this one is informatin for the splitted nodes, especially this array is different on each PE
		for (std::uint32_t p = 0; p < size; p++)
		{
			nodes_to_PE_assignment assign = compressed_nodes_to_PE_assignments[p];
			std::string start_part_node = get_cut_string(assign.start_part_node, assign.start_part_node_start_degree);
			std::string end_part_node = get_cut_string(assign.end_part_node, assign.end_part_node_end_degree);

			if (assign.node_start_index == assign.node_end_index)
			{
				if (assign.start_part_node == -1)
				{
					//dann muss assign.end_part_node != -1
					if (rank < cut_map[end_part_node])
						node_ranges[p] = {static_cast<std::uint64_t>(assign.end_part_node), static_cast<std::uint64_t>(assign.end_part_node +1)};
					else
						node_ranges[p] = {static_cast<std::uint64_t>(assign.end_part_node), static_cast<std::uint64_t>(assign.end_part_node)};
				}
				else if (assign.end_part_node == -1)
				{
					if (rank < cut_map[start_part_node])
						node_ranges[p] = {static_cast<std::uint64_t>(assign.start_part_node), static_cast<std::uint64_t>(assign.start_part_node)};
					else
						node_ranges[p] = {static_cast<std::uint64_t>(assign.start_part_node), static_cast<std::uint64_t>(assign.start_part_node + 1)};
				}
				else if (assign.start_part_node == assign.end_part_node)
				{
					if ((cut_map[start_part_node] <= rank) && (rank < cut_map[end_part_node]))
						node_ranges[p] = {static_cast<std::uint64_t>(assign.start_part_node), static_cast<std::uint64_t>(assign.start_part_node + 1)};
					else if (rank < cut_map[start_part_node])
						node_ranges[p] = {static_cast<std::uint64_t>(assign.start_part_node+1), static_cast<std::uint64_t>(assign.start_part_node+1)};
					else
						node_ranges[p] = {static_cast<std::uint64_t>(assign.end_part_node), static_cast<std::uint64_t>(assign.end_part_node)};
				}
				else //hier haben wir soetwas wie [(0,20),(0,0),(1,10)], insbesondere assign.end_part_node == assign.start_part_node+1
				{
					std::uint64_t node_start_index = assign.start_part_node +1;
					if (rank >= cut_map[start_part_node])
						node_start_index = assign.start_part_node;
					std::uint64_t node_end_index = assign.end_part_node;
					if (rank < cut_map[end_part_node])
						node_end_index = assign.end_part_node +1;
					node_ranges[p] = {node_start_index, node_end_index};
				}
			}
			else
			{
				std::uint64_t node_start_index = assign.node_start_index;
				if ((assign.start_part_node != -1) && (rank >= cut_map[start_part_node]))
					node_start_index = assign.node_start_index -1;
				
				std::uint64_t node_end_index = assign.node_end_index;
				if ((assign.end_part_node != -1) && (rank < cut_map[end_part_node]))
					node_end_index = assign.node_end_index +1;

				node_ranges[p] = {node_start_index, node_end_index};
			}
		}
		
		/*
		std::string output = std::to_string(rank) + " with node_ranges:\n";
		for (int i = 0; i < size; i++)
			output += "[" + std::to_string(node_ranges[i].node_start_index) + "," + std::to_string(node_ranges[i].node_end_index) + "]\n";
		std::cout << output;*/
		timer.add_checkpoint("lb_values");

		
		std::vector<std::uint64_t> lb_num_local_vertices_per_PE(size);
		std::vector<std::uint64_t> lb_prefix_sum_num_local_vertices_per_PE(size+1,0);
		std::vector<node_range> real_node_ranges(size); //this one tells us which nodes live on which PE, especially this array is the same on each PE
		std::vector<std::uint64_t> node_with_lowest_index_per_PE(size); //this one tells us the node with the lowest index per PE, including pseudonodes
		for (std::uint32_t p = 0; p < size; p++)
		{
			nodes_to_PE_assignment assign = compressed_nodes_to_PE_assignments[p];
			std::uint64_t lb_num_local_vertices = assign.node_end_index - assign.node_start_index;
			std::uint64_t start_node, end_node;
			start_node = assign.node_start_index;
			std::uint64_t node_with_lowest_index = start_node;
			if (assign.start_part_node != -1)
			{
				node_with_lowest_index = assign.start_part_node;
				if (assign.end_part_node != -1)
				{
					if (assign.start_part_node == assign.end_part_node)
					{
						lb_num_local_vertices = 1;
						end_node = ++start_node;
					}
					else
					{
						end_node = assign.end_part_node+1;
						lb_num_local_vertices += 2;
					}
				}
				else
				{
					end_node = assign.node_end_index;
					lb_num_local_vertices += 1;
				}
			}
			else
			{
				end_node = assign.node_end_index;
				if (assign.end_part_node != -1)
				{
					end_node = assign.end_part_node+1;
					lb_num_local_vertices += 1;
				}
			}
			real_node_ranges[p] = {start_node, end_node};
			node_with_lowest_index_per_PE[p] = node_with_lowest_index;
			//if (rank == 0) std::cout << p << " hat real node range [" << start_node <<"," << end_node << "]" << std::endl;
			//if (rank == 0) std::cout << p << " hat node with lowest index " << node_with_lowest_index << std::endl;
			lb_num_local_vertices_per_PE[p] = lb_num_local_vertices;
			lb_prefix_sum_num_local_vertices_per_PE[p+1] = lb_prefix_sum_num_local_vertices_per_PE[p] + lb_num_local_vertices;
		}
			

		nodes_to_PE_assignment assign = compressed_nodes_to_PE_assignments[rank];
		std::uint64_t lb_num_local_vertices = lb_num_local_vertices_per_PE[rank];

		std::vector<std::uint64_t> lb_s(lb_num_local_vertices, -1);
		std::vector<std::int64_t> lb_r(lb_num_local_vertices);
		std::vector<std::uint32_t> lb_targetPEs(lb_num_local_vertices);
		std::vector<std::int64_t> lb_real_node(lb_num_local_vertices);
		
		if (assign.start_part_node != -1)
		{
			lb_real_node[0] = -1;
			
			for (std::uint64_t i = 0; i < assign.node_end_index - assign.node_start_index; i++)
			{
				lb_real_node[1 + i] = assign.node_start_index + i;
			}
			
			if ((assign.end_part_node != -1) && (assign.start_part_node != assign.end_part_node))
				lb_real_node.back() = assign.end_part_node;
		}
		else
		{
			for (std::uint64_t i = 0; i < assign.node_end_index - assign.node_start_index; i++)
			{
				lb_real_node[i] = assign.node_start_index + i;
			}
			if (assign.end_part_node != -1)
			{
				lb_real_node.back() = assign.end_part_node;
			}
		}
		/*
		std::cout << rank << " with real nodes:\n";
		for (int i = 0; i < lb_num_local_vertices; i++)
			std::cout << lb_real_node[i] << " ";
		std::cout << std::endl;*/
		
		struct node_with_targetPE {
			std::uint64_t source;
			std::uint64_t destination;
			std::uint32_t targetPE_i;
			std::uint32_t targetPE_s;
		};
		
		struct node {
			std::uint64_t source;
			std::uint64_t destination;
			std::uint32_t targetPE;
		};
		
		std::vector<node_with_targetPE> node_with_targetPEs(num_local_vertices);
		//jetzt wird erst mit den real_node_ranges die nodes an die richtige PEs geschickt
		
		
		timer.add_checkpoint("calculate_edges");

		
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::uint64_t dynamic_lower_bound = 0; // inclusive
			std::uint64_t dynamic_upper_bound = size; //exclusive
			
			while (dynamic_upper_bound - dynamic_lower_bound > 1)
			{
				std::uint64_t middle = (dynamic_lower_bound + dynamic_upper_bound) / 2;
				node_range node_range = real_node_ranges[middle];
				
				if ((node_range.node_start_index <= i + node_offset) && (i + node_offset < node_range.node_end_index))
				{
					dynamic_lower_bound = middle;
					break;
				}
				else if (i+node_offset < node_range.node_start_index)
				{
					dynamic_upper_bound = middle;
				}
				else
				{
					dynamic_lower_bound = middle +1;
				}
			}
			std::uint32_t targetPE_i = dynamic_lower_bound; //this tells us, on which PE this node i' exists
			//now we need to compute where it points
			
			
			dynamic_lower_bound = 0; // inclusive
			dynamic_upper_bound = size; //exclusive
			
			while (dynamic_upper_bound - dynamic_lower_bound > 1)
			{
				std::uint64_t middle = (dynamic_lower_bound + dynamic_upper_bound) / 2;
				node_range node_range = node_ranges[middle];
				
				if ((node_range.node_start_index <= s[i]) && (s[i] < node_range.node_end_index))
				{
					dynamic_lower_bound = middle;
					break;
				}
				else if (s[i] < node_range.node_start_index)
				{
					dynamic_upper_bound = middle;
				}
				else
				{
					dynamic_lower_bound = middle +1;
				}
			}
			std::uint32_t targetPE_s = dynamic_lower_bound; //this tells us, on which PE the successor of node i' exists

			
			node_with_targetPEs[i] = {i + node_offset, s[i], targetPE_i, targetPE_s};
			//std::cout << "node " << i + node_offset << " wird nach " << targetPE <<  " geschickt" << std::endl;
		}
		
		timer.add_checkpoint("send_edges");

		
		std::vector<node> nodes(num_local_vertices);
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::uint32_t targetPE = node_with_targetPEs[i].targetPE_i;
			num_packets_per_PE[targetPE]++;
		}
		calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::uint32_t targetPE = node_with_targetPEs[i].targetPE_i;
			std::uint64_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			nodes[packet_index] = {node_with_targetPEs[i].source, node_with_targetPEs[i].destination, node_with_targetPEs[i].targetPE_s};
		}
		auto recv = comm.alltoallv(kamping::send_buf(nodes), kamping::send_counts(num_packets_per_PE)).extract_recv_buffer();
		timer.add_checkpoint("write_edges");

		for (int i =  0; i < recv.size(); i++)
		{
			node node = recv[i];
			
			std::uint32_t targetPE = node.targetPE;
			lb_s[node.source - node_with_lowest_index_per_PE[rank]] = node.destination - node_with_lowest_index_per_PE[targetPE] + lb_prefix_sum_num_local_vertices_per_PE[targetPE];
			lb_targetPEs[node.source - node_with_lowest_index_per_PE[rank]] = targetPE;
			lb_r[node.source - node_with_lowest_index_per_PE[rank]] = 1;
			
			if (node.source == node.destination)
				lb_r[node.source - node_with_lowest_index_per_PE[rank]] = 0;
			
		}
		
		//INFO: This was with pointer_doubling in mind. For forest_ruling_set it might be more efficient for every pseudonode to point to the real node, and not to nearest pseudonode
		if (assign.start_part_node != -1)
		{
			lb_s[0] = lb_prefix_sum_num_local_vertices_per_PE[rank] - 1; //this is the last node on the prev PE
			lb_targetPEs[0] = rank -1;
			lb_r[0] = 0;
		}
		
		timer.add_checkpoint("rekursion");

		//irregular_pointer_doubling algorithm(lb_s, lb_r, lb_targetPEs, lb_prefix_sum_num_local_vertices_per_PE);
		//std::vector<std::int64_t> ranks = algorithm.start(comm);
		std::vector<std::uint64_t> unnötig(lb_num_local_vertices);
		
		forest_irregular_ruling_set2 recursion(comm_rounds);
		recursion.start(lb_s, lb_r, lb_targetPEs, lb_prefix_sum_num_local_vertices_per_PE, comm, unnötig);
			
		std::vector<std::int64_t> ranks = recursion.result_dist;
		timer.add_checkpoint("final_ranks");

		struct result {
			std::uint64_t node;
			std::int64_t r;
		};
		
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
		for (std::uint64_t i = 0; i < lb_num_local_vertices; i++)
		{
			if (lb_real_node[i] != -1)
			{
				std::uint32_t targetPE = lb_real_node[i] / num_local_vertices;
				num_packets_per_PE[targetPE]++;	
			}
		}
		calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
		std::vector<result> results(send_displacements[size]);
		for (std::uint64_t i = 0; i < lb_num_local_vertices; i++)
		{
			if (lb_real_node[i] != -1)
			{
				std::uint32_t targetPE = lb_real_node[i] / num_local_vertices;
				std::uint64_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
				results[packet_index] = {static_cast<std::uint64_t>(lb_real_node[i]), ranks[i]};
			}
		}
		auto recv_final_ranks = comm.alltoallv(kamping::send_buf(results), kamping::send_counts(num_packets_per_PE)).extract_recv_buffer();
		
		std::vector<std::int64_t> final_ranks(num_local_vertices);
		for (std::uint64_t i =0; i < recv_final_ranks.size(); i++)
		{
			result result = recv_final_ranks[i];
			final_ranks[result.node - node_offset] = result.r;
		}
		
		timer.finalize(comm, "real_load_balance");

		
		/*
		std::cout << rank << " with final r arr:";
		for (int i = 0; i < num_local_vertices; i++)
			std::cout << final_ranks[i] << " ";
		std::cout << std::endl;*/
		
		test::regular_test_ranks(comm, s, final_ranks);
		
	
	}
	
	std::vector<std::uint64_t> calculate_indegrees(std::vector<std::uint64_t>& s, kamping::Communicator<>& comm, karam::mpi::GridCommunicator grid_comm)
	{
		struct packet {
			std::uint64_t node;
			std::uint64_t indegree;
		};
		
		std::vector<std::int32_t> send_counts(size,0);
		std::vector<std::int32_t> send_displacements(size + 1,0);
		
		std::unordered_map<std::uint64_t, std::uint64_t> local_node_indegrees;

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
			send_counts[targetPE]++;
		}
		
		calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, send_counts);
		
		std::vector<packet> send_buf(send_displacements[size]);
		for (const auto& [key, value] : local_node_indegrees)
		{
			std::int32_t targetPE = key / num_local_vertices;
			std::int64_t packet_index = send_displacements[targetPE] + send_counts[targetPE]++;
			send_buf[packet_index].node = key;
			send_buf[packet_index].indegree = value;
		}

		std::vector<std::int32_t> send_counts_row = std::vector<std::int32_t>(grid_comm.row_comm().size(),0);
		for (std::int32_t p = 0; p < comm.size(); p++)
		{
			std::int32_t targetPE = grid_comm.proxy_col_index(static_cast<std::size_t>(p));
			send_counts_row[targetPE] += send_counts[p];
		}
		
		auto send_displacements3 = send_counts_row;
		std::exclusive_scan(send_counts_row.begin(), send_counts_row.end(), send_displacements3.begin(), 0ull);
		auto       index_displacements = send_displacements3;
		
		karam::utils::default_init_vector<karam::mpi::IndirectMessage<packet>> contiguous_send_buf(send_buf.size()); 
		
		std::uint64_t index = 0;
		for (std::int32_t p = 0; p < comm.size(); p++)
		{

			for (std::uint64_t i = 0; i < send_counts[p]; i++)
			{
				auto const final_destination = p;
				auto const destination_in_row = grid_comm.proxy_col_index(static_cast<std::size_t>(final_destination));
				auto const idx = index_displacements[destination_in_row]++;

				contiguous_send_buf[static_cast<std::size_t>(idx)] = karam::mpi::IndirectMessage<packet>(
					static_cast<std::uint32_t>(comm.rank()),
					static_cast<std::uint32_t>(final_destination),
					send_buf[index++]
				  );
				
				
			}
			
		}
		
		auto mpi_result_rowwise = grid_comm.row_comm().alltoallv(
		kamping::send_buf(contiguous_send_buf),
		kamping::send_counts(send_counts_row));
		
		auto rowwise_recv_buf    = mpi_result_rowwise.extract_recv_buffer();
		
		local_node_indegrees = std::unordered_map<std::uint64_t, std::uint64_t>();
		std::unordered_map<std::uint64_t, std::uint32_t> node_targetPE;
		
		for (std::uint64_t i = 0; i < rowwise_recv_buf.size(); i++)
		{
			if (local_node_indegrees.contains(rowwise_recv_buf[i].payload().node))
				local_node_indegrees[rowwise_recv_buf[i].payload().node] += rowwise_recv_buf[i].payload().indegree;
			else
				local_node_indegrees[rowwise_recv_buf[i].payload().node] = rowwise_recv_buf[i].payload().indegree;
			node_targetPE[rowwise_recv_buf[i].payload().node] = rowwise_recv_buf[i].get_destination();
		}
		
		std::vector<std::int32_t> num_packets_per_PE2(size,0);
		std::vector<std::int32_t> send_displacements2(size + 1,0);
		
		for (const auto& [key, value] : local_node_indegrees)
		{
			std::int32_t targetPE = node_targetPE[key];
			num_packets_per_PE2[targetPE]++;
		}
		calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements2, num_packets_per_PE2);
		std::vector<karam::mpi::IndirectMessage<packet>> send(send_displacements2[size]);
		for (const auto& [key, value] : local_node_indegrees)
		{
			std::int32_t targetPE = node_targetPE[key];
			std::int64_t packet_index = send_displacements2[targetPE] + num_packets_per_PE2[targetPE]++;
		
			send[packet_index] = karam::mpi::IndirectMessage<packet>(
					static_cast<std::uint32_t>(comm.rank()),
					static_cast<std::uint32_t>(targetPE),
					{key,value}
				  );
		}
		
		std::vector<karam::mpi::IndirectMessage<packet>> recv = columnwise_exchange(send, grid_comm).extract_recv_buffer();
		//auto rowwise_recv_buf    = mpi_result_rowwise.extract_recv_buffer();
		
		/*
		std::cout << rank << " with: ";
		for (int i = 0; i < recv.size(); i++)
			std::cout << recv[i].payload().node << " hat indegree " << recv[i].payload().indegree << std::endl;
		*/
		local_node_indegrees = std::unordered_map<std::uint64_t, std::uint64_t>();
		for (int i = 0; i < recv.size(); i++)
		{
			if (local_node_indegrees.contains(recv[i].payload().node))
				local_node_indegrees[recv[i].payload().node] += recv[i].payload().indegree;
			else
				local_node_indegrees[recv[i].payload().node] = recv[i].payload().indegree;
		}
		
		std::vector<std::uint64_t> indegrees(num_local_vertices,0);
		
		for (const auto& [key, value] : local_node_indegrees)
		{
			indegrees[key - node_offset] = value;
		}
		return indegrees;
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
	std::int32_t rank, size;
};