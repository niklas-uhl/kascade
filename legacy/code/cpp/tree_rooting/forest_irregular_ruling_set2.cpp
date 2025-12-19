#pragma once

#include "forest_irregular_pointer_doubling.cpp"


//the idea of base of this code is the generalization of "ultimate parallel list ranking"
class forest_irregular_ruling_set2 //this is for forest
{
	struct packet{
		std::uint64_t ruler_source;
		std::uint64_t destination;
		std::int64_t distance;
		std::uint32_t ruler_source_PE;
	};
	
	struct result{
			std::uint64_t root;
			std::int64_t distance;
	};
	
	public:
	
	forest_irregular_ruling_set2(std::uint64_t comm_rounds, std::uint32_t num_iterations, bool grid)
	{
		this->comm_rounds = comm_rounds;
		this->num_iterations = num_iterations;
		this->grid = grid;
	}

	
	
	void start(std::vector<std::uint64_t>& s, std::vector<std::int64_t>& r, std::vector<std::uint32_t>& targetPEs, std::vector<std::uint64_t>& prefix_sum_num_vertices_per_PE, kamping::Communicator<>& comm, std::vector<std::uint64_t>& local_rulers)
	{
		std::vector<std::string> categories = {"local_work", "communication", "other"};
		timer timer("graph_umdrehen", categories, "local_work", "deprecated!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
		
		karam::mpi::GridCommunicator grid_comm;

		
		size = comm.size();
		rank = comm.rank();
		num_local_vertices = s.size();
		
		timer.add_info(std::string("comm_rounds"), std::to_string(comm_rounds));
		timer.add_info(std::string("num_local_vertices"), std::to_string(num_local_vertices), true);
		timer.add_info("grid", std::to_string(grid));
		
		node_offset = prefix_sum_num_vertices_per_PE[rank];
		//now turn around s array
		calculate_adj_arr(s, r, targetPEs, comm, grid_comm, node_offset);
		start(comm, grid_comm, timer);	
	}
	
	
	//targetPEs[i] is targetPE von s[i]
	void start(std::vector<std::uint64_t>& s, std::vector<std::int64_t>& r, std::vector<std::uint32_t>& targetPEs, std::vector<std::uint64_t>& prefix_sum_num_vertices_per_PE, kamping::Communicator<>& comm, karam::mpi::GridCommunicator& grid_comm, std::vector<std::uint64_t>& local_rulers)
	{
		std::vector<std::string> categories = {"local_work", "communication", "other"};
		timer timer("graph_umdrehen", categories, "local_work", "forest_irregular_ruling_set2");
		
		
		
		size = comm.size();
		rank = comm.rank();
		num_local_vertices = s.size();
		
		timer.add_info(std::string("comm_rounds"), std::to_string(comm_rounds));
		timer.add_info(std::string("num_local_vertices"), std::to_string(num_local_vertices), true);
		timer.add_info("grid", std::to_string(grid));
		
		node_offset = prefix_sum_num_vertices_per_PE[rank];
		//now turn around s array
		calculate_adj_arr(s, r, targetPEs, comm, grid_comm, node_offset);
		start(comm, grid_comm, timer);	
	}
	
	
	
	//all edges will be turn around, therefore we have indegree 1 and outdegree can be any integer. Additionally adj_arr will have no self edges
	void calculate_adj_arr(std::vector<std::uint64_t>& s, std::vector<std::int64_t>& r, std::vector<std::uint32_t>& targetPEs, kamping::Communicator<>& comm, karam::mpi::GridCommunicator& grid_comm, std::uint64_t node_offset)
	{
		
		
		
		struct edge{
			std::uint64_t source;
			std::uint64_t destination;
			std::int64_t weight;
			std::uint32_t source_PE;
		};
		std::vector<std::int32_t> num_packets_per_PE(size,0);
		std::vector<std::int32_t> send_displacements(size + 1,0);
		for (std::uint64_t i = 0; i < s.size(); i++)
		{
			if (i + node_offset == s[i])
				continue;
			
			std::int32_t targetPE = targetPEs[i];
			num_packets_per_PE[targetPE]++;		
		}
		calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
		std::vector<edge> send_edges(s.size());
		
		for (std::uint64_t i = 0; i < s.size(); i++)
		{
			if (i + node_offset == s[i])
				continue;
			
			std::int32_t targetPE = targetPEs[i];
			std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			
			send_edges[packet_index].source = i + node_offset;
			send_edges[packet_index].destination = s[i];
			send_edges[packet_index].weight = r[i];
			send_edges[packet_index].source_PE = rank;
		}
		
	

		
		
		auto recv = comm.alltoallv(kamping::send_buf(send_edges), kamping::send_counts(num_packets_per_PE)).extract_recv_buffer();

	
		edges = std::vector<std::uint64_t>(recv.size());
		edges_weights = std::vector<std::int64_t>(recv.size());
		edges_targetPEs = std::vector<std::uint32_t>(recv.size());
		
		std::vector<std::uint64_t> edges_per_node(s.size(),0);

		for (std::uint64_t i = 0; i < recv.size(); i++)
		{
			edges_per_node[recv[i].destination - node_offset]++;
		}
		bounds = std::vector<std::uint64_t>(s.size() + 1, 0);
		for (std::uint64_t i = 1; i <= s.size(); i++)
			bounds[i] = bounds[i-1] + edges_per_node[i-1];
		std::fill(edges_per_node.begin(), edges_per_node.end(), 0);
		for (std::uint64_t i = 0; i < recv.size(); i++)
		{
			std::uint64_t target_node = recv[i].destination - node_offset;
			std::uint64_t packet_index = bounds[target_node] + edges_per_node[target_node]++;
			edges[packet_index] = recv[i].source;
			edges_targetPEs[packet_index] = recv[i].source_PE;
			edges_weights[packet_index] = recv[i].weight;
			
		}
		
		/*
		//print for testing
		std::cout << "PE " << rank << " with s: ";
		for (int i = 0; i < s.size(); i++)
			std::cout << s[i] << ",";
		std::cout << "\nbounds: ";
		for (int i = 0; i < bounds.size(); i++)
			std::cout << bounds[i] << ",";
		std::cout << "\nturned edges: ";
		for (int i = 0; i < turned_edges.size(); i++)
			std::cout << turned_edges[i] << ",";
		std::cout << std::endl;*/

	}
	
	
	void start(kamping::Communicator<>& comm, karam::mpi::GridCommunicator& grid_comm, timer& timer)
	{
		
		timer.add_checkpoint("ruler_pakete_senden");

		std::uint64_t expected_num_packets = num_local_vertices/comm_rounds;
		std::vector<packet> out_buffer(0);
		
		std::vector<std::int32_t> num_packets_per_PE(size,0);
		std::vector<std::int32_t> send_displacements(size + 1,0);
		
		std::vector<std::uint64_t> local_rulers(0);
		std::uint64_t ruler_index = 0; //this means that first free rulers has an index >= ruler_index
	
		std::uint64_t num_packets = 0;
		while (num_packets < expected_num_packets && ruler_index < num_local_vertices)
		{
			local_rulers.push_back(ruler_index);
		
			
			for (std::uint64_t i = unmask(bounds[ruler_index]); i < unmask(bounds[ruler_index+1]); i++)
			{
				std::int32_t targetPE =  edges_targetPEs[i];
				num_packets_per_PE[targetPE]++;
				num_packets++;
			}
			ruler_index++;
		}
		calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
		out_buffer.resize(send_displacements[size]);
		for (std::uint64_t i = 0; i < local_rulers.size(); i++)
		{
			std::uint64_t local_index = local_rulers[i];
			
			for (std::uint64_t i = unmask(bounds[local_index]); i < unmask(bounds[local_index+1]); i++)
			{
				std::int32_t targetPE = edges_targetPEs[i];
				std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
				
				out_buffer[packet_index].ruler_source = local_index + node_offset;
				out_buffer[packet_index].destination = edges[i];
				out_buffer[packet_index].distance = edges_weights[i];
				out_buffer[packet_index].ruler_source_PE = rank;
			}
			
		
			
			
			
			
			mark_as_ruler(local_index);
		}

	
		std::vector<std::uint64_t> mst(num_local_vertices);
		std::vector<std::uint32_t> mst_PE(num_local_vertices,rank);
		std::iota(mst.begin(),mst.end(),node_offset); 
		std::vector<std::int64_t> del(num_local_vertices,0);
	

		
		timer.add_checkpoint("pakete_verfolgen");
		bool work_left = true;
		std::uint32_t iteration = 0;
		while (any_PE_has_work(comm, work_left))
		{
			

			/*
			std::cout << rank << " in iteration " << iteration << " with following packages:\n";
			for (packet& packet: out_buffer)
				std::cout << "(" << packet.ruler_source << "," << packet.destination << "," << packet.distance << "),";
			std::cout << std::endl;*/
			

			
			std::vector<packet> recv_buffer = comm.alltoallv(kamping::send_buf(out_buffer), kamping::send_counts(num_packets_per_PE)).extract_recv_buffer();

			std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
			
			work_left = recv_buffer.size() > 0;
			
			std::uint64_t num_forwarded_packages = 0;
			for (packet& packet: recv_buffer)
			{
				
				std::uint64_t local_index = packet.destination - node_offset;
				
				
				mark_as_reached(local_index);
				if (!is_ruler(local_index))
				{
					for (std::uint64_t i = unmask(bounds[local_index]); i < unmask(bounds[local_index+1]); i++)
					{
						std::int32_t targetPE = edges_targetPEs[i];
						num_packets_per_PE[targetPE]++;
						num_forwarded_packages++;
						
					}
				}

			}
			
			std::vector<std::uint64_t> rulers_to_send_packages(0);
			while (num_forwarded_packages < expected_num_packets)
			{
				while (ruler_index < num_local_vertices && (is_reached(ruler_index) || is_ruler(ruler_index))) ruler_index++;
				
				if (ruler_index == num_local_vertices)
				{
					break;
				}
				local_rulers.push_back(ruler_index);
				mark_as_ruler(ruler_index);
				rulers_to_send_packages.push_back(ruler_index);
				for (std::uint64_t i = unmask(bounds[ruler_index]); i < unmask(bounds[ruler_index+1]); i++)
				{
					std::int32_t targetPE = edges_targetPEs[i];
					num_packets_per_PE[targetPE]++;
					num_forwarded_packages++;
		
				}
		
			}
			
			calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
			out_buffer.resize(send_displacements[size]); 
			
			for (packet& packet: recv_buffer) 
			{
				std::uint64_t local_index = packet.destination - node_offset;
				mst[local_index] = packet.ruler_source;
				del[local_index] = packet.distance;
				mst_PE[local_index] = packet.ruler_source_PE;
				
				
				if (!is_ruler(local_index))
				{
					for (std::uint64_t i = unmask(bounds[local_index]); i < unmask(bounds[local_index+1]); i++)
					{
						std::int32_t targetPE = edges_targetPEs[i];
						std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;

						out_buffer[packet_index].ruler_source = packet.ruler_source; 
						out_buffer[packet_index].destination = unmask(edges[i]);
						out_buffer[packet_index].distance = packet.distance + edges_weights[i];
						out_buffer[packet_index].ruler_source_PE = packet.ruler_source_PE;
						//std::cout << packet_to_string(out_buffer[packet_index]) << " forwarded in iteration " << iteration << "\n";

					}					
				}	

			}
			
			for (std::uint64_t i = 0; i < rulers_to_send_packages.size(); i++)
			{
				std::uint64_t local_index = rulers_to_send_packages[i];
				for (std::uint64_t i = unmask(bounds[local_index]); i < unmask(bounds[local_index+1]); i++)
				{
					
					
					std::int32_t targetPE = edges_targetPEs[i];
					std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;

					out_buffer[packet_index].ruler_source = local_index + node_offset;
					out_buffer[packet_index].destination = unmask(edges[i]);
					out_buffer[packet_index].distance = edges_weights[i];
					out_buffer[packet_index].ruler_source_PE = rank;
					//std::cout << packet_to_string(out_buffer[packet_index]) << " started in iteration " << iteration << "\n";

				}
				
				
			}
			
		}	
		timer.add_checkpoint("rekursion_vorbereiten");
		
		std::vector<std::uint64_t> num_local_vertices_per_PE;

		comm.allgather(kamping::send_buf(local_rulers.size()), kamping::recv_buf<kamping::resize_to_fit>(num_local_vertices_per_PE));

		std::vector<std::uint64_t> prefix_sum_num_vertices_per_PE_rec(size + 1,0);
		for (std::uint32_t i = 1; i < size + 1; i++)
		{
			prefix_sum_num_vertices_per_PE_rec[i] = prefix_sum_num_vertices_per_PE_rec[i-1] + num_local_vertices_per_PE[i-1];
		}
		
		
		std::vector<std::uint64_t> map_ruler_to_its_index(num_local_vertices);
		std::vector<std::uint64_t> s_rec(local_rulers.size());
		std::vector<std::int64_t> r_rec(local_rulers.size());
		std::vector<std::uint32_t> targetPEs_rec(local_rulers.size());
	
		
		std::vector<std::uint64_t> requests(local_rulers.size());
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
		for (std::uint64_t i = 0; i < local_rulers.size(); i++)
		{
			map_ruler_to_its_index[local_rulers[i]] = i;
			std::int32_t targetPE = mst_PE[local_rulers[i]];
			targetPEs_rec[i] = targetPE;
			num_packets_per_PE[targetPE]++;
		}
		calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
		for (std::uint64_t i = 0; i < local_rulers.size(); i++)
		{
			std::int32_t targetPE = mst_PE[local_rulers[i]];
			std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			requests[packet_index] = mst[local_rulers[i]];
		}
		
		
		auto recv = comm.alltoallv(kamping::send_buf(requests), kamping::send_counts(num_packets_per_PE));
		
		
		std::vector<std::uint64_t> recv_requests = recv.extract_recv_buffer();
		
		
		//answers können inplace in requests eingetragen werden
		for (std::uint64_t i = 0; i < recv_requests.size(); i++)
		{
			recv_requests[i] = map_ruler_to_its_index[recv_requests[i]-node_offset] + prefix_sum_num_vertices_per_PE_rec[rank];
		}
		std::vector<std::uint64_t> recv_answers = comm.alltoallv(kamping::send_buf(recv_requests), kamping::send_counts(recv.extract_recv_counts())).extract_recv_buffer();
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
		for (std::uint64_t i = 0; i < local_rulers.size(); i++)
		{
			std::int32_t targetPE = mst_PE[local_rulers[i]];
			std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			s_rec[i] = recv_answers[packet_index];
			r_rec[i] = del[local_rulers[i]];
			
			//if (r_rec[i] == 0) s_rec[i] = i + prefix_sum_num_vertices_per_PE_rec[rank];
		}
		
		/*
		for (int i = 0; i < local_rulers.size(); i++)
			std::cout << i + prefix_sum_num_vertices_per_PE[rank] << " s_rec:" << s_rec[i] << ", r_rec:" << r_rec[i] << std::endl;
		*/
		
		
		timer.add_checkpoint("rekursion");
		std::vector<std::uint64_t> local_rulers_global_index = local_rulers;
		for (std::uint32_t i = 0; i < local_rulers.size(); i++)
			local_rulers_global_index[i] += node_offset;
		/*
		std::cout << "PE " << rank << " with rulers :";
		for (int i = 0; i < local_rulers.size(); i++)
			std::cout << local_rulers[i] + node_offset << " ";
		std::cout << std::endl;
		std::cout << "PE " << rank << " with s_rec :";
		for (int i = 0; i < s_rec.size(); i++)
			std::cout << s_rec[i] << " ";
		std::cout << std::endl;
		std::cout << "PE " << rank << " with r_rec :";
		for (int i = 0; i < r_rec.size(); i++)
			std::cout << r_rec[i] << " ";
		std::cout << std::endl;
		std::cout << "PE " << rank << " with targetPEs_rec :";
		for (int i = 0; i < targetPEs_rec.size(); i++)
			std::cout << targetPEs_rec[i] << " ";
		std::cout << std::endl;*/
		
		
		
		std::vector<std::uint64_t> recursive_roots;
		std::vector<std::int64_t> recursive_ranks;
		
		if (num_iterations == 1)
		{
			forest_irregular_pointer_doubling recursion(s_rec, r_rec, targetPEs_rec, prefix_sum_num_vertices_per_PE_rec, local_rulers_global_index, true, true);
			recursion.start(comm, grid_comm);
		
			recursive_roots = recursion.local_rulers;
			recursive_ranks = recursion.r;
			
		}
		else
		{
			forest_irregular_ruling_set2 recursion(comm_rounds,num_iterations - 1, false);
			recursion.start(s_rec, r_rec, targetPEs_rec, prefix_sum_num_vertices_per_PE_rec, comm, grid_comm, local_rulers_global_index);
			
			recursive_roots = recursion.result_root;
			recursive_ranks = recursion.result_dist;
		}
		
		
		
		
		//forest_irregular_pointer_doubling recursion(s_rec, r_rec, targetPEs_rec, prefix_sum_num_vertices_per_PE_rec, comm, local_rulers_global_index);
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
		timer.add_checkpoint("finalen_ranks_berechnen");
		timer.switch_category("local_work");

		//recursive_roots = recursion.local_rulers;
		//recursive_ranks = recursion.r;

		//TODO finalen ranks berechnen mit request und inplace answer mit recursion.local_rulers und damit bessere Zeit wie gestern. git reset --hard, wenn bei wood_irregular_pointer_doubling was falsch
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::int32_t targetPE = mst_PE[i];
			num_packets_per_PE[targetPE]++;
		}
		calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
		std::vector<std::uint64_t> request(num_local_vertices);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::int32_t targetPE = mst_PE[i];
			std::uint64_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			request[packet_index] = mst[i];
		}

	
		auto recv_request = comm.alltoallv(kamping::send_buf(request), kamping::send_counts(num_packets_per_PE));
		std::vector<std::uint64_t> recv_request_buffer = recv_request.extract_recv_buffer();
		
		struct answer{
			std::uint64_t global_root_index;
			std::int64_t distance;
		};
		std::vector<answer> answers(recv_request_buffer.size());
		for (std::uint64_t i = 0; i < recv_request_buffer.size(); i++)
		{	
			answers[i].global_root_index = recursive_roots[map_ruler_to_its_index[recv_request_buffer[i] - node_offset]];
			answers[i].distance = recursive_ranks[map_ruler_to_its_index[recv_request_buffer[i] - node_offset]];
			
			//std::cout << "request " << recv_request_buffer[i] << " wird beantwortet mit " <<  answers[i].distance << std::endl;
		}
		timer.switch_category("communication");
		auto recv_answers_buffer = comm.alltoallv(kamping::send_buf(answers), kamping::send_counts(recv_request.extract_recv_counts()))	.extract_recv_buffer();
		timer.switch_category("local_work");
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);

		result_dist = std::vector<std::int64_t>(num_local_vertices);
		result_root = std::vector<std::uint64_t>(num_local_vertices);
		
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::int32_t targetPE = mst_PE[i];
			std::uint64_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;

			
			result_root[i] = recv_answers_buffer[packet_index].global_root_index;
			result_dist[i] = recv_answers_buffer[packet_index].distance + del[i];
			
			//std::cout << i + node_offset << " hat final rank " << recv_answers_buffer[packet_index].distance << "+" << del[i] << "=" << result_dist[i] << std::endl;
		}
		
		timer.finalize(comm, "forest_irregular_ruling_set2");
	}
	
	
	
	std::string packet_to_string(packet packet)
	{
		return "(" + std::to_string(packet.ruler_source) + "," + std::to_string(packet.destination) + "," + std::to_string(packet.distance) + ")";
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
	
	void mark_as_ruler(std::uint64_t local_index)
	{
		bounds[local_index] =  mark(bounds[local_index],0);
	}
	
	bool is_ruler(std::uint64_t local_index)
	{
		return is_marked(bounds[local_index],0);
	}
	
	void mark_as_reached(std::uint64_t local_index)
	{
		bounds[local_index] =  mark(bounds[local_index],1);
	}
	
	bool is_reached(std::uint64_t local_index)
	{
		return is_marked(bounds[local_index],1);
	}
	
	bool is_leaf(std::uint64_t local_index)
	{
		return unmask(bounds[local_index]) == unmask(bounds[local_index + 1]);
	}
	
	std::uint64_t unmask(std::uint64_t value)
	{
		return value & 0xfffffffffffffff;
	}
	
	// the nth most significant bit will be marked, n>= 0
	std::uint64_t mark(std::uint64_t index, int n)
	{
		return index | (((std::uint64_t) 0x8000000000000000) >> n);
	}
	
	std::uint64_t unmark(std::uint64_t index, int n)
	{
		return index & (0xffffffffffffffff & (~(((std::uint64_t) 0x8000000000000000) >> n)));
	}
	
	bool is_marked(std::uint64_t index, int n)
	{
		return (index & (((std::uint64_t) 0x8000000000000000) >> n)) != 0;
	}
	
 
	void calculate_send_displacements_and_reset_num_packets_per_PE(std::vector<std::int32_t>& send_displacements, std::vector<std::int32_t	>& num_packets_per_PE)
	{
		send_displacements[0]=0;
		for (std::int32_t i = 1; i < size + 1; i++)
			send_displacements[i] = send_displacements[i-1] + num_packets_per_PE[i-1];
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
	}
	
	
	public:
	std::uint64_t node_offset;
	std::uint64_t num_local_vertices;
	std::uint64_t rank, size;
	std::vector<std::uint64_t> edges;
	std::vector<std::int64_t> edges_weights;
	std::vector<std::uint32_t> edges_targetPEs;
	std::vector<std::uint64_t> bounds;
	
	std::vector<std::int64_t> result_dist;
	std::vector<std::uint64_t> result_root;
	
	std::uint64_t comm_rounds;
	int num_iterations;
	bool grid;
};


	