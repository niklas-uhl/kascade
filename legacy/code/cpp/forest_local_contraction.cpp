#include "list_ranking/irregular_pointer_doubling.cpp"

class forest_local_contraction
{

	public:
	
	forest_local_contraction()
	{

	}
	
	
	std::vector<std::int64_t> start(kamping::Communicator<>& comm, std::vector<std::uint64_t>& s)
	{
		size = comm.size();
		rank = comm.rank();
		num_local_vertices = s.size();
		node_offset = rank * num_local_vertices;
		/*
		std::cout << rank << " with s arr: ";
		for (int i = 0; i < num_local_vertices; i++)
			std::cout << s[i] << " ";
		std::cout << std::endl;*/
		
		std::vector<std::string> categories = {"local_work", "communication"};
		timer timer("nodes_scannen", categories, "local_work", "forest_local_contraction");
		
		timer.add_info(std::string("num_local_vertices"), std::to_string(num_local_vertices), true);
		
		std::vector<bool> has_local_successor(num_local_vertices, false);
		std::vector<bool> has_local_predecessor(num_local_vertices, false);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::int32_t targetPE = s[i] / num_local_vertices;
			
			if (targetPE == rank && i + node_offset != s[i])
			{
				has_local_successor[i] = true;
				has_local_predecessor[s[i] - node_offset] = true;
				//std::cout << i << " has local predecessor" << std::endl;
			}
		}
		struct reduced_node {
			std::uint64_t i;
			std::uint64_t s;
			std::int64_t r;
		};
		struct removed_node {
			std::uint64_t i;
			std::uint64_t source;
			std::int64_t r;
		};
		std::vector<reduced_node> reduced_nodes(0);
		std::vector<removed_node> removed_nodes(0);
		
		timer.add_checkpoint("remove_and_reduce");

		
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			if (!has_local_predecessor[i])
			{

				std::uint64_t succ = i + node_offset;
				std::int64_t r = 0;
				while (((succ / num_local_vertices) == rank) && (succ != s[succ - node_offset])) 
				{
					succ = s[succ - node_offset];
					r++;
					if ((i + node_offset != succ) && (succ / num_local_vertices == rank)) {
						removed_nodes.push_back({ succ,i + node_offset, r});
						//std::cout << "removed node " << succ << " zeigt auf " << i + node_offset << " with dist " << r << std::endl;
					}
					

				}
				
				if (succ / num_local_vertices == rank)
				{
					//that means we have final node
					reduced_nodes.push_back({i + node_offset, i + node_offset, r});
					//std::cout << "final node " << i + node_offset << " zeigt auf " <<  i + node_offset << " with dist " << r << std::endl;
					
				}
				else
				{
					reduced_nodes.push_back({i + node_offset, succ, r});
					//std::cout << "final node " << i + node_offset << " zeigt auf " << succ << " with dist " << r << std::endl;
				}
				
				
			
				
			}			
		}
		
		timer.add_checkpoint("request_indices");

		std::uint64_t num_local_vertices_reduced = reduced_nodes.size();
		std::vector<std::uint64_t> num_vertices_per_PE_reduced(size);
		comm.allgather(kamping::send_buf(num_local_vertices_reduced), kamping::recv_buf<kamping::resize_to_fit>(num_vertices_per_PE_reduced));
		
		std::vector<std::uint64_t> prefix_sum_num_vertices_per_PE_reduced(size+1,0);
		for (std::uint32_t i = 1; i < size + 1; i++)
			prefix_sum_num_vertices_per_PE_reduced[i] = prefix_sum_num_vertices_per_PE_reduced[i-1] + num_vertices_per_PE_reduced[i-1];
		
		std::unordered_map<std::uint64_t,std::uint64_t> source_map; // when node i+node_offset gets gemoved and points to source j, then map[i+node_offset] = j;
		std::unordered_map<std::uint64_t,std::int64_t> r_map; //this tells us the distance to the source
		for (std::uint64_t i = 0; i < removed_nodes.size(); i++)
		{
			removed_node node = removed_nodes[i];
			source_map[node.i]=node.source;
			r_map[node.i]=node.r;
		}
	
		std::vector<std::uint64_t> map_reduced_nodes_to_its_index(num_local_vertices,-1);
		for (std::uint64_t i = 0; i < num_local_vertices_reduced; i++)
		{
			map_reduced_nodes_to_its_index[reduced_nodes[i].i - node_offset] = i;	
		}
		std::vector<std::int32_t> num_packets_per_PE(size,0);
		std::vector<std::int32_t> send_displacements(size + 1,0);
		std::vector<std::uint64_t> request(num_local_vertices_reduced);
		for (std::uint64_t i = 0; i < num_local_vertices_reduced; i++)
		{
			std::int32_t targetPE = reduced_nodes[i].s / num_local_vertices;
			num_packets_per_PE[targetPE]++;
		}
		calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
		for (std::uint64_t i = 0; i < num_local_vertices_reduced; i++)
		{
			std::int32_t targetPE = reduced_nodes[i].s / num_local_vertices;
			std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			
			request[packet_index] = reduced_nodes[i].s;
		}
		
		timer.add_checkpoint("send_request");

		auto recv = comm.alltoallv(kamping::send_buf(request), kamping::send_counts(num_packets_per_PE));
		num_packets_per_PE = recv.extract_recv_counts();
		std::vector<std::uint64_t> recv_request = recv.extract_recv_buffer();
		/*
		std::cout << rank << " with recv request: ";
		for (int i = 0; i <recv_request.size(); i++)
			std::cout << recv_request[i] << " ";
		std::cout << std::endl;*/
		
		struct answer {
			std::uint64_t source;
			std::int64_t r;
		};
		timer.add_checkpoint("process_request");

		std::vector<answer> answers(recv_request.size());
		for (std::uint64_t i = 0; i <recv_request.size(); i++)
		{
			if (source_map.contains(recv_request[i])) //iff der requested node removed wurde
			{
				answers[i] = {map_reduced_nodes_to_its_index[source_map[recv_request[i]]-node_offset] + prefix_sum_num_vertices_per_PE_reduced[rank], r_map[recv_request[i]]};
			}
			else
			{
				answers[i] = {map_reduced_nodes_to_its_index[recv_request[i] - node_offset] + prefix_sum_num_vertices_per_PE_reduced[rank], 0};
			}
		}
		timer.add_checkpoint("send_answers");

		std::vector<answer> recv_answers = comm.alltoallv(kamping::send_buf(answers), kamping::send_counts(num_packets_per_PE)).extract_recv_buffer();
		timer.add_checkpoint("process_answers");

		std::vector<std::uint64_t> s_reduced(num_local_vertices_reduced);
		std::vector<std::int64_t> r_reduced(num_local_vertices_reduced);
		std::vector<std::uint32_t> targetPEs_reduced(num_local_vertices_reduced);
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);	
		for (std::uint64_t i = 0; i < num_local_vertices_reduced; i++)
		{
			std::int32_t targetPE = reduced_nodes[i].s / num_local_vertices;
			std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			//std::cout << "node " << i + prefix_sum_num_vertices_per_PE_reduced[rank] << " has s=" << request[packet_index] << " - " << targetPE*num_local_vertices << " + " << prefix_sum_num_vertices_per_PE_reduced[targetPE] << std::endl;
			s_reduced[i] = recv_answers[packet_index].source;
			r_reduced[i] = reduced_nodes[i].r - recv_answers[packet_index].r;
			targetPEs_reduced[i] = targetPE;
			
			//std::cout << "node for recursion (i,s,r,targetPE): (" << i + prefix_sum_num_vertices_per_PE_reduced[rank] << "," << s_reduced[i] << "," << r_reduced[i] << "," << targetPEs_reduced[i] << ")" <<std::endl;

		}
		
		timer.add_info(std::string("num_local_vertices_reduced"), std::to_string(num_local_vertices_reduced), true);

		
				//if (rank == 0) std::cout << "Instanzgröße von " << size * num_local_vertices << " auf " << prefix_sum_num_vertices_per_PE_reduced[size] * 100 / (size * num_local_vertices) << "% reduziert" << std::endl;

		timer.add_checkpoint("irregular_pointer_doubling");

		
		irregular_pointer_doubling algorithm(s_reduced, r_reduced, targetPEs_reduced, prefix_sum_num_vertices_per_PE_reduced);
		std::vector<std::int64_t> ranks = algorithm.start(comm);
		/*
		std::cout << rank << " with ranks out of recursion :";
		for (std::uint64_t i = 0; i < ranks.size(); i++)
			std::cout << ranks[i] << " ";
		std::cout << std::endl;
		*/
		timer.add_checkpoint("calculate_final_ranks");

		
		std::vector<std::int64_t> final_ranks(num_local_vertices);
		for (std::uint64_t i = 0; i < num_local_vertices_reduced; i++)
		{
			final_ranks[reduced_nodes[i].i - node_offset] = ranks[i];
		}
		
		for (std::uint64_t i = 0; i < removed_nodes.size(); i++)
		{
			final_ranks[removed_nodes[i].i - node_offset] =  final_ranks[removed_nodes[i].source - node_offset] - removed_nodes[i].r;
		}
		
		timer.finalize(comm, "forest_local_contraction");


		return final_ranks;
		
	}
	
	
	void calculate_send_displacements_and_reset_num_packets_per_PE(std::vector<std::int32_t>& send_displacements, std::vector<std::int32_t>& num_packets_per_PE)
	{
		send_displacements[0]=0;
		for (std::int32_t i = 1; i < size + 1; i++)
			send_displacements[i] = send_displacements[i-1] + num_packets_per_PE[i-1];
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
	}
		
	private:
	
	std::uint64_t node_offset;
	std::uint64_t num_local_vertices;
	std::int32_t rank, size;
};