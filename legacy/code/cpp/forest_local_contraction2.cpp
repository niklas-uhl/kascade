#include "list_ranking/irregular_pointer_doubling.cpp"
#include "tree_rooting/forest_irregular_optimized_ruling_set.cpp"


//this algorithm compressed by compressing every local subtree to its local root, in contrairy to forest_local_contraction which compresses every local subtree to its leaves
class forest_local_contraction2
{

	public:
	
	forest_local_contraction2()
	{

	}
	
	void add_timer_info(std::string info)
	{
		this->info = "\"" + info + "\"";
	}
	
	
	std::vector<std::int64_t> start(kamping::Communicator<>& comm, karam::mpi::GridCommunicator& grid_comm, std::vector<std::uint64_t>& s, bool grid=true, bool aggregate=false, bool call_ruling_set=false)
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
		timer.add_info(std::string("num_local_vertices"), std::to_string(num_local_vertices));
		timer.add_info(std::string("call_ruling_set"), std::to_string(call_ruling_set));
		timer.add_info(std::string("grid"), std::to_string(grid));
		
		if (info.size() > 0)
			timer.add_info(std::string("additional_info"), info);

	
		
		std::vector<std::int64_t> local_root(num_local_vertices,-1); //1->1->2->3->4, with node 0-3 on this PE, then local_root[0]=local_root[1]=3
		std::vector<std::uint64_t> dist_local_root(num_local_vertices,-1); //1->1->2->3->4, with node 0-3 on this PE, then local_root[0]=local_root[1]=3

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
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			if (local_root[i] != -1)
				continue;
			
			std::vector<std::uint64_t> indices_on_path_to_local_root(0); //this array contains local indices
			
			std::uint64_t node = i + node_offset;
			//std::cout << rank << " bearbeitet gerade node " << node << std::endl;
			while (node / num_local_vertices == rank)
			{
				if (s[node -  node_offset] == node)
				{

					for (std::uint64_t i = 0; i < indices_on_path_to_local_root.size(); i++)
					{
						local_root[indices_on_path_to_local_root[i]] = node;
						dist_local_root[indices_on_path_to_local_root[i]] = indices_on_path_to_local_root.size() - i;
					}
					local_root[node - node_offset] = node;
					dist_local_root[node - node_offset] = 0;
					break;
				}
				else if (local_root[node - node_offset] != -1)
				{

					
					for (std::uint64_t i = 0; i < indices_on_path_to_local_root.size(); i++)
					{
						local_root[indices_on_path_to_local_root[i]] = local_root[node - node_offset];
						dist_local_root[indices_on_path_to_local_root[i]] = indices_on_path_to_local_root.size() - i + dist_local_root[node - node_offset];
					}
					break;
				}
				else
				{
					indices_on_path_to_local_root.push_back(node - node_offset);
					node = s[node - node_offset];
				}
			}
			
			if (node / num_local_vertices != rank)
			{
				std::uint64_t root = indices_on_path_to_local_root.back() + node_offset;
				for (std::uint64_t i = 0; i < indices_on_path_to_local_root.size() - 1; i++)
				{
					local_root[indices_on_path_to_local_root[i]] = root;
					dist_local_root[indices_on_path_to_local_root[i]] = indices_on_path_to_local_root.size() - i - 1;
					
				}
				local_root[root - node_offset] = root;
				dist_local_root[root - node_offset] = 0;
			}
			
		}
		
		for (int i = 0; i < num_local_vertices; i++)
		{
			if (s[i] == i + node_offset)
			{
				reduced_nodes.push_back({i + node_offset, i + node_offset, 0});
				//std::cout << "final node " << i + node_offset << " zeigt auf " <<  i + node_offset << " with dist " << 0 << std::endl;

			}
			else if (s[i] / num_local_vertices == rank)
			{
				removed_nodes.push_back({static_cast<std::uint64_t>(i + node_offset), static_cast<std::uint64_t>(local_root[i]), static_cast<std::int64_t>(dist_local_root[i])});
				//std::cout << "removed node " << i + node_offset << " zeigt auf " << local_root[i] << " with dist " << dist_local_root[i] << std::endl;

			}
			else
			{
				reduced_nodes.push_back({i + node_offset, s[i], 1});
				//std::cout << "final node " << i + node_offset << " zeigt auf " << s[i] << " with dist " << 1 << std::endl;

			}
			
		}
		
		
		
		timer.add_checkpoint("calculate_inices");


		
		std::uint64_t node_offset_rec = comm.exscan(kamping::send_buf((std::uint64_t)reduced_nodes.size()), kamping::op(kamping::ops::plus<>())).extract_recv_buffer()[0];
		std::uint64_t num_local_vertices_reduced = reduced_nodes.size();
		std::uint64_t num_global_vertices_rec = node_offset_rec + reduced_nodes.size(); //das hier stimmt nur für rank = size -1
		comm.bcast_single(kamping::send_recv_buf(num_global_vertices_rec), kamping::root(size-1));
		
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
		
		struct answer {
			std::uint64_t source;
			std::int64_t r;
		};

		std::function<answer(const std::uint64_t)> lambda = [&] (std::uint64_t request) {
			answer answer;
			if (source_map.contains(request))
			{
				answer.source = map_reduced_nodes_to_its_index[source_map[request]-node_offset] + node_offset_rec;
				answer.r = r_map[request];
			}
			else
			{
				answer.source = map_reduced_nodes_to_its_index[request - node_offset] + node_offset_rec;
				answer.r = 0;
			}
			return answer;
		};
		std::function<std::uint64_t(const std::uint64_t)> request_assignment =  [](std::uint64_t request) {return request;};
		std::vector<answer> recv_answers = request_reply(timer, request, num_packets_per_PE, lambda, comm, grid_comm, grid, aggregate, request_assignment);
		
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
			r_reduced[i] = reduced_nodes[i].r + recv_answers[packet_index].r;
			targetPEs_reduced[i] = targetPE;
			
			//std::cout << "node for recursion (i,s,r,targetPE): (" << i + prefix_sum_num_vertices_per_PE_reduced[rank] << "," << s_reduced[i] << "," << r_reduced[i] << "," << targetPEs_reduced[i] << ")" <<std::endl;

		}
		
		timer.add_info(std::string("average_num_local_vertices_reduced"), std::to_string(num_global_vertices_rec/size));

		
				//if (rank == 0) std::cout << "Instanzgröße von " << size * num_local_vertices << " auf " << prefix_sum_num_vertices_per_PE_reduced[size] * 100 / (size * num_local_vertices) << "% reduziert" << std::endl;

		timer.add_checkpoint("tree_rooting");

		
		std::vector<std::uint64_t> local_rulers(s_reduced.size());

		
		std::vector<std::int64_t> ranks;
		
		if (call_ruling_set)
		{
			forest_irregular_optimized_ruling_set recursion(100, 10, grid, aggregate);
			recursion.start(s_reduced, r_reduced, targetPEs_reduced, node_offset_rec, num_global_vertices_rec, comm, grid_comm, s_reduced);
			ranks = recursion.result_dist;
		}
		else
		{
			irregular_pointer_doubling algorithm(s_reduced, r_reduced, targetPEs_reduced, grid, node_offset_rec, num_global_vertices_rec);
			ranks = algorithm.start(comm, grid_comm);
		}
		/*
		if (num_global_vertices_rec / size < 10000)
		{
			irregular_pointer_doubling algorithm(s_reduced, r_reduced, targetPEs_reduced, grid, node_offset_rec, num_global_vertices_rec);
			ranks = algorithm.start(comm, grid_comm);
		}
		else
		{
			forest_irregular_optimized_ruling_set recursion(100, 10, grid, aggregate);
			recursion.start(s_reduced, r_reduced, targetPEs_reduced, node_offset_rec, num_global_vertices_rec, comm, grid_comm, s_reduced);
			ranks = recursion.result_dist;
		}*/
		
		
		
		
		timer.add_checkpoint("calculate_final_ranks");

		
		std::vector<std::int64_t> final_ranks(num_local_vertices);
		
		
		for (std::uint64_t i = 0; i < num_local_vertices_reduced; i++)
		{
			final_ranks[reduced_nodes[i].i - node_offset] = ranks[i];
		}
		
		for (std::uint64_t i = 0; i < removed_nodes.size(); i++)
		{
			final_ranks[removed_nodes[i].i - node_offset] =  final_ranks[removed_nodes[i].source - node_offset] + removed_nodes[i].r;
		}
		/*
		for (int i = 0; i < num_local_vertices; i++)
			std::cout << i + node_offset << " hat finalen rank " << final_ranks[i] << std::endl;*/
		
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
	std::string info = "";

	std::uint64_t node_offset;
	std::uint64_t num_local_vertices;
	std::int32_t rank, size;
};