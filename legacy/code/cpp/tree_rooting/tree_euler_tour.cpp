/*
This class calculates from a given sucessor array of a regular tree with the euler tour tree rooting
*/

#include "../list_ranking/irregular_ruling_set2.cpp"

class tree_euler_tour
{

	public:
	
	tree_euler_tour(kamping::Communicator<>& comm, std::vector<std::uint64_t>& s, std::uint32_t dist_rulers, bool grid)
	{
		this->grid = grid;
		num_local_vertices = s.size();
		size = comm.size();
		rank = comm.rank();
		node_offset = rank * num_local_vertices;
		this->dist_rulers = dist_rulers;
		

	}
	
	void add_timer_info(std::string info)
	{
		this->info = "\"" + info + "\"";
	}
	
	std::vector<std::int64_t> start(kamping::Communicator<>& comm, karam::mpi::GridCommunicator& grid_comm, std::vector<std::uint64_t>& s)
	{
		std::vector<std::string> categories = {"local_work", "communication", "other"};
		timer timer("graph umdrehen", categories, "local_work", "tree_euler_tour");
		
		timer.add_info(std::string("dist_rulers"), std::to_string(dist_rulers));
		timer.add_info(std::string("num_local_vertices"), std::to_string(num_local_vertices));
		timer.add_info("grid", std::to_string(grid));
		
		if (info.size() > 0)
			timer.add_info(std::string("additional_info"), info);
		
		std::int64_t local_root = -1;
		
		//first calculate adjacency array by also turning edges
		struct edge{
			std::uint64_t source;
			std::uint64_t destination;
		};
		std::vector<std::int32_t> num_packets_per_PE(size,0);
		std::vector<std::int32_t> send_displacements(size + 1,0);
		for (std::uint64_t i = 0; i < s.size(); i++)
		{
			if (i + node_offset == s[i])
			{
				local_root = i;
				continue;
			}
			std::int32_t targetPE = s[i] / s.size();
			num_packets_per_PE[targetPE]++;		
		}
		calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
		std::vector<edge> edges(s.size());
		
		for (std::uint64_t i = 0; i < s.size(); i++)
		{
			if (i + node_offset == s[i])
				continue;
			
			std::int32_t targetPE = s[i] / s.size();
			std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			
			edges[packet_index].source = i + node_offset;
			edges[packet_index].destination = s[i];
		}
		timer.switch_category("communication");
		
		//auto recv = comm.alltoallv(kamping::send_buf(edges), kamping::send_counts(num_packets_per_PE)).extract_recv_buffer();
		
		auto recv = alltoall(timer, edges, num_packets_per_PE, comm, grid_comm, grid);

		
		timer.switch_category("local_work");
	
		all_edges = std::vector<std::uint64_t>(recv.size() + s.size());
		std::vector<std::uint64_t> edges_per_node(s.size(),0);
		
		for (std::uint64_t i = 0; i < recv.size(); i++)
		{
			edges_per_node[recv[i].destination - node_offset]++;
		}
		for (std::uint64_t i = 0; i < s.size(); i++)
			edges_per_node[i]++; //for only not turned edge, since it was originally a tree
		
		bounds = std::vector<std::uint64_t>(s.size() + 1, 0);
		for (std::uint64_t i = 1; i <= s.size(); i++)
			bounds[i] = bounds[i-1] + edges_per_node[i-1];
		std::fill(edges_per_node.begin(), edges_per_node.end(), 0);
		for (std::uint64_t i = 0; i < s.size(); i++)
		{
			std::uint64_t packet_index = bounds[i] + edges_per_node[i]++;
			all_edges[packet_index] = s[i];
			
		}
		for (std::uint64_t i = 0; i < recv.size(); i++)
		{
			std::uint64_t target_node = recv[i].destination - node_offset;
			std::uint64_t packet_index = bounds[target_node] + edges_per_node[target_node]++;
			all_edges[packet_index] = recv[i].source;
		}
		
		
		timer.add_checkpoint("edge_weights");
		all_edges_weights = std::vector<std::int64_t>(recv.size() + s.size(), -1);

		
		// graph sucessfully added all back edges to adjacency array
		num_local_edges = all_edges.size();
		
		
		std::uint64_t node_offset_rec = comm.exscan(kamping::send_buf((std::uint64_t)num_local_edges), kamping::op(kamping::ops::plus<>())).extract_recv_buffer()[0];
		std::uint64_t num_global_vertices_rec = node_offset_rec + num_local_edges; //das hier stimmt nur für rank = size -1
		comm.bcast_single(kamping::send_recv_buf(num_global_vertices_rec), kamping::root(size-1));
				
		
		//idea, if we want the following edge of (a,b) in the euler tour, we need to fing the edge (b,a) in adjacency array to answer this
		//so we sort adjacency array to perfom binary sorts, so for every node i the values all_edges[bounds[i]], ..., all_edges[bounds[i+1]-1] are in ascending order
		
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
			std::sort(&all_edges[bounds[i]], &all_edges[bounds[i+1]]);
		
		for (std::uint64_t i = 0; i < s.size(); i++)
		{
			std::uint64_t dynamic_lower_bound = bounds[i];
			std::uint64_t dynamic_upper_bound = bounds[i+1];
			
			
			while (dynamic_upper_bound - dynamic_lower_bound > 1)
			{
				std::uint64_t middle = (dynamic_lower_bound + dynamic_upper_bound) / 2;
				
				if (all_edges[middle] > s[i])
					dynamic_upper_bound = middle;
				else
					dynamic_lower_bound = middle;
				
			}
			std::uint64_t result = dynamic_lower_bound;
			
			all_edges_weights[result] = 1;
		}
	
		timer.add_checkpoint("transform_into_successor_arr");

		
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);

		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		for (std::uint64_t j = bounds[i]; j < bounds[i+1]; j++)
		{
			std::int32_t targetPE = all_edges[j] / num_local_vertices;
			num_packets_per_PE[targetPE]++;	
		}
		calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
		std::vector<edge> request(send_displacements[size]);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		for (std::uint64_t j = bounds[i]; j < bounds[i+1]; j++)
		{
			std::int32_t targetPE = all_edges[j] / num_local_vertices;
			std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			
			request[packet_index].source = i + node_offset;
			request[packet_index].destination = all_edges[j];
			

			
		}
		
		std::function<std::uint64_t(const edge)> lambda = [&](edge edge) {
			std::uint64_t lower_bound = bounds[edge.destination - node_offset];
			std::uint64_t upper_bound = bounds[edge.destination - node_offset + 1];
			
			std::uint64_t dynamic_lower_bound = lower_bound;
			std::uint64_t dynamic_upper_bound = upper_bound;
			
			
			while (dynamic_upper_bound - dynamic_lower_bound > 1)
			{
				std::uint64_t middle = (dynamic_lower_bound + dynamic_upper_bound) / 2;
				
				if (all_edges[middle] > edge.source)
					dynamic_upper_bound = middle;
				else
					dynamic_lower_bound = middle;
				
			}
			std::uint64_t result = dynamic_lower_bound; //all_edges[result] == recv_buffer[i].source
			
			return node_offset_rec + lower_bound + ((result - lower_bound + 1) % (upper_bound - lower_bound));
		};
		std::vector<std::uint64_t> recv_answers = request_reply(timer, request, num_packets_per_PE, lambda, comm, grid_comm, grid);

	
		/*
		timer.switch_category("communication");
			
		auto recv_request = comm.alltoallv(kamping::send_buf(request), kamping::send_counts(num_packets_per_PE));
		timer.switch_category("local_work");
		std::vector<edge> recv_buffer = recv_request.extract_recv_buffer();

		std::vector<std::uint64_t> answers(recv_buffer.size());
		for (std::uint64_t i = 0; i < recv_buffer.size(); i++)
		{
			std::uint64_t lower_bound = bounds[recv_buffer[i].destination - node_offset];
			std::uint64_t upper_bound = bounds[recv_buffer[i].destination - node_offset + 1];
			
			std::uint64_t dynamic_lower_bound = lower_bound;
			std::uint64_t dynamic_upper_bound = upper_bound;
			
			
			while (dynamic_upper_bound - dynamic_lower_bound > 1)
			{
				std::uint64_t middle = (dynamic_lower_bound + dynamic_upper_bound) / 2;
				
				if (all_edges[middle] > recv_buffer[i].source)
					dynamic_upper_bound = middle;
				else
					dynamic_lower_bound = middle;
				
			}
			std::uint64_t result = dynamic_lower_bound; //all_edges[result] == recv_buffer[i].source
			
			
			
			answers[i] = node_offset_rec + lower_bound + ((result - lower_bound + 1) % (upper_bound - lower_bound));
			
			
		}
		
		
		timer.switch_category("communication");

		auto recv_answers = comm.alltoallv(kamping::send_buf(answers), kamping::send_counts(recv_request.extract_recv_counts())).extract_recv_buffer();

*/


		timer.switch_category("local_work");
		std::vector<std::uint64_t> s_edges(num_local_edges);
	
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
			
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		for (std::uint64_t j = bounds[i]; j < bounds[i+1]; j++)
		{
			std::int32_t targetPE = all_edges[j] / num_local_vertices;
			std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			
			s_edges[j] = recv_answers[packet_index];
			
			
		}
		
		if (local_root != -1)
		{
			std::uint64_t dynamic_lower_bound = bounds[local_root];
			std::uint64_t dynamic_upper_bound = bounds[local_root+1];
			
			
			while (dynamic_upper_bound - dynamic_lower_bound > 1)
			{
				std::uint64_t middle = (dynamic_lower_bound + dynamic_upper_bound) / 2;
				
				if (all_edges[middle] > local_root + node_offset)
					dynamic_upper_bound = middle;
				else
					dynamic_lower_bound = middle;
				
			}
			std::uint64_t result = dynamic_lower_bound;
			
			s_edges[result] = result + node_offset_rec;
			all_edges_weights[result] = 0;
		}
		
		/*
		std::cout << "PE " << rank << " with s arr: ";
		for (int i = 0; i < num_local_vertices; i++)
			std::cout << s[i] << ",";
		std::cout << std::endl;
		std::cout << "PE " << rank << " with bounds arr: ";
		for (int i = 0; i < bounds.size(); i++)
			std::cout << bounds[i] << ",";
		std::cout << std::endl;
		std::cout << "PE " << rank << " with all_edges arr: ";
		for (int i = 0; i < all_edges.size(); i++)
			std::cout << all_edges[i] << ",";
		std::cout << std::endl;
		std::cout << "PE " << rank << " with s_edges arr: ";
		for (int i = 0; i < s_edges.size(); i++)
			std::cout << s_edges[i] << ",";
		std::cout << std::endl;
		std::cout << "PE " << rank << " with all_edges_weights arr: ";
		for (int i = 0; i < all_edges_weights.size(); i++)
			std::cout << all_edges_weights[i] << ",";
		std::cout << std::endl;
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		for (std::uint64_t j = bounds[i]; j < bounds[i+1]; j++)
			std::cout << "(" << i + node_offset<< "," << all_edges[j] << "," << all_edges_weights[j] << "),";
		std::cout << std::endl;*/
		
		std::vector<std::uint32_t> targetPEs(num_local_edges);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		for (std::uint64_t j = bounds[i]; j < bounds[i+1]; j++)
			targetPEs[j] = all_edges[j] / num_local_vertices; 
		
		
		timer.add_checkpoint("ruling_set");
		timer.switch_category("other");


		//irregular_pointer_doubling algorithm(s_edges,all_edges_weights,targetPEs,prefix_sum_num_edges_per_PE);
		irregular_ruling_set2 algorithm(s_edges, all_edges_weights, targetPEs, dist_rulers, node_offset_rec, num_global_vertices_rec, 1, grid);
		algorithm.add_timer_info(info);

		std::vector<std::int64_t> ranks = algorithm.start(comm, grid_comm);
		timer.switch_category("local_work");
		
		timer.add_checkpoint("final_ranks_berechnen");

		
		std::vector<std::int64_t> final_ranks(num_local_vertices);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
			final_ranks[i] = ranks[bounds[i]];
		
		/*
		std::cout << "PE " << rank << " with rank array:";
		for (int i = 0; i < final_ranks.size(); i++)
			std::cout << final_ranks[i] << ",";
		std::cout << std::endl;
		*/
		timer.finalize(comm, "euler_tour");
		
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
	
	bool grid = false;
	
	std::uint64_t size;
	std::uint64_t rank;
	std::uint64_t num_local_vertices;
	std::uint64_t node_offset;
	std::uint32_t dist_rulers;
	
	std::uint64_t num_local_edges;
	
	std::vector<std::uint64_t> all_edges;
	std::vector<std::int64_t> all_edges_weights;
	std::vector<std::uint64_t> bounds;
	std::string info = "";

	
};