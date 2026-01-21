#include "kamping/checking_casts.hpp"
#include "kamping/collectives/allgather.hpp"
#include "kamping/communicator.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/environment.hpp"
#include "kamping/named_parameters.hpp"

#include "../timer.cpp"
#include "regular_pointer_doubling.cpp"



/*
here every PE must have the same number of nodes aka the length of successors is the same
also dist_rulers >= 3
*/
class regular_ruling_set_rec
{
	struct packet_rec {
		std::int32_t ruler_source;
		std::int32_t destination;
		std::int32_t ruler_distance;
	};

	struct node_packet_rec {
		std::int32_t source;
		std::int32_t destination;
		std::int32_t distance;
	};
	
	public:
	
	regular_ruling_set_rec(std::vector<std::uint64_t>& successors, std::vector<std::int64_t>& ranks, std::uint64_t local_index_final, std::int32_t dist_rulers, int iterations, bool grid)
	{
		s = successors;
		r = ranks;
		num_local_vertices = s.size();
		num_local_rulers = num_local_vertices / dist_rulers;
		distance_rulers = dist_rulers;
		local_index_final_node = local_index_final;
		this->iterations = iterations;
		this->grid = grid;
	}
	
	
	std::vector<std::int64_t> start(kamping::Communicator<>& comm, karam::mpi::GridCommunicator grid_comm)
	{
		
		std::vector<std::string> categories = {"local_work", "communication"};
		timer timer("ruler_pakete_senden", categories, "local_work", "regular_ruling_set_rec");
		
		timer.add_info("num_local_vertices", std::to_string(num_local_vertices));
		timer.add_info("dist_rulers", std::to_string(distance_rulers));
		timer.add_info("iterations", std::to_string(iterations));
		timer.add_info("grid", std::to_string(grid));
		
		
		size = comm.size();
		rank = comm.rank();
		num_global_vertices = num_local_vertices * size;
		node_offset = num_local_vertices * rank;
		
		/*
		std::cout << rank << " mit successor array:\n";
		for (int i = 0; i < num_local_vertices; i++)
			std::cout <<"(" << s[i] << "," << r[i] << "),";
		std::cout <<", rulers sind die ersten " << num_local_rulers << " nodes" << std::endl;*/
		
		std::vector<std::int32_t> num_packets_per_PE(size,0);
		std::vector<std::int32_t> send_displacements(size + 1,0);
		
		std::int32_t dist_from_last_ruler = -1;
		
		for (std::int32_t local_index = 0; local_index < num_local_rulers; local_index++)
		{
			if (!is_final(local_index))
			{
				std::int32_t targetPE = calculate_targetPE(s[local_index]);
				num_packets_per_PE[targetPE]++;
			}

		}
	
		calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
		//now packets are written
		std::vector<packet_rec> out_buffer(send_displacements[size]);
		for (std::int32_t local_index = 0; local_index < num_local_rulers; local_index++)
		{
			if (!is_final(local_index))
			{
				std::int32_t targetPE = calculate_targetPE(s[local_index]);
				std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
				
				out_buffer[packet_index].ruler_source = local_index + node_offset;
				out_buffer[packet_index].destination = s[local_index];
				out_buffer[packet_index].ruler_distance = r[local_index];
			}
		}
		
		std::vector<packet_rec> recv_buffer = alltoall(timer, out_buffer, num_packets_per_PE, comm, grid_comm, grid);

		std::vector<std::int64_t> mst(num_local_vertices, -1); //previous ruler
		std::vector<std::int64_t> del(num_local_vertices, -1); //dist to previous ruler
		
		std::int32_t num_reached_nodes = 0;
		bool more_nodes_reached = false;
		timer.add_checkpoint("pakete_verfolgen");

	
		std::int64_t max_iteration = distance_rulers * std::log(num_global_vertices / distance_rulers);
		std::int64_t iteration=0;
		
		while (iteration++ < max_iteration  || any_PE_has_work(comm, grid_comm, timer, more_nodes_reached, grid))
		{
			//if (rank ==  0) std::cout << "iteration " << iteration << " mit " << num_local_vertices - num_reached_nodes << std::endl;
			
			
			
			std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
			more_nodes_reached = false;
			
			for (packet_rec& packet: recv_buffer)
			{
				
				
				
				if (packet_will_be_forwarded(packet))
				{
					std::int32_t local_index = packet.destination - node_offset;
					std::int32_t target_node = s[local_index];
					std::int32_t targetPE = calculate_targetPE(target_node);
					num_packets_per_PE[targetPE]++;
				}
			}
			calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
	
			out_buffer.resize(send_displacements[size]);
		
			for (packet_rec& packet: recv_buffer)
			{
				std::int32_t local_index = packet.destination - node_offset;
				
				mst[local_index] = packet.ruler_source;
				del[local_index] = packet.ruler_distance;
				 
				
				num_reached_nodes++;
				more_nodes_reached = true;
				
				if (packet_will_be_forwarded(packet))
				{
					std::int32_t target_node = s[local_index];
					std::int32_t targetPE = calculate_targetPE(target_node);
					std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
					
					out_buffer[packet_index].ruler_source = packet.ruler_source;
					out_buffer[packet_index].destination = target_node;
					out_buffer[packet_index].ruler_distance = packet.ruler_distance + r[local_index];
				}
			}
			recv_buffer = alltoall(timer, out_buffer, num_packets_per_PE, comm, grid_comm, grid);

		}
		
		
		timer.add_checkpoint("rekursion_vorbereiten");

		
		//wir müssen noch die insgesamte distanz der anfangsknoten vor dem unreached ruler zählen und dann die gesamtzahl als rank setzten
		
		std::int64_t sum_r = 0;
		for (std::uint64_t local_index = 0; local_index < num_local_rulers; local_index++)
			sum_r += r[local_index];
		
		std::int64_t local_partial_distance = 0;
		for (std::int32_t local_index = num_local_rulers; local_index < num_local_vertices; local_index++)
		{
			sum_r += r[local_index];
			if (mst[local_index] == -1)
				local_partial_distance += r[local_index];
		}
		

		std::vector<std::int64_t> recv_partial_distance = allgatherv(timer, local_partial_distance, comm, grid_comm, grid);

		
		std::int32_t sum = 0; 
		for (std::int32_t i = 0; i < size; i++)
			sum+= recv_partial_distance[i];
	
		std::vector<std::int64_t> recv_partial_sum_r = allgatherv(timer, sum_r, comm, grid_comm, grid);

		sum_r = 0; 
		for (std::int32_t i = 0; i < size; i++)
			sum_r+= recv_partial_sum_r[i];
		
		std::int32_t local_index_final_node_rec = -1;
		for (std::int32_t local_index = 0; local_index < num_local_rulers; local_index++)
			if (mst[local_index] == -1)
			{
				
				local_index_final_node_rec = local_index;
				mst[local_index] = local_index + node_offset;
				del[local_index] = sum;
			}
			
			
		std::vector<std::uint64_t> s_rec(num_local_rulers);
		std::vector<std::int64_t> r_rec(num_local_rulers);
		for (std::int32_t local_index = 0; local_index < num_local_rulers; local_index++)
		{
			std::int32_t next_ruler = mst[local_index];
			std::int32_t next_ruler_PE = calculate_targetPE(next_ruler);
			s_rec[local_index] = next_ruler - next_ruler_PE * num_local_vertices + next_ruler_PE * num_local_rulers;
			r_rec[local_index] = del[local_index];
		}
		
	
		/*
		std::cout << rank << " mit s_r array:\n";
		for (int i = 0; i < s_rec.size(); i++)
			std::cout <<"(" << s_rec[i] << "," << r_rec[i] << "),";
		std::cout <<", rulers sind die ersten " << num_local_rulers << " nodes" << std::endl;
		*/
		
	
		timer.add_checkpoint("rekursion");

		//regular_pointer_doubling algorithm(s_rec, r_rec, local_index_final_node_rec, communication_mode);
		//std::vector<std::int64_t> result = algorithm.start(comm, grid_comm);
		
		std::vector<std::int64_t> result;
		if (iterations == 1)
		{
			regular_pointer_doubling algorithm(s_rec, r_rec, local_index_final_node_rec, grid);
			result = algorithm.start(comm, grid_comm);
		}
		else
		{
			regular_ruling_set_rec algorithm(s_rec, r_rec, local_index_final_node_rec, distance_rulers, iterations-1, grid);
			result = algorithm.start(comm, grid_comm);
		}
		
		timer.add_checkpoint("finalen_ranks_berechnen");

		for (std::int32_t local_index = 0; local_index < num_local_rulers; local_index++)
		{
			result[local_index] = sum_r  - result[local_index];
		}
	
	
		std::vector<std::int64_t> all_results = allgatherv(timer, result, comm, grid_comm, grid);


		//jetzt müssen werte wiederhergestellt werden
		//dafür müssen alle ruler auf alle PE verteilt werden
		
		result.resize(num_local_vertices);
		
	
		std::vector<node_packet_rec> local_unreached_nodes(num_local_vertices - num_reached_nodes + 1);
		std::int32_t local_unreached_nodes_index = 0;
		
		for (std::int32_t local_index = num_local_rulers; local_index < num_local_vertices; local_index++)
		{
			if (mst[local_index] != -1)
			{
				std::int32_t prev_ruler = mst[local_index];
				std::int32_t prev_ruler_PE = calculate_targetPE(prev_ruler);
				std::int32_t rank_prev_ruler = all_results[prev_ruler - num_local_vertices * prev_ruler_PE + num_local_rulers * prev_ruler_PE];
				result[local_index] =  rank_prev_ruler - del[local_index];
			}
			else 
			{
				local_unreached_nodes[local_unreached_nodes_index].source = local_index + node_offset;
				local_unreached_nodes[local_unreached_nodes_index].destination = s[local_index];
				local_unreached_nodes[local_unreached_nodes_index].distance = r[local_index];
				local_unreached_nodes_index++;
			}
		}
		
		local_unreached_nodes.resize(local_unreached_nodes_index);
		std::vector<node_packet_rec> global_unreached_nodes = allgatherv(timer, local_unreached_nodes, comm, grid_comm, grid);

		
		std::unordered_map<std::int32_t, std::int32_t> node_map; //node_map[source] = destination, für jeden unreached node (source,destination)
		std::unordered_map<std::int32_t, std::int32_t> has_pred_map; //has_pred_map[source] = true, if any node source has any pred
		std::unordered_map<std::int32_t, std::int32_t> r_map;
		std::int32_t start_node;
		
		for (std::int32_t i = 0; i < global_unreached_nodes.size(); i++)
		{
			node_packet_rec node = global_unreached_nodes[i];
			node_map[node.source] = node.destination;
			r_map[node.source] = node.distance;
			has_pred_map[node.destination] = true;
		}
		
		for (std::int32_t i = 0; i < global_unreached_nodes.size(); i++)
		{
			if (!has_pred_map.contains(global_unreached_nodes[i].source))
			{
				start_node = global_unreached_nodes[i].source;
				break;
			}
		}
		std::int32_t node = start_node;
		std::int32_t node_rank = sum_r; //rank of final node, needs to be changed!
		while (node_map.contains(node))
		{
			if (calculate_targetPE(node) == rank)
				result[node - node_offset] = node_rank;
			node_rank-= r_map[node];
			node = node_map[node];
		}
		
		timer.finalize(comm, "regular_ruling_set_rec_" + std::to_string(iterations) + "iterations");
		
		return result;
	}
	
	bool is_global_ruler(std::int32_t global_index)
	{
		std::int32_t targetPE = calculate_targetPE(global_index);
		return is_ruler(global_index - targetPE * num_local_vertices);
	}
	

	bool is_final(std::uint64_t local_index)
	{
		return local_index + node_offset == s[local_index];
	}
	
	void calculate_send_displacements_and_reset_num_packets_per_PE(std::vector<std::int32_t>& send_displacements, std::vector<std::int32_t>& num_packets_per_PE)
	{
		send_displacements[0]=0;
		for (std::int32_t i = 1; i < size + 1; i++)
			send_displacements[i] = send_displacements[i-1] + num_packets_per_PE[i-1];
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
	}
	
	std::int32_t calculate_targetPE(std::uint64_t global_index)
	{
		return global_index / num_local_vertices;
	}
	
	bool is_ruler(std::uint64_t local_index)
	{
		return local_index < num_local_rulers;
	}
	
	//a packet will be forwardef iff it doesn't point to ruler and doesn't point to final node
	bool packet_will_be_forwarded(packet_rec packet)
	{
		std::uint64_t local_index = packet.destination - node_offset;
		return !is_ruler(local_index) && !is_final(local_index);
	}


	
	private: 
	bool grid;
	
	std::vector<std::int64_t> r;
	std::vector<std::uint64_t> s;
	std::uint64_t num_local_vertices;
	std::uint64_t num_local_rulers;
	std::uint64_t distance_rulers;
	std::uint64_t local_index_final_node;
	std::uint32_t iterations;
	
	std::uint64_t size;
	std::uint64_t rank;
	std::uint64_t num_global_vertices;
	std::uint64_t node_offset;
	
	
};