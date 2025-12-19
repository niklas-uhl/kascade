#pragma once

#include <queue>

#include "forest_irregular_pointer_doubling.cpp"
#include "../helper_functions.cpp"
#include "forest_irregular_optimized_ruling_set.cpp"

class forest_regular_optimized_ruling_set //this is for trees
{
	struct packet{
		std::uint64_t ruler_source;
		std::uint64_t destination;
		std::uint32_t distance;
	};
	
	struct result{
			std::uint64_t root;
			std::int64_t distance;
	};
	
	public:
	
	forest_regular_optimized_ruling_set(std::uint64_t comm_rounds, std::uint32_t num_iterations, bool grid, bool aggregate)
	{
		this->comm_rounds = comm_rounds;
		this->num_iterations = num_iterations;
		this->grid = grid;
		this->aggregate = aggregate;
	}
	
	void add_timer_info(std::string info)
	{
		this->info = "\"" + info + "\"";
	}
	
	void start(std::vector<std::uint64_t> s, kamping::Communicator<>& comm, karam::mpi::GridCommunicator& grid_comm)
	{
		std::vector<std::string> categories = {"local_work", "communication", "other"};
		timer timer("graph_umdrehen", categories, "local_work", "forest_regular_optimized_ruling_set");
		
		timer.add_info(std::string("comm_rounds"), std::to_string(comm_rounds));
		timer.add_info(std::string("grid"), std::to_string(grid));
		timer.add_info(std::string("num_iterations"), std::to_string(num_iterations));
		timer.add_info(std::string("aggregate"), std::to_string(aggregate));
		
		if (info.size() > 0)
			timer.add_info(std::string("additional_info"), info);

			
		
		size = comm.size();
		rank = comm.rank();
		num_local_vertices = s.size();
		node_offset = rank * num_local_vertices;
		//now turn around s array
		timer.add_info(std::string("num_local_vertices"), std::to_string(num_local_vertices));
		
		std::vector<std::uint64_t> local_roots(0); //local indices
		
		struct edge{
			std::uint64_t source;
			std::uint64_t destination;
		};
		std::vector<std::int32_t> num_packets_per_PE(size,0);
		std::vector<std::int32_t> send_displacements(size + 1,0);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			if (i + node_offset == s[i])
			{
				local_roots.push_back(i);
			}
			else
			{
				std::int32_t targetPE = s[i] / num_local_vertices;
				num_packets_per_PE[targetPE]++;
			}
		}
		calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
		std::vector<edge> send_edges(send_displacements[size]);
		
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			if (i + node_offset == s[i])
				continue;
			
			std::int32_t targetPE = s[i] / s.size();
			std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			
			send_edges[packet_index].source = i + node_offset;
			send_edges[packet_index].destination = s[i];
		}
		
		std::vector<edge> local_edges = alltoall(timer, send_edges, num_packets_per_PE, comm, grid_comm, grid);
		
		std::vector<std::uint64_t> edges_per_node(num_local_vertices,0);
		
		for (std::uint64_t i = 0; i < local_edges.size(); i++)
		{
			edges_per_node[local_edges[i].destination - node_offset]++;
		}
		
		edges = std::vector<std::uint64_t>(local_edges.size());
		bounds = std::vector<std::uint64_t>(num_local_vertices+1);
		for (std::uint64_t i = 1; i < num_local_vertices + 1; i++)
			bounds[i] = bounds[i-1] + edges_per_node[i-1];
		std::fill(edges_per_node.begin(), edges_per_node.end(), 0);
		for (std::uint64_t i = 0; i < local_edges.size(); i++)
		{
			std::uint64_t target_node = local_edges[i].destination - node_offset;
			std::uint64_t packet_index = bounds[target_node] + edges_per_node[target_node]++;
			edges[packet_index] = local_edges[i].source;
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
		for (int i = 0; i < edges.size(); i++)
			std::cout << edges[i] << ",";
		std::cout << std::endl;
		std::cout << "\nund alle kanten:";
		for (int i = 0; i < num_local_vertices; i++)
			for (int j = bounds[i]; j < bounds[i+1]; j++)
				std::cout << "(" << i + node_offset << "," << edges[j] << "),";
		std::cout << std::endl;*/
		
		std::vector<std::uint64_t> mst(num_local_vertices);
		std::iota(mst.begin(),mst.end(),node_offset); 
		std::vector<std::uint64_t> del(num_local_vertices,0);
		
		
		//definitely_rulers are the local_roots that belongs to trees of size > 1
		definitely_rulers = std::vector<std::uint64_t>(0);
		for (std::uint64_t i = 0; i < local_roots.size(); i++)
		{
			std::uint64_t maybe_ruler = local_roots[i];
						
			if (unmask(bounds[maybe_ruler+1]) - unmask(bounds[maybe_ruler]) > 0)
			{
				mark_as_ruler(maybe_ruler);
				definitely_rulers.push_back(maybe_ruler);
				//std::cout << maybe_ruler + node_offset<< " ist auf jeden fall ruler" << std::endl;
			}
			else
			{
				mark_as_reached(maybe_ruler);
			}
		}
		
		
		std::uint64_t num_packages_per_iteration = edges.size() / comm_rounds;
		//std::cout << rank << " schickt so so viele packets pro iteration " << num_packages_per_iteration << ", bei insgesamt " << edges.size() << " edges und " << local_roots.size() << " local_roots und davon " << definitely_rulers.size() << " roots mit grad > 1 und " << num_local_vertices << " num_local_vertices"<< std::endl;
		std::vector<std::uint64_t> local_rulers(0);
		std::vector<std::uint64_t> rulers_to_send_packages_in_this_iteration(0);
		
		std::uint64_t num_packages_sent = 0;
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
		timer.add_checkpoint("pakete_verfolgen");

		while (num_packages_sent < num_packages_per_iteration)
		{
			std::uint64_t ruler = get_next_ruler();
			
			//std::cout << rank << " startet wave mit folgendem ruler: " << ruler + node_offset << std::endl;
			
			rulers_to_send_packages_in_this_iteration.push_back(ruler);
			num_packages_sent += unmask(bounds[ruler+1]) - unmask(bounds[ruler]);
			local_rulers.push_back(ruler);
			mark_as_ruler(ruler);
			for (std::uint64_t i = unmask(bounds[ruler]); i < unmask(bounds[ruler+1]); i++)
			{
				std::int32_t targetPE = calculate_targetPE(edges[i]);
				num_packets_per_PE[targetPE]++;
			}
			
		}
		calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);

		std::vector<packet> out_buffer(send_displacements[size]);
		
		for (std::uint64_t ruler : rulers_to_send_packages_in_this_iteration)
		{
			for (std::uint64_t i = unmask(bounds[ruler]); i < unmask(bounds[ruler+1]); i++)
			{				
				std::int32_t targetPE = calculate_targetPE(edges[i]);
				std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
				
				out_buffer[packet_index].ruler_source = ruler + node_offset;
				out_buffer[packet_index].destination = edges[i];
				out_buffer[packet_index].distance = 1;
			}
			
		}
		
	
		
		
		//diese queue beinhaltet ruler, die noch ihre pakete senden müssen
		//aber weil outbuffer schon voll, diese erst in späteren runden sie senden
		std::queue<packet> packets_to_forward;
		std::uint64_t packets_to_forward_out_degree = 0;
		timer.add_checkpoint("pakete_verfolgen");

		for (std::uint32_t iteration = 0; iteration < comm_rounds+2; iteration++)
		{
			//std::cout << rank << " in  iteration " << iteration << " hat out_buffer size von " << out_buffer.size() << std::endl;
			/*
			std::cout << rank << " in iteration " << iteration << " with following out_buffer:";
			for (int i = 0; i < out_buffer.size(); i++)
				std::cout << packet_to_string(out_buffer[i]) << ", ";
			std::cout << std::endl;
			*/
			std::vector<packet> recv_buffer = alltoall(timer, out_buffer, num_packets_per_PE, comm, grid_comm, grid);
			/*
			std::cout << rank << " in iteration " << iteration << " with following recv_buffer:";
			for (int i = 0; i < recv_buffer.size(); i++)
				std::cout << packet_to_string(recv_buffer[i]) << ", ";
			std::cout << std::endl;
			*/
			
			std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
			
			for (packet& packet: recv_buffer)
			{
				std::uint64_t local_index = packet.destination - node_offset;
				mst[local_index] = packet.ruler_source;
				del[local_index] = packet.distance;
				if (is_ruler(local_index))
				{
					
				}
				else
				{
					mark_as_reached(local_index);
					packets_to_forward.push(packet);
					packets_to_forward_out_degree += unmask(bounds[local_index+1]) - unmask(bounds[local_index]);
				}
			}
			//jetzt noch ruler in die queue hinzufügen, damit queue genug elemente hat
			
			
			//std::cout << rank << " in iteration " << iteration << " queue hat out degree " << packets_to_forward_out_degree << " und ich brauche " << num_packages_per_iteration << std::endl;
			
			rulers_to_send_packages_in_this_iteration.resize(0);
			if (packets_to_forward_out_degree < num_packages_per_iteration)
			{
				std::uint64_t rulers_to_send_packages_in_this_iteration_out_degree = 0;
				
				while (rulers_to_send_packages_in_this_iteration_out_degree < num_packages_per_iteration - packets_to_forward_out_degree)
				{
					std::uint64_t ruler = get_next_ruler();
				
					if (ruler == -1)
						break;
					
					//std::cout << rank << " in iteration " << iteration << " fügt folgenden ruler in queue " << ruler << std::endl;
					local_rulers.push_back(ruler);
					mark_as_ruler(ruler);
					rulers_to_send_packages_in_this_iteration_out_degree += unmask(bounds[ruler+1]) - unmask(bounds[ruler]);
					rulers_to_send_packages_in_this_iteration.push_back(ruler);
				}
			}
			
			//jetzt werden erst die pakete aus der queue geschickt und dann die weiteren ruler
			//aber zuerst wieder zählen wie viele pakete wohin
			
			std::vector<packet> packets(0);
			
			int degree = packets_to_forward_out_degree - num_packages_per_iteration;
			degree = degree < 0 ? 0 : degree;
			while (packets_to_forward_out_degree > degree)
			{
				packet packet = packets_to_forward.front();
				packets_to_forward.pop();
				
				//std::cout << rank << " in iteration " << iteration << " popped folgendes element aus queue " << packet_to_string(packet) << std::endl;

				packets.push_back(packet);
				
				std::uint64_t local_index = packet.destination - node_offset;
				packets_to_forward_out_degree -= unmask(bounds[local_index+1]) - unmask(bounds[local_index]);
				for (std::uint64_t i = unmask(bounds[local_index]); i < unmask(bounds[local_index+1]); i++)
				{
					std::int32_t targetPE = calculate_targetPE(edges[i]);
					num_packets_per_PE[targetPE]++;
				}
			}
			for (std::uint64_t local_index : rulers_to_send_packages_in_this_iteration)
			{
				for (std::uint64_t i = unmask(bounds[local_index]); i < unmask(bounds[local_index+1]); i++)
				{
					std::int32_t targetPE = calculate_targetPE(edges[i]);
					num_packets_per_PE[targetPE]++;
				}
			}
			calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
			out_buffer.resize(send_displacements[size]);
			for (packet& packet : packets)
			{
				std::uint64_t local_index = packet.destination - node_offset;
				//std::cout << rank << " jetzt wird packet geschickt mit node " << local_index << std::endl;
				for (std::uint64_t i = unmask(bounds[local_index]); i < unmask(bounds[local_index+1]); i++)
				{
					std::int32_t targetPE = calculate_targetPE(edges[i]);
					std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
					
					out_buffer[packet_index].ruler_source = packet.ruler_source;
					out_buffer[packet_index].destination = edges[i];
					out_buffer[packet_index].distance = packet.distance + 1;
				}
			}
			for (std::uint64_t local_index : rulers_to_send_packages_in_this_iteration)
			{
				//std::cout << rank << " jetzt wird neuer ruler geschickt naemlich " << local_index << std::endl;

				for (std::uint64_t i = unmask(bounds[local_index]); i < unmask(bounds[local_index+1]); i++)
				{
					std::int32_t targetPE = calculate_targetPE(edges[i]);
					std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
					
					out_buffer[packet_index].ruler_source = local_index + node_offset;
					out_buffer[packet_index].destination = edges[i];
					out_buffer[packet_index].distance = 1;
				}
			}

			
			
			
			
		}
		
		
		timer.add_checkpoint("rekursion_vorbereiten");

		
		/*
		std::cout << rank << " with mst: ";
		for (int i = 0; i < num_local_vertices; i++)
			std::cout << mst[i] << " ";
		std::cout << std::endl;
		std::cout << rank << " with del: ";
		for (int i = 0; i < num_local_vertices; i++)
			std::cout << del[i] << " ";
		std::cout << std::endl;
		std::cout << rank << " with local rulers: ";
		for (int i = 0; i < local_rulers.size(); i++)
			std::cout << local_rulers[i] + node_offset << " ";
		std::cout << std::endl;	
		*/
		
		
		std::uint64_t node_offset_rec = comm.exscan(kamping::send_buf((std::uint64_t)local_rulers.size()), kamping::op(kamping::ops::plus<>())).extract_recv_buffer()[0];
		std::uint64_t num_global_vertices_rec = node_offset_rec + local_rulers.size(); //das hier stimmt nur für rank = size -1
		comm.bcast_single(kamping::send_recv_buf(num_global_vertices_rec), kamping::root(size-1));
				
		
		timer.add_info("reduction_factor", std::to_string(num_local_vertices * size / num_global_vertices_rec));
		

		
		std::vector<std::uint64_t> map_ruler_to_its_index(num_local_vertices);
		std::vector<std::uint64_t> s_rec(local_rulers.size());
		std::vector<std::int64_t> r_rec(local_rulers.size());
		std::vector<std::uint32_t> targetPEs_rec(local_rulers.size());
	
		
		std::vector<std::uint64_t> requests(local_rulers.size());
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
		for (std::uint64_t i = 0; i < local_rulers.size(); i++)
		{
			map_ruler_to_its_index[local_rulers[i]] = i;
			std::int32_t targetPE = calculate_targetPE(mst[local_rulers[i]]);
			targetPEs_rec[i] = targetPE;
			num_packets_per_PE[targetPE]++;
		}
		calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
		for (std::uint64_t i = 0; i < local_rulers.size(); i++)
		{
			std::int32_t targetPE = calculate_targetPE(mst[local_rulers[i]]);
			std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			requests[packet_index] = mst[local_rulers[i]];
		}
		
		std::function<std::uint64_t(const std::uint64_t)> lambda = [&] (std::uint64_t request) { return map_ruler_to_its_index[request-node_offset] + node_offset_rec;};
		std::function<std::uint64_t(const std::uint64_t)> request_assignment =  [](std::uint64_t request) {return request;};


		std::vector<std::uint64_t> recv_answers = request_reply(timer, requests, num_packets_per_PE, lambda, comm, grid_comm, grid, aggregate, request_assignment);

		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
		for (std::uint64_t i = 0; i < local_rulers.size(); i++)
		{
			std::int32_t targetPE = calculate_targetPE(mst[local_rulers[i]]);
			std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			s_rec[i] = recv_answers[packet_index];
			r_rec[i] = del[local_rulers[i]];
			
			if (r_rec[i] == 0)
				s_rec[i] = i + node_offset_rec;
		}
		
		/*
		for (int i = 0; i < local_rulers.size(); i++)
			std::cout << i + prefix_sum_num_vertices_per_PE[rank] << " s_rec:" << s_rec[i] << ", r_rec:" << r_rec[i] << std::endl;
		*/
		timer.add_checkpoint("rekursion");
		timer.switch_category("other");

		std::vector<std::uint64_t> local_rulers_global_index = local_rulers;
		for (std::uint32_t i = 0; i < local_rulers.size(); i++)
			local_rulers_global_index[i] += node_offset;
		
		std::vector<std::uint64_t> recursive_global_index ;
		std::vector<std::int64_t> recursive_r;
		
		
		if (num_global_vertices_rec / size > 10000)
		{
			forest_irregular_optimized_ruling_set recursion(comm_rounds, num_iterations - 1, grid, aggregate);
			recursion.start(s_rec, r_rec, targetPEs_rec, node_offset_rec, num_global_vertices_rec, comm, grid_comm, local_rulers_global_index);
			
			recursive_global_index = recursion.result_root;
			recursive_r = recursion.result_dist;
		}
		else
		{
			forest_irregular_pointer_doubling recursion(s_rec, r_rec, targetPEs_rec, node_offset_rec, num_global_vertices_rec, local_rulers_global_index, grid, aggregate);
			recursion.start(comm, grid_comm);
			recursive_global_index = recursion.local_rulers;
			recursive_r = recursion.r;
		}
				
		
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
		timer.add_checkpoint("finalen_ranks_berechnen");
		timer.switch_category("local_work");


		result_dist = std::vector<std::int64_t>(num_local_vertices, 0);
		result_root = std::vector<std::uint64_t>(num_local_vertices);
		std::iota(result_root.begin(), result_root.end(), node_offset);
		
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			//vlt nur local roots skippen
			if (s[i] == i + node_offset && unmask(bounds[i]) == unmask(bounds[i+1]))
				continue;
			
			
			std::int32_t targetPE = calculate_targetPE(mst[i]);
			num_packets_per_PE[targetPE]++;
		}
		calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
		std::vector<std::uint64_t> request(num_local_vertices);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			if (s[i] == i + node_offset && unmask(bounds[i]) == unmask(bounds[i+1]))
				continue;
			
			std::int32_t targetPE = calculate_targetPE(mst[i]);
			std::uint64_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			request[packet_index] = mst[i];
		}
		
		struct answer{
			std::uint64_t global_root_index;
			std::int64_t distance;
		};
		std::function<answer(const std::uint64_t)> lambda2 = [&] (std::uint64_t request) { 
			answer answer;
			answer.global_root_index = recursive_global_index[map_ruler_to_its_index[request - node_offset]];
			answer.distance = recursive_r[map_ruler_to_its_index[request - node_offset]];
			return answer;
		};
		

		
		std::vector<answer> recv_answers_buffer = request_reply(timer, request, num_packets_per_PE, lambda2, comm, grid_comm, grid, aggregate, request_assignment);

		
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);

		
		
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			if (s[i] == i + node_offset && unmask(bounds[i]) == unmask(bounds[i+1]))
				continue;
			
			
			std::int32_t targetPE = calculate_targetPE(mst[i]);
			std::uint64_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			
			
			
			result_root[i] = recv_answers_buffer[packet_index].global_root_index;
			result_dist[i] = recv_answers_buffer[packet_index].distance + del[i];
			
			//std::cout << i + node_offset << " hat dist " << result_dist[i] << " von root " << result_root[i] << std::endl;

		}
		
		std::string save_dir = "forest_regular_ruling_set2";
		if (num_iterations == 2)
			save_dir = "forest_regular_ruling_set2_rec";
		timer.finalize(comm, "forest_synchron_ruling_set");


		
	}
	
	std::uint64_t get_next_ruler()
	{
		if (definitely_rulers.size() > 0)
		{
			std::uint64_t next_ruler = definitely_rulers.back();
			definitely_rulers.pop_back();
			return next_ruler;
		}
		//die roots der bäume der größe 1 sind eh schon als reached markiert
		while (ruler_index < num_local_vertices && (is_reached(ruler_index) || is_ruler(ruler_index) || (unmask(bounds[ruler_index+1])==unmask(bounds[ruler_index])))) ruler_index++;
				
		if (ruler_index == num_local_vertices)
			return -1;
		return ruler_index;
	}
	
	std::uint64_t ruler_index = 0;
	std::vector<std::uint64_t> definitely_rulers;
	
	
	std::int32_t calculate_targetPE(std::uint64_t global_index)
	{
		return unmask(global_index) / num_local_vertices;
	}
	
	std::string packet_to_string(packet packet)
	{
		return "(" + std::to_string(packet.ruler_source) + "," + std::to_string(packet.destination) + "," + std::to_string(packet.distance) + ")";
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
	std::vector<std::uint64_t> bounds;
	
	std::vector<std::int64_t> result_dist;
	std::vector<std::uint64_t> result_root;
	
	std::uint64_t comm_rounds;
	std::uint32_t num_iterations;
	bool grid;
	bool aggregate;
	std::string info = "";
};


	