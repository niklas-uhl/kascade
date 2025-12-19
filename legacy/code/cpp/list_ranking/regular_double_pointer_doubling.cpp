#pragma once

#include "../helper_functions.cpp"


class regular_double_pointer_doubling
{

	
	public:
	

	regular_double_pointer_doubling(bool grid)
	{
		this->grid = grid;
	}
	
	std::vector<std::int64_t> start(std::vector<std::uint64_t> s, kamping::Communicator<>& comm, karam::mpi::GridCommunicator& grid_comm)
	{
		std::vector<std::string> categories = {"local_work", "communication"};
		timer timer("kanten_drehen", categories, "local_work", "double_pointer_doubling");
		
		num_local_vertices = s.size();

		timer.add_info(std::string("num_local_vertices"), std::to_string(num_local_vertices));
		timer.add_info("grid", std::to_string(grid));
		
		size = comm.size();
		rank = comm.rank();
	
		num_global_vertices = num_local_vertices * size;
		node_offset = num_local_vertices * rank;
		
		
		
		struct edge{
			std::uint64_t source;
			std::uint64_t destination;
		};
		std::vector<std::int64_t> r_s(num_local_vertices,1);
		std::vector<std::int32_t> num_packets_per_PE(size,0);
		std::vector<std::int32_t> send_displacements(size + 1,0);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			if (i + node_offset == s[i])
			{
				r_s[i]=0;
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
	
		std::vector<std::int64_t> r_p(num_local_vertices,0);
		std::vector<std::uint64_t> p(num_local_vertices);
		std::iota(p.begin(), p.end(), node_offset);

		for (std::uint64_t i = 0; i < local_edges.size(); i++)
		{
			p[local_edges[i].destination - node_offset] = local_edges[i].source;
			r_p[local_edges[i].destination - node_offset] = 1;
		}
		
		timer.add_checkpoint("start_doubling");
		
		struct packet {
			std::uint64_t target_node; //oberstes bit von target node = 0 --> packet an successor, oberstes bit = 1 --> packet an pred
			std::uint64_t new_p_or_S;
			std::int64_t weight;
		};
		

		std::vector<packet> packet_vector(2*num_local_vertices);

		
		
		std::int32_t max_iteration = std::log2(num_global_vertices) +2;
		for (std::uint32_t iteration = 0; iteration < max_iteration; iteration++)
		{
			/*
			std::cout << rank << " in iteration " << iteration << " with q: ";
			for (int i = 0; i < num_local_vertices; i++)
				std::cout <<"("<< i+node_offset << "," <<s[i] << ")" << " ";
			std::cout << "\n and with p:";
			for (int i = 0; i < num_local_vertices; i++)
				std::cout << p[i] << " ";
			std::cout << "\n and with r_s:";
			for (int i = 0; i < num_local_vertices; i++)
				std::cout << r_s[i] << " ";
			std::cout << "\n and with r_p:";
			for (int i = 0; i < num_local_vertices; i++)
				std::cout << r_p[i] << " ";
			std::cout << std::endl;*/
			
			std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
			for (std::uint64_t i = 0; i < num_local_vertices; i++)
			{
				if (r_s[i] == (((std::uint64_t) 1) << iteration))
				{
					std::uint32_t targetPE = s[i] / num_local_vertices;
					num_packets_per_PE[targetPE]++;
				}
				if (r_p[i] == (((std::uint64_t) 1) << iteration))
				{
					std::uint32_t targetPE = p[i] / num_local_vertices;
					num_packets_per_PE[targetPE]++;
				}
			}

			calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);

			packet_vector.resize(send_displacements[size]);

			for (std::uint64_t i = 0; i < num_local_vertices; i++)
			{
				if (r_s[i] == (((std::uint64_t) 1) << iteration))
				{
					std::uint32_t targetPE = s[i] / num_local_vertices;
					std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
					
					packet_vector[packet_index].target_node = s[i];
					packet_vector[packet_index].new_p_or_S = p[i];
					packet_vector[packet_index].weight = r_p[i];
				}
				if (r_p[i] == (((std::uint64_t) 1) << iteration))
				{
					std::uint32_t targetPE = p[i] / num_local_vertices;
					std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
					
					packet_vector[packet_index].target_node = mark(p[i],0);
					packet_vector[packet_index].new_p_or_S = s[i];
					packet_vector[packet_index].weight = r_s[i];
				}
			}

			std::vector<packet> recv_packet = alltoall(timer, packet_vector, num_packets_per_PE, comm, grid_comm, grid);

						
			//hier werden alle empfangenen pakete lokal eingetragen
			
			
			for (std::uint64_t i = 0; i < recv_packet.size(); i++)
			{
				std::uint64_t local_index = unmask(recv_packet[i].target_node) - node_offset;

				if (is_marked(recv_packet[i].target_node,0))//packet an predecessor
				{
					s[local_index] = recv_packet[i].new_p_or_S;
					r_s[local_index] += recv_packet[i].weight;
				}
				else
				{
					p[local_index] = recv_packet[i].new_p_or_S;
					r_p[local_index] += recv_packet[i].weight;
				}
			}
			
		}
		timer.finalize(comm, "double_pointer_doubling");
		
		
		return r_s;
		
	}
	/*
	std::vector<std::int64_t> start2(kamping::Communicator<>& comm, karam::mpi::GridCommunicator grid_comm)
	{
		std::vector<std::string> categories = {"local_work", "communication"};
		timer timer("start", categories, "local_work", "regular_pointer_doubling");
		
		timer.add_info(std::string("num_local_vertices"), std::to_string(num_local_vertices));
		timer.add_info("grid", std::to_string(grid));

		
		size = comm.size();
		rank = comm.rank();
	
		num_global_vertices = num_local_vertices * size;
		node_offset = num_local_vertices * rank;
		
		
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
		
		std::vector<std::uint64_t> p(num_local_vertices);
		std::iota(p.begin(), p.end(), node_offset);

		for (std::uint64_t i = 0; i < num_local_vertices; i++)
			p[local_edges[i].destination - node_offset] = local_edges[i].source;
		
		
		
		
		
		
		struct packet_s {
			std::uint64_t target_node;
			std::uint64_t new_predecessor;
			std::int64_t weight;
		};
		
		struct packet_p {
			std::uint64_t target_node;
			std::uint64_t new_successor;
			std::int64_t weight;
		};
		std::vector<std::int64_t> r_s(num_local_vertices,1);
		
		std::vector<std::int64_t> r_p(num_local_vertices,1);
		std::vector<bool> passive_s(num_local_vertices, false); //this means we do not need to send a packet to our sucessor
		std::vector<bool> passive_p(num_local_vertices, false); //this means we do not need to send a packet to our predecessor
		for (int i = 0; i < num_local_vertices; i++)
		{
			if (s[i] == i + node_offset)
			{
				r_s[i] = 0;
				
				//passive_s[i] = true;
				//passive_p[i] = true;
			}
			if (p[i] == i + node_offset)
			{
				r_p[i] = 0;
				
				//passive_p[i] = true;
				//passive_s[i] = true;
			}
		}
		std::cout << rank << " with s: ";
		for (int i = 0; i < num_local_vertices; i++)
			std::cout << s[i] << " ";
		std::cout << "\n and with p:";
		for (int i = 0; i < num_local_vertices; i++)
			std::cout << p[i] << " ";
		std::cout << "\n and with r_s:";
		for (int i = 0; i < num_local_vertices; i++)
			std::cout << r_s[i] << " ";
		std::cout << "\n and with r_p:";
		for (int i = 0; i < num_local_vertices; i++)
			std::cout << r_p[i] << " ";
		std::cout << std::endl;

		std::vector<packet_s> packet_s_vector(num_local_vertices);
		std::vector<packet_p> packet_p_vector(num_local_vertices);
		
		
		
		std::int32_t max_iteration = std::log2(num_global_vertices) + 2;
		for (std::int32_t iteration = 0; iteration < max_iteration; iteration++)
		{
			std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
			for (std::uint64_t i = 0; i < num_local_vertices; i++)
			{
				if (r_s[i] != my_pow(2,iteration))
					continue;
				
				std::uint32_t targetPE = s[i] / num_local_vertices;
				num_packets_per_PE[targetPE]++;
				
				

			}
			calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
			
			for (std::uint64_t i = 0; i < num_local_vertices; i++)
			{
				if (r_s[i] != my_pow(2,iteration))
					continue;
				
				std::uint32_t targetPE = s[i] / num_local_vertices;
				std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
				
				std::cout << "in iteration " << iteration << " node " << i + node_offset << " sends packet to " << s[i] << std::endl;
				
				packet_s_vector[packet_index].target_node = s[i];
				packet_s_vector[packet_index].new_predecessor = p[i];
				packet_s_vector[packet_index].weight = r_p[i];
			}
			
			std::vector<packet_s> recv_packet_s = alltoall(timer, packet_s_vector, num_packets_per_PE, comm, grid_comm, grid);

			
			std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
			for (std::uint64_t i = 0; i < num_local_vertices; i++)
			{
				if (r_p[i] != my_pow(2,iteration))
					continue;
				
				std::uint32_t targetPE = p[i] / num_local_vertices;
				num_packets_per_PE[targetPE]++;
				
			}
			calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);

			for (std::uint64_t i = 0; i < num_local_vertices; i++)
			{
				if (r_p[i] != my_pow(2,iteration))
					continue;
				
				std::cout << "in iteration " << iteration << " node " << i + node_offset << " sends packet to " << p[i] << std::endl;
				
				std::uint32_t targetPE = p[i] / num_local_vertices;
				std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;

				packet_p_vector[packet_index].target_node = p[i];
				packet_p_vector[packet_index].new_successor = s[i];
				packet_p_vector[packet_index].weight = r_s[i];
			}
			std::vector<packet_p> recv_packet_p = alltoall(timer, packet_p_vector, num_packets_per_PE, comm, grid_comm, grid);
			
			//hier werden alle empfangenen pakete lokal eingetragen
			
			std::fill(passive_p.begin(), passive_p.end(), true);
			std::fill(passive_s.begin(), passive_s.end(), true);

			for (int i = 0; i < recv_packet_s.size(); i++)
			{
				p[recv_packet_s[i].target_node - node_offset] = recv_packet_s[i].new_predecessor;
				r_p[recv_packet_s[i].target_node - node_offset] += recv_packet_s[i].weight;

				//passive_p[recv_packet_s[i].target_node - node_offset] = false;
			}
			for (int i = 0; i < recv_packet_p.size(); i++)
			{
				
				s[recv_packet_p[i].target_node - node_offset] = recv_packet_p[i].new_successor;
				r_s[recv_packet_p[i].target_node - node_offset] += recv_packet_p[i].weight;
				//passive_s[recv_packet_p[i].target_node - node_offset] = false;

			}
			
			std::cout << rank << " int iteration " << iteration << " with s: ";
			for (int i = 0; i < num_local_vertices; i++)
				std::cout << s[i] << " ";
			std::cout << "\n and with p:";
			for (int i = 0; i < num_local_vertices; i++)
				std::cout << p[i] << " ";
			std::cout << "\n and with r_s:";
			for (int i = 0; i < num_local_vertices; i++)
				std::cout << r_s[i] << " ";
			std::cout << "\n and with r_p:";
			for (int i = 0; i < num_local_vertices; i++)
				std::cout << r_p[i] << " ";
			std::cout << std::endl;
			
		}
		
		
		return r_s;
		
	}*/
	
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
	
	int my_pow(int a, int b)// return a^b
	{
		if (b == 0)
			return 1;
		return a*my_pow(a,b-1);
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
	
	
	private:
	bool grid;
	
	std::uint64_t num_local_vertices;

	
	std::int32_t rank;
	std::int32_t size;
	std::uint64_t num_global_vertices;
	std::uint64_t node_offset;

};