#pragma once

#include "../timer.cpp"

class grid_irregular_ruling_set2
{
	struct packet{
		std::uint64_t ruler_source;
		std::uint64_t destination;
		std::uint64_t distance;
		std::uint32_t ruler_PE;
	};


	
	public:
	
	grid_irregular_ruling_set2(std::vector<std::uint64_t>& s, std::vector<std::int64_t> r, std::vector<std::uint32_t> targetPEs, std::uint64_t dist_rulers, std::vector<std::uint64_t> prefix_sum_num_vertices_per_PE, int communication_mode)
	{
		this->s = s;
		this->dist_rulers = dist_rulers;
		this->r = r;
		this->targetPEs = targetPEs;
		this->prefix_sum_num_vertices_per_PE = prefix_sum_num_vertices_per_PE;
		this->communication_mode = communication_mode;
	}
	
	
	std::vector<std::int64_t> start(kamping::Communicator<>& comm, karam::mpi::GridCommunicator grid_comm)
	{
		
		
		rank = comm.rank();
		size = comm.size();
		num_local_vertices = s.size();
		node_offset = prefix_sum_num_vertices_per_PE[rank];
		
		std::uint64_t out_buffer_size = num_local_vertices/dist_rulers;
		std::vector<packet> out_buffer(out_buffer_size);
		
		std::vector<std::string> categories = {"local_work", "communication"};
		timer timer("ruler_pakete_senden", categories, "local_work", "grid_irregular_ruling_set2");
		
		timer.add_info(std::string("dist_rulers"), std::to_string(dist_rulers));
		timer.add_info(std::string("num_local_vertices"), std::to_string(num_local_vertices), true);
		

		std::vector<std::int32_t> num_packets_per_PE(size,0);
		std::vector<std::int32_t> send_displacements(size + 1,0);
		
		std::vector<uint64_t> local_rulers(0);
		std::vector<uint64_t> rulers_to_send_packages(out_buffer_size);
		std::uint64_t ruler_index = 0; //this means that first free rulers has an index >= ruler_index
		
		for (std::uint64_t i = 0; i < out_buffer_size; i++)
		{
			while (is_final(ruler_index)) ruler_index++;
			
			local_rulers.push_back(ruler_index);
			rulers_to_send_packages[i] = ruler_index;
			std::int32_t targetPE = targetPEs[ruler_index];
			num_packets_per_PE[targetPE]++;
			ruler_index++;
		}
		
		
		
		calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
		
		
		for (std::uint64_t i = 0; i < out_buffer_size; i++)
		{
			std::uint64_t local_index = rulers_to_send_packages[i];
			
			std::int32_t targetPE = targetPEs[local_index];
			std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			
			
			out_buffer[packet_index].ruler_source = local_index + node_offset;
			out_buffer[packet_index].destination = s[local_index];
			out_buffer[packet_index].distance = r[local_index];
			out_buffer[packet_index].ruler_PE = rank;
			
			mark_as_ruler(local_index);
		}

		
		std::vector<std::uint32_t> mst_PE(num_local_vertices, rank);
		std::vector<std::uint64_t> mst(num_local_vertices);
		std::iota(mst.begin(),mst.end(),node_offset); 
		std::vector<std::int64_t> del(num_local_vertices,0);
		
		
		for (std::uint64_t iteration = 0; iteration < dist_rulers + 3; iteration++)
		{
			/*
			std::cout << rank << " in iteration " << iteration << " with following packages:\n";
			for (packet& packet: out_buffer)
				std::cout << "(" << packet.ruler_source << "," << packet.destination << "," << packet.distance << "),";
			std::cout << std::endl;*/
			
			std::vector<packet> recv_buffer = alltoall(timer, out_buffer, num_packets_per_PE, comm, grid_comm, communication_mode);
	
			std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
			
			
			std::uint64_t num_forwarded_packages = 0;
			for (int i = 0; i < recv_buffer.size(); i++)
			{
				std::uint64_t local_index = recv_buffer[i].destination - node_offset;
				
				
				mark_as_reached(local_index);
				
				if (!is_final(local_index) && !is_ruler(local_index))
				{
					std::int32_t targetPE = targetPEs[local_index];
					num_packets_per_PE[targetPE]++;
					num_forwarded_packages++;
				}

			}
			
			
			
			//select num_rulers_to_send_packages new rulers if possible
			std::uint64_t num_rulers_to_send_packages = out_buffer_size > num_forwarded_packages ? out_buffer_size - num_forwarded_packages : 0;
			
			rulers_to_send_packages.resize(num_rulers_to_send_packages);
			for (std::uint64_t i = 0; i < num_rulers_to_send_packages; i++)
			{
				while (ruler_index < num_local_vertices && (is_final(ruler_index) || is_reached(ruler_index) || is_ruler(ruler_index))) ruler_index++;
				
				if (ruler_index == num_local_vertices)
				{
					rulers_to_send_packages.resize(i);
					break;
				}
				
				local_rulers.push_back(ruler_index);
				mark_as_ruler(ruler_index);
				rulers_to_send_packages[i] = ruler_index;
				std::int32_t targetPE = targetPEs[ruler_index];
				num_packets_per_PE[targetPE]++;
				
			}
			

			calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
			out_buffer.resize(send_displacements[size]); 
			//now packets are written
			
	 
			for (int i = 0; i < recv_buffer.size(); i++) 
			{
				std::uint64_t local_index = recv_buffer[i].destination - node_offset;
				mst[local_index] = recv_buffer[i].ruler_source;
				del[local_index] = recv_buffer[i].distance;
				mst_PE[local_index] = recv_buffer[i].ruler_PE;
				
		
				if (!is_final(local_index) && !is_ruler(local_index)) 
				{
					std::int32_t targetPE = targetPEs[local_index]; 
					std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
					
					
					
					out_buffer[packet_index].ruler_source = recv_buffer[i].ruler_source; 
					out_buffer[packet_index].destination = unmask(s[local_index]); 
					out_buffer[packet_index].distance = recv_buffer[i].distance + r[local_index];
					out_buffer[packet_index].ruler_PE = recv_buffer[i].ruler_PE;
				}	

			}
		

			
			for (std::uint64_t i = 0; i < rulers_to_send_packages.size(); i++)
			{
				std::uint64_t local_index = rulers_to_send_packages[i];
				
				std::int32_t targetPE = targetPEs[local_index];
				std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
				
				out_buffer[packet_index].ruler_source = local_index + node_offset;
				out_buffer[packet_index].destination = unmask(s[local_index]);
				out_buffer[packet_index].distance = r[local_index];
				out_buffer[packet_index].ruler_PE = rank;
				
			}
			
		}
		
		timer.add_checkpoint("rekursion_vorbereiten");

		//now just the global starting node is unreached, this node is also always a ruler
		std::vector<std::uint64_t> num_local_rulers_per_PE;
		timer.switch_category("communication");
		comm.allgather(kamping::send_buf(local_rulers.size()), kamping::recv_buf<kamping::resize_to_fit>(num_local_rulers_per_PE));
		timer.switch_category("local_work");
		std::vector<std::uint64_t> prefix_sum_num_rulers_per_PE(size + 1,0);
		for (std::uint32_t i = 1; i < size + 1; i++)
		{
			prefix_sum_num_rulers_per_PE[i] = prefix_sum_num_rulers_per_PE[i-1] + num_local_rulers_per_PE[i-1];
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
			//std::int32_t targetPE = targetPEs_rec[i];
			std::int32_t targetPE = mst_PE[local_rulers[i]];

			std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			requests[packet_index] = mst[local_rulers[i]];
		}
		
	
		std::function<std::uint64_t(const std::uint64_t)> lambda = [&] (std::uint64_t request) { return map_ruler_to_its_index[request-node_offset] + prefix_sum_num_rulers_per_PE[rank];};
		std::vector<std::uint64_t> recv_answers = request_reply(timer, requests, num_packets_per_PE, lambda, comm, grid_comm, communication_mode);
		
		timer.switch_category("local_work");
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
		for (std::uint64_t i = 0; i < local_rulers.size(); i++)
		{
			std::int32_t targetPE = targetPEs_rec[i];
			std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			s_rec[i] = recv_answers[packet_index];
			r_rec[i] = del[local_rulers[i]];
		}
		
		timer.add_checkpoint("rekursion");

		grid_irregular_pointer_doubling algorithm(s_rec, r_rec, targetPEs_rec, prefix_sum_num_rulers_per_PE, communication_mode);
		std::vector<std::int64_t> ranks = algorithm.start(comm, grid_comm);
		timer.add_checkpoint("finalen_ranks_berechnen");
		
		/*
		std::cout << "PE " << rank << " with s_rec:";
		for (int i = 0; i < local_rulers.size(); i++)
			std::cout << s_rec[i] << ",";
		std::cout << std::endl;
		std::cout << "PE " << rank << " with r_rec:";
		for (int i = 0; i < local_rulers.size(); i++)
			std::cout << r_rec[i] << ",";
		std::cout << std::endl;
		std::cout << "PE " << rank << " with targetPEs_rec:";
		for (int i = 0; i < local_rulers.size(); i++)
			std::cout << targetPEs_rec[i] << ",";
		std::cout << std::endl;*/
		
		//pefix sum num rulers wurde ja umbenannt
		
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
		requests.resize(num_local_vertices);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::int32_t targetPE = mst_PE[i];
			num_packets_per_PE[targetPE]++;
		}
		calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::int32_t targetPE = mst_PE[i];
			std::uint64_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			requests[packet_index] = mst[i];
		}
		
		std::function<std::uint64_t(const std::uint64_t)> lambda2 = [&] (std::uint64_t request) { return ranks[map_ruler_to_its_index[request - node_offset]];};
		recv_answers = request_reply(timer, requests, num_packets_per_PE, lambda2, comm, grid_comm, communication_mode);

		
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
		
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::int32_t targetPE = mst_PE[i];
			std::uint64_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			del[i] = prefix_sum_num_vertices_per_PE[size] - 1 - (del[i] + recv_answers[packet_index]);
		}
		
		timer.finalize(comm, "grid_irregular_ruling_set2");


	
		return del;
		
		
	}
	
	
	
	
	std::string packet_to_string(packet packet)
	{
		return "(" + std::to_string(packet.ruler_source) + "," + std::to_string(packet.destination) + "," + std::to_string(packet.distance) + ")";
	}
	
	bool packet_will_be_forwarded(packet& packet)
	{
		std::uint64_t local_index = packet.destination - node_offset;
		return !is_final(local_index) && !is_ruler(local_index);
	}
	
	void mark_as_ruler(std::uint64_t local_index)
	{
		s[local_index] =  mark(s[local_index],0);
	}
	
	bool is_ruler(std::uint64_t local_index)
	{
		return is_marked(s[local_index],0);
	}
	
	void mark_as_reached(std::uint64_t local_index)
	{
		s[local_index] =  mark(s[local_index],1);
	}
	
	bool is_reached(std::uint64_t local_index)
	{
		return is_marked(s[local_index],1);
	}
	
	bool is_final(std::uint64_t local_index)
	{
		return local_index + node_offset == unmask(s[local_index]);
	}

	
	//sagen wir mal, die obersten 4 bit stehen zum markieren frei
	
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
	
	
	private:
	int communication_mode;
	
	std::uint64_t node_offset;
	std::uint64_t num_local_vertices;
	std::uint64_t rank, size;
	std::vector<std::uint64_t> s;
	std::vector<std::int64_t> r;
	std::vector<std::uint32_t> targetPEs;
	std::vector<std::uint64_t> prefix_sum_num_vertices_per_PE;
	std::uint64_t dist_rulers;
};