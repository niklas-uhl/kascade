//this class is implemented like in the paper "ultimate parallel list ranking" to minimize number of communication steps in trade of local work

#include "irregular_pointer_doubling.cpp"
#include "irregular_ruling_set2.cpp"

class regular_ruling_set2
{
	struct packet{
		std::uint64_t ruler_source;
		std::uint64_t destination;
		std::uint32_t distance;
	};

	double c_direct = 0.78703;
	int num_iterations_direct = 3;
	
	double c_grid = 0.20366;
	int num_iterations_grid = 3;

	
	public:
	
	
	
	regular_ruling_set2(std::vector<std::uint64_t>& s, double dist_rulers, std::uint32_t num_iterations, bool grid)
	{
		this->s = s;
		this->dist_rulers = dist_rulers;
		this->num_iterations = num_iterations;
		this->grid = grid;
	}
	
	
	std::vector<std::int64_t> start(kamping::Communicator<>& comm, karam::mpi::GridCommunicator& grid_comm)
	{
		rank = comm.rank();
		size = comm.size();
		num_local_vertices = s.size();
		node_offset = rank * num_local_vertices;
		
		std::vector<std::string> categories = {"local_work", "communication", "other"};
		timer timer("pakete_verfolgen", categories, "local_work", "regular_ruling_set2");
		
		timer.add_info(std::string("dist_rulers"), std::to_string(dist_rulers));
		timer.add_info(std::string("num_local_vertices"), std::to_string(num_local_vertices));
		timer.add_info(std::string("iterations"), std::to_string(num_iterations));
		timer.add_info("grid", std::to_string(grid));

		
		//man kann ja wieder die ersten n/dist vielen nodes als ruler setzten. den ruler index speichern. wenn eine packet iteration durch ist, werden erreichte ruler gezählt und genau so viele neue ruler gemacht, in dem rulerindex erhöhrt wird. Dadruch wird nur ein einziges mal extra iteriert
		std::uint64_t out_buffer_size = num_local_vertices/dist_rulers;
		std::vector<packet> out_buffer(out_buffer_size);
		
		
		/*
		std::cout << rank << " mit successor array:\n";
		for (int i = 0; i < num_local_vertices; i++)
			std::cout << s[i] << " ";
		std::cout <<", es werden " << out_buffer_size << " pakete gesendet" << std::endl;*/

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
			std::int32_t targetPE = calculate_targetPE(s[ruler_index]);
			num_packets_per_PE[targetPE]++;
			ruler_index++;
		}
		
		
		
		calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
		
		
		for (std::uint64_t i = 0; i < out_buffer_size; i++)
		{
			std::uint64_t local_index = rulers_to_send_packages[i];
			
			std::int32_t targetPE = calculate_targetPE(s[local_index]);
			std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			
			
			out_buffer[packet_index].ruler_source = local_index + node_offset;
			out_buffer[packet_index].destination = s[local_index];
			out_buffer[packet_index].distance = 1;
			
			mark_as_ruler(local_index);
		}

		
		std::vector<std::uint64_t> mst(num_local_vertices);
		std::iota(mst.begin(),mst.end(),node_offset); 
		std::vector<std::int64_t> del(num_local_vertices,0);
		
		//timer.add_checkpoint("pakete_verfolgen");



		for (std::uint64_t iteration = 0; iteration < dist_rulers+1; iteration++)
		{
			
			/*
			std::cout << rank << " in iteration " << iteration << " with following packages:\n";
			for (packet& packet: out_buffer)
				std::cout << "(" << packet.ruler_source << "," << packet.destination << "," << packet.distance << "),";
			std::cout << std::endl;*/
			//timer.add_checkpoint("iteration " + std::to_string(iteration));
			std::vector<packet> recv_buffer = alltoall(timer, out_buffer, num_packets_per_PE, comm, grid_comm, grid);

			
			
			std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
			
			
			std::uint64_t num_forwarded_packages = 0;
			for (packet& packet: recv_buffer)
			{
				std::uint64_t local_index = packet.destination - node_offset;
				
				
				mark_as_reached(local_index);
				
				if (!is_final(local_index) && !is_ruler(local_index))
				{
					std::int32_t targetPE = calculate_targetPE(s[local_index]);
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
				std::int32_t targetPE = calculate_targetPE(s[ruler_index]);
				num_packets_per_PE[targetPE]++;
				
			}
			

			calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
			out_buffer.resize(send_displacements[size]); 
			//now packets are written
			
	 
			for (packet& packet: recv_buffer) //vlt compileranweisung, dass schleife oft ausgeführt wird?
			{
				std::uint64_t local_index = packet.destination - node_offset;
				mst[local_index] = packet.ruler_source;
				del[local_index] = packet.distance;//ich könnte beide arrays zusammenlegen
				
				
				//das könnte auch in eine bit-operation umgewandelt werden
				if (!is_final(local_index) && !is_ruler(local_index)) //if kosten auch viel zeit, aber da führt glaube kein weg vorbei...
				{
					std::int32_t targetPE = calculate_targetPE(s[local_index]); //targetPE mit shift berechnen, division kostet viel zeit
					std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
					
					
					
					out_buffer[packet_index].ruler_source = packet.ruler_source; 
					out_buffer[packet_index].destination = unmask(s[local_index]); //hier kann man den local index schicken, das spart platz
					out_buffer[packet_index].distance = packet.distance + 1;
					
				}	

			}
		

			
			for (std::uint64_t i = 0; i < rulers_to_send_packages.size(); i++)
			{
				std::uint64_t local_index = rulers_to_send_packages[i];
				
				std::int32_t targetPE = calculate_targetPE(s[local_index]);
				std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
				
				out_buffer[packet_index].ruler_source = local_index + node_offset;
				out_buffer[packet_index].destination = unmask(s[local_index]);
				out_buffer[packet_index].distance = 1;
				
				
			}
			
			/*
			//count how many nodes reached
			std::uint64_t num_nodes_reached = 0;
			for (std::uint64_t i = 0; i < num_local_vertices; i++)
				if (is_reached(i))
					num_nodes_reached++;
				
			if (rank == 0)  std::cout << "iteration " << iteration << " with left " << num_local_vertices - num_nodes_reached << std::endl;
			*/
		}

		
		timer.add_checkpoint("rekursion_vorbereiten");

		
		std::uint64_t node_offset_rec = comm.exscan(kamping::send_buf((std::uint64_t)local_rulers.size()), kamping::op(kamping::ops::plus<>())).extract_recv_buffer()[0];
		std::uint64_t num_global_vertices_rec = node_offset_rec + local_rulers.size(); //das hier stimmt nur für rank = size -1
		comm.bcast_single(kamping::send_recv_buf(num_global_vertices_rec), kamping::root(size-1));
				
		
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
		std::vector<std::uint64_t> recv_answers = request_reply(timer, requests, num_packets_per_PE, lambda, comm, grid_comm, grid);
		
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
		for (std::uint64_t i = 0; i < local_rulers.size(); i++)
		{
			std::int32_t targetPE = calculate_targetPE(mst[local_rulers[i]]);
			std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			s_rec[i] = recv_answers[packet_index];
			r_rec[i] = del[local_rulers[i]];
		}
		

		
		timer.add_checkpoint("rekursion");
		timer.switch_category("other");
		std::vector<std::int64_t> ranks;
		
		std::uint64_t n = num_local_vertices * size;
		std::uint64_t n_reduced = num_global_vertices_rec;
		
		timer.add_info(std::string("reduction_factor"), std::to_string(n/(double) n_reduced));

		//if (rank == 0) std::cout << n_reduced / size << " nodes per PE " << std::endl;
		//std::uint64_t start_rec = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
		
		if (n_reduced / size > 10000)
		{
			irregular_ruling_set2 algorithm(s_rec, r_rec, targetPEs_rec, dist_rulers, node_offset_rec, num_global_vertices_rec, num_iterations - 1, grid);

			ranks = algorithm.start(comm, grid_comm);
			
		}
		else
		{
			irregular_pointer_doubling algorithm(s_rec, r_rec, targetPEs_rec, grid, node_offset_rec, num_global_vertices_rec);
			ranks = algorithm.start(comm, grid_comm);
		}
		
		
		//std::uint64_t end_rec = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
		//if (rank == 0) std::cout << "time = " << end_rec - start_rec << " for iteration " << num_iterations << std::endl;
		
		timer.switch_category("local_work");
		timer.add_checkpoint("finalen_ranks_berechnen");
		


		
		//rank[i + node_offset] = rank[mst[i]] + del[i], and requests[i] = rank[mst[i]] is goal
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
		requests.resize(num_local_vertices);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::int32_t targetPE = calculate_targetPE(mst[i]);
			num_packets_per_PE[targetPE]++;
		}
		calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::int32_t targetPE = calculate_targetPE(mst[i]);
			std::uint64_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			requests[packet_index] = mst[i];
		}
	
		std::function<std::uint64_t(const std::uint64_t)> lambda2 = [&] (std::uint64_t request) { return ranks[map_ruler_to_its_index[request - node_offset]];};
		recv_answers = request_reply(timer, requests, num_packets_per_PE, lambda2, comm, grid_comm, grid);
		
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
		
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::int32_t targetPE = calculate_targetPE(mst[i]);
			std::uint64_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			del[i] = size * num_local_vertices - 1 - (del[i] + recv_answers[packet_index]);
		}
		
		std::string save_dir = "regular_ruling_set2";
		timer.finalize(comm, save_dir);

	
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
	
	std::int32_t calculate_targetPE(std::uint64_t global_index)
	{
		return unmask(global_index) / num_local_vertices;
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
	std::uint64_t node_offset;
	std::uint64_t num_local_vertices;
	std::uint64_t rank, size;
	std::vector<std::uint64_t> s; //s einfach immer übergeben, sonst wird da viel zu viel rumkpiert
	bool grid;
	double dist_rulers;
	int dist_rulers_rec;
	std::uint32_t num_iterations;
};


	