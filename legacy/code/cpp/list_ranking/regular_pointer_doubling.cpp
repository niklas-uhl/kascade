#pragma once

#include "../helper_functions.cpp"


class regular_pointer_doubling
{
	struct node_request {
		std::uint64_t mst;
	};

	struct answer {
		std::int64_t r_of_mst;
		std::uint64_t mst_of_mst;
		bool passive_of_mst;
	};
	
	
	
	public:
	
	//if this PE has final node, then final node is set to a valid value, otherweise it is -1
	regular_pointer_doubling(std::vector<std::uint64_t>& successors, std::vector<std::int64_t>& ranks, std::uint64_t local_index_final_node, bool grid)
	{
		this->grid = grid;
		s = successors;
		num_local_vertices = s.size();
		q = s;
		r = ranks;

		passive = std::vector<bool>(num_local_vertices, false);
		
		if (local_index_final_node != -1)
			passive[local_index_final_node] = true;
	}
	
	
	regular_pointer_doubling(std::vector<std::uint64_t>& successors, kamping::Communicator<>& comm, bool grid)
	{
		this->grid = grid;
		s = successors;
		num_local_vertices = s.size();
		q = s;
		node_offset = num_local_vertices * comm.rank();
		
		r = std::vector<std::int64_t>(num_local_vertices, 1);
		passive = std::vector<bool>(num_local_vertices, false);
		
		for (std::int32_t local_index = 0; local_index < num_local_vertices; local_index++)
			if (q[local_index] == local_index + node_offset)
			{
				r[local_index] = 0;
				passive[local_index] = true;
			}
	}
	
	regular_pointer_doubling(std::vector<std::uint64_t>& successors, kamping::Communicator<>& comm, bool grid, bool aggregate)
	{
		this->aggregate = aggregate;
		this->grid = grid;
		s = successors;
		num_local_vertices = s.size();
		q = s;
		node_offset = num_local_vertices * comm.rank();
		
		r = std::vector<std::int64_t>(num_local_vertices, 1);
		passive = std::vector<bool>(num_local_vertices, false);
		
		for (std::int32_t local_index = 0; local_index < num_local_vertices; local_index++)
			if (q[local_index] == local_index + node_offset)
			{
				r[local_index] = 0;
				passive[local_index] = true;
			}
	}
	
	void add_timer_info(std::string info)
	{
		this->info = "\"" + info + "\"";
	}
	
	
	std::vector<std::int64_t> start(kamping::Communicator<>& comm, karam::mpi::GridCommunicator& grid_comm)
	{
		std::vector<std::string> categories = {"local_work", "communication"};
		timer timer("start", categories, "local_work", "regular_pointer_doubling");
		
		timer.add_info(std::string("num_local_vertices"), std::to_string(num_local_vertices));
		timer.add_info("grid", std::to_string(grid));
		timer.add_info("aggregate", std::to_string(aggregate));

		if (info.size() > 0)
			timer.add_info(std::string("additional_info"), info);

		
		size = comm.size();
		rank = comm.rank();
		
		/*
		std::cout << rank << " mit successor array:\n";
		for (int i = 0; i < num_local_vertices; i++)
			std::cout << s[i] << " ";
		std::cout <<std::endl;
		*/
		num_global_vertices = num_local_vertices * size;
		node_offset = num_local_vertices * rank;
		
		std::vector<node_request> requests(num_local_vertices);
		std::vector<answer> answers(num_local_vertices);

		
		std::vector<std::int32_t> num_packets_per_PE(size,0);
		std::vector<std::int32_t> send_displacements(size + 1,0);
		
		std::int32_t max_iteration = std::log2(num_global_vertices) + 2;
		bool more_nodes_reached = true;
		//for (std::int32_t iteration = 0; iteration < max_iteration; iteration++)
		while (any_PE_has_work(comm, timer, more_nodes_reached))
		{
			more_nodes_reached = false;
			//timer.add_checkpoint("iteration " + std::to_string(iteration));
			
			//zuerst request packets gezählt
			std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
			for (std::int32_t local_index = 0; local_index < num_local_vertices;local_index++)
			{
				if (!passive[local_index])
				{
					more_nodes_reached = true;
					std::int32_t targetPE = calculate_targetPE(q[local_index]);
					num_packets_per_PE[targetPE]++;
				}
			}
			calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
			requests.resize(send_displacements[size]);
			
			//dann requests gefülllt
			for (std::int32_t local_index = 0; local_index < num_local_vertices;local_index++)
			{
				if (!passive[local_index])
				{
					std::int32_t targetPE = calculate_targetPE(q[local_index]);
					std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
					requests[packet_index].mst = q[local_index];
				}
			}
			
			std::function<answer(const node_request)> lambda = [&] (node_request request) { 
				std::int32_t local_index = request.mst - node_offset;
				
				answer answer;
				answer.r_of_mst = r[local_index];
				answer.mst_of_mst = q[local_index];
				answer.passive_of_mst = passive[local_index];
				return answer;
			};
			std::function<std::uint64_t(const node_request)> request_assignment =  [](node_request request) {return request.mst;};

			std::vector<answer> recv_answers = request_reply(timer, requests, num_packets_per_PE, lambda, comm, grid_comm, grid, aggregate, request_assignment);
			
		

				//dann answers eingetragen
			std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
			for (std::int32_t local_index = 0; local_index < num_local_vertices;local_index++)
			{
				if (!passive[local_index])
				{
					std::int32_t targetPE = calculate_targetPE(q[local_index]);
					std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
				
					q[local_index] = recv_answers[packet_index].mst_of_mst;
					r[local_index] = r[local_index] + recv_answers[packet_index].r_of_mst;
					passive[local_index] = recv_answers[packet_index].passive_of_mst;
				}
			}	
		}
		timer.finalize(comm, "regular_pointer_doubling");

		/*
		std::cout << rank << " mit rank array:\n";
		for (int i = 0; i < num_local_vertices; i++)
			std::cout << r[i] << " ";
		std::cout <<std::endl;
		*/
		return r;
		
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
	
	
	
	bool grid;
	bool aggregate = false;
	
	std::uint64_t num_local_vertices;
	std::uint64_t final_node;
	std::vector<std::uint64_t> s;
	std::vector<std::uint64_t> q;
	std::vector<std::int64_t> r;
	std::vector<bool> passive;
	
	std::int32_t rank;
	std::int32_t size;
	std::uint64_t num_global_vertices;
	std::uint64_t node_offset;
	
	std::string info = "";


};