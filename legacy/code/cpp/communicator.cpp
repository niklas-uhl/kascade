

#pragma once
	
	template<typename packet>
	static std::vector<packet> alltoall(timer& timer, std::vector<packet>& send_buf, std::vector<std::int32_t>& send_counts, kamping::Communicator<>& comm, karam::mpi::GridCommunicator& grid_comm, bool grid)
	{
		if (grid)
			return alltoall_grid(timer, send_buf, send_counts, comm, grid_comm);
		else
			return alltoall_normal(timer, send_buf, send_counts, comm);
	}
	
	template<typename request, typename answer>
	static std::vector<answer> request_reply(timer& timer, std::vector<request>& requests, std::vector<std::int32_t>& send_counts, std::function<answer(const request)>& lambda, kamping::Communicator<>& comm, karam::mpi::GridCommunicator& grid_comm, bool grid)
	{
		if (grid)
			return request_reply_grid<request,answer>(timer, requests, send_counts, lambda, comm, grid_comm);
		else
			return request_reply_normal<request,answer>(timer, requests, send_counts, lambda, comm);
	}
	
	template<typename request, typename answer>
	static std::vector<answer> request_reply(timer& timer, std::vector<request>& requests, std::vector<std::int32_t>& send_counts, std::function<answer(const request)>& lambda, kamping::Communicator<>& comm, karam::mpi::GridCommunicator& grid_comm, bool grid, bool aggregate, std::function<std::uint64_t(const request)>& request_assingment)
	{
		if (grid)
		{
			if (aggregate)
				return request_reply_aggregate_grid<request,answer>(timer, requests, request_assingment, send_counts, lambda, comm, grid_comm);
			else
				return request_reply_grid<request,answer>(timer, requests, send_counts, lambda, comm, grid_comm);
		}
		else
		{
			if (aggregate)
				return request_reply_aggregate_normal<request,answer>(timer, requests, request_assingment, send_counts, lambda, comm);
			else
				return request_reply_normal<request,answer>(timer, requests, send_counts, lambda, comm);
		}
	}
	
	template<typename packet>
	static std::vector<packet> allgatherv(timer& timer, std::vector<packet>& send_buf, kamping::Communicator<>& comm, karam::mpi::GridCommunicator grid_comm, bool grid)
	{
		if (grid)
			return allgatherv_grid(timer, send_buf, comm, grid_comm);
		else
			return allgatherv_normal(timer, send_buf, comm);
	}
	
	template<typename packet>
	static std::vector<packet> allgatherv(timer& timer, packet send_buf, kamping::Communicator<>& comm, karam::mpi::GridCommunicator grid_comm, bool grid)
	{
		if (grid)
			return allgatherv_grid(timer, std::vector<packet>(1,send_buf), comm, grid_comm);
		else
			return allgatherv_normal(timer, std::vector<packet>(1,send_buf), comm);
	}
	
	
	template<typename packet>
	static std::vector<packet> allgatherv_grid(timer& timer, std::vector<packet> send_buf, kamping::Communicator<>& comm, karam::mpi::GridCommunicator grid_comm)
	{
		timer.switch_category("communication");
		std::vector<packet> to_return = grid_comm.allgatherv(comm, send_buf);
		timer.switch_category("local_work");
		return to_return;
	}
	
	template<typename packet>
	static std::vector<packet> allgatherv_normal(timer& timer, std::vector<packet> send_buf, kamping::Communicator<>& comm)
	{
		timer.switch_category("communication");
		std::vector<packet> to_return;
		comm.allgatherv(kamping::send_buf(send_buf), kamping::recv_buf<kamping::resize_to_fit>(to_return));
		timer.switch_category("local_work");
		return to_return;
	}

	template<typename packet>
	static std::vector<packet> alltoall_normal(timer& timer, std::vector<packet>& send_buf, std::vector<std::int32_t>& send_counts, kamping::Communicator<>& comm)
	{
		timer.switch_category("communication");
		std::vector<packet> recv =  comm.alltoallv(kamping::send_buf(send_buf), kamping::send_counts(send_counts)).extract_recv_buffer();
		timer.switch_category("local_work");
		return recv;
	}
	
	template<typename packet>
	static std::vector<packet> alltoall_grid(timer& timer, std::vector<packet>& send_buf, std::vector<std::int32_t>& send_counts, kamping::Communicator<>& comm, karam::mpi::GridCommunicator& grid_comm)
	{
		timer.switch_category("communication");
		auto recv = my_grid_all_to_all(send_buf, send_counts, grid_comm, comm).extract_recv_buffer();
		timer.switch_category("local_work");

		std::vector<packet> recv_vector(recv.size());
		for (std::uint64_t i = 0; i < recv.size(); i++)
			recv_vector[i] = recv[i].payload();
		
		return recv_vector;
	}
	

	
	template<typename request, typename answer>
	static std::vector<answer> request_reply_aggregate_grid(timer& timer, std::vector<request> requests, std::function<std::uint64_t(const request)> request_assingment, std::vector<std::int32_t> send_counts, std::function<answer(const request)> lambda, kamping::Communicator<>& comm, karam::mpi::GridCommunicator& grid_comm)
	{		
		struct request_info {
			request request_value;
			std::int32_t targetPE;
		};
		
		std::unordered_map<std::uint64_t, request_info> request_map;
		
		
		std::uint64_t index = 0;
		for (std::int32_t p = 0; p < comm.size(); p++)
		for (std::uint64_t i = 0; i < send_counts[p]; i++)
		{
			if (!request_map.contains(request_assingment(requests[index])))
			{
				request_map[request_assingment(requests[index])] = {requests[index],  p};
			}
			index++;
		}
		std::vector<std::int32_t> send_counts_row = std::vector<std::int32_t>(grid_comm.row_comm().size(),0);
		for (const auto& [key, value] : request_map)
		{
			std::int32_t targetPE = grid_comm.proxy_col_index(static_cast<std::size_t>(value.targetPE));
			send_counts_row[targetPE]++;
		}
		std::vector<std::int32_t> send_displacements_row = std::vector<std::int32_t>(grid_comm.row_comm().size()+1);
		send_displacements_row[0]=0;
		for (std::uint32_t p_row = 0; p_row < grid_comm.row_comm().size(); p_row++)
			send_displacements_row[p_row+1] = send_displacements_row[p_row]+send_counts_row[p_row];
		std::fill(send_counts_row.begin(), send_counts_row.end(),0);
		struct send_request {
			request request_value;
			std::uint64_t key;
		};
		karam::utils::default_init_vector<karam::mpi::IndirectMessage<send_request>> send_buf_row(send_displacements_row[grid_comm.row_comm().size()]);; 
		for (const auto& [key, value] : request_map)
		{
			std::int32_t targetPE = grid_comm.proxy_col_index(static_cast<std::size_t>(value.targetPE));
			std::uint64_t packet_index = send_displacements_row[targetPE] + send_counts_row[targetPE]++;
			
			send_buf_row[packet_index] = karam::mpi::IndirectMessage<send_request>(static_cast<std::uint32_t>(comm.rank()),static_cast<std::uint32_t>(value.targetPE),{value.request_value, key});
		}
		
		timer.switch_category("communication");
		auto mpi_result_rowwise = grid_comm.row_comm().alltoallv(kamping::send_buf(send_buf_row),kamping::send_counts(send_counts_row));	
		timer.switch_category("local_work");

		auto rowwise_recv_buf    = mpi_result_rowwise.extract_recv_buffer();
/*
		std::cout << comm.rank() << " erhält in indirekton folgende request: ";
		for (int i = 0; i < rowwise_recv_buf.size(); i++)
			std::cout << rowwise_recv_buf[i].payload().request_value << " ";
		std::cout << std::endl;*/
		
		request_map.clear();
		for (std::uint64_t i = 0; i < rowwise_recv_buf.size(); i++)
		{
			if (!request_map.contains(rowwise_recv_buf[i].payload().key))
			{	
				request_map[rowwise_recv_buf[i].payload().key] = {rowwise_recv_buf[i].payload().request_value, static_cast<std::int32_t>(rowwise_recv_buf[i].get_destination())};
			}
		}
		/*
		std::cout << comm.rank() << " hat aggregatet request in indirektion: ";
		for (const auto& [key, value] : request_map)
		{
			std::cout << value.request_value << " ";
		}		
		std::cout << std::endl;*/
		std::vector<std::int32_t> send_counts_col = std::vector<std::int32_t>(grid_comm.col_comm().size(),0);
		for (const auto& [key, value] : request_map)
		{
			std::int32_t targetPE = grid_comm.proxy_row_index(static_cast<std::size_t>(value.targetPE));
			send_counts_col[targetPE]++;
		}
		std::vector<std::int32_t> send_displacements_col = std::vector<std::int32_t>(grid_comm.col_comm().size()+1);
		send_displacements_col[0]=0;
		for (std::uint32_t p_col = 0; p_col < grid_comm.col_comm().size(); p_col++)
			send_displacements_col[p_col+1] = send_displacements_col[p_col]+send_counts_col[p_col];
		std::fill(send_counts_col.begin(), send_counts_col.end(),0);
		karam::utils::default_init_vector<karam::mpi::IndirectMessage<send_request>> send_buf_col(send_displacements_col[grid_comm.col_comm().size()]); 
		for (const auto& [key, value] : request_map)
		{
			std::int32_t targetPE = grid_comm.proxy_row_index(static_cast<std::size_t>(value.targetPE));
			std::uint64_t packet_index = send_displacements_col[targetPE] + send_counts_col[targetPE]++;
			
			send_buf_col[packet_index] = karam::mpi::IndirectMessage<send_request>(static_cast<std::uint32_t>(comm.rank()),static_cast<std::uint32_t>(value.targetPE),{value.request_value, key});
		}
		timer.switch_category("communication");
		auto colwise_recv = grid_comm.col_comm().alltoallv(kamping::send_buf(send_buf_col),kamping::send_counts(send_counts_col));
		timer.switch_category("local_work");

		auto colwise_recv_buf = colwise_recv.extract_recv_buffer();
		/*
		std::cout << comm.rank() << " bekomment letztendlich folgende request: ";
		for (int i = 0; i < colwise_recv_buf.size(); i++)
			std::cout << colwise_recv_buf[i].payload().request_value << " ";
		std::cout << std::endl;*/
		struct send_answer {
			answer answer_value;
			std::uint64_t key;
		};
		std::vector<send_answer> send_buf_col_back(colwise_recv_buf.size()); 
		for (std::uint64_t i = 0; i < colwise_recv_buf.size(); i++)
			send_buf_col_back[i] = {lambda(colwise_recv_buf[i].payload().request_value), colwise_recv_buf[i].payload().key};
		
		timer.switch_category("communication");
		auto colwise_recv_buf_back = grid_comm.col_comm().alltoallv(kamping::send_buf(send_buf_col_back),kamping::send_counts(colwise_recv.extract_recv_counts())).extract_recv_buffer();
		timer.switch_category("local_work");
		/*
		std::cout << comm.rank() << " bekommt in indirektion folgende answer: ";
		for (int i = 0; i < colwise_recv_buf_back.size(); i++)
			std::cout << colwise_recv_buf_back[i].answer_value << " ";
		std::cout << std::endl;*/
		
		
		std::unordered_map<std::uint64_t, answer> answer_map;
		for (std::uint64_t i = 0; i < colwise_recv_buf_back.size(); i++)
			answer_map[colwise_recv_buf_back[i].key] = colwise_recv_buf_back[i].answer_value;
			
		
		std::vector<send_answer> answers_row(rowwise_recv_buf.size());
		for (std::uint64_t i = 0; i < answers_row.size(); i++)
			answers_row[i] = {answer_map[rowwise_recv_buf[i].payload().key], rowwise_recv_buf[i].payload().key};

		timer.switch_category("communication");
		auto rowwise_recv_buf_back = grid_comm.row_comm().alltoallv(kamping::send_buf(answers_row),kamping::send_counts(mpi_result_rowwise.extract_recv_counts())).extract_recv_buffer();
		timer.switch_category("local_work");

		answer_map.clear();
		
		//std::cout << comm.rank() << " bekommt aggregated engtülgite answers: ";
		for (std::uint64_t i = 0; i < rowwise_recv_buf_back.size(); i++)
		{
			//std::cout << rowwise_recv_buf_back[i].answer_value << " ";
			answer_map[rowwise_recv_buf_back[i].key] = rowwise_recv_buf_back[i].answer_value;
		}
		//std::cout << std::endl;
		std::vector<answer> answers(requests.size());
		for (std::uint64_t i = 0; i < requests.size(); i++)
			answers[i] = answer_map[request_assingment(requests[i])];
		/*
		std::cout << comm.rank() << " hat engülgite answers: ";
		for (int i = 0; i < answers.size(); i++)
			std::cout << answers[i] << " ";
		std::cout << std::endl;*/
		
		
		
		
		return answers;
	}
	
	template<typename request, typename answer>
	static std::vector<answer> request_reply_aggregate_normal(timer& timer, std::vector<request> requests, std::function<std::uint64_t(const request)> request_assingment, std::vector<std::int32_t> send_counts, std::function<answer(const request)> lambda, kamping::Communicator<>& comm)
	{
		
		std::vector<std::int32_t> targetPEs(requests.size());
		std::uint64_t index = 0;
		for (std::uint32_t p = 0; p < comm.size(); p++)
			for (std::uint64_t i = 0; i < send_counts[p]; i++)
				targetPEs[index++] = p;
		
		struct value {
			request request_value;
			std::int32_t targetPE;
		};
		
		std::unordered_map<std::uint64_t, value> request_map;
		
		
		for (std::uint64_t i = 0; i < requests.size(); i++)
		{
			if (!request_map.contains(request_assingment(requests[i])))
			{
				request_map[request_assingment(requests[i])] = {requests[i],  targetPEs[i]};
			}
			
		}
		
		
		std::vector<std::int32_t> num_packets_per_PE(comm.size(),0);
		for (const auto& [key, value] : request_map)
		{
			std::uint32_t targetPE = value.targetPE;
			num_packets_per_PE[targetPE]++;
		}
		std::vector<std::int32_t> send_displacements(comm.size()+1);
		send_displacements[0]=0;
		for (std::int32_t i = 1; i < comm.size() + 1; i++)
			send_displacements[i] = send_displacements[i-1] + num_packets_per_PE[i-1];
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);

		struct send_request {
			request request_value;
			std::uint64_t key;
		};
			
		std::vector<send_request> send_requests(send_displacements[comm.size()]);
		for (const auto& [key, value] : request_map)
		{
			std::uint32_t targetPE = value.targetPE;
			std::uint64_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			//std::cout << comm.rank() << " requestet " << value.request_value << std::endl;
			send_requests[packet_index] = { value.request_value, key};
		}
	
		//hier kann auch request reply benutzt werden. Dann annahme; recv_answers in selber reihenfolge wie send_requests
		timer.switch_category("communication");
		auto recv = comm.alltoallv(kamping::send_buf(send_requests), kamping::send_counts(num_packets_per_PE));
		timer.switch_category("local_work");
	
		std::vector<send_request> recv_requests = recv.extract_recv_buffer();
		if (recv_requests.size() > 10*send_requests.size())	std::cout << comm.rank() << " sendet " << send_requests.size() << " requests und bekommt " << recv_requests.size() << " requests " << std::endl;
		
		struct send_answer {
			answer answer_value;
			std::uint64_t key;
		};
		
		std::vector<send_answer> send_answers(recv_requests.size());
		for (std::uint64_t i = 0; i < recv_requests.size(); i++)
			send_answers[i] = {lambda(recv_requests[i].request_value), recv_requests[i].key};
		
		timer.switch_category("communication");
		auto recv_answers = comm.alltoallv(kamping::send_buf(send_answers), kamping::send_counts(recv.extract_recv_counts())).extract_recv_buffer();
		timer.switch_category("local_work");

		std::unordered_map<std::uint64_t, answer> answer_map;
		for (std::uint64_t i = 0; i < recv_answers.size(); i++)
			answer_map[recv_answers[i].key] = recv_answers[i].answer_value;
		std::vector<answer> answers(requests.size());
		for (std::uint64_t i = 0; i < answers.size(); i++)
			answers[i] = answer_map[request_assingment(requests[i])];
		return answers;
		
	}

	
	template<typename request, typename answer>
	static std::vector<answer> request_reply_normal(timer& timer, std::vector<request>& requests, std::vector<std::int32_t>& send_counts, std::function<answer(const request)> lambda, kamping::Communicator<>& comm)
	{
		timer.switch_category("communication");
		auto recv = comm.alltoallv(kamping::send_buf(requests), kamping::send_counts(send_counts));
		timer.switch_category("local_work");
		
		std::vector<request> recv_request = recv.extract_recv_buffer();
		
		std::uint64_t size = recv_request.size();
		std::vector<answer> answers(size);
		
		
		
		
		for (std::uint64_t i = 0; i < size; i++)
		{
			answers[i] = lambda(recv_request[i]);
		}
		
		
		timer.switch_category("communication");
		std::vector<answer> recv_answers = comm.alltoallv(kamping::send_buf(answers), kamping::send_counts(recv.extract_recv_counts())).extract_recv_buffer();
		timer.switch_category("local_work");

		return recv_answers;
	}
	
	
	
	template<typename request, typename answer>
	static std::vector<answer> request_reply_grid(timer& timer, std::vector<request>& requests, std::vector<std::int32_t>& send_counts, std::function<answer(const request)> lambda, kamping::Communicator<>& comm, karam::mpi::GridCommunicator grid_comm)
	{
		timer.switch_category("communication");
		auto recv_request = my_grid_all_to_all(requests, send_counts, grid_comm, comm).extract_recv_buffer();
		timer.switch_category("local_work");

		std::uint64_t size = recv_request.size();
		std::vector<answer> answers(recv_request.size());
		
		std::vector<std::int32_t> recv_counts(comm.size(), 0);
		for (std::uint64_t i = 0; i < size; i++)
		{
			recv_counts[recv_request[i].get_source()]++;
		}
		std::vector<std::uint64_t> recv_displacements(comm.size(), 0);
		for (std::int32_t p = 1; p < comm.size(); p++)
			recv_displacements[p] = recv_displacements[p-1] + recv_counts[p-1];
		std::fill(recv_counts.begin(), recv_counts.end(), 0);

		for (std::uint64_t i = 0; i < size; i++)
		{
			std::int32_t targetPE = recv_request[i].get_source();
			std::uint64_t packet_index = recv_displacements[targetPE] + recv_counts[targetPE]++;
			answers[packet_index] = lambda(recv_request[i].payload());
			
		}
		timer.switch_category("communication");
		auto recv_indirect_answers = my_grid_all_to_all(answers, recv_counts, grid_comm, comm).extract_recv_buffer();
		timer.switch_category("local_work");
		std::vector<std::uint64_t> send_displacements(comm.size(),0);
		for (std::int32_t p = 1; p < comm.size(); p++)
			send_displacements[p] = send_displacements[p-1] + send_counts[p-1];
		
		std::fill(send_counts.begin(), send_counts.end(), 0);

		size = recv_indirect_answers.size();
		std::vector<answer> recv_answers(size);
		for (std::uint64_t i = 0; i < size; i++)
		{	
			std::int32_t sourcePE = recv_indirect_answers[i].get_source();
			std::uint64_t packet_index = send_displacements[sourcePE] + send_counts[sourcePE]++;
			recv_answers[packet_index] = recv_indirect_answers[i].payload();
		}
		return  recv_answers;
	}
	
	
	
	static void test(kamping::Communicator<>& comm, karam::mpi::GridCommunicator grid_comm)
	{
		int rank = comm.rank();
		int size = comm.size();
		
		std::vector<std::uint64_t> requests(size, rank);
		std::vector<std::int32_t> send_counts(size,1);
		
		auto lambda = [&](std::uint64_t i ) {return i + rank;}; 
		/*
		std::vector<std::uint64_t> reply = request_reply_normal<std::uint64_t,std::uint64_t>(requests, send_counts, lambda, comm);
		
		
		std::cout << rank << " with: ";
		for (int i = 0; i < reply.size(); i++)
			std::cout << reply[i] << " ";
		std::cout << std::endl;
		
		return;
		int i = 0;
		
		auto test = [&](float a) {
            return a + i;
        };
		
		std::cout << test(0) << std::endl;
		
		i = 1;
		
		std::cout << test(0) << std::endl;
		*/
		
	}
		
	
