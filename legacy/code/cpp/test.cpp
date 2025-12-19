#pragma once

class test
{
	public:
	
	static void regular_test_ranks(kamping::Communicator<>& comm, std::vector<std::uint64_t>& s, std::vector<std::int64_t>& d)
	{
		std::uint64_t rank = comm.rank();
		std::uint64_t size = comm.size();
		std::uint64_t num_local_vertices = s.size();
		std::uint64_t node_offset = rank * num_local_vertices;
		
		std::vector<std::int64_t> r(num_local_vertices, 1);
		
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
			if (s[i] == i + node_offset)
				r[i] = 0;
		
		
		regular_test_ranks(comm,s,r,d);
	}
	
	static void regular_test_ranks_and_roots(kamping::Communicator<>& comm, std::vector<std::uint64_t>& s, std::vector<std::int64_t>& d, std::vector<std::uint64_t>& roots)
	{
		std::uint64_t rank = comm.rank();
		std::uint64_t size = comm.size();
		std::uint64_t num_local_vertices = s.size();
		std::uint64_t node_offset = rank * num_local_vertices;
		
		std::vector<std::int64_t> r(num_local_vertices, 1);
		
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
			if (s[i] == i + node_offset)
				r[i] = 0;
		
		
		regular_test_ranks_and_roots(comm,s,r,d,roots);
	}
	
	//s is sucessor vector which must  have same dimension on every PE, r is the weight of the edges, obviously same dimension as s, and d is the result vector, also same dimension as the others
	static void regular_test_ranks(kamping::Communicator<>& comm, std::vector<std::uint64_t>& s, std::vector<std::int64_t>& r, std::vector<std::int64_t>& d)
	{
		start_test(comm);
		
		std::uint64_t rank = comm.rank();
		std::uint64_t size = comm.size();
		std::uint64_t num_local_vertices = s.size();
		std::uint64_t node_offset = rank * num_local_vertices;
		
		//first test if input and output dimensions are the same
		std::vector<std::uint64_t> send_num(1,s.size() != d.size());
		std::vector<std::uint64_t> recv_num;
		comm.allgather(kamping::send_buf(send_num), kamping::recv_buf<kamping::resize_to_fit>(recv_num));
		std::uint64_t sum = 0;
		for (std::uint64_t i = 0; i < size; i++)
			sum += recv_num[i];
		
		if (sum > 0)
		{
			error(comm, "input and output dimensions are not the same!");
			return;
		}	
		else
		{
			output(comm, "input and output dimensions are the same");
		}
		
		//now test if the ranks are correct
		
		std::vector<std::int32_t> num_packets_per_PE(size,0);
		std::vector<std::int32_t> send_displacements(size + 1,0);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::int32_t targetPE = s[i] / num_local_vertices;
			num_packets_per_PE[targetPE]++;	
		}
		calculate_send_displacements_and_reset_num_packets_per_PE(comm, send_displacements, num_packets_per_PE);
		std::vector<std::int64_t> request(num_local_vertices);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::int32_t targetPE = s[i] / num_local_vertices;
			std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			
			request[packet_index] = s[i];
		}
		
		auto recv = comm.alltoallv(kamping::send_buf(request), kamping::send_counts(num_packets_per_PE));
		std::vector<std::int64_t> recv_request = recv.extract_recv_buffer();
		
		//request können wieder inplace ausgefüllt werden
		for (std::uint64_t i = 0; i < recv_request.size(); i++)
		{
			recv_request[i] = d[recv_request[i] - node_offset];
		}
		
		auto recv_answers = comm.alltoallv(kamping::send_buf(recv_request), kamping::send_counts(recv.extract_recv_counts())).extract_recv_buffer();
		
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
		bool correct = true;
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::int32_t targetPE = s[i] / num_local_vertices;
			std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			
			correct &= d[i] - r[i] ==  recv_answers[packet_index];
		}
		
		send_num[0] = !correct;
		comm.allgather(kamping::send_buf(send_num), kamping::recv_buf<kamping::resize_to_fit>(recv_num));
		sum = 0;
		for (std::uint64_t i = 0; i < size; i++)
			sum += recv_num[i];
		
		if (sum > 0)
		{
			error(comm, "ranks are not correct!");
			return;
		}	
		else
		{
			output(comm, "ranks are correct");
		}
	}
	
	//s is sucessor vector which must  have same dimension on every PE, r is the weight of the edges, obviously same dimension as s, and d is the result vector, also same dimension as the others
	static void regular_test_ranks_and_roots(kamping::Communicator<>& comm, std::vector<std::uint64_t>& s, std::vector<std::int64_t>& r, std::vector<std::int64_t>& d, std::vector<std::uint64_t>& roots)
	{
		start_test(comm);
		
		std::uint64_t rank = comm.rank();
		std::uint64_t size = comm.size();
		std::uint64_t num_local_vertices = s.size();
		std::uint64_t node_offset = rank * num_local_vertices;
		
		//first test if input and output dimensions are the same
		std::vector<std::uint64_t> send_num(1,(s.size() != d.size()) + (s.size() != roots.size()));
		std::vector<std::uint64_t> recv_num;
		comm.allgather(kamping::send_buf(send_num), kamping::recv_buf<kamping::resize_to_fit>(recv_num));
		std::uint64_t sum = 0;
		for (std::uint64_t i = 0; i < size; i++)
			sum += recv_num[i];
		
		if (sum > 0)
		{
			error(comm, "input and output dimensions are not the same!");
			return;
		}	
		else
		{
			output(comm, "input and output dimensions are the same");
		}
		
		//now test if the ranks and roots are correct
		
		std::vector<std::int32_t> num_packets_per_PE(size,0);
		std::vector<std::int32_t> send_displacements(size + 1,0);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::int32_t targetPE = s[i] / num_local_vertices;
			num_packets_per_PE[targetPE]++;	
		}
		calculate_send_displacements_and_reset_num_packets_per_PE(comm, send_displacements, num_packets_per_PE);
		std::vector<std::int64_t> request(num_local_vertices);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::int32_t targetPE = s[i] / num_local_vertices;
			std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			
			request[packet_index] = s[i];
		}
		
		auto recv = comm.alltoallv(kamping::send_buf(request), kamping::send_counts(num_packets_per_PE));
		std::vector<std::int64_t> recv_request = recv.extract_recv_buffer();
		
		struct answer {
			std::int64_t rank;
			std::uint64_t root;
		};
		std::vector<answer> answers(recv_request.size());
		
		//request können wieder inplace ausgefüllt werden
		for (std::uint64_t i = 0; i < recv_request.size(); i++)
		{
			answers[i].rank = d[recv_request[i] - node_offset];
			answers[i].root = roots[recv_request[i] - node_offset];
		}
		
		auto recv_answers = comm.alltoallv(kamping::send_buf(answers), kamping::send_counts(recv.extract_recv_counts())).extract_recv_buffer();
		//test ranks first
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
		bool correct = true;
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::int32_t targetPE = s[i] / num_local_vertices;
			std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			
			//std::cout << "testing: " << i + node_offset << " where " << d[i] << " - " << r[i] << " == " <<recv_answers[packet_index].rank<<std::endl;
			
			correct &= d[i] - r[i] ==  recv_answers[packet_index].rank;
			
			if (s[i] == i + node_offset)
				correct &= d[i] == 0;
		}
		
		send_num[0] = !correct;
		
		comm.allgather(kamping::send_buf(send_num), kamping::recv_buf<kamping::resize_to_fit>(recv_num));
		sum = 0;
		for (std::uint64_t i = 0; i < size; i++)
			sum += recv_num[i];
		
		if (sum > 0)
		{
			error(comm, "ranks are not correct!");
			return;
		}
		else
			output(comm, "ranks are correct");
		
		//test roots second
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
		correct = true;
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			if (s[i] == i + node_offset)
				correct &= roots[i] == s[i];
			
			std::int32_t targetPE = s[i] / num_local_vertices;
			std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			
			correct &= roots[i] ==  recv_answers[packet_index].root;
		}
		
		send_num[0] = !correct;
		comm.allgather(kamping::send_buf(send_num), kamping::recv_buf<kamping::resize_to_fit>(recv_num));
		sum = 0;
		for (std::uint64_t i = 0; i < size; i++)
			sum += recv_num[i];
		
		if (sum > 0)
		{
			error(comm, "roots are not correct!");
			return;
		}
		else
			output(comm, "roots are correct");
	}
	
	static void start_test(kamping::Communicator<>& comm)
	{
		if (comm.rank() == 0) std::cout << "##########################\n########## test ##########\n##########################\n\n";
	}
	
	static void calculate_send_displacements_and_reset_num_packets_per_PE(kamping::Communicator<>& comm, std::vector<std::int32_t>& send_displacements, std::vector<std::int32_t>& num_packets_per_PE)
	{
		send_displacements[0]=0;
		for (std::int32_t i = 1; i < comm.size() + 1; i++)
			send_displacements[i] = send_displacements[i-1] + num_packets_per_PE[i-1];
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
	}
	
	
	static void error(kamping::Communicator<>& comm, std::string output)
	{
		if (comm.rank() == 0)
			std::cout << "ERROR: " << output << std::endl;
	}
	
	static void output(kamping::Communicator<>& comm, std::string output)
	{
		if (comm.rank() == 0)
			std::cout << "INFO: " << output << std::endl;
	}


};