#include <fmt/format.h>
#include <fmt/ranges.h>
#include <message-queue/buffered_queue.hpp>
#include <message-queue/concepts.hpp>
#include <message-queue/indirection.hpp>
#include <random>

#include "irregular_pointer_doubling.cpp"
#include "irregular_ruling_set2.cpp"

class asynchron_ruling_set2
{
	public:
	
	asynchron_ruling_set2()
	{
		
	}

	std::vector<std::int64_t> test(std::vector<std::uint64_t>& s, std::uint64_t dist_rulers, kamping::Communicator<>& comm)
	{
		
		this->s = s;
		
		
		
		rank = comm.rank();
		size = comm.size();
		num_local_vertices = s.size();
		node_offset = rank * num_local_vertices;
		std::uint64_t out_buffer_size = num_local_vertices/dist_rulers;
		
		std::vector<std::string> categories = {"local_work", "communication", "other"};
		timer timer("ruler_pakete_senden", categories, "local_work", "regular_ruling_set2");
		
		timer.add_info(std::string("dist_rulers"), std::to_string(dist_rulers));
		timer.add_info(std::string("num_local_vertices"), std::to_string(num_local_vertices));
		timer.add_info(std::string("iterations"), std::to_string(num_iterations));
	/*
		std::cout << rank << " mit successor array:\n";
		for (int i = 0; i < num_local_vertices; i++)
			std::cout << s[i] << " ";
		std::cout << std::endl;
	*/
	
		std::vector<uint64_t> local_rulers(0);
		std::uint64_t ruler_index = 0; //this means that first free rulers has an index >= ruler_index
		
		/*struct packet{
		std::uint64_t ruler_source;
		std::uint64_t destination;
		std::uint32_t distance;
		*/
		auto queue = message_queue::make_buffered_queue<std::tuple<std::uint64_t,std::uint64_t,std::uint64_t>>(MPI_COMM_WORLD, [](auto& buf, message_queue::PEID receiver) {});
		
		std::vector<std::uint64_t> mst(num_local_vertices);
		std::iota(mst.begin(),mst.end(),node_offset); 
		std::vector<std::int64_t> del(num_local_vertices,0);
		
		std::vector<std::tuple<std::uint64_t,std::uint64_t,std::uint64_t>> work(0);
		
		auto on_message = [&](message_queue::Envelope<std::tuple<std::uint64_t,std::uint64_t,std::uint64_t>> auto envelope) {
			queue.reactivate();

			for (std::uint64_t i = 0; i < envelope.message.size(); i++)
			{
				work.push_back(envelope.message[i]);
				
				std::uint64_t ruler_source =std::get<0>(envelope.message[i]);
				std::uint64_t destination = std::get<1>(envelope.message[i]);
				std::uint64_t distance = std::get<2>(envelope.message[i]);
				
				//std::cout << rank << " bekommt packet (" << ruler_source << "," << destination << "," << distance << ")" << std::endl;
			}
        };
		
		for (std::uint64_t i = 0; i < out_buffer_size; i++)
		{
			while (is_final(ruler_index)) ruler_index++;
			
			local_rulers.push_back(ruler_index);
			std::int32_t targetPE = calculate_targetPE(s[ruler_index]);
			mark_as_ruler(ruler_index);
			
			queue.post_message(std::tuple{ruler_index + node_offset, unmask(s[ruler_index]), 1}, targetPE);
			//std::cout << rank << " sendet packet (" << ruler_index + node_offset << "," << unmask(s[ruler_index]) << "," << 1 << ") an PE " << targetPE << std::endl;
			
			ruler_index++;
		}
		timer.add_checkpoint("pakete_verfolgen");
		
		/*
		do {
    while (!local_queue.empty()) {
      message_queue.poll(on_message);
      auto [source, target, label] = local_queue.pop();
      KASSERT(graph.is_local_vertex(target), "Vertex " << target << " is not local");
      if (!distances.relax(source, target, label, on_settle)) {
        continue;
      };
      auto nbhs = graph.neighbours(target);
      for (auto const& edge : nbhs) {
        if (distances.would_relax(edge.id, label + 1)) {
          if (graph.is_local_vertex(edge.id)) {
            // insert local vertices into the local queue if their label will be updated
            local_queue.emplace(target, edge.id, label + 1);
          } else {
            // otherwise, send the message to the appropriate rank
            distances.relax(source, target, label + 1, on_settle);
            message_queue.post_message({target, edge.id, label + 1}, edge.rank);
          }
        }
      }
    }
  } while (!message_queue.terminate(on_message));*/

		int i = 0;
		std::uint64_t num_reached_nodes = 0;
		do
		{
			//if (rank == 0) std::cout << "iteration " << i++ << std::endl;
			queue.poll(on_message);
			
			for (std::uint64_t i = 0; i < work.size(); i++)
			{
				num_reached_nodes++;
				std::uint64_t ruler_source =std::get<0>(work[i]);
				std::uint64_t destination = std::get<1>(work[i]);
				std::uint64_t distance = std::get<2>(work[i]);
				
				//std::cout << rank << " verarbeitet packet (" << ruler_source << "," << destination << "," << distance << ")" << std::endl;
				
				std::uint64_t local_index = destination - node_offset;
				mark_as_reached(local_index);
				mst[local_index] = ruler_source;
				del[local_index] = distance;
				if (!is_ruler(destination - node_offset) && !is_final(destination - node_offset))
				{
					std::uint64_t targetPE = calculate_targetPE(s[destination - node_offset]);
					//std::cout << rank << " on_message makes new (" << ruler_source << "," << s[destination - node_offset] << ") to PE " << targetPE << std::endl
					queue.post_message(std::tuple{ruler_source, unmask(s[destination - node_offset]), distance+1}, targetPE);
				}
				else
				{
					while (ruler_index < num_local_vertices && (is_final(ruler_index) || is_reached(ruler_index) || is_ruler(ruler_index))) ruler_index++;
					if (ruler_index == num_local_vertices) continue;
										
					std::uint64_t targetPE = calculate_targetPE(s[ruler_index]);
					mark_as_ruler(ruler_index);
					local_rulers.push_back(ruler_index);
					queue.post_message(std::tuple{ruler_index + node_offset, unmask(s[ruler_index]), 1}, targetPE);
					
				}
			}
			work.resize(0);
			
		} while (!queue.terminate(on_message));// || (ruler_index < num_local_vertices));
		
		
		//std::cout << rank << ", " << num_reached_nodes << " sollte " << num_local_vertices << std::endl;
		//return std::vector<std::int64_t>(0);
		timer.add_checkpoint("rekursion_vorbereiten");
		
		/*
		std::cout << rank << " hat folgende ruler:";
		for (int i = 0; i < local_rulers.size(); i++)
			std::cout << local_rulers[i] << " ";
		std::cout << "\n und hat folgendes mst array: ";
		for (int i = 0; i < num_local_vertices; i++)
			std::cout << mst[i] << " ";
		std::cout << "\n und hat folgendes del array: ";
		for (int i = 0; i < num_local_vertices; i++)
			std::cout << del[i] << " ";
		std::cout << std::endl;*/
			
		//###############################
		
		//now just the global starting node is unreached, this node is also always a ruler
		std::vector<std::uint64_t> num_local_vertices_per_PE;
		comm.allgather(kamping::send_buf(local_rulers.size()), kamping::recv_buf<kamping::resize_to_fit>(num_local_vertices_per_PE));
		std::vector<std::uint64_t> prefix_sum_num_vertices_per_PE(size + 1,0);
		for (std::uint32_t i = 1; i < size + 1; i++)
		{
			prefix_sum_num_vertices_per_PE[i] = prefix_sum_num_vertices_per_PE[i-1] + num_local_vertices_per_PE[i-1];
		}
		
		if (rank == 0) std::cout << num_local_vertices * size << " reduziert auf " << prefix_sum_num_vertices_per_PE[size] << std::endl;
		
		std::vector<std::uint64_t> map_ruler_to_its_index(num_local_vertices);
		std::vector<std::uint64_t> s_rec(local_rulers.size());
		std::vector<std::int64_t> r_rec(local_rulers.size());
		std::vector<std::uint32_t> targetPEs_rec(local_rulers.size());
	
		
		std::vector<std::uint64_t> requests(local_rulers.size());
		std::vector<std::int32_t> num_packets_per_PE(size,0);
		std::vector<std::int32_t> send_displacements(size + 1,0);
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
		
		auto recv = comm.alltoallv(kamping::send_buf(requests), kamping::send_counts(num_packets_per_PE));

		std::vector<std::uint64_t> recv_requests = recv.extract_recv_buffer();
		
		
		//answers können inplace in requests eingetragen werden
		for (std::uint64_t i = 0; i < recv_requests.size(); i++)
		{
			recv_requests[i] = map_ruler_to_its_index[recv_requests[i]-node_offset] + prefix_sum_num_vertices_per_PE[rank];
		}
		std::vector<std::uint64_t> recv_answers = comm.alltoallv(kamping::send_buf(recv_requests), kamping::send_counts(recv.extract_recv_counts())).extract_recv_buffer();
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
		for (std::uint64_t i = 0; i < local_rulers.size(); i++)
		{
			std::int32_t targetPE = calculate_targetPE(mst[local_rulers[i]]);
			std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			s_rec[i] = recv_answers[packet_index];
			r_rec[i] = del[local_rulers[i]];
		}
		
		timer.add_checkpoint("rekursion");
		std::vector<std::int64_t> ranks;
		if (num_iterations == 2)
		{
			irregular_ruling_set2 algorithm(s_rec, r_rec, targetPEs_rec, dist_rulers,  prefix_sum_num_vertices_per_PE);
			ranks = algorithm.start(comm);	
		}
		else
		{
			irregular_pointer_doubling algorithm(s_rec, r_rec, targetPEs_rec, prefix_sum_num_vertices_per_PE);
			ranks = algorithm.start(comm);	
		}
		
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
		
		auto recv2 = comm.alltoallv(kamping::send_buf(requests), kamping::send_counts(num_packets_per_PE));
		recv_requests = recv2.extract_recv_buffer();
		num_packets_per_PE = recv2.extract_recv_counts();
		for (std::uint64_t i = 0; i < recv_requests.size(); i++)
		{
			recv_requests[i] = ranks[map_ruler_to_its_index[recv_requests[i] - node_offset]];
		}
		
		recv_answers = comm.alltoallv(kamping::send_buf(recv_requests), kamping::send_counts(num_packets_per_PE)).extract_recv_buffer();
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
		
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::int32_t targetPE = calculate_targetPE(mst[i]);
			std::uint64_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			del[i] = size * num_local_vertices - 1 - (del[i] + recv_answers[packet_index]);
		}
		
		std::string save_dir = "regular_ruling_set2";
		if (num_iterations == 2)
			save_dir = "regular_ruling_set2_rec";
		timer.finalize(comm, save_dir);

	
		return del;
	}
	
	std::uint64_t sum_it(std::uint64_t value, kamping::Communicator<>& comm)
	{
		std::vector<std::uint64_t> recv;
		comm.allgather(kamping::send_buf(value), kamping::recv_buf<kamping::resize_to_fit>(recv));
		std::uint64_t sum = 0;
		for (int i = 0; i < comm.size(); i++)
			sum += recv[i];
		return sum;
		
	}
	
	void calculate_send_displacements_and_reset_num_packets_per_PE(std::vector<std::int32_t>& send_displacements, std::vector<std::int32_t	>& num_packets_per_PE)
	{
		send_displacements[0]=0;
		for (std::int32_t i = 1; i < size + 1; i++)
			send_displacements[i] = send_displacements[i-1] + num_packets_per_PE[i-1];
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
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
	
	private:
	
	std::uint64_t node_offset;
	std::uint64_t num_local_vertices;
	std::uint64_t rank, size;
	std::vector<std::uint64_t> s;
	std::uint64_t dist_rulers;
	std::uint32_t num_iterations;


};