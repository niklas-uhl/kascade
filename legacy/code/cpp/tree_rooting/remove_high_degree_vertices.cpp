#include <set>
#include "../list_ranking/regular_pointer_doubling.cpp"
#include "../list_ranking/irregular_pointer_doubling.cpp"
#include "forest_irregular_pointer_doubling.cpp"
#include "../test.cpp"

class remove_high_degree_vertices 
{
	
	
	public:
	
	remove_high_degree_vertices(bool grid, bool aggregate)
	{
		this->grid = grid;
		this->aggregate = aggregate;
	}
	
	void add_timer_info(std::string info)
	{
		this->info = "\"" + info + "\"";
	}
	

	std::vector<std::int64_t>  start(std::vector<std::uint64_t>& s, kamping::Communicator<>& comm, karam::mpi::GridCommunicator& grid_comm)
	{
		size = comm.size();
		rank = comm.rank();
		num_local_vertices = s.size();
		node_offset = rank * num_local_vertices;
		
		std::vector<std::string> categories = {"local_work", "communication", "other"};
		timer timer("calculate out degree", categories, "local_work", "remove_high_degree_vertices");
		timer.add_info("num_local_vertices", std::to_string(num_local_vertices));
		timer.add_info("grid", std::to_string(grid));
		timer.add_info("aggregate", std::to_string(aggregate));
		
		if (info.size() > 0)
			timer.add_info(std::string("additional_info"), info);

		
		std::function<bool(const std::uint64_t)> is_high_degree = [&](std::uint64_t degree) {return degree >= 5;};
		
		//std::cout << " einfach abbrechen, wenn keine high degree vertices gefunden wurden" << std::endl;
		
	/*	
		std::cout << rank << " with s: ";
		for (int i = 0; i < num_local_vertices; i++)
			std::cout << s[i] << " ";
		std::cout << std::endl;*/
		
		calculate_high_degree_vertices(s, comm, grid_comm);
		
		//jetzt müssen wir wissen ob s[i] high degree vertex ist
		
		timer.add_checkpoint("maniulate_instance");
		
		std::vector<std::int32_t> num_packets_per_PE(size,0);
		std::vector<std::int32_t> send_displacements(size + 1,0);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::int32_t targetPE = s[i] / num_local_vertices;
			num_packets_per_PE[targetPE]++;
		}
		calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
		std::vector<std::uint64_t> requests(num_local_vertices);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::int32_t targetPE = s[i] / num_local_vertices;
			std::uint64_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			requests[packet_index] = s[i];
		}
		
		std::function<std::uint32_t(const std::uint64_t)> answer_is_high_degree = [&] (std::uint64_t request) { return local_node_indegrees.contains(request) && is_high_degree(local_node_indegrees[request]);};
		std::function<std::uint64_t(const std::uint64_t)> request_assignment =  [](std::uint64_t request) {return request;};
		//hier wird berechnet, ob s[i] high degree ist
		std::vector<std::uint32_t> recv_is_high_degree = request_reply(timer, requests, num_packets_per_PE, answer_is_high_degree, comm, grid_comm, grid, aggregate, request_assignment);
		std::vector<bool> succ_is_high_degree(num_local_vertices);
		
		std::vector<std::uint64_t> new_s(num_local_vertices);
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::int32_t targetPE = s[i] / num_local_vertices;
			std::uint64_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			succ_is_high_degree[i] = recv_is_high_degree[packet_index];
			if (succ_is_high_degree[i])
			{
				
				new_s[i] = i + node_offset;
			
			}
			else
				new_s[i] = s[i];
		}
		
		
		
		timer.add_checkpoint("rank_maniulated_instance");

		
		//hier wir manipulated instance geranked
		regular_pointer_doubling algorithm(new_s, comm, grid);
		
		//forest_regular_optimized_ruling_set algorithm(100,2,grid,aggregate);
		
		//algorithm.start(new_s, comm, grid_comm);
		//std::vector<std::int64_t> ranks = algorithm.result_dist;
		//std::vector<std::uint64_t> roots = algorithm.result_root;
			
		std::vector<std::int64_t> ranks = algorithm.start(comm, grid_comm);

		std::vector<std::uint64_t> roots = algorithm.q;
		/*
		std::cout << rank << " with manipulated instance s: ";
		for (int i = 0; i < num_local_vertices; i++)
			std::cout << new_s[i] << " ";
		std::cout << "\n und dist vector: ";
		for (int i = 0; i < num_local_vertices; i++)
			std::cout << ranks[i] << " ";
		std::cout << "\n und root vector: ";
		for (int i = 0; i < num_local_vertices; i++)
			std::cout << roots[i] << " ";
		std::cout << std::endl;*/
		timer.add_checkpoint("restore_maniulate_instance");

		
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::uint32_t targetPE = roots[i] / num_local_vertices;
			num_packets_per_PE[targetPE]++;
		}
		calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::uint32_t targetPE = roots[i] / num_local_vertices;
			std::uint64_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
		
			requests[packet_index] = roots[i];
		}
		
		struct answer {
			std::uint64_t root;
			std::int64_t dist;
		};
		std::function<answer(const std::uint64_t)> restore_pointer_to_high_degree_vertex = [&](std::uint64_t request) {
			answer answer;
			if (succ_is_high_degree[request - node_offset] && (s[request - node_offset] != request))
			{
				answer.root = s[request - node_offset];
				answer.dist = ranks[request - node_offset] + 1;
			}
			else
			{
				answer.root = roots[request - node_offset];
				answer.dist = ranks[request - node_offset];
			}
			return answer;
		};
		
		//hier werden die pointer wieder zu den high degree vertices wiederhergestellt
		std::vector<answer> answers_here = request_reply(timer, requests, num_packets_per_PE, restore_pointer_to_high_degree_vertex, comm, grid_comm, grid, aggregate, request_assignment);
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::uint32_t targetPE = roots[i] / num_local_vertices;
			std::uint64_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			if (roots[i] != answers_here[packet_index].root)
			{
				roots[i] = answers_here[packet_index].root;
				ranks[i]  += answers_here[packet_index].dist;
				
				//std::cout << answers_here[packet_index].dist << std::endl;
			}
		}
		/*
		std::cout << rank << " with restored instance s: ";
		for (int i = 0; i < num_local_vertices; i++)
			std::cout << new_s[i] << " ";
		std::cout << "\n und dist vector: ";
		for (int i = 0; i < num_local_vertices; i++)
			std::cout << ranks[i] << " ";
		std::cout << "\n und root vector: ";
		for (int i = 0; i < num_local_vertices; i++)
			std::cout << roots[i] << " ";
		std::cout << std::endl;*/
		//jetzt muss instanz der high degree vertices erstellt werden
		
		timer.add_checkpoint("create_instance_of_high_degree_vertices");

		
		std::vector<std::uint64_t> local_high_degree_vertices(0);
		std::vector<std::uint64_t> map_local_high_degree_vertice_to_consecutive_index(num_local_vertices,0);
		std::uint64_t consecutive_index = 0;
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			if (local_node_indegrees.contains(i + node_offset) && is_high_degree(local_node_indegrees[i + node_offset]))
			{
				//std::cout << i + node_offset << " ist hdg vertex " << std::endl;
				local_high_degree_vertices.push_back(i);
				map_local_high_degree_vertice_to_consecutive_index[i] = consecutive_index++;
			}
			
		}
		
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
		for (std::uint64_t i = 0; i < local_high_degree_vertices.size(); i++)
		{
			std::int32_t targetPE = roots[local_high_degree_vertices[i]] / num_local_vertices;
			num_packets_per_PE[targetPE]++;
		}
		calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
		requests.resize(send_displacements[size]);
		for (std::uint64_t i = 0; i < local_high_degree_vertices.size(); i++)
		{			
			std::int32_t targetPE = roots[local_high_degree_vertices[i]] / num_local_vertices;
			std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			
			//std::cout << rank << " requested für high degree vertex " << local_high_degree_vertices[i]+node_offset<< " die request " << roots[local_high_degree_vertices[i]] << std::endl;
			
			requests[packet_index] = roots[local_high_degree_vertices[i]] ;
		}
		
		std::function<answer(const std::uint64_t)> lambda = [&] (std::uint64_t request) {
			answer answer;
			
			//wenn request ein hdg vertex ist, dann trivial antworten
			if (local_node_indegrees.contains(request) && is_high_degree(local_node_indegrees[request]))
			{
				answer.root = request;
				answer.dist = 0;
			}
			else
			{
				answer.root = roots[request-node_offset];
				answer.dist = ranks[request-node_offset];
			}
			//std::cout << comm.rank() << " hier wird request " << request << " beantwortet mit (" << answer.root << "," << answer.dist << ")" << std::endl;

			return answer;
		};
		/*
		std::cout << rank << " das sind die requests: ";
		for (int i = 0;  i < requests.size(); i++)
			std::cout << requests[i] << " ";
		std::cout << "\n und das die send counts: ";
		for (int  i = 0; i < num_packets_per_PE.size(); i++)
			std::cout << num_packets_per_PE[i] << " ";
		std::cout << std::endl;*/
		
		

		//hier wird berechnet für die high degree vertices v nämich root[root[v]] berechnet, da sich das bei restored instance geändert haben kann
		std::vector<answer> answers = request_reply(timer, requests, num_packets_per_PE, lambda, comm, grid_comm, grid, aggregate, request_assignment);	
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
		
		
		std::vector<std::uint64_t> s_high_degree_vertices(local_high_degree_vertices.size());
		std::vector<std::int64_t> r_high_degree_vertices(local_high_degree_vertices.size());
		std::vector<std::uint32_t> targetPEs_high_degree_vertices(local_high_degree_vertices.size());
		for (int i = 0; i < answers.size(); i++)
		{
			std::int32_t targetPE = roots[local_high_degree_vertices[i]] / num_local_vertices;
			std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			
			s_high_degree_vertices[i] = answers[packet_index].root;
			r_high_degree_vertices[i] = answers[packet_index].dist + ranks[local_high_degree_vertices[i]];
		}
		//jetzt muss die antwort auch noch in consecutive_index übersetzt werden
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
		for (std::uint64_t i = 0; i < local_high_degree_vertices.size(); i++)
		{
			std::int32_t targetPE = s_high_degree_vertices[i] / num_local_vertices;
			num_packets_per_PE[targetPE]++;
		}
		calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
		for (std::uint64_t i = 0; i < local_high_degree_vertices.size(); i++)
		{
			std::int32_t targetPE = s_high_degree_vertices[i] / num_local_vertices;
			std::uint64_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			
			requests[packet_index] = s_high_degree_vertices[i];
		}
		
		std::vector<std::uint64_t> num_high_degree_vertices_per_PE;
		comm.allgather(kamping::send_buf(local_high_degree_vertices.size()), kamping::recv_buf<kamping::resize_to_fit>(num_high_degree_vertices_per_PE));
		std::vector<std::uint64_t> prefix_sum_high_degree_vertices_per_PE(size +1,0);
		for (std::uint64_t p = 0; p < size; p++)
			prefix_sum_high_degree_vertices_per_PE[p+1] = prefix_sum_high_degree_vertices_per_PE[p] + num_high_degree_vertices_per_PE[p];
		
		struct answer2 {
			bool is_high_degree;
			std::uint64_t recursive_index;
		};
		std::function<answer2(const std::uint64_t)> lambda2 = [&](std::uint64_t request) {
			answer2 answer;
			answer.is_high_degree = local_node_indegrees.contains(request) && is_high_degree(local_node_indegrees[request]);
			if (answer.is_high_degree)
				answer.recursive_index = map_local_high_degree_vertice_to_consecutive_index[request - node_offset] + prefix_sum_high_degree_vertices_per_PE[rank];
			return answer;
		};
		std::vector<answer2> answers2 = request_reply(timer, requests, num_packets_per_PE, lambda2, comm, grid_comm, grid, aggregate, request_assignment);	
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
		
		
		std::vector<std::uint64_t> s_initial_high_degree_vertices = s_high_degree_vertices;
		for (std::uint64_t i = 0; i < local_high_degree_vertices.size(); i++)
		{
			std::int32_t targetPE = s_high_degree_vertices[i] / num_local_vertices;
			std::uint64_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			
			if (answers2[packet_index].is_high_degree)
			{
				targetPEs_high_degree_vertices[i] = targetPE;
				s_high_degree_vertices[i] = answers2[packet_index].recursive_index;
			}
			else
			{
				s_high_degree_vertices[i] = i + prefix_sum_high_degree_vertices_per_PE[rank];
				targetPEs_high_degree_vertices[i] = rank;
			}
		}
		/*
		std::cout << rank << " with s_high_degree_vertices: ";
		for (int i = 0; i < s_high_degree_vertices.size(); i++)
			std::cout << s_high_degree_vertices[i] << " ";
		std::cout << "\n and with r_high_degree_vertices: ";
		for (int i = 0; i < r_high_degree_vertices.size(); i++)
			std::cout << r_high_degree_vertices[i] << " ";
		std::cout << "\n and with targetPEs_high_degree_vertices: ";
		for (int i = 0; i < targetPEs_high_degree_vertices.size(); i++)
			std::cout << targetPEs_high_degree_vertices[i] << " ";
		std::cout << std::endl;*/
		timer.add_checkpoint("rank_instance_of_high_degree_vertices");

		forest_irregular_pointer_doubling algorithm2(s_high_degree_vertices, r_high_degree_vertices, targetPEs_high_degree_vertices, prefix_sum_high_degree_vertices_per_PE[rank], prefix_sum_high_degree_vertices_per_PE[size], s_initial_high_degree_vertices, grid, false);
		
		algorithm2.start(comm, grid_comm);
		
		timer.add_checkpoint("create_final_ranks");

		
		std::vector<std::uint64_t> high_degree_vertices_roots = algorithm2.local_rulers;
		std::vector<std::int64_t> high_degree_vertices_ranks = algorithm2.r;
		/*
		std::cout << rank << " with high_degree_vertices_roots: ";
		for (int i = 0; i < high_degree_vertices_roots.size(); i++)
			std::cout << high_degree_vertices_roots[i] << " ";
		std::cout << "\n and with targetPEs_high_degree_vertices: ";
		for (int i = 0; i < high_degree_vertices_ranks.size(); i++)
			std::cout << high_degree_vertices_ranks[i] << " ";
		std::cout << std::endl;*/
		
		//jetzt muss jeder seine root fragen, ob die sich geändert hat, weil ja high vertices jetzt geranked
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::uint32_t targetPE = roots[i] / num_local_vertices;
			num_packets_per_PE[targetPE]++;
		}
		calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
		requests.resize(num_local_vertices);
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::uint32_t targetPE = roots[i] / num_local_vertices;
			std::uint64_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
		
			requests[packet_index] = roots[i];
		}
		
		
		std::function<answer(const std::uint64_t)> ask_root_for_his_root = [&](std::uint64_t request) {
			answer answer;
			if (local_node_indegrees.contains(request) && is_high_degree(local_node_indegrees[request]))
			{
				answer.root = high_degree_vertices_roots[map_local_high_degree_vertice_to_consecutive_index[request-node_offset]];
				answer.dist = high_degree_vertices_ranks[map_local_high_degree_vertice_to_consecutive_index[request-node_offset]];
			}
			else
			{
				//einfach zurückschicken
				answer.root = request;
				answer.dist = 0;
			}
			//std::cout << "request " << request << " wird beantwortet mit (" << answer.root << "," << answer.dist << ")" <<std::endl;
			return answer;
		};
		
		//hier werden die pointer wieder zu den high degree vertices wiederhergestellt
		answers_here = request_reply(timer, requests, num_packets_per_PE, ask_root_for_his_root, comm, grid_comm, grid, aggregate, request_assignment);
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
		std::vector<std::int64_t> final_ranks(num_local_vertices);
		std::vector<std::uint64_t> final_roots(num_local_vertices);
		
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			std::uint32_t targetPE = roots[i] / num_local_vertices;
			std::uint64_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			
			final_ranks[i] = answers_here[packet_index].dist + ranks[i];
			final_roots[i] = answers_here[packet_index].root;
			
		}
		timer.finalize(comm, "remove_high_degree_vertices");
		
		/*
		std::cout << rank << " with final dist vector: ";
		for (int i = 0; i < num_local_vertices; i++)
			std::cout << final_ranks[i] << " ";
		std::cout << "\n und final root vector: ";
		for (int i = 0; i < num_local_vertices; i++)
			std::cout << final_roots[i] << " ";
		std::cout << std::endl;*/
		
		test::regular_test_ranks_and_roots(comm, s, final_ranks, final_roots);

		
		
		/*
		std::vector<std::uint64_t> local_high_degree_vertices = calculate_high_degree_vertices(s, comm, grid_comm);
		std::vector<std::uint64_t> global_high_degree_vertices;
		comm.allgatherv(kamping::send_buf(local_high_degree_vertices), kamping::recv_buf<kamping::resize_to_fit>(global_high_degree_vertices));
		
		std::set<std::uint64_t> global_high_degree_vertices_set;
		std::vector<std::uint64_t> high_degree_vertices_per_PE(size,0);
		std::unordered_map<std::uint64_t,std::uint64_t> map_high_degree_vertice_to_consecutive_index;
		for (std::uint64_t i = 0; i < global_high_degree_vertices.size(); i++)
		{
			global_high_degree_vertices_set.insert(global_high_degree_vertices[i]);
			high_degree_vertices_per_PE[global_high_degree_vertices[i] / num_local_vertices]++;
			map_high_degree_vertice_to_consecutive_index[global_high_degree_vertices[i]]=i;
		}
		std::vector<std::uint64_t> prefix_sum_high_degree_vertices_per_PE(size+1,0);
		for (std::uint32_t p = 0; p < size; p++)
			prefix_sum_high_degree_vertices_per_PE[p+1] = prefix_sum_high_degree_vertices_per_PE[p] + high_degree_vertices_per_PE[p];
		
		
		std::vector<std::uint64_t> new_s(num_local_vertices);
		for (std::uint64_t i = 0; i < s.size(); i++)
			if (global_high_degree_vertices_set.contains(s[i]))
				new_s[i] = i + node_offset;
			else
				new_s[i] = s[i];
			
		
		regular_pointer_doubling algorithm(new_s, comm, grid);
			
		std::vector<std::int64_t> ranks = algorithm.start(comm, grid_comm);

		std::vector<std::uint64_t> roots = algorithm.q;
		std::cout << rank << " with manipulated instance s: ";
		for (int i = 0; i < num_local_vertices; i++)
			std::cout << new_s[i] << " ";
		std::cout << "\n und dist vector: ";
		for (int i = 0; i < num_local_vertices; i++)
			std::cout << ranks[i] << " ";
		std::cout << "\n und root vector: ";
		for (int i = 0; i < num_local_vertices; i++)
			std::cout << roots[i] << " ";
		std::cout << std::endl;
		
		
		//jetzt müssen pointer zu high degree vertices wieder hergestellt werden
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			if (global_high_degree_vertices_set.contains(s[i]) && i+node_offset != s[i])
			{
				ranks[i]++;
				//std::cout << "node " << i + node_offset << " wird von selbst pointer zu pointer auf " << s[i] << std::endl;
				roots[i] = s[i];
			}
		}
		
		std::cout << rank << " with restored manipulated instance s: ";
		for (int i = 0; i < num_local_vertices; i++)
			std::cout << new_s[i] << " ";
		std::cout << "\n und dist vector: ";
		for (int i = 0; i < num_local_vertices; i++)
			std::cout << ranks[i] << " ";
		std::cout << "\n und root vector: ";
		for (int i = 0; i < num_local_vertices; i++)
			std::cout << roots[i] << " ";
		std::cout << std::endl;
		
		//jetzt muss für jede high degree vertex v nochmal root[root[v]] requested werden, da sich das grade ja verändert hat
		std::vector<std::int32_t> num_packets_per_PE(size,0);
		std::vector<std::int32_t> send_displacements(size + 1,0);
		for (std::uint64_t i = 0; i < local_high_degree_vertices.size(); i++)
		{
			std::int32_t targetPE = roots[local_high_degree_vertices[i]-node_offset] / num_local_vertices;
			num_packets_per_PE[targetPE]++;
		}
		calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
		std::vector<std::uint64_t> requests(send_displacements[size]);
		for (std::uint64_t i = 0; i < local_high_degree_vertices.size(); i++)
		{
			std::int32_t targetPE = roots[local_high_degree_vertices[i]-node_offset] / num_local_vertices;
			std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			
			std::cout << rank << " requested für high degree vertex " << local_high_degree_vertices[i] << std::endl;
			
			requests[packet_index] = roots[local_high_degree_vertices[i]-node_offset];
		}
		struct answer {
			std::uint64_t root;
			std::int64_t dist;
		};
		std::function<answer(const std::uint64_t)> lambda = [&] (std::uint64_t request) {
			answer answer;
			answer.root = roots[request-node_offset];
			answer.dist = ranks[request-node_offset];
			return answer;
		};
		std::function<std::uint64_t(const std::uint64_t)> request_assignment =  [](std::uint64_t request) {return request;};
		
		std::vector<answer> answers = request_reply(timer, requests, num_packets_per_PE, lambda, comm, grid_comm, grid, false, request_assignment);	
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
		
		std::vector<std::uint64_t> s_high_degree_vertices(local_high_degree_vertices.size());
		std::vector<std::int64_t> r_high_degree_vertices(local_high_degree_vertices.size());
		std::vector<std::uint32_t> targetPEs_high_degree_vertices(local_high_degree_vertices.size());

		//also entweder zeigt unser high degree vertex auch auf einen high degree vertex
		//falls nicht, machen wir ihn zu root in unserer high_degree_vertex_instanz
		
		for (std::uint64_t i = 0; i < local_high_degree_vertices.size(); i++)
		{
			std::int32_t targetPE = roots[local_high_degree_vertices[i]-node_offset] / num_local_vertices;
			std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			
			std::cout << rank << " bekommt für high degree vertex " << local_high_degree_vertices[i] << " die antwort (" << answers[packet_index].root << "," << answers[packet_index].dist << ") für eine request auf " << roots[local_high_degree_vertices[i]-node_offset] << std::endl;

			r_high_degree_vertices[i] = ranks[local_high_degree_vertices[i]-node_offset] + answers[packet_index].dist;
			if (map_high_degree_vertice_to_consecutive_index.contains(answers[packet_index].root))
			{
				s_high_degree_vertices[i] = map_high_degree_vertice_to_consecutive_index[answers[packet_index].root];
				
				targetPEs_high_degree_vertices[i] = answers[packet_index].root / num_local_vertices; 
			}
			else
			{
				s_high_degree_vertices[i] = i + prefix_sum_high_degree_vertices_per_PE[rank];
				targetPEs_high_degree_vertices[i] = rank;
				
			}
			
			
			
			//requests[packet_index] = roots[local_high_degree_vertices[i]-node_offset];
		}
		
		std::cout << rank << " with s_high_degree_vertices: ";
		for (int i = 0; i < s_high_degree_vertices.size(); i++)
			std::cout << s_high_degree_vertices[i] << " ";
		std::cout << "\n and with r_high_degree_vertices: ";
		for (int i = 0; i < r_high_degree_vertices.size(); i++)
			std::cout << r_high_degree_vertices[i] << " ";
		std::cout << "\n and with targetPEs_high_degree_vertices: ";
		for (int i = 0; i < targetPEs_high_degree_vertices.size(); i++)
			std::cout << targetPEs_high_degree_vertices[i] << " ";
		std::cout << std::endl;
		//hier wieder forest pointer double da wir auc nicht high degree indices eingeben können
		irregular_pointer_doubling algorithm2(s_high_degree_vertices, r_high_degree_vertices, targetPEs_high_degree_vertices, prefix_sum_high_degree_vertices_per_PE);
			
		std::vector<std::int64_t> ranks_high_degree_vertices = algorithm2.start(comm, grid_comm);

		std::vector<std::uint64_t> roots_high_degree_vertices = algorithm2.q;
		
		*/
		return std::vector<std::int64_t>(0);
	}
	
	void calculate_high_degree_vertices(std::vector<std::uint64_t>& s, kamping::Communicator<>& comm, karam::mpi::GridCommunicator& grid_comm)
	{
		std::function<bool(const std::uint64_t)> is_high_degree = [&] (std::uint64_t degree) { return degree >= comm.size();};
		
		
		std::vector<std::string> categories = {"local_work", "communication", "other"};
		timer timer("calculate out degree", categories, "local_work", "regular_ruling_set");
		timer.add_info("num_local_vertices", std::to_string(num_local_vertices));
		timer.add_info("grid", std::to_string(grid));
		
		struct packet {
			std::uint64_t node;
			std::uint64_t indegree;
		};
		
		
		
		//std::unordered_map<std::uint64_t, std::uint64_t> local_node_indegrees; //key ist node und value ist indegree
		local_node_indegrees = std::unordered_map<std::uint64_t,std::uint64_t>();
		

		
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{	
			if (local_node_indegrees.contains(s[i]))
				local_node_indegrees[s[i]] = local_node_indegrees[s[i]] + 1;
			else
				local_node_indegrees[s[i]] = 1;
		}
		
		if (grid)
		{
			timer.add_checkpoint("row_communicator");
			std::vector<std::int32_t> num_packets_per_PE_row(grid_comm.row_comm().size(),0);
			std::vector<std::int32_t> send_displacements_row(grid_comm.row_comm().size() + 1,0);
			for (const auto& [key, value] : local_node_indegrees)
			{
				std::int32_t targetPE = key / num_local_vertices;
				targetPE = grid_comm.proxy_col_index(targetPE);
				num_packets_per_PE_row[targetPE]++;
			}
			calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements_row, num_packets_per_PE_row);
			std::vector<packet> send_packets(send_displacements_row[send_displacements_row.size()-1]);
			for (const auto& [key, value] : local_node_indegrees)
			{
				std::int32_t targetPE = key / num_local_vertices;
				targetPE = grid_comm.proxy_col_index(targetPE);
				std::int32_t packet_index = send_displacements_row[targetPE] + num_packets_per_PE_row[targetPE]++;
				
				send_packets[packet_index].node = key;
				send_packets[packet_index].indegree = value;
			}
			std::vector<packet> recv_packets = grid_comm.row_comm().alltoallv(kamping::send_buf(send_packets),kamping::send_counts(num_packets_per_PE_row)).extract_recv_buffer();
			timer.add_checkpoint("col_communicator");

			std::vector<std::int32_t> num_packets_per_PE_col(grid_comm.col_comm().size(),0);
			std::vector<std::int32_t> send_displacements_col(grid_comm.col_comm().size() + 1,0);
			local_node_indegrees.clear();
			for (std::uint64_t i = 0; i < recv_packets.size(); i++)
			{
				std::uint64_t node = recv_packets[i].node;
				std::uint64_t indegree = recv_packets[i].indegree;
				if (local_node_indegrees.contains(node))
					local_node_indegrees[node] = local_node_indegrees[node] + indegree;
				else
					local_node_indegrees[node] = indegree;
			}
			for (const auto& [key, value] : local_node_indegrees)
			{
				std::int32_t targetPE = key / num_local_vertices;
				targetPE = grid_comm.proxy_row_index(targetPE);
				num_packets_per_PE_col[targetPE]++;
			}
			calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements_col, num_packets_per_PE_col);
			send_packets.resize(send_displacements_col[send_displacements_col.size()-1]);
			for (const auto& [key, value] : local_node_indegrees)
			{
				std::int32_t targetPE = key / num_local_vertices;
				targetPE = grid_comm.proxy_row_index(targetPE);
				std::int32_t packet_index = send_displacements_col[targetPE] + num_packets_per_PE_col[targetPE]++;
				
				send_packets[packet_index].node = key;
				send_packets[packet_index].indegree = value;
			}

			recv_packets = grid_comm.col_comm().alltoallv(kamping::send_buf(send_packets),kamping::send_counts(num_packets_per_PE_col)).extract_recv_buffer();
			timer.add_checkpoint("ergebnisse_eintragen");
			local_node_indegrees.clear();

			for (std::uint64_t i = 0; i < recv_packets.size(); i++)
			{
				std::uint64_t node = recv_packets[i].node;
				std::uint64_t indegree = recv_packets[i].indegree;
				if (local_node_indegrees.contains(node))
					local_node_indegrees[node] = local_node_indegrees[node] + indegree;
				else
					local_node_indegrees[node] = indegree;
			}
			
			
			
		}
		else
		{
			timer.add_checkpoint("communicator");

			std::vector<std::int32_t> num_packets_per_PE(size,0);
			std::vector<std::int32_t> send_displacements(size + 1,0);
			for (const auto& [key, value] : local_node_indegrees)
			{
				std::int32_t targetPE = key / num_local_vertices;
				num_packets_per_PE[targetPE]++;
			}
			
			calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
			std::vector<packet> send_packets(send_displacements[size]);
			for (const auto& [key, value] : local_node_indegrees)
			{
				std::int32_t targetPE = key / num_local_vertices;
				std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
				
				send_packets[packet_index].node = key;
				send_packets[packet_index].indegree = value;
			}
			std::vector<packet> recv_packets = alltoall(timer, send_packets, num_packets_per_PE, comm, grid_comm, grid);
			timer.add_checkpoint("ergebnisse_eintragen");

			local_node_indegrees.clear();
			for (std::uint64_t i = 0; i < recv_packets.size(); i++)
			{
				std::uint64_t node = recv_packets[i].node;
				std::uint64_t indegree = recv_packets[i].indegree;
				if (local_node_indegrees.contains(node))
					local_node_indegrees[node] = local_node_indegrees[node] + indegree;
				else
					local_node_indegrees[node] = indegree;
			}
			
			
			
			
			
		}
		//timer.finalize(comm, "remove_high_degree_vertices");
		
	
		
	}
	
	void calculate_send_displacements_and_reset_num_packets_per_PE(std::vector<std::int32_t>& send_displacements, std::vector<std::int32_t>& num_packets_per_PE)
	{
		send_displacements[0]=0;
		for (std::int32_t i = 1; i < send_displacements.size(); i++)
			send_displacements[i] = send_displacements[i-1] + num_packets_per_PE[i-1];
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
	}
	
	std::unordered_map<std::uint64_t,std::uint64_t> local_node_indegrees;
	
	private:
	std::string info = "";

	bool grid;
	bool aggregate;
	std::uint64_t node_offset;
	std::uint64_t num_local_vertices;
	std::int32_t rank, size;
};