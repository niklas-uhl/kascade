#pragma once

#include <queue>

#include "forest_irregular_pointer_doubling.cpp"
#include "../helper_functions.cpp"
#include "forest_irregular_ruling_set2.cpp"

class forest_irregular_double_pointer_doubling //this is for trees
{
	struct packet{
		std::uint64_t ruler_source;
		std::int32_t ruler_sourcePE;
		std::uint64_t destination;
		std::uint64_t distance;
	};
	
	struct result{
			std::uint64_t root;
			std::int64_t distance;
	};
	
	public:
	
	forest_irregular_double_pointer_doubling(std::uint64_t comm_rounds, std::uint32_t num_iterations, bool grid)
	{
		this->comm_rounds = comm_rounds;
		this->num_iterations = num_iterations;
		this->grid = grid;
	}
	
	
	
	void start(std::vector<std::uint64_t> s, std::vector<std::int64_t> r, std::vector<std::uint32_t> targetPEs, std::uint64_t node_offset, kamping::Communicator<>& comm, karam::mpi::GridCommunicator& grid_comm, std::vector<std::uint64_t> non_recursive_indices)
	{
		
		
		size = comm.size();
		rank = comm.rank();
		num_local_vertices = s.size();
		node_offset = prefix_sum_num_vertices_per_PE[rank];
		//now turn around s array
		
		std::vector<std::string> categories = {"local_work", "communication", "other"};
		timer timer("graph_umdrehen", categories, "local_work", "forest_irregular_optimized_ruling_set");
		
		timer.add_info(std::string("comm_rounds"), std::to_string(comm_rounds));
		timer.add_info(std::string("grid"), std::to_string(grid));
		timer.add_info(std::string("num_iterations"), std::to_string(num_iterations));

		timer.add_info(std::string("num_local_vertices"), std::to_string(num_local_vertices), true);
		
		std::vector<std::uint64_t> local_roots(0); //local indices
		
		struct edge{
			std::uint64_t source;
			std::uint64_t destination;
			std::int64_t weight;
			std::uint32_t source_PE;
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
				std::int32_t targetPE = targetPEs[i];
				num_packets_per_PE[targetPE]++;
			}
		}
		calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
		std::vector<edge> send_edges(send_displacements[size]);
		
		for (std::uint64_t i = 0; i < num_local_vertices; i++)
		{
			if (i + node_offset == s[i])
				continue;
			
			std::int32_t targetPE = targetPEs[i];;
			std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			
			send_edges[packet_index].source = i + node_offset;
			send_edges[packet_index].destination = s[i];
			send_edges[packet_index].weight = r[i];
			send_edges[packet_index].source_PE = rank;
		}
		
		std::vector<edge> local_edges = alltoall(timer, send_edges, num_packets_per_PE, comm, grid_comm, grid);
		
		std::vector<std::uint64_t> edges_per_node(num_local_vertices,0);
		
		for (std::uint64_t i = 0; i < local_edges.size(); i++)
		{
			edges_per_node[local_edges[i].destination - node_offset]++;
		}
		
		std::vector<std::uint64_t> edges(local_edges.size());
		std::vector<std::uint64_t> bounds(num_local_vertices+1);
		std::vector<std::int64_t> edges_weights(local_edges.size());
		std::vector<std::uint32_t> edges_targetPEs(local_edges.size());
		for (std::uint64_t i = 1; i < num_local_vertices + 1; i++)
			bounds[i] = bounds[i-1] + edges_per_node[i-1];
		std::fill(edges_per_node.begin(), edges_per_node.end(), 0);
		for (std::uint64_t i = 0; i < local_edges.size(); i++)
		{
			std::uint64_t target_node = local_edges[i].destination - node_offset;
			std::uint64_t packet_index = bounds[target_node] + edges_per_node[target_node]++;
			edges[packet_index] = local_edges[i].source;
			edges_weights[packet_index] = local_edges[i].weight;
			edges_targetPEs[packet_index] = local_edges[i].source_PE;
		}
		
		
		std::vector<std::int64_t> r_s = r; //das hier gibt den rank vektor für die ausgabe an
		std::vector<std::uint64_t> d_s(num_local_vertices, 1); //das hier gibt die absolute distanz zu successor an, wird für contention reduction gebraucht
		std::vector<std::uint64_t> s_initial = non_recursive_indices; //das hier gibt initial inidices an
		std::vector<std::uint32_t> 
		
		std::vector<std::
		
		
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

	
	std::vector<std::int64_t> result_dist;
	std::vector<std::uint64_t> result_root;

	bool grid;
	

};


	