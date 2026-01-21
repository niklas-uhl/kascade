#include "tree_rooting/forest_regular_optimized_ruling_set.cpp"
#include "list_ranking/regular_pointer_doubling.cpp"

#include "kamping/checking_casts.hpp"
#include "kamping/collectives/alltoall.hpp"
#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/collectives/allgather.hpp"
#include "kamping/collectives/bcast.hpp"
#include "kamping/collectives/gather.hpp"

class analyze_instances
{
	public:
	
	static void analyze_regular_instance2(std::vector<std::uint64_t>& s, kamping::Communicator<>& comm, karam::mpi::GridCommunicator& grid_comm, std::string save_dir)
	{
		
		
		//regular_pointer_doubling algorithm(s, comm, true, true);
		//std::vector<std::int64_t> result_dist= algorithm.start(comm, grid_comm);				
		//std::vector<std::uint64_t> result_root = algorithm.q;
		
		forest_regular_optimized_ruling_set algorithm(100,10,true,true);
		algorithm.start(s, comm, grid_comm);
		std::vector<std::uint64_t> result_root = algorithm.result_root;
		std::vector<std::int64_t> result_dist = algorithm.result_dist;
		analyze_regular_instance2(result_root, result_dist, comm, grid_comm, save_dir);
	}
	
	static void analyze_regular_instance2(std::vector<std::uint64_t>& result_root, std::vector<std::int64_t>& result_dist, kamping::Communicator<>& comm, karam::mpi::GridCommunicator& grid_comm, std::string save_dir)
	{
		if (comm.rank() == 0) std::cout << "##########################\n######### analyse ########\n##########################\n\n";
		std::uint64_t num_local_vertices = result_root.size();
		std::uint32_t rank = comm.rank();
		std::uint32_t size = comm.size();
		std::uint64_t node_offset = num_local_vertices *rank;
		
		
		std::unordered_map<std::uint64_t, std::uint64_t> local_subtree_size; //local_subtree_size[i] = j iff there are j nodes on this PE with the root node i
		std::unordered_map<std::uint64_t, std::int64_t> local_max_depth_size;
		for (std::uint32_t i = 0; i < num_local_vertices; i++)
		{
			local_subtree_size[result_root[i]] = 0;
			local_max_depth_size[result_root[i]] = 0;
		}
		for (std::uint32_t i = 0; i < num_local_vertices; i++)
		{
			local_subtree_size[result_root[i]]++;
			
			local_max_depth_size[result_root[i]] = std::max(local_max_depth_size[result_root[i]], result_dist[i]);
		}
		
		
		std::vector<std::int32_t> num_packets_per_PE_row(grid_comm.row_comm().size(),0);
		std::vector<std::int32_t> send_displacements_row(grid_comm.row_comm().size() + 1,0);
		for (const auto& [key, value] : local_subtree_size)
		{
			std::int32_t targetPE = key / num_local_vertices;
			targetPE = grid_comm.proxy_col_index(targetPE);
			num_packets_per_PE_row[targetPE]++;
		}
		calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements_row, num_packets_per_PE_row);
		struct packet {
			std::uint64_t target_node;
			std::uint64_t subtree_size;
			std::uint64_t subtree_depth;
		};
		std::vector<packet> send_packets(send_displacements_row[send_displacements_row.size()-1]);
		for (const auto& [key, value] : local_subtree_size)
		{
			std::int32_t targetPE = key / num_local_vertices;
			targetPE = grid_comm.proxy_col_index(targetPE);
			std::int32_t packet_index = send_displacements_row[targetPE] + num_packets_per_PE_row[targetPE]++;
			
			//std::cout << comm.rank() << " hier hat root " << key << " subtreesize " << local_subtree_size[key] << " und maxdeph " << local_max_depth_size[key] << std::endl;
			
			send_packets[packet_index].target_node = key;
			send_packets[packet_index].subtree_size = local_subtree_size[key];
			send_packets[packet_index].subtree_depth = local_max_depth_size[key];
		}
		std::vector<packet> recv_packets = grid_comm.row_comm().alltoallv(kamping::send_buf(send_packets),kamping::send_counts(num_packets_per_PE_row)).extract_recv_buffer();

		std::vector<std::int32_t> num_packets_per_PE_col(grid_comm.col_comm().size(),0);
		std::vector<std::int32_t> send_displacements_col(grid_comm.col_comm().size() + 1,0);
		local_subtree_size.clear();
		local_max_depth_size.clear();
		for (std::uint64_t i = 0; i < recv_packets.size(); i++)
		{
			std::uint64_t node = recv_packets[i].target_node;
						
			if (local_subtree_size.contains(node))
			{
				local_subtree_size[node] += recv_packets[i].subtree_size;
				
				std::uint64_t currenct_max_depth = local_max_depth_size[recv_packets[i].target_node];
				std::uint64_t max_deph = std::max(currenct_max_depth, recv_packets[i].subtree_depth);
				
				//std::cout << "max(" << local_max_depth_size[node] << "," << recv_packets[i].subtree_depth << ") = " << max_deph << std::endl;
				
				local_max_depth_size[node]  = max_deph;// local_max_depth_size[node] < recv_packets[i].subtree_depth ? recv_packets[i].subtree_depth : local_max_depth_size[node];
			}
			else
			{
				local_subtree_size[node] = recv_packets[i].subtree_size;
				local_max_depth_size[node]  = recv_packets[i].subtree_depth;
			}
		}
		for (const auto& [key, value] : local_subtree_size)
		{
			std::int32_t targetPE = key / num_local_vertices;
			targetPE = grid_comm.proxy_row_index(targetPE);
			num_packets_per_PE_col[targetPE]++;
		}
		calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements_col, num_packets_per_PE_col);
		send_packets.resize(send_displacements_col[send_displacements_col.size()-1]);
		for (const auto& [key, value] : local_subtree_size)
		{
			std::int32_t targetPE = key / num_local_vertices;
			targetPE = grid_comm.proxy_row_index(targetPE);
			std::int32_t packet_index = send_displacements_col[targetPE] + num_packets_per_PE_col[targetPE]++;
			//std::cout << comm.rank() << " indirektion hier hat root " << key << " subtreesize " << local_subtree_size[key] << " und maxdeph " << local_max_depth_size[key] << std::endl;

			send_packets[packet_index].target_node = key;
			send_packets[packet_index].subtree_size = local_subtree_size[key];
			send_packets[packet_index].subtree_depth = local_max_depth_size[key];
		}

		recv_packets = grid_comm.col_comm().alltoallv(kamping::send_buf(send_packets),kamping::send_counts(num_packets_per_PE_col)).extract_recv_buffer();
		local_subtree_size.clear();
		local_max_depth_size.clear();
		
		for (std::uint64_t i = 0; i < recv_packets.size(); i++)
		{
			std::uint64_t node = recv_packets[i].target_node;
			
			
			if (local_subtree_size.contains(node))
			{
				local_subtree_size[node] += recv_packets[i].subtree_size;
				local_max_depth_size[node]  = local_max_depth_size[node] < recv_packets[i].subtree_depth ? recv_packets[i].subtree_depth : local_max_depth_size[node];
			}
			else
			{
				local_subtree_size[node] = recv_packets[i].subtree_size;
				local_max_depth_size[node]  = recv_packets[i].subtree_depth;
			}
		}
		
		//we are just interested in tree dephts
		
		std::vector<std::uint64_t> outliers;
		std::uint64_t array_length = 100000;
		std::vector<std::uint64_t> sub_tree_dephs(array_length,0);

		
		
		std::uint64_t num_trees = 0;
		for (const auto& [key, value] : local_max_depth_size)
		{
			num_trees++;
			
			if (value >= array_length)
			{
				outliers.push_back(value);
			}
			else
			{
				sub_tree_dephs[value]++;
			}
		}
		
		
		std::vector<uint64_t> total_num_trees = comm.reduce(kamping::send_buf(num_trees), kamping::op(kamping::ops::plus<>())).extract_recv_buffer();
		std::vector<std::uint64_t> recv_sub_tree_dephs = comm.reduce(kamping::send_buf(sub_tree_dephs), kamping::op(kamping::ops::plus<>())).extract_recv_buffer();
		std::vector<std::uint64_t> all_outliers;
		comm.allgatherv(kamping::send_buf(outliers), kamping::recv_buf<kamping::resize_to_fit>(all_outliers));
		
		if (comm.rank() == 0)
		{
			std::string output = "p=" + std::to_string(comm.size()) + ",n/p=" + std::to_string(result_root.size()) + "\n";
			output += "there are in total " + std::to_string(total_num_trees[0]) + " tree with tree dephs:\n[";
			for (int i = 0; i < recv_sub_tree_dephs.size(); i++)
			{
				if (recv_sub_tree_dephs[i] > 0)
					output +=  "[" + std::to_string(i) + "," + std::to_string(recv_sub_tree_dephs[i]) + "],";
			}
			for (int i = 0; i < all_outliers.size(); i++)
				output += "[" + std::to_string(all_outliers[i]) + ",1],";
			
			output.pop_back();
			output += "\n";
			
			std::cout << output << std::endl;
			std::ofstream myfile;
			myfile.open (save_dir + ".txt",  std::ios::app);
			myfile << output;
			myfile.close();
				
		}
		
	}
	
	
	
	static void analyze_regular_instance(std::vector<std::uint64_t>& s, kamping::Communicator<>& comm)
	{
		if (comm.rank() == 0) std::cout << "##########################\n######### DEPRECATED ########\n##########################\n\n";

		karam::mpi::GridCommunicator grid_comm;

		
		std::uint64_t num_local_vertices = s.size();
		std::uint32_t rank = comm.rank();
		std::uint32_t size = comm.size();
		std::uint64_t node_offset = num_local_vertices *rank;
		
		std::uint64_t dist_rulers = 500;
	//	if (dist_rulers > num_local_vertices) dist_rulers = num_local_vertices / 3;
//		forest_regular_ruling_set2 algorithm = forest_regular_ruling_set2(dist_rulers,1, false);
		
		forest_regular_optimized_ruling_set algorithm(100,10,true,true);
		algorithm.start(s, comm, grid_comm);
	
		

		std::vector<std::uint64_t> result_root = algorithm.result_root;
		std::vector<std::int64_t> result_dist = algorithm.result_dist;
		
		std::unordered_map<std::uint64_t, std::uint64_t> local_subtree_size; //local_subtree_size[i] = j iff there are j nodes on this PE with the root node i
		std::unordered_map<std::uint64_t, std::int64_t> local_max_depth_size;
		for (std::uint32_t i = 0; i < num_local_vertices; i++)
		{
			local_subtree_size[result_root[i]] = 0;
			local_max_depth_size[result_root[i]] = 0;
		}
		for (std::uint32_t i = 0; i < num_local_vertices; i++)
		{
			local_subtree_size[result_root[i]]++;
	
			local_max_depth_size[result_root[i]] = std::max(local_max_depth_size[result_root[i]], result_dist[i]);
		}
		
		
		std::vector<std::int32_t> num_packets_per_PE(size,0);
		std::vector<std::int32_t> send_displacements(size + 1,0);
		for (const auto& [key, value] : local_subtree_size)
		{
			std::int32_t targetPE = key / num_local_vertices;
			num_packets_per_PE[targetPE]++;
		}
		calculate_send_displacements_and_reset_num_packets_per_PE(send_displacements, num_packets_per_PE);
		struct packet
		{
			std::uint64_t root;
			std::uint64_t local_subtree_size;
			std::int64_t local_max_depth_size;
		};
		std::vector<packet> send_buffer(send_displacements[size]);
		
		for (const auto& [key, value] : local_subtree_size)
		{
			std::int32_t targetPE = key / num_local_vertices;
			std::int32_t packet_index = send_displacements[targetPE] + num_packets_per_PE[targetPE]++;
			send_buffer[packet_index].local_subtree_size = local_subtree_size[key];
			send_buffer[packet_index].local_max_depth_size = local_max_depth_size[key];
			send_buffer[packet_index].root = key;
		}
		
		std::vector<packet> recv_buffer = comm.alltoallv(kamping::send_buf(send_buffer), kamping::send_counts(num_packets_per_PE)).extract_recv_buffer();

		std::unordered_map<std::uint64_t, std::int64_t> max_depth_size;
		std::unordered_map<std::uint64_t, std::uint64_t> subtree_size; 
		for (std::uint32_t i = 0; i < num_local_vertices; i++)
			if (s[i] == i + node_offset)
			{
				max_depth_size[i + node_offset] == 0;
				subtree_size[i + node_offset] == 0;
			}
			
		for (std::uint32_t i = 0; i < recv_buffer.size(); i++)
		{
			max_depth_size[recv_buffer[i].root] = std::max(recv_buffer[i].local_max_depth_size, max_depth_size[recv_buffer[i].root]);
			subtree_size[recv_buffer[i].root] += recv_buffer[i].local_subtree_size;
		}
		std::vector<std::uint64_t> subtree_sizes(0);
		std::vector<std::uint64_t> subtree_depths(0);		
		for (const auto& [key, value] : subtree_size)
		{
			subtree_sizes.push_back(subtree_size[key]);
			subtree_depths.push_back(max_depth_size[key]);
			
			//std::cout << "PE " << rank <<  "has root " << key << " has size " << value << " and a max depth of " << max_depth_size[key] << std::endl;
		}
		
		
		
		std::vector<std::uint64_t> all_subtree_sizes(0);
		std::vector<std::uint64_t> all_subtree_depths(0);	
		comm.gatherv(kamping::send_buf(subtree_sizes), kamping::recv_buf<kamping::resize_to_fit>(all_subtree_sizes), kamping::root(0));
		comm.gatherv(kamping::send_buf(subtree_depths), kamping::recv_buf<kamping::resize_to_fit>(all_subtree_depths), kamping::root(0));


		if (rank == 0)
		{
			int parts = 10;
			
			if (parts >= all_subtree_sizes.size())
			{
				
				std::cout << "there are " << all_subtree_depths.size() << " trees\n"; 
				
				std::cout << "tree sizes:[" << all_subtree_sizes[0] ;
				for (int i = 1; i < all_subtree_sizes.size(); i++)
					std::cout <<"," << all_subtree_sizes[i];
				std::cout << "]\ntree depths:[" << all_subtree_depths[0] ;
				for (int i = 1; i < all_subtree_depths.size(); i++)
					std::cout <<"," << all_subtree_depths[i];
				std::cout << "]" << std::endl;
				
			}
			else
			{
				std::vector<std::uint64_t> quantil_subtree_sizes(parts);
				std::vector<std::uint64_t> quantil_subtree_depths(parts);

				
				
				std::sort(all_subtree_sizes.begin(), all_subtree_sizes.end());
				std::sort(all_subtree_depths.begin(), all_subtree_depths.end());

				for (int i = 0; i < parts; i++)
				{
					int index = (i * (all_subtree_sizes.size() - 1)) / (parts - 1);
					quantil_subtree_sizes[i] = all_subtree_sizes[index];
					quantil_subtree_depths[i] = all_subtree_depths[index];
				}
				
				std::cout << "there are " << all_subtree_depths.size() << " trees\n"; 
				
				std::cout << parts << " quantils of tree sizes:[" << quantil_subtree_sizes[0] ;
				for (int i = 1; i < quantil_subtree_sizes.size(); i++)
					std::cout <<"," << quantil_subtree_sizes[i];
				std::cout << "]\n" << parts << " quantils of tree depths:[" << quantil_subtree_depths[0] ;
				for (int i = 1; i < quantil_subtree_sizes.size(); i++)
					std::cout <<"," << quantil_subtree_depths[i];
				std::cout << "]" << std::endl;
			}
			
			
		}
		



	}
	
	private:
	
	static void calculate_send_displacements_and_reset_num_packets_per_PE(std::vector<std::int32_t>& send_displacements, std::vector<std::int32_t	>& num_packets_per_PE)
	{
		send_displacements[0]=0;
		for (std::int32_t i = 1; i < num_packets_per_PE.size() + 1; i++)
			send_displacements[i] = send_displacements[i-1] + num_packets_per_PE[i-1];
		std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
	}
	
};