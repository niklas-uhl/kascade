#pragma once


#include <chrono>

#include <iostream>
#include <fstream>

#include "kamping/checking_casts.hpp"
#include "kamping/collectives/alltoall.hpp"
#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/collectives/allgather.hpp"
#include "kamping/collectives/bcast.hpp"
#include "kamping/collectives/gather.hpp"

#include <algorithm>
#include <cmath>
#include <vector>



class timer
{
	public:
	
	timer(std::string first_checkpoint, std::vector<std::string> categories, std::string start_category, std::string algorithm)
	{

		times = std::vector<uint64_t>(1);
		times[0] =  get_time();

		names = std::vector<std::string>(1);
		names[0] = first_checkpoint;
		
		info_values = std::vector<std::string>(0);
		info_names = std::vector<std::string>(0);
		
		current_category = start_category;
		category_timestamp = get_time();
		
		map = std::unordered_map<std::string, uint64_t>();
		for (int i = 0; i < categories.size(); i++)
			map[categories[i]] = 0;
		
		add_info("algorithm", quote(algorithm));
	}
	
	void switch_category(std::string category)
	{
		std::uint64_t current_time = get_time();
		map[current_category] = map[current_category] +  current_time - category_timestamp;
		category_timestamp = current_time;
		current_category = category;
		
	}
	
	
	void add_info(std::string name, std::string value)
	{
		add_info(name, value, false);
	}
	
	void add_info(std::string name, std::string value, bool output_for_every_PE)
	{
		info_values.push_back(value);
		info_names.push_back(name);
		info_bool.push_back(output_for_every_PE);
	}
	
	void add_checkpoint(std::string checkpoint)
	{
		times.push_back(get_time());
		names.push_back(checkpoint);
	}
	
	
	std::vector<std::string> split (const std::string &s, char delim) {
		std::vector<std::string> result;
		std::stringstream ss (s);
		std::string item;
		while (getline (ss, item, delim))
			result.push_back (item);
		return result;
	}
	
	std::string quote(std::string input)
	{
		return "\"" + input + "\"";
	}
	
	void finalize(kamping::Communicator<>& comm, std::string file_name)
	{	
		//measure finish time instantly
		times.push_back(get_time());
		add_info("p", std::to_string(comm.size()));
		
		std::string output = "{\n";
		//first print total time
		std::uint64_t total_time = times[times.size()-1] - times[0];
		std::vector<std::uint64_t> total_times;
		comm.gather(kamping::send_buf(total_time), kamping::recv_buf<kamping::resize_to_fit>(total_times), kamping::root(0));
		if (comm.rank() == 0)
		{
			output += quote("total_time") + ":" + get_output_string(total_times) +",\n";
		}
		
		
		//second print infos
		std::string info_string = "";
		for (std::uint32_t i = 0; i < info_values.size(); i++)
			info_string += info_values[i] + ",";
		std::vector<char> data(info_string.begin(), info_string.end());
		std::vector<char> recv_data;
		comm.gatherv(kamping::send_buf(data), kamping::recv_buf<kamping::resize_to_fit>(recv_data), kamping::root(0));
	
		if (comm.rank() == 0)
		{
			std::vector<std::string> all_info_values = split(std::string(recv_data.begin(), recv_data.end()), ',');
			
			output += "\"info_names\":[" + quote(info_names[0]);
			for (std::uint32_t i = 1; i < info_names.size(); i++)
				output += "," + quote(info_names[i]);
			output += "],\n";
			
			for (std::uint32_t i = 0; i < info_names.size(); i++)
			{
				if (info_bool[i])
				{
					output += quote(info_names[i]) + ":[";
					for (std::uint32_t j = 0; j < comm.size(); j++)
						output += all_info_values[i + j*info_names.size()] + ",";
					output.pop_back();
					output += "],\n";
				} 
				else
				{
					output += quote(info_names[i]) + ":" + info_values[i] + ",\n";
				}
				
				
			}
		}
		
	
		//third print time steps	
		std::vector<uint64_t> final_times(names.size());
		for (std::int32_t i = 0; i < names.size(); i++)
			final_times[i] = times[i+1] - times[i];
		
		
		
		std::vector<uint64_t> all_final_times;
		comm.gather(kamping::send_buf(final_times), kamping::recv_buf<kamping::resize_to_fit>(all_final_times), kamping::root(0));

		if (comm.rank() == 0)
		{
			output += "\"time_step_names\":[" + quote(names[0]);
			for (std::uint32_t i = 1; i < names.size(); i++) //-1, damit total time nicht als time step gewählt wird
				output += "," + quote(names[i]);
			output += "],\n";
			
			std::vector<uint64_t> all_final_times_from_one_checkpoint(comm.size());
			for (int i = 0; i < names.size(); i++)
			{
				
				for (int j = 0; j < comm.size(); j++)
					all_final_times_from_one_checkpoint[j] = all_final_times[i + j*names.size()];
				
				
				/*
				output += "\"" + names[i] + "\"" + ":[" + std::to_string(all_final_times_from_one_checkpoint[0]);
				for (int i = 1; i < comm.size(); i++)
					output += "," + std::to_string(all_final_times_from_one_checkpoint[i]);
				output += "],\n";*/
				
				output += quote(names[i]) + ":" + get_output_string(all_final_times_from_one_checkpoint) + ",\n";
				
			}
			
		
		}
		
		//third print categories
		std::vector<uint64_t> all_categorial_times;
		std::vector<uint64_t> local_categorial_times(0);
		std::vector<std::string> categorial_names(0);
		
		for (const auto& [key, value] : map)
		{
			local_categorial_times.push_back(value);
			categorial_names.push_back(key);
		}
		comm.gather(kamping::send_buf(local_categorial_times), kamping::recv_buf<kamping::resize_to_fit>(all_categorial_times), kamping::root(0));
		if (comm.rank() == 0)
		{
			output += quote("all_categories") + ":[";
			
			for (std::uint32_t i = 0; i < categorial_names.size(); i++)
			{
				output += quote(categorial_names[i]) + ",";
			}
			output.pop_back();
			output += "],\n";
			
			std::vector<uint64_t> all_final_times_from_one_category(comm.size());
			for (int i = 0; i < local_categorial_times.size(); i++)
			{
				output += quote(categorial_names[i]) + ":";
				for (int j = 0; j < comm.size(); j++)
					all_final_times_from_one_category[j] = all_categorial_times[i + j*local_categorial_times.size()];
				
				output += get_output_string(all_final_times_from_one_category);
				output += ",\n";
			}
			
			
			/*
			for (int i = 0; i < local_categorial_times.size(); i++)
			{
				output += quote(categorial_names[i]) + ":[";
				for (int j = 0; j < comm.size(); j++)
					output += std::to_string(all_categorial_times[i + j*local_categorial_times.size()]) + ",";
				output.pop_back();
				output += "],\n";
			}*/
			
			output.pop_back();
			output.pop_back();
			output += "\n}\n,\n";
			std::cout << output;
			std::ofstream myfile;
			myfile.open (file_name + ".txt",  std::ios::app);
			myfile << output;
			myfile.close();
		}	
			
		
		
		
	}
	
	
	std::string get_output_string(std::vector<std::uint64_t> to_output)
	{
		bool everything = false; //output values from all PEs iff everything otherwise output [min, lower_quartil, median, upper_quartil, max]
		std::string output = "[";
		if (everything)
		{
			for (int i = 0; i < to_output.size(); i++)
				output += std::to_string(to_output[i]) + ",";
			output.pop_back();
			output += "]";	
		}
		else
		{
			std::sort(to_output.begin(), to_output.end());
			int parts = 5;
			for (int i = 0; i < parts; i++)
			{
				int index = (i * (to_output.size() - 1)) / (parts - 1);
				output += std::to_string(to_output[index]) + ",";
			}
			output.pop_back();
			output += "]";
		}
		
		return output;
		
	}

	uint64_t get_time()
	{
		return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	}

	private:
	std::string current_category;
	std::uint64_t category_timestamp;
	std::unordered_map<std::string, uint64_t> map;
	
	std::vector<std::string> info_names;
	std::vector<std::string> info_values;
	std::vector<bool> info_bool;
	
	std::vector<uint64_t> times;
	std::vector<std::string> names;
		
	
};