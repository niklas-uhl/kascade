#very important: this skript handles json objects, the syntax is obviously json, but more specific 
#{"data":[data_point1, data_point2]}

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import math
 


work_communication_dir = "work_communication/"
time_step_graphs_dir = "time_step_graphs/"
os.mkdir(work_communication_dir)
os.mkdir(time_step_graphs_dir)


 

def to_json(path):
    with open(path, 'r+') as file: 
        file_data = file.read() 
        file.seek(0, 0) 
        file.write("{\n\"data\":[\n" + file_data[:-2] + "\n]}") 


#this function groups the json elements in path by the value "p" and returns the mean total_time
def get_PE_total_time_tuple(path):
    f = open(path)
    data = json.load(f)
   
    processors = [data_point['p'] for data_point in data['data']]
    processors = list(dict.fromkeys(processors))
    processors.sort() #so we have unique sorted processors array
    
    times = []
    for p in processors:
        relevant_data_points = [data_point for data_point in data['data'] if data_point['p'] == p]
        relevant_total_times = [np.amax(data_point['total_time'])  for data_point in relevant_data_points]
        times.append(np.mean(relevant_total_times))
    
    return processors, times


def generate_time_step_graph(path, algorithm_name, extra_info):
    f = open(path)
    data = json.load(f)
   
    processors = [data_point['p'] for data_point in data['data']]
    processors = list(dict.fromkeys(processors))
    processors.sort()
    
    steps = data['data'][0]['time_step_names']
    
    times = [0 for i in range(len(processors))]
    prefix_sum_times = [times];
    
    #first find for every p in processors one data_point to print
    data_points = []
    for p in processors:
        data_points.append([data_point for data_point in data['data'] if data_point['p'] == p][0])
        
    time_matrix = []
    for step in steps:
        pe = 0 #0 <= pe < p
        time_matrix.append([data_point[step][pe] for data_point in data_points])
    
    times = [0 for i in range(len(processors))]
    prefix_sum_times = [times];
    for i in range(len(steps)):
        plt.xscale("log")
        width = [p / 10 for p in processors]       
    
        plt.bar(processors, time_matrix[i], bottom=times, label=steps[i], width = width)
        times = np.add(times,time_matrix[i])
        prefix_sum_times.append(times)
    
    
    plt.xlabel("processors")
    plt.ylabel("time in ms")
    plt.title("time of different steps in " + algorithm_name + " algorithm\n" + extra_info)
    plt.legend()
    
 

    plt.savefig(time_step_graphs_dir + "time_step_" + algorithm_name + ".pdf")
    plt.clf()
    
def generate_work_communication_graph(path, algorithm_name, extra_info):
    f = open(path)
    data = json.load(f)
   
    processors = [data_point['p'] for data_point in data['data']]
    processors = list(dict.fromkeys(processors))
    processors.sort()
    
    
    
    #first find for every p in processors one data_point to print
    data_points = []
    for p in processors:
        data_points.append([data_point for data_point in data['data'] if data_point['p'] == p][0])
    
    values = []
    names = ["min_work","max_work", "min_communication", "max_communication"]
    local_work_min_times = [data_point['local_work'][0] for data_point in data_points]
    local_work_max_times = [data_point['local_work'][-1] for data_point in data_points]
    local_communication_min_times = [data_point['communication'][0] for data_point in data_points]
    local_communication_max_times = [data_point['communication'][-1] for data_point in data_points]
    values.append(local_work_min_times)
    values.append(local_work_max_times)
    values.append(local_communication_min_times)
    values.append(local_communication_max_times)
    width = 20
    for i in range(len(names)):
           value = values[i]
    
           plt.xscale("log")
           width = [p / 10 for p in processors]
           processors_shifed = [processors[j] + i * width[j] for j in range(len(processors))]
       
    
    
           plt.bar(processors_shifed, value, label=names[i], width = width)
    
    
    plt.xlabel("processors")
    plt.ylabel("time in ms")
    plt.title("work and communication\n" + extra_info)
    plt.legend()

    plt.savefig(work_communication_dir + "work_and_communication_of_" + algorithm_name + ".pdf")
    plt.clf() 


path1 = 'grid_regular_ruling_set2.txt'
path2 = 'grid_regular_ruling_set2_rec.txt'    

path3 = 'grid_regular_ruling_set2.txt'
path4 = 'grid_regular_ruling_set2_rec.txt'

path5 = 'grid_regular_ruling_set2.txt'
path6 = 'grid_regular_ruling_set2_rec.txt'


paths = [path1, path2, path3, path4, path5, path6]

names = [path[:-4] for path in paths]

names = ["com_mode_0_ruling_set2", "com_mode_0_ruling_set2_rec", "com_mode_1_ruling_set2", "com_mode_1_ruling_set2_rec","com_mode_2_ruling_set2", "com_mode_2_ruling_set2_rec" ]



save_dir1 = '../../other/supermuc_auswertung2/aufschrieb/communication_mode_0/'
save_dir2 = '../../other/supermuc_auswertung2/aufschrieb/communication_mode_1/'
save_dir3 = '../../other/supermuc_auswertung2/aufschrieb/communication_mode_2/'

paths = [save_dir1 + path1, save_dir1 + path2, save_dir2 + path3, save_dir2 + path4,save_dir3 + path5, save_dir3 + path6]


extra_info = "input=random_liset with 1.000.000 per PE"
##########WICHTIG########################
#for path in paths:
    #to_json(path)




for i in range(len(paths)):
    generate_time_step_graph(paths[i], names[i], extra_info)
    generate_work_communication_graph(paths[i], names[i], extra_info)

for i in range(len(paths)):    
    processors, times = get_PE_total_time_tuple(paths[i])
    
    plt.xscale("log")
    width = [p / 10 for p in processors]
    processors_shifed = [processors[j] + i * width[j] for j in range(len(processors))]
       
    
    xticks = ["2^" + str(int(math.log(p,2))) for p in processors]
    
    plt.xticks(processors, processors, rotation=90)
    
    plt.bar(processors_shifed, times, label=names[i], width = width)



plt.xlabel("processors")
plt.ylabel("time in ms")
#plt.title("time of different algorithms for different number of PEs\n"+extra_info)
plt.legend()
plt.savefig("all.pdf")
plt.clf()
