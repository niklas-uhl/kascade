with open('../../other/supermuc_auswertung/results_regular_ruling_set.txt', 'r+') as file: 
 file_data = file.read() 
 file.seek(0, 0) 
 file.write("{\n\"data\":[\n" + file_data[:-2] + "\n}]}") 