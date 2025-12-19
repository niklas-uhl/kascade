#/usr/bin/bash



for n in {1000000,10000000}; do
	for p in {2,4,6,8,10,12,16,20,24,28,32,36,40,44,48,52,56,60,65,70,75,80,85,90,95,100,105,110,115,120,128}; do
		for dist in {5,6,7,8,9,10,12,14,16,18,20,23,26,30}; do
			echo "mpirun -np " $p " build/Code ruling_set " $n " " $dist
			mpirun -np $p build/Code ruling_set $n $dist
		done
	done
done 