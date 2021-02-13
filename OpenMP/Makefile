all: openmp 

openmp_db:
	g++ -std=c++17 -g -Wall -Wextra -pedantic -O0 -fopenmp ms.cpp -o mean_shift_openmp_db
	g++ -std=c++17 -g -Wall -Wextra -pedantic -O0 -fopenmp ms_static.cpp -o mean_shift_openmp_static_db

openmp:
	g++ -std=c++17 -Wall -Wextra -pedantic -O3 -fopenmp ms.cpp -o mean_shift_openmp
	g++ -std=c++17 -Wall -Wextra -pedantic -O3 -fopenmp ms_static.cpp -o mean_shift_openmp_static

clean:
	rm -f mean_shift_openmp*