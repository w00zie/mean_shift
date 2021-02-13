all: openmp 

openmp_db: ms.cpp
	g++ -std=c++17 -g -Wall -Wextra -pedantic -O0 -ms.cpp -o mean_shift_sequential_db

openmp: ms.cpp
	g++ -std=c++17 -Wall -Wextra -pedantic -O3 ms.cpp -o mean_shift_sequential

clean:
	rm -f mean_shift_openmp*
