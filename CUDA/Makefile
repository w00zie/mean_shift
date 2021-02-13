all: sm naive
sm:
	/usr/local/cuda-11/bin/nvcc sm.cu -std=c++17 -o sm
naive:
	/usr/local/cuda-11/bin/nvcc naive.cu -std=c++17 -o naive
debug:
	/usr/local/cuda-11/bin/nvcc test.cu -g -G -std=c++17 -o test
clean:
	rm sm && rm naive