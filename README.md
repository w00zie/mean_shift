# Mean Shift Clustering

<p align="center">
<img width="460" height="300" src="https://ml-explained.com/_nuxt/img/mean_shift.9ca3b90.gif">
</p>

# Table of Contents
- [Mean Shift Clustering](#mean-shift-clustering)
- [Table of Contents](#table-of-contents)
  - [Brief description](#brief-description)
  - [Code](#code)
    - [Sequential and OpenMP](#sequential-and-openmp)
      - [Running](#running)
    - [CUDA](#cuda)
      - [Running CUDA](#running-cuda)

## Brief description

Mean shift is a popular non-parametric clustering technique. It is used when the number of cluster centers is unknown a-priori.
Based on kernel density estimation, it aims to discover the modes of a density of samples. With the help of a kernel function, mean shift works by updating each sample to be the mean of the points within a given region. More details can be found here [1].

The average complexity is given by *O(N * N * I)*, were *N* is the number of samples and *I* is the number of iteration.

## Code

This repository contains the code for three implementations of the mean shift algorithm:
1. Sequential
2. OpenMP
3. CUDA

All these implementations are self-contained, at the expense of code duplications. Sequential and OpenMP implementations are *header-only* C++17 libraries.

Mean shift works for Euclidean spaces of arbitrary dimensionality (obviously suffering the [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)) and in this implementation this number will be not fixed a-priori. 

A few simple synthetic datasets of various dimensionalities (2 and 3) and data size (500 to 10000 data points) are placed under `datasets/`. For example 

```
./datasets/2d/500_samples_3_centers
```

is a directory containing a dataset of 500 points in 2D synthetically generated around 3 cluster centers (centroids). Inside every directory there will be 2 `csv` files:

1. `centroids.csv`: The real centroids (that should match the ones computed by mean shift).
2. `data.csv`: The actual data to be clustered.

This implementation passes [cppcheck](https://github.com/danmar/cppcheck), [valgrind](https://valgrind.org/) and [CUDA-MEMCHECK](https://docs.nvidia.com/cuda/cuda-memcheck/index.html) tests.

### Sequential and OpenMP

Under `sequential/` and `OpenMP` you will find:

- `benchmark/`: Sub-directory used for running benchmarks.
- `test/`: Sub-directory used for (*sort of*) unit-testing the algorithm. Has a separate README file.
- `include/`: Sub-directory containing the actual implementation's header files.
- `Makefile`: Very (very) simple makefile needed just for alternating between debug and release mode. 
- `ms.cpp`: Example source file where the code functionality is shown.

The `OpenMP/include` directory contains **two versions**: one with **dynamic** workload scheduling (`OpenMP/include/meanshift.h`) between the threads, managed by OpenMP, and one with **static** workload scheduling (`OpenMP/include/meanshift_static`). The number of desired threads must be set by the user in this latter version.

#### Running

Since this implementation is centered on **performance** the user has to specify every time the "*hyperparameters*" of the algorithm, in order to help the compiler optimize the code. This is done in the `main` function inside `ms.cpp`, where these lines have to be edited each time

```cpp
// Hyperparameters
const float bandwidth = 3;
const float radius = 30;
const float min_distance = 60;
const size_t niter = 50;
const double eps = 0;
// I/O
const size_t num_points = 5000;
const size_t dim = 2;
const std::string data_path = "../datasets/2d/5000_samples_3_centers/data.csv";
const size_t num_threads = 4; // Only in OpenMP / Optional
```

In particular

- `bandwidth` is the standard deviation of the gaussian used to compute the kernel.
- `radius` is used for determining the neighbors of a point.
- `min_distance` is the minimum (L2 squared) distance between two points to be considered belonging to the same clusters.
- `num_points` is the number of data points (**Warning**: this number needs to match the one preceding `_samples` in the `data_path` variable).
- `dim` is the dimensionality of the data set (**Warning**: this number needs to match the one indicating the sub-directory in the `data_path` variable).
- `niter` is the number of iterations that the algorithm will run through.
- `data_path` is the string containing the path to the `data.csv` that the user wants to cluster.
- `eps` is the tolerance value for establishing the convergence of the algorithm. When all the points have moved of a distance <= eps wrt the previous step, then the algorithm will stop. If set to a value less or equal to zero this feature will not be used and the algorithm will run for `niter`.
- `num_threads` (**only in OpenMP**) is used in case the user wants to specify the number of threads in the execution (hence using the static worload balancing).

Once these are set up the user has to run (Sequential)

```
$ make
$ ./mean_shift_sequential
```
or (OpenMP)
```
$ make
$ ./mean_shift_openmp
```

and the algorithm will execute. In this early version the centroids computed by mean shift will be printed to the console. The main function (`mean_shift::seq::mean_shift` or `mean_shift::omp::mean_shift`) will be timed and the elapsed time will be displayed as well.

### CUDA

Under `CUDA/` you will find:

- `benchmark/`: Sub-directory used for running benchmarks.
- `test/`: Sub-directory used for (*sort of*) unit-testing the algorithm. Has a separate README file.
- `constants.h`: Header file containing the constants used for the algorithm executions (and I/O as well).
- `utils.h`: Header file containing some utility functions.
- `Makefile`: Very (very) simple makefile needed just for compiling the two source files. 
- `naive.cu`: Source file containing the naive implementation.
- `sm.cu`: Source file containing the shared memory implementation.

**Warning**: Inside the Makefile you will have to change the compilation command to match your local installation of `nvcc`. My installation was located under `/usr/local/cuda-11/` but yours could be placed into another location.

#### Running CUDA

Inside `CUDA/constants.h` you will find something like

```cpp
// Hyperparameters
constexpr float RADIUS = 60;
constexpr float SIGMA = 4;
constexpr float MIN_DISTANCE = 60;
constexpr float DBL_SIGMA_SQ = (2 * SIGMA * SIGMA);
constexpr size_t NUM_ITER = 50;
// Dataset
const std::string PATH_TO_DATA = "datasets/3d/5000_samples_3_centers/data.csv";
constexpr int N = 5000;
constexpr int D = 3;
// Device
constexpr int THREADS = 64;
constexpr int BLOCKS = (N + THREADS - 1) / THREADS;
constexpr int TILE_WIDTH = THREADS;
```

these are almost the same constants as in the previous cases (sequential and openmp). Names are different and, in a future commit, I will try to use the same nomenclature between the implementations.

- `SIGMA` is the standard deviation of the gaussian used to compute the kernel.
- `RADIUS` is used for determining the neighbors of a point.
- `MIN_DISTANCE` is the minimum (L2 squared) distance between two points to be considered belonging to the same clusters.
- `N` is the number of data points (**Warning**: this number needs to match the one preceding `_samples` in the `PATH_TO_DATA` variable).
- `D` is the dimensionality of the data set (**Warning**: this number needs to match the one indicating the sub-directory in the `PATH_TO_DATA` variable).
- `NUM_ITER` is the number of iterations that the algorithm will run through.
- `PATH_TO_DATA` is the string containing the path to the `data.csv` that the user wants to cluster.
- `THREADS` is the number of threads in a block.

After setting these constants you run
```
$ make
```
decide which version (naive or tiled) to execute and run
```
$ ./naive
```
or
```
$ ./sm
```

The compiler I used is [g++](https://man7.org/linux/man-pages/man1/g++.1.html) 9.3 with `-std=c++17`.
CUDA version was 11.

---

[1]: https://en.wikipedia.org/wiki/Mean_shift
