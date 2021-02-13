#include <cassert>
#include <chrono>
#include "../constants_bench_3d.h"
#include <cuda.h>
#include <iostream>
#include "../../../../utils.h"

// Dataset
constexpr auto N = mean_shift::cuda::bench_3d::case_5000::N;
constexpr auto D = mean_shift::cuda::bench_3d::D;
constexpr auto M = mean_shift::cuda::bench_3d::M;
const auto PATH_TO_DATA = mean_shift::cuda::bench_3d::case_5000::PATH_TO_DATA; 
const auto PATH_TO_CENTROIDS = mean_shift::cuda::bench_3d::case_5000::PATH_TO_CENTROIDS;
const auto LOG_SM = mean_shift::cuda::bench_3d::case_5000::LOG_SM;
// Hyperparams
constexpr auto RADIUS = mean_shift::cuda::bench_3d::case_5000::RADIUS;
constexpr auto NUM_ITER = mean_shift::cuda::bench_3d::NUM_ITER;
constexpr auto DBL_SIGMA_SQ = mean_shift::cuda::bench_3d::case_5000::DBL_SIGMA_SQ;
constexpr auto MIN_DISTANCE = mean_shift::cuda::bench_3d::case_5000::MIN_DISTANCE;
// Device
constexpr auto THREADS = mean_shift::cuda::bench_3d::THREADS;
constexpr auto BLOCKS = mean_shift::cuda::bench_3d::case_5000::BLOCKS;
constexpr auto TILE_WIDTH = mean_shift::cuda::bench_3d::TILE_WIDTH;

// Benchmarking
constexpr auto NUM_TRIALS = mean_shift::cuda::bench_3d::NUM_TRIALS;

__global__ void mean_shift_tiling(float *data, float *data_next) {

    // Shared memory allocation
    __shared__ float local_data[TILE_WIDTH * D];
    // A few convenient variables
    int last_tile = (N - 1) / TILE_WIDTH + 1;
    int local_row = threadIdx.x * D;
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int row = tid * D;
    float new_position[D] = {0.};
    float tot_weight = 0.;
    // Load data in shared memory
    for (int t = 0; t < last_tile; ++t) {
        int tid_in_tile = t * TILE_WIDTH + threadIdx.x;
        if (tid_in_tile < N) {
            int row_in_tile = tid_in_tile * D;
            for (int j = 0; j < D; ++j) {
                local_data[local_row + j] = data[row_in_tile + j];
            }
        }
        else {
            for (int j = 0; j < D; ++j) {
                local_data[local_row + j] = 0;
            }
        }
        __syncthreads();
        if (tid < N) {
            for (int i = 0; i < TILE_WIDTH; ++i) {
                int local_row_tile = i * D;
                float sq_dist = 0.;
                for (int j = 0; j < D; ++j) {
                    sq_dist += (data[row + j] - local_data[local_row_tile + j]) * (data[row + j] - local_data[local_row_tile + j]);
                }
                if (sq_dist <= RADIUS) {
                    float weight = expf(-sq_dist / DBL_SIGMA_SQ);
                    for (int j = 0; j < D; ++j) {
                        new_position[j] += weight * local_data[local_row_tile + j];
                    }
                    tot_weight += weight;
                }
            }
        }
        __syncthreads();
    }
    if (tid < N) {
        for (int j = 0; j < D; ++j) {
            data_next[row + j] = new_position[j] / tot_weight;
        }
    }
    return;
}

double run_once() {
    // Load data
    std::array<float, N * D> data = mean_shift::cuda::utils::load_csv<N, D>(PATH_TO_DATA, ',');
    std::array<float, N * D> data_next {};
    float *dev_data;
    float *dev_data_next;
    // Allocate GPU memory
    size_t data_bytes = N * D * sizeof(float);
    cudaMalloc(&dev_data, data_bytes);
    cudaMalloc(&dev_data_next, data_bytes);
    // Copy to GPU memory
    cudaMemcpy(dev_data, data.data(), data_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_data_next, data_next.data(), data_bytes, cudaMemcpyHostToDevice);
    // Run mean shift clustering
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < NUM_ITER; ++i) {
        mean_shift_tiling<<<BLOCKS, THREADS>>>(dev_data, dev_data_next);
        cudaDeviceSynchronize();
        mean_shift::cuda::utils::swap(dev_data, dev_data_next);
    }
    cudaMemcpy(data.data(), dev_data, data_bytes, cudaMemcpyDeviceToHost);
    const auto centroids = mean_shift::cuda::utils::reduce_to_centroids<N, D>(data, MIN_DISTANCE);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // Check if correct number
    assert(centroids.size() == M);
    
    return duration;
}

int main() {

    std::array<double, NUM_TRIALS> exec_times;

    for (auto i = 0; i < NUM_TRIALS; ++i)
        exec_times[i] = run_once();

    mean_shift::cuda::utils::write_csv<double, NUM_TRIALS>(exec_times, LOG_SM, ',');

    return 0;

}