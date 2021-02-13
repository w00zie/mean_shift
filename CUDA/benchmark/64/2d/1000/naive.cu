#include <cassert>
#include <chrono>
#include "../constants_bench_2d.h"
#include <cuda.h>
#include <iostream>
#include "../../../../utils.h"

// Dataset
constexpr auto N = mean_shift::cuda::bench_2d::case_1000::N;
constexpr auto D = mean_shift::cuda::bench_2d::D;
constexpr auto M = mean_shift::cuda::bench_2d::M;
const auto PATH_TO_DATA = mean_shift::cuda::bench_2d::case_1000::PATH_TO_DATA; 
const auto PATH_TO_CENTROIDS = mean_shift::cuda::bench_2d::case_1000::PATH_TO_CENTROIDS;
const auto LOG_NAIVE = mean_shift::cuda::bench_2d::case_1000::LOG_NAIVE;
// Hyperparams
constexpr auto RADIUS = mean_shift::cuda::bench_2d::case_1000::RADIUS;
constexpr auto NUM_ITER = mean_shift::cuda::bench_2d::NUM_ITER;
constexpr auto DBL_SIGMA_SQ = mean_shift::cuda::bench_2d::case_1000::DBL_SIGMA_SQ;
constexpr auto MIN_DISTANCE = mean_shift::cuda::bench_2d::case_1000::MIN_DISTANCE;
// Device
constexpr auto THREADS = mean_shift::cuda::bench_2d::THREADS;
constexpr auto BLOCKS = mean_shift::cuda::bench_2d::case_1000::BLOCKS;

// Benchmarking
constexpr auto NUM_TRIALS = mean_shift::cuda::bench_2d::NUM_TRIALS;

__global__ void mean_shift_naive(float *data, float *data_next) {
    size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < N) {
        size_t row = tid * D;
        float new_position[D] = {0.};
        float tot_weight = 0.;
        for (size_t i = 0; i < N; ++i) {
            size_t row_n = i * D;
            float sq_dist = 0.;
            for (size_t j = 0; j < D; ++j) {
                sq_dist += (data[row + j] - data[row_n + j]) * (data[row + j] - data[row_n + j]);
            }
            if (sq_dist <= RADIUS) {
                float weight = expf(-sq_dist / DBL_SIGMA_SQ);
                for (size_t j = 0; j < D; ++j) {
                    new_position[j] += weight * data[row_n + j];
                }
                tot_weight += weight;
            }
        }
        for (size_t j = 0; j < D; ++j) {
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
        mean_shift_naive<<<BLOCKS, THREADS>>>(dev_data, dev_data_next);
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

    mean_shift::cuda::utils::write_csv<double, NUM_TRIALS>(exec_times, LOG_NAIVE, ',');

    return 0;

}