#include <chrono>
#include "constants.h"
#include <cuda.h>
#include <iostream>
#include "utils.h"

namespace mean_shift::cuda {

    __global__ void mean_shift(float *data, float *data_next) {
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

}

int main() {

    constexpr auto N = mean_shift::cuda::N;
    constexpr auto D = mean_shift::cuda::D;
    constexpr auto M = mean_shift::cuda::M;
    constexpr auto THREADS = mean_shift::cuda::THREADS;
    constexpr auto BLOCKS = mean_shift::cuda::BLOCKS;
    const auto PATH_TO_DATA = mean_shift::cuda::PATH_TO_DATA; 
    const auto PATH_TO_CENTROIDS = mean_shift::cuda::PATH_TO_CENTROIDS;
    constexpr auto DIST_TO_REAL = mean_shift::cuda::DIST_TO_REAL;

    mean_shift::cuda::utils::print_info(PATH_TO_DATA, N, D, BLOCKS, THREADS);

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
    // Run mean shift clustering and time the execution
    const auto before = std::chrono::system_clock::now();
    for (size_t i = 0; i < mean_shift::cuda::NUM_ITER; ++i) {
        mean_shift::cuda::mean_shift<<<BLOCKS, THREADS>>>(dev_data, dev_data_next);
        cudaDeviceSynchronize();
        mean_shift::cuda::utils::swap(dev_data, dev_data_next);
    }
    cudaMemcpy(data.data(), dev_data, data_bytes, cudaMemcpyDeviceToHost);
    const auto centroids = mean_shift::cuda::utils::reduce_to_centroids<N, D>(data, mean_shift::cuda::MIN_DISTANCE);
    const auto after = std::chrono::system_clock::now();
    const std::chrono::duration<double, std::milli> duration = after - before;
    std::cout << "\nNaive took " << duration.count() << " ms\n" << std::endl;

    cudaFree(dev_data);
    cudaFree(dev_data_next);
    mean_shift::cuda::utils::print_data<D>(centroids);
    // Check if correct number
    assert(centroids.size() == M);
    // Check if these centroids are sufficiently close to real ones
    const std::array<float, M * D> real = mean_shift::cuda::utils::load_csv<M, D>(PATH_TO_CENTROIDS, ',');
    const bool are_close = mean_shift::cuda::utils::are_close_to_real<M, D>(centroids, real, DIST_TO_REAL);
    assert(are_close);
    std::cout << "SUCCESS!\n";

    return 0;
}