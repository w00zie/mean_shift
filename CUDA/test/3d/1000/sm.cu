#include "../constants_test_3d.h"
#include <cuda.h>
#include <iostream>
#include "../../../utils.h"

// Dataset
constexpr auto N = mean_shift::cuda::test_3d::case_1000::N;
constexpr auto D = mean_shift::cuda::test_3d::case_1000::D;
constexpr auto M = mean_shift::cuda::test_3d::M;
const auto PATH_TO_DATA = mean_shift::cuda::test_3d::case_1000::PATH_TO_DATA; 
const auto PATH_TO_CENTROIDS = mean_shift::cuda::test_3d::case_1000::PATH_TO_CENTROIDS;
// Hyperparams
constexpr auto RADIUS = mean_shift::cuda::test_3d::case_1000::RADIUS;
constexpr auto MIN_DISTANCE = mean_shift::cuda::test_3d::case_1000::MIN_DISTANCE;
constexpr auto NUM_ITER = mean_shift::cuda::test_3d::NUM_ITER;
constexpr auto DBL_SIGMA_SQ = mean_shift::cuda::test_3d::case_1000::DBL_SIGMA_SQ;
constexpr auto DIST_TO_REAL = mean_shift::cuda::test_3d::DIST_TO_REAL;
// Device
constexpr auto THREADS = mean_shift::cuda::test_3d::THREADS;
constexpr auto BLOCKS = mean_shift::cuda::test_3d::case_1000::BLOCKS;
constexpr auto TILE_WIDTH = mean_shift::cuda::test_3d::case_1000::TILE_WIDTH;

__global__ void mean_shift_tiling(const float* data, float* data_next) {

    // Shared memory allocation
    __shared__ float local_data[TILE_WIDTH * D];
    __shared__ float valid_data[TILE_WIDTH];
    // A few convenient variables
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int row = tid * D;
    int local_row = threadIdx.x * D;
    float new_position[D] = {0.};
    float tot_weight = 0.;
    // Load data in shared memory
    for (int t = 0; t < BLOCKS; ++t) {
        int tid_in_tile = t * TILE_WIDTH + threadIdx.x;
        if (tid_in_tile < N) {
            int row_in_tile = tid_in_tile * D;
            for (int j = 0; j < D; ++j) {
                local_data[local_row + j] = data[row_in_tile + j];
            }
            valid_data[threadIdx.x] = 1;
        }
        else {
            for (int j = 0; j < D; ++j) {
                local_data[local_row + j] = 0;
                valid_data[threadIdx.x] = 0;
            }
        }
        __syncthreads();
        for (int i = 0; i < TILE_WIDTH; ++i) {
            int local_row_tile = i * D;
            float valid_radius = RADIUS * valid_data[i];
            float sq_dist = 0.;
            for (int j = 0; j < D; ++j) {
                sq_dist += (data[row + j] - local_data[local_row_tile + j]) * (data[row + j] - local_data[local_row_tile + j]);
            }
            if (sq_dist <= valid_radius) {
                float weight = expf(-sq_dist / DBL_SIGMA_SQ);
                for (int j = 0; j < D; ++j) {
                    new_position[j] += (weight * local_data[local_row_tile + j]);
                }
                tot_weight += (weight * valid_data[i]);
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

int main() {
    mean_shift::cuda::utils::print_info(PATH_TO_DATA, N, D, BLOCKS, THREADS, TILE_WIDTH);
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
    for (size_t i = 0; i < NUM_ITER; ++i) {
        mean_shift_tiling<<<BLOCKS, THREADS>>>(dev_data, dev_data_next);
        cudaDeviceSynchronize();
        mean_shift::cuda::utils::swap(dev_data, dev_data_next);
    }
    cudaMemcpy(data.data(), dev_data, data_bytes, cudaMemcpyDeviceToHost);
    // Copy from GPU and de-allocate
    cudaFree(dev_data);
    cudaFree(dev_data_next);
    // Reduce data to cluster centers
    const auto centroids = mean_shift::cuda::utils::reduce_to_centroids<N, D>(data, MIN_DISTANCE);
    // Check if correct number
    assert(centroids.size() == M);
    // Check if these centroids are sufficiently close to real ones
    const std::array<float, M * D> real = mean_shift::cuda::utils::load_csv<M, D>(PATH_TO_CENTROIDS, ',');
    const bool are_close = mean_shift::cuda::utils::are_close_to_real<M, D>(centroids, real, DIST_TO_REAL);
    assert(are_close);
    std::cout << "\nSUCCESS!\n";
    return 0;

}