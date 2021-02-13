#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <string>

namespace mean_shift::cuda {

    // Hyperparameters
    constexpr float RADIUS = 60;
    constexpr float SIGMA = 4;
    constexpr float DBL_SIGMA_SQ = (2 * SIGMA * SIGMA);
    constexpr float MIN_DISTANCE = 60;
    constexpr size_t NUM_ITER = 50;
    constexpr float DIST_TO_REAL = 10;
    // Dataset
    const std::string PATH_TO_DATA = "../datasets/3d/5000_samples_3_centers/data.csv";
    const std::string PATH_TO_CENTROIDS = "../datasets/3d/5000_samples_3_centers/centroids.csv";
    constexpr int N = 5000;
    constexpr int D = 3;
    constexpr int M = 3;
    // Device
    constexpr int THREADS = 64;
    constexpr int BLOCKS = (N + THREADS - 1) / THREADS;
    constexpr int TILE_WIDTH = THREADS;

}

#endif