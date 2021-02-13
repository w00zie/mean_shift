#ifndef CONSTANTS_TEST_2D_H
#define CONSTANTS_TEST_2D_H

#include <string>

namespace mean_shift::cuda::test_2d {

    constexpr size_t THREADS = 256;
    constexpr size_t NUM_ITER = 50;
    constexpr float DIST_TO_REAL = 10;
    constexpr size_t M = 3;     // 3 cluster centers

    namespace case_500 {

        // Hyperparameters
        constexpr float RADIUS = 30;   // 20
        constexpr float SIGMA = 3;     // 3
        constexpr float DBL_SIGMA_SQ = (2 * SIGMA * SIGMA);
        constexpr float MIN_DISTANCE = 60; // 60
        // Dataset
        const std::string PATH_TO_DATA = "../../../../datasets/2d/500_samples_3_centers/data.csv";
        const std::string PATH_TO_CENTROIDS = "../../../../datasets/2d/500_samples_3_centers/centroids.csv";
        constexpr size_t N = 500;
        constexpr size_t D = 2;
        // Device
        constexpr size_t BLOCKS = (N + THREADS - 1) / THREADS;
        constexpr size_t TILE_WIDTH = THREADS;

    }

    namespace case_1000 {

        // Hyperparameters
        constexpr float RADIUS = 30;
        constexpr float SIGMA = 3;
        constexpr float DBL_SIGMA_SQ = (2 * SIGMA * SIGMA);
        constexpr float MIN_DISTANCE = 60;
        // Dataset
        const std::string PATH_TO_DATA = "../../../../datasets/2d/1000_samples_3_centers/data.csv";
        const std::string PATH_TO_CENTROIDS = "../../../../datasets/2d/1000_samples_3_centers/centroids.csv";
        constexpr size_t N = 1000;
        constexpr size_t D = 2;
        // Device
        constexpr size_t BLOCKS = (N + THREADS - 1) / THREADS;
        constexpr size_t TILE_WIDTH = THREADS;

    }

    namespace case_2000 {

        // Hyperparameters
        constexpr float RADIUS = 30;
        constexpr float SIGMA = 3;
        constexpr float DBL_SIGMA_SQ = (2 * SIGMA * SIGMA);
        constexpr float MIN_DISTANCE = 60;
        // Dataset
        const std::string PATH_TO_DATA = "../../../../datasets/2d/2000_samples_3_centers/data.csv";
        const std::string PATH_TO_CENTROIDS = "../../../../datasets/2d/2000_samples_3_centers/centroids.csv";
        constexpr size_t N = 2000;
        constexpr size_t D = 2;
        // Device
        constexpr size_t BLOCKS = (N + THREADS - 1) / THREADS;
        constexpr size_t TILE_WIDTH = THREADS;

    }

    namespace case_5000 {

        // Hyperparameters
        constexpr float RADIUS = 60;
        constexpr float SIGMA = 3;
        constexpr float DBL_SIGMA_SQ = (2 * SIGMA * SIGMA);
        constexpr float MIN_DISTANCE = 60;
        // Dataset
        const std::string PATH_TO_DATA = "../../../../datasets/2d/5000_samples_3_centers/data.csv";
        const std::string PATH_TO_CENTROIDS = "../../../../datasets/2d/5000_samples_3_centers/centroids.csv";
        constexpr size_t N = 5000;
        constexpr size_t D = 2;
        // Device
        constexpr size_t BLOCKS = (N + THREADS - 1) / THREADS;
        constexpr size_t TILE_WIDTH = THREADS;

    }

}

#endif