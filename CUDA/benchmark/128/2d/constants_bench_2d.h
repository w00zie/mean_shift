#ifndef CONSTANTS_BENCH_2D_H
#define CONSTANTS_BENCH_2D_H

#include <string>

namespace mean_shift::cuda::bench_2d {

    constexpr size_t THREADS = 128;
    
    constexpr size_t TILE_WIDTH = THREADS;
    constexpr size_t NUM_ITER = 50;
    constexpr float DIST_TO_REAL = 10;
    constexpr size_t M = 3;
    constexpr size_t NUM_TRIALS = 10;
    constexpr size_t D = 2;

    namespace case_500 {

        // Hyperparameters
        constexpr float RADIUS = 30;
        constexpr float SIGMA = 3;
        constexpr float DBL_SIGMA_SQ = (2 * SIGMA * SIGMA);
        constexpr float MIN_DISTANCE = 60;
        // Dataset
        const std::string PATH_TO_DATA = "../../../../../datasets/2d/500_samples_3_centers/data.csv";
        const std::string PATH_TO_CENTROIDS = "../../../../../datasets/2d/500_samples_3_centers/centroids.csv";
        const std::string LOG_NAIVE = "../../timings/naive/2d_500.csv";
        const std::string LOG_SM = "../../timings/sm/2d_500.csv";
        constexpr size_t N = 500;
        // Device
        constexpr size_t BLOCKS = (N + THREADS - 1) / THREADS;

    }

    namespace case_1000 {

        // Hyperparameters
        constexpr float RADIUS = 30;
        constexpr float SIGMA = 3;
        constexpr float DBL_SIGMA_SQ = (2 * SIGMA * SIGMA);
        constexpr float MIN_DISTANCE = 60;
        // Dataset
        const std::string PATH_TO_DATA = "../../../../../datasets/2d/1000_samples_3_centers/data.csv";
        const std::string PATH_TO_CENTROIDS = "../../../../../datasets/2d/1000_samples_3_centers/centroids.csv";
        const std::string LOG_NAIVE = "../../timings/naive/2d_1000.csv";
        const std::string LOG_SM = "../../timings/sm/2d_1000.csv";
        constexpr size_t N = 1000;
        // Device
        constexpr size_t BLOCKS = (N + THREADS - 1) / THREADS;

    }

    namespace case_2000 {

        // Hyperparameters
        constexpr float RADIUS = 30;
        constexpr float SIGMA = 3;
        constexpr float DBL_SIGMA_SQ = (2 * SIGMA * SIGMA);
        constexpr float MIN_DISTANCE = 60;
        // Dataset
        const std::string PATH_TO_DATA = "../../../../../datasets/2d/2000_samples_3_centers/data.csv";
        const std::string PATH_TO_CENTROIDS = "../../../../../datasets/2d/2000_samples_3_centers/centroids.csv";
        const std::string LOG_NAIVE = "../../timings/naive/2d_2000.csv";
        const std::string LOG_SM = "../../timings/sm/2d_2000.csv";
        constexpr size_t N = 2000;
        // Device
        constexpr size_t BLOCKS = (N + THREADS - 1) / THREADS;

    }

    namespace case_5000 {

        // Hyperparameters
        constexpr float RADIUS = 30;
        constexpr float SIGMA = 3;
        constexpr float DBL_SIGMA_SQ = (2 * SIGMA * SIGMA);
        constexpr float MIN_DISTANCE = 60;
        // Dataset
        const std::string PATH_TO_DATA = "../../../../../datasets/2d/5000_samples_3_centers/data.csv";
        const std::string PATH_TO_CENTROIDS = "../../../../../datasets/2d/5000_samples_3_centers/centroids.csv";
        const std::string LOG_NAIVE = "../../timings/naive/2d_5000.csv";
        const std::string LOG_SM = "../../timings/sm/2d_5000.csv";
        constexpr size_t N = 5000;
        // Device
        constexpr size_t BLOCKS = (N + THREADS - 1) / THREADS;

    }

}

#endif