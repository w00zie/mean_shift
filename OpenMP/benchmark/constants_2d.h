#ifndef CONSTANTS_BENCH_2D
#define CONSTANTS_BENCH_2D

#include <string>

namespace constants {

    constexpr size_t niter = 50;
    constexpr size_t dim = 2;
    constexpr size_t num_centroids = 3;
    constexpr size_t num_trials = 10;

    namespace case_500 {
        // Dataset
        constexpr size_t num_points = 500;
        // Hyperparameters
        constexpr float bandwidth = 3;
        constexpr float radius = 30;
        constexpr float min_distance = 60;
    }

    namespace case_1000 {
        // Dataset
        constexpr size_t num_points = 1000;
        // Hyperparameters
        constexpr float bandwidth = 3;
        constexpr float radius = 30;
        constexpr float min_distance = 60;
    }

    namespace case_2000 {
        // Dataset
        constexpr size_t num_points = 2000;
        // Hyperparameters
        constexpr float bandwidth = 3;
        constexpr float radius = 30;
        constexpr float min_distance = 60;
    }

    namespace case_5000 {
        // Dataset
        constexpr size_t num_points = 5000;
        // Hyperparameters
        constexpr float bandwidth = 3;
        constexpr float radius = 30;
        constexpr float min_distance = 60;
    }

} // namespace constants

#endif