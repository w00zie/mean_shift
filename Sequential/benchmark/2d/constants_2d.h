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
        const std::string data_path = "../../../datasets/2d/500_samples_3_centers/data.csv";
        const std::string centroids_path = "../../../datasets/2d/500_samples_3_centers/centroids.csv";
        const std::string out_file = "../timings/2d_500.csv";

        // Hyperparameters
        constexpr float bandwidth = 3;
        constexpr float radius = 30;
        constexpr float min_distance = 60;
    }

    namespace case_1000 {
        // Dataset
        constexpr size_t num_points = 1000;
        const std::string data_path = "../../../datasets/2d/1000_samples_3_centers/data.csv";
        const std::string centroids_path = "../../../datasets/2d/1000_samples_3_centers/centroids.csv";
        const std::string out_file = "../timings/2d_1000.csv";
        // Hyperparameters
        constexpr float bandwidth = 3;
        constexpr float radius = 30;
        constexpr float min_distance = 60;
    }

    namespace case_2000 {
        // Dataset
        constexpr size_t num_points = 2000;
        const std::string data_path = "../../../datasets/2d/2000_samples_3_centers/data.csv";
        const std::string centroids_path = "../../../datasets/2d/2000_samples_3_centers/centroids.csv";
        const std::string out_file = "../timings/2d_2000.csv";

        // Hyperparameters
        constexpr float bandwidth = 3;
        constexpr float radius = 30;
        constexpr float min_distance = 60;
    }

    namespace case_5000 {
        // Dataset
        constexpr size_t num_points = 5000;
        const std::string data_path = "../../../datasets/2d/5000_samples_3_centers/data.csv";
        const std::string centroids_path = "../../../datasets/2d/5000_samples_3_centers/centroids.csv";
        const std::string out_file = "../timings/2d_5000.csv";

        // Hyperparameters
        constexpr float bandwidth = 3;
        constexpr float radius = 30;
        constexpr float min_distance = 60;
    }

} // namespace constants

#endif