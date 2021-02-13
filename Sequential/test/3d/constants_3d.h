#ifndef CONSTANTS_BENCH_3D
#define CONSTANTS_BENCH_3D

#include <string>

namespace constants {

    constexpr size_t niter = 50;
    constexpr size_t dim = 3;
    constexpr size_t num_centroids = 3;

    namespace case_500 {
        // Dataset
        constexpr size_t num_points = 500;
        const std::string data_path = "../../../../datasets/3d/500_samples_3_centers/data.csv";
        const std::string centroids_path = "../../../../datasets/3d/500_samples_3_centers/centroids.csv";
        const std::string out_file = "../timings/3d_500.csv";

        // Hyperparameters
        constexpr float bandwidth = 4;
        constexpr float radius = 60;
        constexpr float min_distance = 60;
        constexpr float eps = 1e-4;
    }

    namespace case_1000 {
        // Dataset
        constexpr size_t num_points = 1000;
        const std::string data_path = "../../../../datasets/3d/1000_samples_3_centers/data.csv";
        const std::string centroids_path = "../../../../datasets/3d/1000_samples_3_centers/centroids.csv";
        const std::string out_file = "../timings/3d_1000.csv";
        // Hyperparameters
        constexpr float bandwidth = 4;
        constexpr float radius = 60;
        constexpr float min_distance = 60;
        constexpr float eps = 1e-4;
    }

    namespace case_2000 {
        // Dataset
        constexpr size_t num_points = 2000;
        const std::string data_path = "../../../../datasets/3d/2000_samples_3_centers/data.csv";
        const std::string centroids_path = "../../../../datasets/3d/2000_samples_3_centers/centroids.csv";
        const std::string out_file = "../timings/3d_2000.csv";

        // Hyperparameters
        constexpr float bandwidth = 4;
        constexpr float radius = 60;
        constexpr float min_distance = 60;
        constexpr float eps = 1e-4;
    }

    namespace case_5000 {
        // Dataset
        constexpr size_t num_points = 5000;
        const std::string data_path = "../../../../datasets/3d/5000_samples_3_centers/data.csv";
        const std::string centroids_path = "../../../../datasets/3d/5000_samples_3_centers/centroids.csv";
        const std::string out_file = "../timings/3d_5000.csv";

        // Hyperparameters
        constexpr float bandwidth = 4;
        constexpr float radius = 60;
        constexpr float min_distance = 60;
        constexpr float eps = 1e-4;
    }

} // namespace constants

#endif