#ifndef THREADS_6
#define THREADS_6

#include <string>

namespace constants {

    constexpr size_t num_threads = 6;
    
    namespace case_500 {
        const std::string data_path = "../../../../../datasets/2d/500_samples_3_centers/data.csv";
        const std::string centroids_path = "../../../../../datasets/2d/500_samples_3_centers/centroids.csv";
        const std::string out_file_2d = "../timings/2d_500.csv";
        const std::string out_file_3d = "../timings/3d_500.csv";
    }

    namespace case_1000 {
        const std::string data_path = "../../../../../datasets/2d/1000_samples_3_centers/data.csv";
        const std::string centroids_path = "../../../../../datasets/2d/1000_samples_3_centers/centroids.csv";
        const std::string out_file_2d = "../timings/2d_1000.csv";
        const std::string out_file_3d = "../timings/3d_1000.csv";
    }

    namespace case_2000 {
        const std::string data_path = "../../../../../datasets/2d/2000_samples_3_centers/data.csv";
        const std::string centroids_path = "../../../../../datasets/2d/2000_samples_3_centers/centroids.csv";
        const std::string out_file_2d = "../timings/2d_2000.csv";
        const std::string out_file_3d = "../timings/3d_2000.csv";
    }

    namespace case_5000 {
        const std::string data_path = "../../../../../datasets/2d/5000_samples_3_centers/data.csv";
        const std::string centroids_path = "../../../../../datasets/2d/5000_samples_3_centers/centroids.csv";
        const std::string out_file_2d = "../timings/2d_5000.csv";
        const std::string out_file_3d = "../timings/3d_5000.csv";
    }

}

#endif