#ifndef FILES_BENCH_3D
#define FILES_BENCH_3D

#include <string>

namespace constants {
    
    namespace case_500 {
        const std::string data_path = "../../../../datasets/3d/500_samples_3_centers/data.csv";
        const std::string centroids_path = "../../../../datasets/3d/500_samples_3_centers/centroids.csv";
        const std::string out_file = "../timings/3d_500.csv";
    }

    namespace case_1000 {
        const std::string data_path = "../../../../datasets/3d/1000_samples_3_centers/data.csv";
        const std::string centroids_path = "../../../../datasets/3d/1000_samples_3_centers/centroids.csv";
        const std::string out_file = "../timings/3d_1000.csv";
    }

    namespace case_2000 {
        const std::string data_path = "../../../../datasets/3d/2000_samples_3_centers/data.csv";
        const std::string centroids_path = "../../../../datasets/3d/2000_samples_3_centers/centroids.csv";
        const std::string out_file = "../timings/3d_2000.csv";
    }

    namespace case_5000 {
        const std::string data_path = "../../../../datasets/3d/5000_samples_3_centers/data.csv";
        const std::string centroids_path = "../../../../datasets/3d/5000_samples_3_centers/centroids.csv";
        const std::string out_file = "../timings/3d_5000.csv";
    }

} // namespace constants


#endif