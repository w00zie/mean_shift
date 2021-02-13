#ifndef THREADS_1
#define THREADS_1

#include <string>

namespace constants {

    constexpr size_t num_threads = 1;
    
    namespace case_500 {
        const std::string data_path_2d = "../../../../../datasets/2d/500_samples_3_centers/data.csv";
        const std::string data_path_3d = "../../../../../datasets/3d/500_samples_3_centers/data.csv";
        const std::string out_file_2d = "../timings/2d_500.csv";
        const std::string out_file_3d = "../timings/3d_500.csv";
    }

    namespace case_1000 {
        const std::string data_path_2d = "../../../../../datasets/2d/1000_samples_3_centers/data.csv";
        const std::string data_path_3d = "../../../../../datasets/3d/1000_samples_3_centers/data.csv";
        const std::string out_file_2d = "../timings/2d_1000.csv";
        const std::string out_file_3d = "../timings/3d_1000.csv";
    }

    namespace case_2000 {
        const std::string data_path_2d = "../../../../../datasets/2d/2000_samples_3_centers/data.csv";
        const std::string data_path_3d = "../../../../../datasets/3d/2000_samples_3_centers/data.csv";
        const std::string out_file_2d = "../timings/2d_2000.csv";
        const std::string out_file_3d = "../timings/3d_2000.csv";
    }

    namespace case_5000 {
        const std::string data_path_2d = "../../../../../datasets/2d/5000_samples_3_centers/data.csv";
        const std::string data_path_3d = "../../../../../datasets/3d/5000_samples_3_centers/data.csv";
        const std::string out_file_2d = "../timings/2d_5000.csv";
        const std::string out_file_3d = "../timings/3d_5000.csv";
    }

}

#endif