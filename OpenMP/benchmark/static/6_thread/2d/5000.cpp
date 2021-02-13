#include <chrono>
#include "../../../../include/meanshift_static.h"
#include "../../../../include/container_io.h"
#include "../../../constants_2d.h"
#include "../thd_const.h"
#include <thread>

// Hyperparameters
constexpr auto bandwidth = constants::case_5000::bandwidth;
constexpr auto radius = constants::case_5000::radius;
constexpr auto min_distance = constants::case_5000::min_distance;
constexpr auto niter = constants::niter;
constexpr auto num_trials = constants::num_trials;
constexpr auto num_threads = constants::num_threads;
// I/O
constexpr auto num_points = constants::case_5000::num_points;
constexpr auto dim = constants::dim;
constexpr auto num_centroids = constants::num_centroids;
const auto data_path = constants::case_5000::data_path;
const auto out_file = constants::case_5000::out_file_2d;

double run_once() {

    mean_shift::mat<float, num_points, dim> data = mean_shift::io::load_csv<float, num_points, dim>(data_path, ',');    
    auto start = std::chrono::high_resolution_clock::now();
    const std::vector<mean_shift::vec<float, dim>> centroids = mean_shift::omp::cluster_points<float, num_points, dim>(data, niter, bandwidth, radius, min_distance, num_threads);    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return duration;
}

int main(){
    mean_shift::vec<double, num_trials> exec_times;
    for (size_t i = 0; i < num_trials; ++i) {
        exec_times[i] = run_once();
    }

    std::this_thread::sleep_for (std::chrono::seconds(1));

    mean_shift::io::write_csv<double, num_trials>(exec_times, out_file);

    return 0;
}