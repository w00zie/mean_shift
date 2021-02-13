#include <cassert>
#include <chrono>
#include "../../../include/meanshift.h"
#include "../../../include/container_io.h"
#include "../constants_3d.h"

int main(int argc, char *argv[]){

    const float bandwidth = std::stof(argv[1]);
    const float radius = std::stof(argv[2]);
    const float min_distance = std::stof(argv[3]);
    const float eps_to_real = std::stof(argv[4]);
    // Hyperparameters
    constexpr auto niter = constants::niter;
    // I/O
    constexpr auto num_points = constants::case_5000::num_points;
    constexpr auto dim = constants::dim;
    constexpr auto num_centroids = constants::num_centroids;
    const auto data_path = constants::case_5000::data_path;
    const auto centroids_path = constants::case_5000::centroids_path;

    mean_shift::mat<float, num_points, dim> data = mean_shift::io::load_csv<float, num_points, dim>(data_path, ',');
    const mean_shift::mat<float, num_centroids, dim> real_centroids = mean_shift::io::load_csv<float, num_centroids, dim>(centroids_path, ',');
    const std::vector<mean_shift::vec<float, dim>> centroids = mean_shift::seq::cluster_points<float, num_points, dim>(data, niter, bandwidth, radius, min_distance);    
    assert(centroids.size() == num_centroids);
    bool are_close = mean_shift::are_close_to_real<float, num_centroids, dim>(centroids, real_centroids, eps_to_real);
    assert(are_close);

    return 0;
}