#include <cassert>
#include "../../../include/meanshift.h"
#include "../../../include/container_io.h"

int main(int argc, char *argv[]) {

    // Hyperparameters
    const float bandwidth = std::stof(argv[1]);
    const float radius = std::stof(argv[2]);
    const float min_distance = std::stof(argv[3]);
    const size_t niter = 100;
    // I/O
    const size_t num_points = 10000;
    const size_t dim = 3;
    const std::string data_path = "../../../../datasets/3d/10000_samples_3_centers/data.csv";
    const std::string centroids_path = "../../../../datasets/3d/10000_samples_3_centers/centroids.csv";
    // Final check
    const size_t num_centroids = 3;
    const double eps_to_real = std::stod(argv[4]);

    mean_shift::mat<float, num_points, dim> data = mean_shift::io::load_csv<float, num_points, dim>(data_path, ',');
    const mean_shift::mat<float, num_centroids, dim> real_centroids = mean_shift::io::load_csv<float, num_centroids, dim>(centroids_path, ',');
    const std::vector<mean_shift::vec<float, dim>> centroids = mean_shift::omp::cluster_points<float, num_points, dim>(data, niter, bandwidth, radius, min_distance);    
    assert(centroids.size() == num_centroids);
    bool are_close = mean_shift::are_close_to_real<float, num_centroids, dim>(centroids, real_centroids, eps_to_real);
    assert(are_close);

    return 0;
}