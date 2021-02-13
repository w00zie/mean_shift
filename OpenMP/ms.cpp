#include <cassert>
#include <chrono>
#include "include/meanshift.h"
#include "include/container_io.h"

int main() {

    // Hyperparameters
    const float bandwidth = 3;
    const float radius = 30;
    const float min_distance = 60;
    const size_t niter = 50;
    const double eps = 0;
    // I/O
    const size_t num_points = 5000;
    const size_t dim = 2;
    const std::string data_path = "../datasets/2d/5000_samples_3_centers/data.csv";
    const std::string centroids_path = "../datasets/2d/5000_samples_3_centers/centroids.csv";
    // Final check
    const size_t num_centroids = 3;
    const double eps_to_real = 10;

    mean_shift::mat<float, num_points, dim> data = mean_shift::io::load_csv<float, num_points, dim>(data_path, ',');
    const mean_shift::mat<float, num_centroids, dim> real_centroids = mean_shift::io::load_csv<float, num_centroids, dim>(centroids_path, ',');
    if (eps > 0) {
        auto start = std::chrono::high_resolution_clock::now();
        const std::vector<mean_shift::vec<float, dim>> centroids = mean_shift::omp::cluster_points<float, num_points, dim>(data, niter, bandwidth, radius, min_distance, eps);    
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Duration: " << duration << " ms" << "\n\n";
        mean_shift::io::print_mat(centroids);
        std::cout << "There are " << centroids.size() << " centroids.\n";
        assert(centroids.size() == num_centroids);
        bool are_close = mean_shift::are_close_to_real<float, num_centroids, dim>(centroids, real_centroids, eps_to_real);
        assert(are_close);
    } else {
        auto start = std::chrono::high_resolution_clock::now();
        const std::vector<mean_shift::vec<float, dim>> centroids = mean_shift::omp::cluster_points<float, num_points, dim>(data, niter, bandwidth, radius, min_distance);    
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Duration: " << duration << " ms" << "\n\n";
        mean_shift::io::print_mat(centroids);
        std::cout << "There are " << centroids.size() << " centroids.\n";
        assert(centroids.size() == num_centroids);
        bool are_close = mean_shift::are_close_to_real<float, num_centroids, dim>(centroids, real_centroids, eps_to_real);
        assert(are_close);
    }
    return 0;
}