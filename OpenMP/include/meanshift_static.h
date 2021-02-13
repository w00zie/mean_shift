#ifndef MEAN_SHIFT_STATIC_H
#define MEAN_SHIFT_STATIC_H

#include <algorithm>
#include <cmath>
#include "container.h"
#include "container_io.h"
#include <iostream>
#include "utils.h"

namespace mean_shift {

    namespace omp {

        template <typename T, const size_t N, const size_t D>
        std::vector<vec<T, D>> cluster_points(mat<T, N, D>& data, 
                                              const size_t niter, 
                                              const float bandwidth,
                                              const float radius,
                                              const float min_distance, 
                                              const double eps,
                                              const size_t num_threads) {
            const float double_sqr_bdw = 2 * bandwidth * bandwidth;
            vec<bool, N> has_stopped {false};
            std::vector<vec<T, D>> centroids;
            mat<T, N, D> new_data;
            for (size_t i = 0; i < niter; ++i) {
                #pragma omp parallel for default(none) \
                shared(data, niter, bandwidth, eps, radius, double_sqr_bdw, has_stopped, centroids, new_data, min_distance) \
                schedule(static) num_threads(num_threads)
                for (size_t p = 0; p < N; ++p) {
                    if (has_stopped[p]) {
                        #pragma omp critical
                        {
                            if ((centroids.size() == 0) || (is_centroid(centroids, data[p], min_distance))) {
                                centroids.emplace_back(data[p]);
                            }
                        }
                        continue;
                    }
                    vec<T, D> new_position {};
                    float sum_weights = 0.;
                    for (size_t q = 0; q < N; ++q) {
                        double dist = calc_distance(data[p], data[q]);
                        if (dist <= radius) {
                            float gaussian = std::exp(- dist / double_sqr_bdw);
                            new_position = new_position + data[q] * gaussian;
                            sum_weights += gaussian;
                        }
                    }
                    new_position = new_position / sum_weights;
                    double shift = calc_distance(data[p], new_position);
                    if (shift <= eps) {
                        #pragma omp atomic write
                        has_stopped[p] = true;
                    }
                    #pragma omp critical
                    new_data[p] = new_position;
                }
                data = new_data;
                if (std::all_of(has_stopped.begin(), has_stopped.end(), [](bool b) {return b;})) {
                    std::cout << "With eps = " << eps << " took " << i << " iterations!\n";
                    return centroids;
                }
            }
            return centroids;
        }

        template <typename T, const size_t N, const size_t D>
        std::vector<vec<T, D>> cluster_points(mat<T, N, D>& data, 
                                              const size_t niter, 
                                              const float bandwidth,
                                              const float radius,
                                              const float min_distance,
                                              const size_t num_threads) {
            const float double_sqr_bdw = 2 * bandwidth * bandwidth;
            mat<T, N, D> new_data;
            for (size_t i = 0; i < niter; ++i) {
                #pragma omp parallel for default(none) \
                shared(data, niter, bandwidth, radius, double_sqr_bdw, new_data) \
                schedule(static) num_threads(num_threads)
                for (size_t p = 0; p < N; ++p) {
                    vec<T, D> new_position {};
                    float sum_weights = 0.;
                    for (size_t q = 0; q < N; ++q) {
                        double dist = calc_distance(data[p], data[q]);
                        if (dist <= radius) {
                            float gaussian = std::exp(- dist / double_sqr_bdw);
                            new_position = new_position + data[q] * gaussian;
                            sum_weights += gaussian;
                        }
                    }
                    #pragma omp critical
                    new_data[p] = new_position / sum_weights;
                }
                data = new_data;
            }
            return reduce_to_centroids(data, min_distance);
        }

    } // namespace omp

} // namespace mean_shift

#endif