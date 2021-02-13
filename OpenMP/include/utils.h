#ifndef UTILS_H
#define UTILS_H

#include "container.h"
#include <vector>
#include <numeric> // iota
        
namespace mean_shift {

    template<typename T, const size_t D>
    double calc_distance(const vec<T, D>& p, const vec<T, D>& q) {
        double sum = 0.0;
        for (size_t i = 0; i < D; ++i)
            sum += ((p[i] - q[i]) * (p[i] - q[i]));
        return sum;
    }

    template<typename T, const size_t M, const size_t D>
    bool are_close_to_real(const std::vector<vec<T, D>>& centroids, const mat<T, M, D>& real, const double eps_to_real) {
        vec<bool, M> are_close {false};
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < M; ++j) {
                if (calc_distance(centroids[i], real[j]) <= eps_to_real)
                    are_close[i] = true;
            }
        }
        return std::all_of(are_close.begin(), are_close.end(), [](const bool b){return b;});   
    }

    template <typename T, const size_t  D>
    bool is_centroid(std::vector<vec<T, D>>& curr_centroids, const vec<T, D>& point, const double eps_clust) {
        return std::none_of(curr_centroids.begin(), 
                            curr_centroids.end(), 
                            [&](auto& c) {return calc_distance(c, point) <= eps_clust;});
    }

    template <typename T, const size_t N, const size_t D>
    std::vector<vec<T, D>> reduce_to_centroids(mat<T, N, D>& data, const float min_distance) {
        std::vector<vec<T, D>> centroids = {data[0]};
        for (const auto& p : data) {
            if (is_centroid(centroids, p, min_distance))
                centroids.emplace_back(p);
        }
        return centroids;
    }

    template <typename T, const size_t N, const size_t D>
    std::vector<vec<T, D>> reduce_to_centroids_abs(mat<T, N, D>& data, const float min_distance) {
        std::vector<vec<T, D>> centroids = {data[0]};
        vec<size_t, D> range;
        std::iota(range.begin(), range.end(), 0);
        for (const auto& p : data) {
            bool at_least_one_close = false;
            for (const auto& c : centroids) {
                if (std::all_of(range.begin(), range.end(), [&] (size_t j) {return std::abs(p[j] - c[j]) <= min_distance;})) {
                    at_least_one_close = true;
                    continue;    
                }
            }
            if (not at_least_one_close) {
                centroids.emplace_back(p);
            }
        }
        return centroids;
    }
}

#endif