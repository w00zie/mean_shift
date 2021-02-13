#ifndef CONT_H
#define CONT_H

#include <cstddef>
#include <array>

namespace mean_shift {

    template <typename T, const size_t D>
    using vec = std::array<T, D>;

    template <typename T, const size_t N, const size_t D>
    using mat = std::array<vec<T, D>, N>;

    template <typename T, const size_t D>
    vec<T, D> operator+(const vec<T, D>& a, const vec<T, D>& b) {
        vec<T, D> res;
        for (size_t i = 0; i < D; ++i)
            res[i] = a[i] + b[i];
        return res;
    }

    template <typename T, const size_t D>
    vec<T, D> operator-(const vec<T, D>& a, const vec<T, D>& b) {
        vec<T, D> res;
        for (size_t i = 0; i < D; ++i)
            res[i] = a[i] - b[i];
        return res;
    }

    template <typename T, const size_t D>
    vec<T, D> operator/(const vec<T, D>& a, const vec<T, D>& b) {
        vec<T, D> res;
        for (size_t i = 0; i < D; ++i)
            res[i] = a[i] / b[i];
        return res;
    }

    template <typename T, typename U, const size_t D>
    vec<T, D> operator/(const vec<T, D>& a, const U b) {
        vec<T, D> res;
        for (size_t i = 0; i < D; ++i)
            res[i] = a[i] / b;
        return res;
    }

    template <typename T, typename U, const size_t D>
    vec<T, D> operator*(const vec<T, D>& a, const U b) {
        vec<T, D> res;
        for (size_t i = 0; i < D; ++i)
            res[i] = a[i] * b;
        return res;
    }

} // namespace mean_shift

#endif