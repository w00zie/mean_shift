#ifndef CONTAINER_IO_H
#define CONTAINER_IO_H

#include <cassert>
#include "container.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace mean_shift::io {

    template <typename T, const size_t N, const size_t D>
    mat<T, N, D> load_csv(const std::string& path, const char delim) {
        assert(std::filesystem::exists(path));
        std::ifstream file(path);
        std::string line;
        mat<T, N, D> data_matrix;
        for (size_t i = 0; i < N; ++i) {
            std::getline(file, line);
            std::stringstream line_stream(line);
            std::string cell;
            vec<T, D> point;
            for (size_t j = 0; j < D; ++j) {
                std::getline(line_stream, cell, delim);
                //trovare il modo di usare from_chars
                //T val;
                //auto res = std::from_chars(cell.data(), cell.data() + cell.size(), val);
                point[j] = static_cast<T>(std::stod(cell));
            }
            data_matrix[i] = point;
        }
        file.close();
        return data_matrix;
    }

    template <typename T, const size_t N, const size_t D>
    void write_csv(const mat<T, N, D>& data, const std::string& path, const char delim) {
        std::ofstream output(path);
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < D - 1; ++j)
                output << data[i][j] << delim;
            output << data[i][D-1] << '\n';
        }
        output.close();
        return;
    }

    template <typename T, const size_t D>
    void write_csv(const vec<T, D>& vector, const std::string& path) {
        std::ofstream output(path);
        for (size_t i = 0; i < D; ++i)
            output << vector[i] << '\n';
        output.close();
        return;
    }


    template <typename T, const size_t D>
    void print_vec(const vec<T, D>& vector) {
        for (auto v : vector)
            std::cout << v << ' ';
        std::cout << '\n';
        return;
    }

    template <typename T>
    void print_vec(const std::vector<T>& vector) {
        for (auto v : vector)
            std::cout << v << ' ';
        std::cout << '\n';
        return;
    }

    template <typename T, const size_t N, const size_t D>
    void print_mat(const mat<T, N, D>& matrix) {
        for (const auto& vector : matrix)
            print_vec<T, D>(vector);
        std::cout << '\n';
        return;
    }

    template <typename T, const size_t D>
    void print_mat(const std::vector<vec<T, D>>& matrix) {
        for (const vec<T, D>& vector : matrix)
            print_vec<T, D>(vector);
        std::cout << '\n';
        return;
    }

} // namespace io

#endif