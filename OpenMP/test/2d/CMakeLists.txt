project(tester)

cmake_minimum_required(VERSION 2.8.0)

set (CMAKE_CXX_FLAGS "-std=c++17 -O3 -fopenmp")

add_executable(2d_500_full 500/full_iterations.cpp)
add_executable(2d_500_eps 500/with_epsilon.cpp)

add_executable(2d_1000_full 1000/full_iterations.cpp)
add_executable(2d_1000_eps 1000/with_epsilon.cpp)

add_executable(2d_2000_full 2000/full_iterations.cpp)
add_executable(2d_2000_eps 2000/with_epsilon.cpp)

add_executable(2d_5000_full 5000/full_iterations.cpp)
add_executable(2d_5000_eps 5000/with_epsilon.cpp)


enable_testing()

add_test(NAME 2d_500_full COMMAND 2d_500_full 3 20 40 10)
add_test(NAME 2d_500_eps COMMAND 2d_500_eps 3 20 40 10)

add_test(NAME 2d_1000_full COMMAND 2d_1000_full 3 20 40 10)
add_test(NAME 2d_1000_eps COMMAND 2d_1000_eps 3 20 40 10)

add_test(NAME 2d_2000_full COMMAND 2d_2000_full 3 20 60 10)
add_test(NAME 2d_2000_eps COMMAND 2d_2000_eps 3 20 60 10)

add_test(NAME 2d_5000_full COMMAND 2d_5000_full 3 30 60 10)
add_test(NAME 2d_5000_eps COMMAND 2d_5000_eps 3 30 60 10)
