project(tester)

cmake_minimum_required(VERSION 2.8.0)

set (CMAKE_CXX_FLAGS "-std=c++17 -O3")

add_executable(3d_500_full 500/full_iterations.cpp)
add_executable(3d_500_eps 500/with_epsilon.cpp)

add_executable(3d_1000_full 1000/full_iterations.cpp)
add_executable(3d_1000_eps 1000/with_epsilon.cpp)

add_executable(3d_2000_full 2000/full_iterations.cpp)
add_executable(3d_2000_eps 2000/with_epsilon.cpp)

add_executable(3d_5000_full 5000/full_iterations.cpp)
add_executable(3d_5000_eps 5000/with_epsilon.cpp)


enable_testing()

add_test(NAME 3d_500_full COMMAND 3d_500_full 4 30 60 10)
add_test(NAME 3d_500_eps COMMAND 3d_500_eps 4 30 60 10)

add_test(NAME 3d_1000_full COMMAND 3d_1000_full 4 30 60 10)
add_test(NAME 3d_1000_eps COMMAND 3d_1000_eps 4 30 60 10)

add_test(NAME 3d_2000_full COMMAND 3d_2000_full 4 30 60 10)
add_test(NAME 3d_2000_eps COMMAND 3d_2000_eps 4 30 60 10)

add_test(NAME 3d_5000_full COMMAND 3d_5000_full 4 60 60 10)
add_test(NAME 3d_5000_eps COMMAND 3d_5000_eps 4 60 60 10)
