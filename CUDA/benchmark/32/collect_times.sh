#!/bin/bash

rm -rf timings
mkdir timings
mkdir timings/naive
mkdir timings/sm

cd 2d
./build_tests.sh
./run_tests.sh
cd ../3d
./build_tests.sh
./run_tests.sh

