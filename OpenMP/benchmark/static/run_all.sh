#!/bin/bash

declare -a num_threads=("1" "2" "3" "4" "5" "6" "7" "8")

for t in ${num_threads[@]}; do
    cd ${t}_thread
    pwd
    ./run_bench.sh
    cd ../
done