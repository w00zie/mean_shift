#!/bin/bash

declare -a num_threads=("1" "2" "3" "4" "5" "6" "7" "8")

rm -rf gathered_timings
mkdir gathered_timings

cd gathered_timings

mkdir dynamic
cp ../../benchmark/dynamic/timings/*.csv dynamic/

mkdir static
for t in ${num_threads[@]}; do
    mkdir static/${t}_thread
    cp ../../benchmark/static/${t}_thread/timings/*.csv static/${t}_thread
done
