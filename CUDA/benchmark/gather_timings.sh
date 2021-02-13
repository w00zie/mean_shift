#!/bin/bash

declare -a threads=("32" "64" "128" "256" "512" "1024")

rm -rf gathered_timings
mkdir gathered_timings
cd gathered_timings

for t in ${threads[@]}; do
    mkdir ${t}
    mkdir ${t}/naive
    mkdir ${t}/sm
    cp ../${t}/timings/naive/*.csv ${t}/naive
    cp ../${t}/timings/sm/*.csv ${t}/sm
done
