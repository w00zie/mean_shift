#!/bin/bash

declare -a threads=("32" "64" "128" "256" "512" "1024")

for t in ${threads[@]}; do
    echo "${t}"
    cd ${t}
    ./collect_times.sh
    cd ../
done