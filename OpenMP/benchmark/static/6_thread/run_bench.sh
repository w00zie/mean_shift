#!/bin/bash

declare -a dim=("2d" "3d")
declare -a size=("500" "1000" "2000" "5000")

rm -rf bin/
rm -rf timings/
mkdir bin/
mkdir timings/

cd 2d
make
cd ../3d
make
cd ../bin

for d in ${dim[@]}; do
    for s in ${size[@]}; do
        sleep 3
        echo "Running ${d}_${s}"
        ./${d}_${s}
    done
done