#!/bin/bash
declare -a Directories=("500" "1000" "2000" "5000")

rm -rf bin
mkdir bin

for subdir in ${Directories[@]}; do
    cd $subdir
    make
    cd ../
done