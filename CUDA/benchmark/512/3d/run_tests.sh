#!/bin/bash
declare -a Directories=("500" "1000" "2000" "5000")

cd bin
nv="naive"
sm="sm"

for subdir in ${Directories[@]}; do
    cmd_naive="${nv}_${subdir}"
    cmd_sm="${sm}_${subdir}"
    printf "\nRunning ${cmd_naive}...\n"
    ./${cmd_naive}
    printf "\nRunning ${cmd_sm}...\n"
    ./${cmd_sm}
done