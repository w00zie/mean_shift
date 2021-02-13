#!/bin/bash

cd 2d

rm -rf build
mkdir build
cd build && cmake ../
make
ctest

cd ../../3d

rm -rf build
mkdir build
cd build && cmake ../
make
ctest

