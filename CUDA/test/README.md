
## How to run

Launch

```
./run_tests.sh
```

## Details

In each subdirectory (e.g. `2d/500`) you will find two test cases: `naive.cu` and `sm.cu`.
Both are structured as following:

- Initialize hyperparameters.
- Run mean shift and return the computed cluster centers.
- Compare these cluster centers to the real ones.
- Pass the test if the distance of each computed centroid wrt the (relative) real one is less or equal than some epslion.

Every test case is run with 64 threads per block.

**WARNING**:
In each subdirectory you will find a Makefile that compiles the CUDA source files. The compilation command uses NVCC under `/usr/local/cuda-11/bin`. This might (probabily) will be different for you! If this is the case you will have to edit each Makefile (2x4=8 files) and fix the path to your NVCC.

I'm sorry for this.