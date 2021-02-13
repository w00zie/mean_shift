
## How to run

Launch

```
./run_tests.sh
```

## Details

In each subdirectory (e.g. `2d/500`) you will find two test cases: `full_iterations.cpp` and `with_epsilon.cpp`.
Both are structured as following:

- Initialize hyperparameters.
- Run mean shift and return the computed cluster centers.
- Compare these cluster centers to the real ones.
- Pass the test if the distance of each computed centroid wrt the (relative) real one is less or equal than some epslion.

The difference between `full_iterations.cpp` and `with_epsilon.cpp` is minimal: the first one does not check for convergence and runs straight for the fixed number of iterations (set by the user). The latter, instead, checks for convergence by measuring the distance between the previously-calculated cluster center and the currently-calculated cluster center: if it is less or equal than some epsilon (set by the user), then the convergence is reached and the algorithm stops.