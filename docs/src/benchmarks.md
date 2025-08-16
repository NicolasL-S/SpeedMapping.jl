Here are a few benchmarks to help decide which specification of SpeedMapping may work best, and compare them to other packages with similar functionalities. The stopping criterion is the `|residual norm| < 1e-7`, except for a few more difficult/long to compute problems. The best efforts have been put forth to compare different packages fairly, suggestions for improvements are always welcome.

## Accelerating convergent mapping iterations

For mapping applications, the **ACX** algorithm can be compared with **AA** with certain criteria
- with or without adaptive relaxation, 
- composite or not. 
- For mapping applications with objective function available, monotonicity (with tolerance for objective deterioration) can also be imposed.

Two fixed-point acceleration packages are available in Julia: `FixedPoint` and `FixedPointAcceleration` (with algorithms `SEA`, `VEA`, `MPE`, `RRE` and `Anderson`).

The benchmarks will be a series of 15 problems from physics, statistics, genetics and social sciences from the FixedPointTestProblems library, ranging from 3 to 52 812 parameters. 
![Mapping results](https://raw.githubusercontent.com/NicolasL-S/SpeedMapping.jl/refs/heads/MajorRefactor/docs/assets/mapping_benchmarks.svg)
Each marker shows how much longer each algorithm took relative to the fastest for each problem. The color scale shows the same for the number of iterations.

**ACX** is very fast and reliable, as well as monotonic **AA**. A note of caution, for the most challenging problems like PH with interval censoring and the Ancestry problem, results can vary significantly depending on the artificial dataset and starting point.

## Solving non-linear systems of equations

For solving non-linear equations, only one specification of `SpeedMapping` is available: standard **AA**. SciML has already performed [extensive benchmarks](https://docs.sciml.ai/SciMLBenchmarksOutput/stable/NonlinearProblem/nonlinear_solver_23_tests/) of nonlinear solvers using the [library of 23 challenging problems](https://github.com/SciML/DiffEqProblemLibrary.jl/blob/master/lib/NonlinearProblemLibrary/src/NonlinearProblemLibrary.jl). A simple version of these tests are performed here for NonlinearSolve and NLSolve (with default choice of ad backend to show how they perform out of the box).
![Problems](https://raw.githubusercontent.com/NicolasL-S/SpeedMapping.jl/refs/heads/MajorRefactor/docs/assets/nonlinear_benchmarks.svg)
Each marker shows how much longer each algorithm took relative to the fastest for each problem. The color scale shows the same for the number of iterations.

SpeedMapping is generally reliable and fast. NLSolve trust_region and newton are also surprisingly fast, despite their higher number of function evaluations. An obvious caveat is that all problems are 10-variables or less; lager-scale problems should be added to draw a more complete picture.

## Minimizing a function

#### Without constraint

Since **ACX** shines for problems where the gradient is available and does not rely on the Hessian matrix, a natural comparison is with [Optim](https://julianlsolvers.github.io/Optim.jl/stable/)'s [L-BFGS](https://julianlsolvers.github.io/Optim.jl/stable/algo/lbfgs/) and [conjugate gradient](https://julianlsolvers.github.io/Optim.jl/stable/algo/cg/) algorithms. To avoid relying on autodiff or libraries external to Julia, a good test set is the 124 unconstrained test problems from [ArtificialLandscapes](https://github.com/NicolasL-S/ArtificialLandscapes.jl), most of which come from the [CUTEst](https://github.com/ralna/CUTEst) test suite. 

![Performance, Optim](https://raw.githubusercontent.com/NicolasL-S/SpeedMapping.jl/refs/heads/MajorRefactor/docs/assets/optimization_performance.svg)

The previous graph shows the proportion of problems solved within a factor $\pi$ of the time taken by the fastest solver. The most reliable is obviously LBFGS which solved nearly all problems. **ACX** converged for approximately 80% of them. **ACX** can struggle for poorly conditioned problems, which abound in CUTEst.  However, it was the fastest 55% of the time, making it potentially useful for regular well-conditioned problems, especially with manually-coded gradients.

#### With box constraints

An advantage of **ACX** is that box constraints have little impacts on performance. There are no standard constrained problems in [ArtificialLandscapes](https://github.com/NicolasL-S/ArtificialLandscapes.jl), but it is easy to add some manually (whether or not they make sense). For each problem with starting point `x0` of length `n`, a lower bound 
```Julia 
lb = [-Inf * ones(n ÷ 2); x0[(n ÷ 2 + 1):end] .- 1]
``` 
will be added. For `Optim`, lower and upper bounds are added using [`fminbox`](https://julianlsolvers.github.io/Optim.jl/stable/user/minimization/#Box-Constrained-Optimization). 

![Performance, Optim, constraint](https://raw.githubusercontent.com/NicolasL-S/SpeedMapping.jl/refs/heads/MajorRefactor/docs/assets/optimization_constr_performance.svg)

Here, the speed advantage for **ACX** are very good. It would have been interesting to consider other algorithms for constrainted like [`trunk`] or (https://jso.dev/JSOSolvers.jl/stable/solvers/#JSOSolvers.trunk), [`tron`](https://jso.dev/JSOSolvers.jl/stable/solvers/#JSOSolvers.tron) from [JSOSolvers](https://jso.dev/JSOSolvers.jl/stable/#Home). Unfortunately, the problems of [ArtificialLandscapes](https://github.com/NicolasL-S/ArtificialLandscapes.jl) are not always compatible with `autodiff` which would be required.