# Benchmarks

The following benchmarks compare the performances of various specification of SpeedMapping with those of other packages with similar functionalities. Given the great diversity of problems, the criterion for convergence is always `|residual norm| < abstol`, even though different algorithms may converge to different fixed points or minima. `abstol` is set to 1e-7, except for a few especially long or difficult problems. 

Every effort has been made to ensure a fair comparison of the packages. 
- The same algorithms are run with the same parameters (e.g. the maximum number of lags for Anderson Acceleration is set to 30). 
- Otherwise, default options are preferred to see how each algorithm performs out of the box. 
- Initialization time is included in compute time. 

The code for the benchmarking scripts is available [here](https://github.com/NicolasL-S/SpeedMapping.jl/tree/MajorRefactor/docs/benchmarking_code).

## Accelerating convergent mapping iterations

For mapping applications, the **ACX** algorithm can be compared with **AA**:
- with or without adaptive relaxation
- with or without composition. 
- For mapping applications with objective function available, monotonicity (with tolerance for 
objective deterioration) can also be imposed or not.

Two fixed-point acceleration packages are available in Julia: `FixedPoint` and `FixedPointAcceleration` (with algorithms `SEA`, `VEA`, `MPE`, `RRE` and `Anderson`).

The benchmarks are based on 15 problems from physics, statistics, genetics and social sciences from the FixedPointTestProblems library. 

![Mapping results](https://raw.githubusercontent.com/NicolasL-S/SpeedMapping.jl/refs/heads/MajorRefactor/docs/assets/mapping_benchmarks.svg)

The height of each marker indicates how much longer each algorithm took relative to the fastest algorithm for each problem. The color scale shows the same for the number of iterations. The left-most algorithms are the most reliable and the quickest.

**ACX** and monotonic **AA** perform well. While **AA** tends to need fewer iterations, **ACX** is lighter and tends to be quicker.

## Solving non-linear systems of equations

SciML has already performed [extensive benchmarking](https://docs.sciml.ai/SciMLBenchmarksOutput/stable/NonlinearProblem/nonlinear_solver_23_tests/) of nonlinear solvers using the [library of 23 challenging problems](https://github.com/SciML/DiffEqProblemLibrary.jl/blob/master/lib/NonlinearProblemLibrary/src/NonlinearProblemLibrary.jl). A simple version of these tests are performed here for NonlinearSolve and NLSolve (with default choice of ad backend to show how they perform out of the box). In `SpeedMapping`, only standard **AA** is available for solving non-linear equations.

![Problems](https://raw.githubusercontent.com/NicolasL-S/SpeedMapping.jl/refs/heads/MajorRefactor/docs/assets/nonlinear_benchmarks.svg)

Each marker's height shows how much longer each algorithm took relative to the fastest for each problem. The color scale shows the same for the number of iterations.

SpeedMapping is generally reliable and fast. NLSolve trust_region and newton are also surprisingly fast, despite their higher number of function evaluations. A caveat to these tests is that all problems have 10 variables or less. Lager-scale problems should be added to paint a more complete picture.

## Minimizing a function

#### Without constraint

**ACX** shines for problems where the gradient is available and does not rely on the Hessian matrix. A natural comparison is thus with [Optim](https://julianlsolvers.github.io/Optim.jl/stable/)'s [L-BFGS](https://julianlsolvers.github.io/Optim.jl/stable/algo/lbfgs/) and [conjugate gradient](https://julianlsolvers.github.io/Optim.jl/stable/algo/cg/) algorithms. To avoid relying on autodiff or libraries external to Julia, a good test set is the unconstrained test problems from [ArtificialLandscapes](https://github.com/NicolasL-S/ArtificialLandscapes.jl), most of which come from the [CUTEst](https://github.com/ralna/CUTEst) test suite. 

![Performance, Optim](https://raw.githubusercontent.com/NicolasL-S/SpeedMapping.jl/refs/heads/MajorRefactor/docs/assets/optimization_performance.svg)

The previous graph shows the fraction of problems solved within a factor $\pi$ of the time taken by the fastest solver. When no algorithm converged in time for a problem, it is removed from the list. 

The most reliable is obviously LBFGS which converged to minima reasonably fast for nearly all the problems. **ACX** and Conjugate gradient. converged for approximately 80% of them. While **ACX** can struggle for poorly conditioned problems (which abound in CUTEst), it was the fastest 55% of the time, making it potentially useful for regular well-conditioned problems, especially with manually-coded gradients.

#### With box constraints

There are no constrained problems in [ArtificialLandscapes](https://github.com/NicolasL-S/ArtificialLandscapes.jl), but we can easily add arbitrary ones for testing purposes. For each problem with starting point `x0` of length an lower bound will be added equal to `x0 .+ 0.5`, and no lower bound. 

An advantage of **ACX** is that box constraints add little extra computation. For `Optim`, lower and upper bounds are added using [`fminbox`](https://julianlsolvers.github.io/Optim.jl/stable/user/minimization/#Box-Constrained-Optimization). Since `fminbox` adds a barrier penalty, it can increase computation time significantly. An interesting option to explore is [LBFGSB.jl](https://github.com/Gnimuc/LBFGSB.jl), a wrapper around a Fortran implementation of [L-BFGS-B](https://digital.library.unt.edu/ark:/67531/metadc666315/), which handles constraints differently. 

A problem considered having converged when the |gradient_i| < 1e-7 for all i for which a constraint was not binding.

![Performance, Optim, constraint](https://raw.githubusercontent.com/NicolasL-S/SpeedMapping.jl/refs/heads/MajorRefactor/docs/assets/optimization_constr_performance.svg)

Among the four algorithms tested, there seems to be no reason not to use ACX. It would be interesting to test other bound-constrained solvers like tron from [JSOSolvers.jl](https://jso.dev/JSOSolvers.jl/stable/solvers/).