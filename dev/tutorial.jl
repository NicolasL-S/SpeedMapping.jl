# `speedmapping(x0; kwargs...)`  solves three types of problems:
# 1. [Accelerating convergent mapping iterations](#Accelerate-convergent-mapping-iterations)
# 2. [Solving non-linear systems of equations](#Solve-non-linear-systems-of-equations)
# 3. [Minimizing a function, possibly with box constraints](#Minimize-a-function)
#
# using two algorithms:
# - Alternating cyclic extrapolations (**ACX**) [Lepage-Saucier, 2024](https://www.sciencedirect.com/science/article/abs/pii/S0377042723005514)
# - Anderson Acceleration (**AA**) [Anderson, 1964](https://dl.acm.org/doi/10.1145/321296.321305)

#
# This tutorial will display its main functionality on simple problems. To see which specification may be more performant for your problem, the Benchmarks section compares all of them, along with other Julia packages
# with similar functionalities.
#
# # Accelerating convergent mapping iterations
#
# Let $F:\mathbb{R}^n\rightarrow\mathbb{R}^n$ and $x \in \mathbb{R}^n$ be a reasonable starting 
# point. If the series $x, F(x), F(F(x)),...$ converges to $x^*$ where $F(x^*) = x^*$, the **ACX** algorithm 
# and [**AA** with adative relaxation](https://arxiv.org/abs/2408.16920) can be used to accelerate the 
# convergence. If $F$ does not converge, the problem can simply be redefined as 
# [solving $G(x) = F(x) - x = 0$](#Solving-non-linear-systems-of-equations).
#
# Let's find the dominant eigenvalue of a matrix $A$ using the accelerated [Power iteration](https://en.wikipedia.org/wiki/Power_iteration).

using LinearAlgebra

n = 10;
A = ones(n,n) .+ Diagonal(1:n);

## An in-place mapping function to avoid allocations
function power_iteration!(xout, xin, A)
    mul!(xout, A, xin)
    maxabs = 0.
    for xi in xout
        abs(xi) > maxabs && (maxabs = abs(xi))
    end
    xout ./= maxabs
end;
x0 = ones(n);

# Speedmapping has one mandatory argument: the starting point ``x0``. The mapping is specified with the keyword argument ``m!``.
using SpeedMapping
res = speedmapping(x0; m! = (xout, xin) -> power_iteration!(xout, xin, A));
display(res)

# The dominant eigenvalue is:
v = res.minimizer; ## The dominant eigenvector
dominant_eigenvalue = v'A*v/v'v;
eigen(A).values[10] ≈ dominant_eigenvalue

# With `m!`, the default algorithm is `algo = :acx`. To switch, set `algo = :aa`.
res = speedmapping(x0; m! = (xout, xin) -> power_iteration!(xout, xin, A), algo = :aa);

# By default, **AA** uses [adaptive relaxation](https://arxiv.org/abs/2408.16920), which can 
# reduce the number of iterations. It is specified by the keyword argument 
# `adarelax = :minimum_distance`. For constant relaxation, set `adarelax = :none`.
res = speedmapping(x0; m! = (xout, xin) -> power_iteration!(xout, xin, A), algo = :aa, adarelax = :none);

# Another recent development for **AA** is **Composite AA** by [Chen and Vuik, 2022](https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.7096).
# A one-step **AA** iteration (using 2 maps) is inserted between 2 full **AA** steps, which reduces 
# the computation and can speed-up some applications. The default is 
# `composite = :none`. Two versions are available:
res = speedmapping(x0; m! = (xout, xin) -> power_iteration!(xout, xin, A), algo = :aa, composite = :aa1);
res = speedmapping(x0; m! = (xout, xin) -> power_iteration!(xout, xin, A), algo = :aa, composite = :acx2);

# Some mapping iterations maximize or minimize a certain objective function. Since some **AA** steps 
# can deteriorate the objective, it would be best to avoid them by falling back to the last map. This 
# can be done by supplying an objective function (assumed to be a minimization 
# problem) using `f` as keyword argument. Here is an illustrative example
# using the [EM-algorithm](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm).

using FixedPointTestProblems
EMx0, EMmap!, EMobj = testproblems["Hasselblad, Poisson mixtures"]();
speedmapping(EMx0; m! = EMmap!, algo = :aa);
speedmapping(EMx0; m! = EMmap!, f = EMobj, algo = :aa);

# ## Avoiding memory allocation
#
# For similar problems solved many times, it is possible to preallocate working 
# memory and feed it using the `cache` keyword argument. Each algorithm has its own cache:

acx_cache = AcxCache(x0);
aa_cache = AaCache(x0);

# Note that ``x0`` must still be supplied to speedmapping.

# For small-sized problems with **ACX**, heap-allocation can be avoided by supplying a static array 
# or a tuple as starting point and using the keyword argument `m` for the mapping function, offering 
# additional speed gains.

using StaticArrays

function power_iteration(xin, A)
    xout = A * xin;
    maxabs = 0.;
    for xi in xout
        abs(xi) > maxabs && (maxabs = abs(xi))
    end;
    return xout / maxabs;
end;

As = @SMatrix ones(n,n);
As += Diagonal(1:n);
x0s = @SVector ones(n);

speedmapping(x0s; m = x -> power_iteration(x, As));

# Comparing speed gains

using BenchmarkTools
t_eigen = @benchmark eigen($A)
t_alloc = @benchmark speedmapping($x0; m! = (xout, xin) -> power_iteration!(xout, xin, $A)); # Allocating
t_prealloc = @benchmark speedmapping($x0; m! = (xout, xin) -> power_iteration!(xout, xin, $A), cache = $acx_cache); # Pre-allocated
t_nonalloc = @benchmark speedmapping($x0s; m = x -> power_iteration(x, $As)) # Non allocating
[median(t) for t in (t_eigen, t_alloc, t_prealloc, t_nonalloc)]

# ## Working with scalars
#
# `m` also accepts scalar functions.

speedmapping(0.5; m = cos);

# # Solving non linear systems of equations
#
# For non-linear systems of equations (finding $x^*$ such that $G(x^*) = 0$), only **AA** with 
# constant relaxation should be used (and is set by default). The keyword argument to supply $G$ is 
# `r!`.

function r!(res, x)
	res[1] = x[1]^2;
	res[2] = (x[2] + x[1])^3;
end

speedmapping([1.,2.]; r! = r!);

# # Minimizing a function

# To minimize a function (using **ACX**), the function and its in-place gradient are supplied with 
# the keyword arguments `f` and `g!`. The Hessian cannot be supplied.
# Compared to other quasi-Newton algorithms like L-BFGS, **ACX** iterations are very fast, but the 
# algorithm may struggle for ill-conditioned problems.

f_Rosenbrock(x) = 100 * (x[1]^2 - x[2])^2 + (x[1] - 1.)^2;

function g_Rosenbrock!(grad, x) # Rosenbrock gradient
	grad[1] = 400 * (x[1]^2 - x[2]) * x[1] + 2 * (x[1] - 1);
	grad[2] = -200 * (x[1]^2 - x[2]);
end

speedmapping([-1.2, 1.]; f = f_Rosenbrock, g! = g_Rosenbrock!);

# The function objective is only used to compute a safer initial learning rate. It can be omitted.
speedmapping([-1.2, 1.]; g! = g_Rosenbrock!);

# If only the objective is supplied, the gradient is computed using using ForwardDiff.
speedmapping([-1.2, 1.]; f = f_Rosenbrock);

# The keyword argument g can be used with static arrays or tuple to supply a non-allocating gradient.

using StaticArrays
g_Rosenbrock(x :: StaticArray) = SA[400 * (x[1]^2 - x[2]) * x[1] + 2 * (x[1] - 1), -200 * (x[1]^2 - x[2])];
speedmapping(SA[-1.2, 1.]; g = g_Rosenbrock);

g_Rosenbrock(x :: Tuple) = (400 * (x[1]^2 - x[2]) * x[1] + 2 * (x[1] - 1), -200 * (x[1]^2 - x[2]));
speedmapping((-1.2, 1.); g = g_Rosenbrock);

# Scalar functions can also be supplied. E.g. $f(x) = e^x + x^2$

speedmapping(0.; f = x -> exp(x) + x^2, g = x -> exp(x) + 2x);

# ## Adding box constraint
# 
# An advantage of **ACX** is that constraints on parameters have little impact on estimation speed. 
# They are added with the keyword arguments `lower` and `upper` (`= nothing` by default). The 
# starting point does not need to be in the feasible domain, but, if supplied, upper / lower need
# to be of type x0.

speedmapping([-1.2, 1.]; f = f_Rosenbrock, g! = g_Rosenbrock!, lower = [2, -Inf]);
speedmapping(0.; g = x -> exp(x) + 2x, upper = -1.);