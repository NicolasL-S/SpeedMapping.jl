#    SpeedMapping
`speedmapping(x₀; m!, kwargs...)` accelerates the convergence of a mapping 
`m!(x_out, x_in)` to a fixed point of `m!` by the `Alternating cyclic 
extrapolation algorithm` (`ACX`). Since gradient descent is an example 
of such mapping, `speedmapping(x0; g!, kwargs...)` can also perform multivariate 
optimization based on the gradient function `g!(∇, x)`.

Reference:
N. Lepage-Saucier, _Alternating cyclic extrapolation methods for optimization 
algorithms_, arXiv:2104.04974 (2021). https://arxiv.org/abs/2104.04974

### Arguments

*   `x₀ :: AbstractArray`: The starting point; must be of eltype `Float`.

Main keyword arguments:
*   `m!(x_out, x_in)`: A map for which a fixed-point must be found.
*   `g!(∇, x)`: The gradient of a function to be minimized.
*   `f(x)`: The objective of the function to be minimized. It is useful to *i*)
    compute a good initial α (learning  rate) for the gradient descent, *ii*)
    optimize using autodiff or *iii*) track the progress of the algorithm.
*   `lower, upper :: Union{AbstractArray, Nothing} = nothing`: Box constraints 
    for the optimization. NOTE: When appropriate, it should also be used with 
    `m!`. Even if `m!` always keeps `x_out` within bounds, an extrapolation 
    step could throw `x` out of bounds.
*   `tol :: Float64 = 1e-10, Lp :: Real = 2` When using `m!`, the algorithm 
    stops when `norm(F(xₖ) - xₖ, Lp) ≤ tol`. When using `g!`, the algorithm 
    stops when `norm(∇f(xₖ), Lp) ≤ tol`. 
*   `maps_limit :: Real = 1e6`: Maximum number of `m!` calls or `g!` calls. 
*   `time_limit :: Real = 1000`: Maximum time in seconds.

Minor keyword arguments:
*   `ord :: Int = 2 ∈ {1,2}` sets whether `ACX` alternates between `1`: `[3, 2]` 
    or `2`: `[3, 3, 2]` extrapolations. For highly non-linear applications, `2` 
    may be better, but there is no general rule (see paper).
*   `check_obj :: Bool = false`: In case of `NaN` or `Inf` values, the algorithm
    restarts at the best past iterate. If `check_obj = true`, progress is 
    monitored with the value of the objective (requires `f`). 
    Otherwise, it is monitored with `norm(F(xₖ) - xₖ, Lp)`. Advantages of 
    `check_obj = true`: more precise and if the algorithm convergences on a bad 
    local minimum, the algorithm instead returns the best past iterate. 
    Advantages of `check_obj = false`: for well-behaved applications, it avoids 
    the effort and time of providing `f` and calling it at every 
    iteration.
*   `store_info :: Bool = false`: Stores `xₖ`, `σₖ` and `αₖ` (see paper).
*   `buffer :: Float64 = 0.01` If `xₖ` goes out of bounds, it is brought back in
    with a buffer. Ex. `xₖ = buffer * xₖ₋₁ + (1 - buffer) * upper`. Setting 
    `buffer = 0.001` may speed-up box-constrained optimization.

Keyword arguments to fine-tune fixed-point mapping acceleration (using `m!`):
*   `σ_min :: Real = 0.0`: Setting to `1` may avoid stalling when using `m!`
    (see paper).
*   `stabilize :: Bool = false`: performs a stabilization mapping before 
    extrapolating. Setting to `true` may improve the performance for 
    applications like the EM or MM algorithm (see paper).

Keyword arguments to fine-tune optimization:
*   `init_descent :: Float64 = 0.01` In case `f` is not provided, `α` (the 
    learning rate) is initialized to init_descent.
*   `init_descent_manually :: Bool = false`: Forces the use of `init_descent`
    even if `f` is provided.

# Example: Finding a dominant eigenvalue
```jldoctest
julia> using LinearAlgebra

julia> using SpeedMapping

julia> A = ones(10) * ones(10)' + Diagonal(1:10);

julia> A = A + A';

julia> function power_iteration!(x_out, x_in, A)
           mul!(x_out, A, x_in)
           x_out ./= norm(x_out, Inf)
       end;

julia> res = speedmapping(ones(10); m! = (x_out, x_in) -> power_iteration!(x_out, x_in, A))
(minimizer = [0.412149140776937, 0.4409506066686425, 0.47407986422778675, 0.512591614439504, 0.557913573459803, 0.6120273722477959, 0.6777660401470581, 0.7593262779332898, 0.8632012008883913, 1.0], maps = 20, f_calls = 0, converged = true, norm_∇ = 3.0946706265554256e-11)

julia> V = res.minimizer;

julia> dominant_eigenvalue = V'A * V / V'V
32.6200113815844

```
# Example: Minimizing a multidimensional Rosenbrock
```
julia> function f(x)
           f_out = 0.0
           for i ∈ 1:size(x,1)
                   f_out += 100 * (x[i,1]^2 - x[i,2])^2 + (x[i,1] - 1)^2
           end
           return f_out
       end;

julia> function g!(∇, x)
           ∇[:,1] .=  400(x[:,1].^2 .- x[:,2]) .* x[:,1] .+ 2(x[:,1] .- 1)
           ∇[:,2] .= -200(x[:,1].^2 .- x[:,2])
           return nothing
       end;

julia> x0 = 1.0 * [-4 -3 -2 -1; 0 1 2 3]';
```

Optimizing, providing f and g!
```
julia> speedmapping(x0; f, g!)
(minimizer = [0.9999999999406816 0.9999999998811234; 0.9999999999699053 0.9999999999396917; 0.9999999999608367 0.9999999999215149; 0.9999999999704666 0.9999999999408166], maps = 115, f_calls = 8, converged = true, norm_∇ = 8.257289534183707e-11)
```

Optimizing without objective
```
julia> speedmapping(x0; g!)
[ Info: α initialized to 0.01 automatically. For stability, provide an objective function or set α manually using init_descent.
(minimizer = [0.9999999999667474 0.9999999999333617; 0.9999999999560251 0.9999999999118742; 0.9999999999727485 0.9999999999453878; 0.9999999999142999 0.9999999998282568], maps = 148, f_calls = 0, converged = true, norm_∇ = 9.438023320576503e-11)
```

Optimizing without gradient
```
julia> speedmapping(x0; f)
[ Info: minimizing f using gradient descent and ForwardDiff
(minimizer = [0.9999999999362467 0.9999999998722411; 0.9999999999678646 0.999999999935602; 0.9999999999581914 0.9999999999162139; 0.9999999999687575 0.9999999999373889], maps = 107, f_calls = 8, converged = true, norm_∇ = 8.789493864861286e-11)
```

Optimizing with a box constraint
```
julia> upper = [0.5ones(5) Inf * ones(5)];

julia> speedmapping(x0; f, g!, upper)
(minimizer = [0.5 0.2499999999998795; 0.5 0.24999999999979697; 0.5 0.24999999999990052; 0.5 0.24999999999996994], maps = 91, f_calls = 8, converged = true, norm_∇ = 9.705761719383147e-11)
```
