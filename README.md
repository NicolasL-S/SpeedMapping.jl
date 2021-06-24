# SpeedMapping

[![Build Status](https://github.com/NicolasL-S/SpeedMapping.jl/workflows/CI/badge.svg)](https://github.com/NicolasL-S/SpeedMapping.jl/actions)
[![codecov](https://codecov.io/gh/NicolasL-S/SpeedMapping.jl/branch/main/graph/badge.svg?token=UKzBbD3WeQ)](https://codecov.io/gh/NicolasL-S/SpeedMapping.jl)

SpeedMapping accelerates the convergence of a mapping to a fixed point by the Alternating cyclic extrapolation algorithm. Since gradient descent is an example of such mapping, it can also perform multivariate optimization based on the gradient function. Typical uses are

Accelerating a fixed-point mapping
```julia
julia> using SpeedMapping
julia> function power_iteration!(x_out, x_in)
           mul!(x_out, [1 2;2 3], x_in)
           x_out ./= maximum(abs.(x_out))
      end;
julia> dominant_eigenvector = speedmapping(ones(2); m! = power_iteration!).minimizer
2-element Vector{Float64}:
 0.6180339887498947
 1.0
```

Optimizing a function
```julia
julia> using SpeedMapping
julia> rosenbrock(x) =  (1 - x[1])^2 + 100(x[2] - x[1]^2)^2;
julia> solution = speedmapping(zeros(2); f = rosenbrock).minimizer
[ Info: minimizing f using gradient descent acceleration and ForwardDiff
2-element Vector{Float64}:
 0.999999999999982
 0.9999999999999639
```
### The Alternating cyclic extrapolation algorithm

Let *F* : ℝⁿ → ℝⁿ denote a mapping which admits continuous, bounded partial derivatives. A  *p*-order cyclic extrapolation may be expressed as

<img src="https://github.com/NicolasL-S/SpeedMapping.jl/blob/main/Extra.svg">
where
<img src="https://github.com/NicolasL-S/SpeedMapping.jl/blob/main/explanation.svg">

The extrapolation step size is σ⁽ᴾ⁾ and Δᴾ follows Aitken's notation. The algorithm alternates between *p* = 3 and *p* = 2. For gradient descent acceleration, σ⁽ᴾ⁾ is used to adjust the learning rate dynamically.

### Documentation

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://USER_NAME.github.io/PACKAGE_NAME.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://USER_NAME.github.io/PACKAGE_NAME.jl/dev)

Reference:
N. Lepage-Saucier, _Alternating cyclic extrapolation methods for optimization algorithms_, arXiv:2104.04974 (2021). https://arxiv.org/abs/2104.04974
