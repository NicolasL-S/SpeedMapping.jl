# SpeedMapping

[![Build Status](https://github.com/NicolasL-S/SpeedMapping.jl/workflows/CI/badge.svg)](https://github.com/NicolasL-S/SpeedMapping.jl/actions)
[![codecov](https://codecov.io/gh/NicolasL-S/SpeedMapping.jl/branch/main/graph/badge.svg?token=UKzBbD3WeQ)](https://codecov.io/gh/NicolasL-S/SpeedMapping.jl)

SpeedMapping accelerates the convergence of a mapping to a fixed point by the Alternating cyclic extrapolation algorithm. Since gradient descent is an example of such mapping, it can also perform multivariate optimization based on the gradient function. Typical uses are

Accelerating a fixed-point mapping
```julia
julia> using SpeedMapping, LinearAlgebra
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
## Documentation

[Documentation](https://NicolasL-S.github.io/SpeedMapping.jl/)

## The Alternating cyclic extrapolation algorithm

# Let $F : ℝⁿ → ℝⁿ$ $F : \mathbb{R}^n \rightarrow \mathbb{R}^n$ denote a mapping 
# which admits continuous, bounded partial derivatives. A  $p$-order cyclic 
# extrapolation may be expressed as

# ```math
# x_{k+1}=\sum_{i=0}^{p}\tbinom{p}{i}(\sigma _{k}^{(p)})^{i}\Delta ^{i}x_{k}%
# \text{\qquad }p\geq 2
# ```

# where $\sigma _{k}^{(p)}=\frac{|\langle \Delta ^{p},\Delta ^{p-1}\rangle |}{%
# \left\Vert \Delta ^{p}\right\Vert ^{2}}$, $\tbinom{p}{i}=\frac{p!}{i!(p-i)!}$
# is a binomial coefficient and, following Aitken's notation, $\Delta
# x_{k}=F(x_{k})-x_{k}$, and $\Delta x_{k}^{p}=\Delta ^{p-1}F(x_{k})-\Delta
# ^{p-1}x_{k}$.

# The extrapolation step size is $\sigma^{(p)}$ and $\Delta^{p}$ follows 
# Aitken's notation. The algorithm alternates between $p=3$ and $p=2$. For 
# gradient descent acceleration, $\sigma^{(p)}$ is used to adjust the learning 
# rate dynamically.

Reference:
N. Lepage-Saucier, _Alternating cyclic extrapolation methods for optimization algorithms_, arXiv:2104.04974 (2021). https://arxiv.org/abs/2104.04974
