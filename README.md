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
julia> speedmapping(ones(2); m! = power_iteration!)
(minimizer = [0.6180339887498947, 1.0], maps = 7, f_calls = 0, converged = true, norm_∇ = 3.1086244689504383e-15)
```
Optimizing a function
```julia
julia> using SpeedMapping
julia> rosenbrock(x) =  (1 - x[1])^2 + 100(x[2] - x[1]^2)^2;
julia> speedmapping(zeros(2); f = rosenbrock)
[ Info: minimizing f using gradient descent acceleration and ForwardDiff
(minimizer = [0.999999999999982, 0.9999999999999639], maps = 108, f_calls = 8, converged = true, norm_∇ = 8.360473284759195e-14)
```
### The Alternating cyclic extrapolation algorithm

Let *F* : ℝⁿ → ℝⁿ denote a mapping which admits continuous, bounded partial derivatives. A  *p*-order cyclic extrapolation may be expressed as

<img src="https://github.com/NicolasL-S/SpeedMapping.jl/blob/main/Extra.svg">
<img src="https://github.com/NicolasL-S/SpeedMapping.jl/blob/main/explanation.svg">
<table style="width:100%">
  <tr>
    <td><img src="https://github.com/NicolasL-S/SpeedMapping.jl/blob/main/sigma.svg"></td>
    <td>is the extrapolation stepsize</td>
  </tr>
  <tr>
    <td>is the extrapolation step size</td>
    <td>allo</td>
  </tr>
</table>

### Documentation

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://USER_NAME.github.io/PACKAGE_NAME.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://USER_NAME.github.io/PACKAGE_NAME.jl/dev)

Reference:
N. Lepage-Saucier, _Alternating cyclic extrapolation methods for optimization algorithms_, arXiv:2104.04974 (2021). https://arxiv.org/abs/2104.04974
