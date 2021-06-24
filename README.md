# SpeedMapping

[![Build Status](https://github.com/NicolasL-S/SpeedMapping.jl/workflows/CI/badge.svg)](https://github.com/NicolasL-S/SpeedMapping.jl/actions)
[![codecov](https://codecov.io/gh/NicolasL-S/SpeedMapping.jl/branch/main/graph/badge.svg?token=UKzBbD3WeQ)](https://codecov.io/gh/NicolasL-S/SpeedMapping.jl)

`speedmapping(x₀; m!, kwargs...)` accelerates the convergence of a mapping `m!(x_out, x_in)` to a fixed point of `m!` by the `Alternating cyclic extrapolation algorithm` (`ACX`). Since gradient descent is an example of such mapping, `speedmapping(x0; g!, kwargs...)` can also perform multivariate optimization based on the gradient function `g!(∇, x)`.

### The Alternating cyclic extrapolation algorithm

Let ``F:\mathbb{R}^{n} \rightarrow \mathbb{R}^{n}`` with ``n\in N^{+}`` denote a mapping which admits continuous, bounded partial derivatives. A  ``p``-order cyclic extrapolation may be synthesized as

<img src="https://github.com/NicolasL-S/SpeedMapping.jl/blob/main/Extra.svg">

where `\sigma_{k}^{(p_{k})}=|\langle\Delta^{p_{k}},\Delta^{p_{k}-1}\rangle|/\left\Vert \Delta^{p_{k}}\right\Vert ^{2}`, 

### Documentation

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://USER_NAME.github.io/PACKAGE_NAME.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://USER_NAME.github.io/PACKAGE_NAME.jl/dev)

Reference:
N. Lepage-Saucier, _Alternating cyclic extrapolation methods for optimization algorithms_, arXiv:2104.04974 (2021). https://arxiv.org/abs/2104.04974

