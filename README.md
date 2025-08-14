# SpeedMapping

[![Build Status](https://github.com/NicolasL-S/SpeedMapping.jl/workflows/CI/badge.svg)](https://github.com/NicolasL-S/SpeedMapping.jl/actions)
[![codecov](https://codecov.io/gh/NicolasL-S/SpeedMapping.jl/branch/main/graph/badge.svg?token=UKzBbD3WeQ)](https://codecov.io/gh/NicolasL-S/SpeedMapping.jl)

`speedmapping(x0; kwargs...)` solves three types of problems:
1. [Accelerating convergent mapping iterations](#Accelerate-convergent-mapping-iterations)
2. [Solving non-linear systems of equations](#Solve-non-linear-systems-of-equations)
3. [Minimizing a function, possibly with box constraints](#Minimize-a-function)

using two algorithms:
- Alternating cyclic extrapolations (**ACX**) [Lepage-Saucier, 2024](https://www.sciencedirect.com/science/article/abs/pii/S0377042723005514)
- Anderson Acceleration (**AA**) [Anderson, 1964](https://dl.acm.org/doi/10.1145/321296.321305)

SpeedMapping implements algorithms that have been developed very recently. Extensive benchmarks show 
that it is competitive with Julia packages solving similar problems.
