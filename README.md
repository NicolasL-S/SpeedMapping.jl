# SpeedMapping

[![Build Status](https://github.com/NicolasL-S/SpeedMapping.jl/workflows/CI/badge.svg)](https://github.com/NicolasL-S/SpeedMapping.jl/actions)
[![codecov](https://codecov.io/gh/NicolasL-S/SpeedMapping.jl/branch/main/graph/badge.svg?token=UKzBbD3WeQ)](https://codecov.io/gh/NicolasL-S/SpeedMapping.jl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://NicolasL-S.github.io/SpeedMapping.jl/dev/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://NicolasL-S.github.io/SpeedMapping.jl/dev/)


SpeedMapping solves three types of problems:
1. Accelerating convergent mapping iterations
2. Solving non-linear systems of equations
3. Minimizing a function, possibly with box constraints

using two algorithms: [Alternating cyclic extrapolations](https://www.sciencedirect.com/science/article/abs/pii/S0377042723005514) and [Anderson Acceleration](https://en.wikipedia.org/wiki/Anderson_acceleration). With recently developed features, SpeedMapping [is competitive with similar Julia packages](https://nicolasl-s.github.io/SpeedMapping.jl/dev/benchmarks/).