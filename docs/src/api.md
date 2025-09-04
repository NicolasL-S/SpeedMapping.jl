## Notation

Problems:
- **MAP**: Accelerating convergent mapping iterations
- **NLS**: Solving a non-linear systems of equations
- **MIN**: Minimizing a function, possibly with box-constraints

Algorithms:
- **ACX**: Alternating cyclic extrapolations
- **AA**: Anderson Acceleration

## speedmapping

`speedmapping(x0; kwargs...) :: SpeedMappingResult`

`x0 :: T` is the starting point and defines the type:
- For **ACX**: `x0` can be of type `Real` or `Complex` with mutable or immutable containers of different shapes like `AbstractArray`, `StaticArray`, `Real`, or `Tuple`.
- For **AA**: `x0` should be a mutable `AbstractArray{AbstractFloat}`.

## Keyword arguments defining the problem
One and _only one_ of the following argument should be supplied. They are all of type `FN = Union{Function, Nothing}`.

`m! :: FN = nothing` in-place mapping function for **MAP** with mutable arrays as input. 
```Julia
speedmapping([1.,1.]; m! = (xout, xin) -> xout .= 0.9xin)
```
`r! :: FN = nothing` in-place residual function for **NLS** with mutable arrays as input. 
```Julia
speedmapping([1.,1.]; r! = (resid, x) -> resid .= -0.1x)
```
`g! :: FN = nothing` in-place gradient function for **MIN** with mutable arrays as input. 
```Julia
speedmapping([1.,1.]; g! = (grad, x) -> grad .= 4x.^3)
```
`m` and `g` are versions of `m!` and `g!` with immutable types like `Real`, `StaticArray`, or `Tuple` as input and output.
```Julia
using StaticArrays
speedmapping(1.; m = x -> 0.9x)
speedmapping(SA[1.,1.]; m = x -> 0.9x)
speedmapping(1.; g = x -> 4x^3)
speedmapping((1.,1.); g = x -> (x[1] - 2, x[2]^3))

```
## Other important keyword arguments

`algo :: Symbol = r! !== nothing ? :aa : :acx` determines the method used.
- `:acx` can be used to solve **MAP** or **MIN** (the default for **MAP**).
- `:aa` can be used to solve **MAP** or **NLS**. 

`f :: FN = nothing` computes an objective function. 
- For **MIN**, `f` is be used to initialize the learning rate better.
- For **MAP** using **AA**, `f` is be used ensure monotonicity of the algorithm. 
- For **NLS**, `f` is ignored.

`lower, upper = nothing` define bounds on parameters which can be used with any problem.
```Julia
speedmapping([1., 1.]; g! = (grad, x) -> grad .= 4x.^3, lower = [-Inf,2.])
```
## Other keyword arguments
### Affecting both **ACX** and **AA**
`cache :: Union{AcxCache, AaCache, Nothing} = nothing`
  pre-allocates memory for **ACX** or **AA** with mutable input
```Julia
c = AaCache([1.,1.])
speedmapping([1.,1.]; m! = (xout, xin) -> xout .= 0.9xin, algo = :aa, cache = c)
```

`abstol :: AbstractFloat = 1e-8`
  The absolute tolerance used as stopping criterion. 
  - For **MAP**, the algorithm stops when `‖xout - xin‖ < abstol` (from `m!(xout, xin)`)
  - For **NLS**, the algorithm stops when `‖res‖ < abstol` (from `r!(res, x)`)
  - For **MIN**, the algorithm stops when `‖gradient‖ < abstol` (from `g!(gradient, x)`)

`pnorm :: Real = 2.`
  The norm used for the stopping criterion. Typically `1`, `2` or `Inf`.

`maps_limit :: Real = 1_000_000_000`
  The maximum number of main function evaluation (`m!`, `r!`, or `g!`) before the algorithm terminates.

`iter_limit :: Real = 1_000_000_000`
  The maximum number of iterations before the algorithm terminates.

`time_limit :: Real = Inf`
  The time limit before stopping (if `time_limit == Inf`, `time()` will not be called at each iteration).

`reltol_resid_grow :: Real = algo == :aa ? 4. : (g! !== nothing || g !== nothing) ? 1e5 : 100.`
  `reltol_resid_grow` is a problem-specific stabilizing parameter. After a mapping/descent step/iteration, the distance between the current `x` and the previous `x` is reduced until the residual norm (`‖xout - xin‖`, `‖res‖`, or `‖grad‖`) does not increase more than a by a factor `reltol_resid_grow`. It is set to 4 for **AA** because this algorithm is more sensitive to low-quality iterations and because **NLS** may involve highly divergent functions. For **ACX** it is set to 100 for **MAP**, and for 1e5 for **MIN**.

`buffer :: AbstractFloat = (m! !== nothing || m !== nothing) ? 0.05 : 0.`
  `buffer`is used in conjunction with `lower` or `upper`. If an iterate `x` lands outside a constraint, `buffer` leaves some distance between `x` and the constraint. It is set by default to `0.05` for **MAP** because constraints may be used to avoid landing immediately on bad values like saddle points at which the algorithm would stall. 

`store_trace :: Bool = false`
  To store information on each iteration of the solving process. The trace depends on the algorithm. For a SpeedMapping result `res`, the trace is 
  - `res.acx_trace` for **ACX**;
  - `res.aa_trace` for **AA**.

### Affecting only **ACX**
`orders = (2,3,3)`
  The extrapolation orders. (2,3,3) is extremely reliable, but others like (2,3), or (2,) could be considered.

`initial_learning_rate :: Real = 1.`
  The initial learning rate used for **MIN**. If `initialize_learning_rate == true`, it is the starting point for the initialization.

`initialize_learning_rate :: Bool = true`
  To find a suitable learning rate to start **MIN** for which the residual norm does not increase too fast and the change in the objective (if supplied) respects the Armijo condition.

### Affecting only **AA**
`lags :: Integer = 30`
  The maximum number of past residuals used to compute the next iterate

`condition_max :: Real = 1e6`
  The maximum condition number of the matrix of past residuals used to compute the next iterate. Setting it too high increases the risk of numerical imprecision.

`relax_default :: Real = 1.`
  The default relaxation parameter (also referred to as damping or mixing parameter).

`ada_relax :: Symbol = m! !== nothing ? :minimum_distance : :none`
  Adaptive relaxation. For now, only `:minimum_distance` is implemented (see [Lepage-Saucier, 2024](https://arxiv.org/abs/2408.16920) although changes were made to the regularization). It is set to `:none` for **NLS** since `:minimum_distance` requires convergent mapping to be useful.

`composite :: Symbol = :none`
  Composite Anderson Acceleration by [Chen and Vuik, 2022](https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.7096). A one-step **AA** iteration (using 2 maps) is inserted between 2 full **AA** steps, which reduces the computation and can offer interesting speed-up for some applications. Two types are implemented: `:aa1` and `acx2` (which inserts an ACX, order 2 step).

`abstol_obj_grow :: Real = √abstol` 
  If `f` is supplied with **AA**, the objective is not allowed to increase by more than `abstol_obj_grow` between iterations (otherwise, it fall back on the last map). Set `abstol_obj_grow = 0` for tight monotonicity.

## SpeedMappingResult
`SpeedMappingResult` has fields
  - `minimizer :: typeof(x0)`: The solution
  - `residual_norm :: AbstractFloat`: The norm of the residual, which would be ‖xout - xin‖ for **MAP**, ‖residual‖ for **NLS**, and ‖∇f(x)‖ for **MIN** (only for non-binding components of the gradient).
  - `maps`: the number of maps, function evaluations or gradient evaluations
  - `f_calls`: The number of objective function evaluations
  - `iterations`: The number of iteration
  - `status :: Symbol ∈ (:first_order, :max_iter, :max_eval, :max_time, :failure)` should be `:first_order` if a solution has been found successfully.
  - `algo ∈ (:acx, :aa)`
  - `acx_trace` A vector of `AcxState` if `algo == :acx && store_trace == true`, `nothing` otherwise.
  - `aa_trace` A vector of `AaState` if `algo == :aa && store_trace == true`, `nothing` otherwise.
  - `last_learning_rate :: AbstractFloat` The last learning rate, only meaningful for **MIN**.