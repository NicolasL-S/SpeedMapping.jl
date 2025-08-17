@inline isbad(x :: AbstractFloat) = isnan(x) || isinf(x)
function isbad(x)
    s = sum(x)
    return isnan(s) || isinf(s) # For some reason, this is the fastest
end

# Fast common norms for common types
@inline function inf_norm(x :: AbstractArray{T}) :: T where T <: Union{Float64, Float32}
    max_abs_xi = 0
    for xi in x
        abs(xi) > max_abs_xi && (max_abs_xi = abs(xi))
    end
    max_abs_xi
end

@inline sumsq(x) = sum(xi -> xi * xi, x)

@inline function lpnorm(x, pnorm)
    pnorm == 2 && return √sumsq(x)
    pnorm == 1 && return sum(xi -> abs(xi), x)
    pnorm == Inf && return inf_norm(x)
    return norm(x, pnorm)
end

# Accurate dot product to improve the precision of extrapolations (at the cost of slightly longer compute time)
# Unfortunately, dot_oro caused compatibility issues
#=
function accurate_dot(x::T, y::T) where T<:Union{AbstractArray{Float32},AbstractArray{Float64}}
    return dot_oro(x, y)
end
accurate_dot(x, y) = dot(x,y)
accurate_cdot(x, y, FT) :: FT = (eltype(x) <: Real && eltype(y) <: Real) ? accurate_dot(x, y) : real(dot(x,y))
accurate_cdot(x, FT) = accurate_cdot(x, x, FT)
=#

cdot(x, y, FT) :: FT = real(dot(x,y)) where FT <: AbstractFloat # To make sure that the compiler infers the type FT, even with unusual inputs like complex arrays. Not entirely sure this is necessary, but better safe than sorry.
cdot(x, FT) :: FT = cdot(x, x, FT) where FT <: AbstractFloat # To make sure that the compiler infers the type FT, even with unusual inputs like complex arrays. Not entirely sure this is necessary, but better safe than sorry.

#####
##### Core functions
#####

# Makes sure x_try[i] stays within boundaries with a gap buffer * (bound[i] - x_old[i]).
function box_constraints!(
    extr, x_try::T, x_old::T, bound :: T, buffer, ip
) where {T}

    if ip
        @simd for i ∈ eachindex(x_try)
            @inbounds x_try[i] = extr(x_try[i], (1 - buffer) * bound[i] + buffer * x_old[i])
        end
        return x_try
    else
        return extr.(x_try, (1 - buffer) .* bound .+ buffer .* x_old)
    end
end

function box_constraint!(
    x_try::T, x_old::T, bounds, buffer, ip
) where {T}
    bounds.l ≠ nothing && (x_try = box_constraints!(max, x_try, x_old, bounds.l, buffer, ip))
    bounds.u ≠ nothing && (x_try = box_constraints!(min, x_try, x_old, bounds.u, buffer, ip))
    return x_try
end

function assess_status(iter_budget, maps_budget, max_time, converged)
    converged && return :first_order
    iter_budget <= 0 && return :max_iter
    maps_budget <= 0 && return :max_eval
    time() >= max_time && return :max_time
    return :failure
end

FN = Union{Function, Nothing}

"""
SpeedMapping solves three types of problems:
1. Accelerating convergent mapping iterations
2. Solving non-linear systems of equations
3. Minimizing a function, possibly with box constraints

using two algorithms:
- Alternating cyclic extrapolations (**ACX**)
- Anderson Acceleration (**AA**)

`speedmapping(x0; kwargs...)`

`x0 :: T` is the starting point and defines the type:
- For **ACX**: `x0` can be of type real or complex with mutable or immutable containers like Abstract Array, Static Array, Scalar, Tuple.
- For **AA**: `x0` should be a mutable AbstractArray{AbstractFloat}.

#### Keyword arguments defining the problem
One and _only one_ of the following argument should be supplied. All of them are of type `FN = Union{Function, Nothing}`.

`m! ::  FN  =  nothing` in-place mapping function for problem **1** with mutable arrays as input. 
```Julia
speedmapping([1.,1.]; m! = (xout, xin) -> xout .=  0.9xin)
```
`r! ::  FN  =  nothing` in-place residual function for problem **2** with mutable arrays as input. 
```Julia
speedmapping([1.,1.]; r! = (resid, x) -> resid .=  -0.1x)
```
`g! ::  FN  =  nothing` in-place gradient function for problem **3** with mutable arrays as input. 
```Julia
speedmapping([1.,1.]; g! = (grad, x) -> grad .=  4x.^3)
```
`m` and `g` are versions of `m!` and `g!` with immutable types like `Real` or `Complex` scalar, `StaticArray` or `Tuple` as input and output.
```Julia
using StaticArrays
speedmapping(1.; m = x -> 0.9x)
speedmapping(SA[1.,1.]; m = x -> 0.9x)
speedmapping(1.; g = x -> 4x^3)
speedmapping((1.,1.); g = x -> (4x[1]^3,3x[2] - 1))
```
`f ::  FN  =  nothing` computes an objective function. 
- For **3**, `f` will be used to initialize the learning rate better.
- For  **1** using **AA**, `f` will be used ensure monotonicity of the algorithm. 

`lower, upper::Union{Nothing, T} = nothing` define bounds on parameters which can be used with any problems and any algorithm.
```Julia
speedmapping([1.,1.]; g!  =  (grad, x)  -> grad .= 4x.^3, lower = [-Inf, 2.])
```
### Other keyword arguments

`algo :: Symbol  = r! !==  nothing  ?  :aa  :  :acx` determines the method used, either `:acx` or `:aa` (default: `:acx`, unless `r!` is used).

- Affecting both **ACX** and **AA**: `cache`,  `abstol`, `pnorm`, `maps_limit`, `iter_limit`, `time_limit`, `reltol_resid_grow`, `buffer`, `store_trace`
- Affecting **ACX**: `orders`, `initial_learning_rate`
- Affecting **AA**: `lags`, `condition_max`, `adarelax`, `relax_default`, `composite`, `abstol_obj_grow`
"""
function speedmapping(
        x0 :: T; f :: FN = nothing, g! :: FN = nothing, g :: FN = nothing, m! :: FN = nothing, 
        m :: FN = nothing, r! :: FN = nothing, algo::Symbol = r! !== nothing ? :aa : :acx, # Note: we don't use r because static arrays are not implemented for aa 
        cache :: Union{AcxCache, AaCache, Nothing} = nothing, 
        orders :: Tuple = (2,3,3), initial_learning_rate :: Real = 1., initialize_learning_rate :: Bool = true,
        lags :: Integer = 30, condition_max :: Real = 1e6, rel_default :: Real = 1., 
        adarelax :: Symbol = m! !== nothing ? :minimum_distance : :none, composite :: Symbol = :none, 
        abstol :: AbstractFloat = 1e-8, pnorm :: Real = 2., 
        maps_limit :: Real = 1_000_000_000, iter_limit = 1_000_000_000, time_limit :: Real = Inf, 
        reltol_resid_grow :: Real = algo == :aa ? 4. : (g! !== nothing || g !== nothing) ? 1e5 : 100, 
        abstol_obj_grow :: Real = √abstol, 
        lower :: Union{Nothing, T} = nothing, upper :: Union{Nothing, T} = nothing, 
        buffer :: AbstractFloat = (m! !== nothing || m !== nothing) ? 0.05 : 0.,
        store_trace :: Bool = false
    ) where {T} # T can typically be a mutable abstract array, or with :acx it could be a scalar, static array, a tuple...

    firstindex(x0) == 1 # Just to avoid trouble for now.
    FT = real(eltype(x0)) # While speedmapping accepts complex types, many quantities like norms, stepsize, and relaxation parameter require AbstractFloats
    FT <: AbstractFloat || throw(ArgumentError("Starting point must be of floating-point type (real or complex)."))

    max_time = time_limit < Inf ? time() + time_limit : Inf

    in_place = !isimmutable(x0)
    (!in_place && (m! ≠ nothing || g! ≠ nothing)) && throw(ArgumentError("Use m or g with scalars or static arrays."))
    (in_place && (m ≠ nothing || g ≠ nothing)) && throw(ArgumentError("Use m! or g! with mutable arrays."))
    n_mapping_functions = (m! ≠ nothing) + (g! ≠ nothing) + (m ≠ nothing) + (g ≠ nothing) + (r! ≠ nothing)
    n_mapping_functions > 1 && throw(ArgumentError("Can't provide more than one of m!, g!, m, or g."))

    params_F = (abstol = FT(abstol), pnorm = FT(pnorm), buffer = FT(buffer), resid_grow = FT(reltol_resid_grow), 
        norm_grow = FT(abstol_obj_grow), initial_learning_rate = FT(initial_learning_rate))
    
    maps_limit = Int(ceil(min(maps_limit, typemax(1))))
    iter_limit = Int(ceil(min(iter_limit, typemax(1))))
    params_I = (iter_limit = iter_limit, maps_limit = maps_limit)
    bounds = (l = lower, u = upper)
    
    if algo == :aa
        # Fix this (inputs, etc)
        m! ≠ nothing || r! ≠ nothing || throw(ArgumentError("m! or r! must be provided with algo :aa")) 
        (g! ≠ nothing || g ≠ nothing) && throw(ArgumentError("for minimization, use algo :acx"))
        r! ≠ nothing && (lower !== nothing || upper !== nothing) && throw(ArgumentError("bounds not available with r!"))

        if in_place && cache === nothing 
            cache = AaCache(x0; lags)
        elseif !(typeof(cache) <: AaCache) # too type many arguments to specify them
            throw(ArgumentError("must supply a cache of type AaCache with the algo :aa"))
        elseif length(x0) != length(cache.x) # checking length because cache.x is shaped as a vector
            throw(ArgumentError("length of cache and size of x0 differ"))
        end

        composite ∉ (:none, :aa1, :acx2) && throw(ArgumentError("Unknown composite type"))

        return aa(f, r!, m!, cache, x0, condition_max, adarelax, FT(rel_default), composite, 
            params_F, params_I, bounds, max_time, store_trace)
    end

    min_orders, max_orders = extrema(orders)
    (min_orders < 2 || max_orders > 3) && throw(ArgumentError("orders must be 2 or 3"))
    min_orders ≠ 2 && throw(ArgumentError("orders must contain at least one 2"))

    if in_place
        if cache === nothing 
            cache = AcxCache(x0)
        elseif !(typeof(cache) <: AcxCache)
            throw(ArgumentError("must supply a cache of type AcxCache with the algo :acx"))
        elseif size(x0) != size(cache.x_now)
            throw(ArgumentError("size of cache and size of x0 differ"))
        end
    end
    if g! === m! === g === m === nothing
        if f === nothing
            throw(ArgumentError("No function supplied."))
        elseif isa(x0, AbstractArray)
            return acx(cache, f, (grad, x) -> ForwardDiff.gradient!(grad, f, x), m!, g, m, x0, 
                orders, params_F, params_I, bounds, max_time, store_trace, initialize_learning_rate)
        else
            return acx(cache, f, g!, m!, x -> ForwardDiff.derivative(f, x), m, x0, 
                orders, params_F, params_I, bounds, max_time, store_trace, initialize_learning_rate)
        end
    else
        return acx(cache, f, g!, m!, g, m, x0, orders, params_F, params_I, bounds, max_time, 
            store_trace, initialize_learning_rate)
    end
end