using AccurateArithmetic: dot_oro
using ForwardDiff
using LinearAlgebra
using MuladdMacro

@inline _isbad(x) = isnan(x) || isinf(x)
@inline _mod1(x, m) = (x - 1) % m + 1

function accurate_dot(x::AbstractArray{T1}, y::AbstractArray{T2}) where {T1<:Union{Float32,Float64},T2<:Union{Float32,Float64}}
    return dot_oro(x, y)
end

function accurate_dot(x, y)
    return dot(x, y)
end

Base.@kwdef mutable struct State{T<:Number,RealT<:Real}

    check_obj::Bool
    has_obj::Bool
    is_map::Bool
    has_constr::Bool
    store_info::Bool

    maps_limit::Int
    time_limit::Int

    buffer::RealT
    Lp::RealT = RealT(2)
    tol::RealT = √eps(RealT)

    go_on::Bool = true
    converged::Bool = false

    maps::Int = zero(Int) # nb of maps or of g! calls if optim == true.
    k::Int = zero(Int)
    f_calls::Int = zero(Int)

    p::Int = zero(Int) # The order of the current extrapolation

    σ::RealT = zero(RealT)
    α::RealT = one(RealT)
    obj_now::RealT = RealT(Inf)
    norm_∇::RealT = RealT(Inf)

    ix::Int = one(Int)
    ord_best::Int = zero(Int)
    i_ord::Int = zero(Int)

    α_best::RealT = one(RealT)
    σ_mult_fail::RealT = one(RealT)
    σ_mult_loop::RealT = one(RealT)
    α_mult::RealT = one(RealT)
    α_boost::RealT = one(RealT)
    norm_best::RealT = RealT(Inf)
    obj_best::RealT = RealT(Inf)

    σs_i::Int = one(Int)
end

#####
##### Core functions
#####

# Makes sure x_try[i] stays within boundaries with a gap buffer * (bound[i] - x_old[i]).

function bound!(
    extr, x_try::ArrT, x_old::ArrT, bound::Union{AbstractArray,Nothing}, s::State
) where {ArrT<:AbstractArray}
    if bound !== nothing
        @simd for i ∈ eachindex(x_try)
            @inbounds x_try[i] = extr(x_try[i], (1 - s.buffer) * bound[i] + s.buffer * x_old[i])
        end
    end
    return nothing
end

function bound_and_update∇!(
    extr, x_try::ArrT, ∇::ArrT, x_old::ArrT, bound::Union{AbstractArray,Nothing}, s::State
) where {ArrT<:AbstractArray}

    if bound !== nothing
        @simd for i ∈ eachindex(x_try)
            @inbounds if (abs(x_old[i] - bound[i]) < s.tol) && (extr(x_try[i], bound[i]) ≠ x_try[i])
                ∇[i] = 0
            end
            @inbounds x_try[i] = extr(x_try[i], (1 - s.buffer) * bound[i] + s.buffer * x_old[i])
        end
    end

    return nothing
end

@inline function descent!(
    x_out::ArrT, ∇::ArrT, s::State,
    x_in::ArrT, lower::Union{AbstractArray,Nothing}, upper::Union{AbstractArray,Nothing}
) where {ArrT<:AbstractArray}
    @muladd @. x_out = x_in - s.α * ∇
    if lower !== nothing
        bound_and_update∇!(max, x_out, ∇, x_in, lower, s)
    end
    if upper !== nothing
        bound_and_update∇!(min, x_out, ∇, x_in, upper, s)
    end
    return nothing
end

@inline function update_x!(
    gm!, x_out::ArrT, ∇::ArrT, s::State, x_in::ArrT,
    lower::Union{AbstractArray,Nothing}, upper::Union{AbstractArray,Nothing}
) where {ArrT<:AbstractArray}

    if !s.is_map
        if s.k > 1 || s.ix > 1 # If s.k == 1 && s.ix ≤ 1, ∇ and x were already computed while initializing α.
            gm!(∇, x_in)
        end
        if s.ix < s.p || s.has_constr || s.store_info # Computing the last x before extrapolation is useless unless we need to check boundaries or store info
            descent!(x_out, ∇, s, x_in, lower, upper)
        end
        s.norm_∇ = norm(∇, s.Lp) # NOTE: to provide an accurate stopping criterion, s.norm_∇ MUST be computed here, where it is the true gradient orthogonal to the binding constraints. After, ∇ is updated to be used in the extrapolation.
        if s.has_constr
            ∇ .= x_in .- x_out # ∇ needs to be updated to be used in the extrapolation in case some constraints were binding.
        end
    else
        ∇ .= x_in
        gm!(x_out, x_in) # Note: I prefer computing ∇ like this (storing x_in before calling gm!(x_out, x_in)) just in case map!(x_out, x_in) provided by the user somehow changes x_in as well as x_out
        ∇ .-= x_out

        s.norm_∇ = norm(∇, s.Lp)
        if s.α < 1
            @muladd @. x_out += (1 - s.α) * ∇ # If the algorithm has reached an infeasible value, it helps stability of the algorithm if we also slow down the change of x from mappings.
        end
    end
    return nothing
end

function check_convergence!(f, s::State, x::AbstractArray)
    if s.norm_∇ <= s.tol # only based on norm_∇ (for now)
        s.go_on = false

        if s.check_obj
            s.obj_now = f(x)
            s.f_calls += 1
        end
        s.converged = !s.check_obj || !_isbad(s.obj_now)
    end
    return nothing
end

@inline function mapping!(
    f, gm!, x_out::ArrT, ∇::ArrT, s::State, info, x_in::ArrT,
    lower::Union{AbstractArray,Nothing}, upper::Union{AbstractArray,Nothing}
) where {ArrT<:AbstractArray}

    if s.go_on
        s.ix += 1
        s.maps += 1
        update_x!(gm!, x_out, ∇, s, x_in, lower, upper)
        !_isbad(s.norm_∇) ? check_convergence!(f, s, x_out) : s.go_on = false
        if s.converged && s.ix == s.p && !s.has_constr # Updating the last x is now useful
            descent!(x_out, ∇, s, x_in, lower, upper)
        end
        store_info!(info, x_out, s, false)
    end
    return nothing
end

function prodΔ(∇s::Vector{T}, xs::Vector{T}, s::State) where {T<:AbstractArray}

    @. xs[1] = ∇s[2] - ∇s[1] # Here xs[1] only serve as temp storage to avoid allocation
    if s.p == 2
        return abs(accurate_dot(xs[1], ∇s[1])), abs(accurate_dot(xs[1], xs[1]))
    elseif s.p == 3
        @muladd @. xs[2] = ∇s[3] - 2∇s[2] + ∇s[1] # Here xs[2] only serve as temp storage to avoid allocation
        return abs(accurate_dot(xs[2], xs[1])), abs(accurate_dot(xs[2], xs[2]))
    end
end

function extrapolate!(
    x_out::ArrT, x_in::ArrT, ∇s::Vector{ArrT}, s::State
) where {ArrT<:AbstractArray}
    α = s.is_map || !s.has_constr ? s.α : 1 # If we are doing constrained optimization, ∇ has been set to x_out - x_in after descent and bound checking, which means α = 1. 
    if s.p == 1
        @muladd @. x_out = x_in - α * ∇s[1] # Just a stabilizing step
    elseif s.p == 2
        c1, c2 = α * (2s.σ - s.σ^2), α * s.σ^2
        @muladd @. x_out = x_in - c1 * ∇s[1] - c2 * ∇s[2]
    elseif s.p == 3
        c1, c2, c3 = α * (3s.σ - 3s.σ^2 + s.σ^3), α * (3s.σ^2 - 2s.σ^3), α * s.σ^3
        @muladd @. x_out = x_in - c1 * ∇s[1] - c2 * ∇s[2] - c3 * ∇s[3]
    end
    return nothing
end

@inline function update_α!(
    s::State{T,RealT}, σ_new::RealT, absΔᵇΔᵇ
) where {T,RealT}

    # Increasing / decreasing α to maintain σ ∈ [1, 2] as much as possible.
    s.α *= RealT(1.5)^((σ_new > 2) - (σ_new < 1))

    if absΔᵇΔᵇ < 1e-100 # if absΔᵇΔᵇ gets really small (fairly rare)...
        s.α_boost *= 2 # Increasing more aggressively each time
        s.α = min(s.α * s.α_boost, one(RealT)) # We boost α
    end
    return nothing
end

#####
##### Functions to handle failures
#####

# update_progress! is used to know which iterate to fall back on when 
# backtracking and, optionally, which x was the smallest minimizer in case the 
# problem has multiple minima.

@inline function update_progress!(
    f, s::State{T,RealT}, x₀::ArrT, x_best::ArrT
) where {ArrT<:AbstractArray,T,RealT}

    if s.go_on && s.check_obj
        s.obj_now = f(x₀)
        s.f_calls += 1
        s.go_on = !_isbad(s.obj_now)
    end

    if s.go_on && ((s.check_obj && s.obj_now < s.obj_best) ||
                   (!s.check_obj && s.norm_∇ < s.norm_best))
        x_best .= x₀
        s.ord_best, s.α_best = (s.i_ord, s.α)
        s.σ_mult_fail = s.α_mult = one(RealT)
        s.check_obj ? s.obj_best = s.obj_now : s.norm_best = s.norm_∇
    end
    return nothing
end

function backtrack!(s::State, x₀::ArrT, x_best::ArrT) where {ArrT<:AbstractArray}

    x₀ .= x_best
    s.i_ord = s.ord_best
    s.σ_mult_fail /= 2
    s.α_mult /= 2
    s.α = s.α_best * s.α_mult
    if s.check_obj
        s.obj_now = s.obj_best
    end
    return nothing
end

# The future of the algorithm is fully determined by x₀, α and i_ord. Since ACX
# non-monotonic, the same (x₀, α, i_ord) could in principle appear twice with 
# non-zero probability, creating an infinite loop. The probability is very small, 
# so little resources should be devoted to it. We verify this by comparing values 
# of σ and norm(∇) over time. If (σ, norm(∇)) appear twice within a certain number 
# of iterations, we divide α and σ by 2.

function check_∞_loop!(
    s::State, last_order::Bool, σs::Vector{T}, norm_∇s::Vector{T}
) where {T<:Number}

    if s.σ > σs[s.σs_i]
        σs[s.σs_i] = s.σ
        norm_∇s[s.σs_i] = s.norm_∇
    end

    l = length(σs)
    if last_order && σs[s.σs_i] > 0.001 && s.σ_mult_loop > 0.01 && s.σs_i == l
        loop = false
        for i ∈ 1:l-1, j ∈ i+1:l
            if σs[i] > 1e-10 && abs(σs[i] - 1) > 1e-10 &&
               abs(1 - σs[j] / σs[i]) < 1e-10 &&
               abs(1 - norm_∇s[j] / norm_∇s[i]) < 1e-10
                loop = true
                s.σ_mult_loop /= 2
                s.α /= 2
                σs .= 0
                norm_∇s .= 0
                break
            end
        end
        if !loop
            s.σ_mult_loop = 1.0
        end
    end
    s.σs_i = _mod1(s.σs_i + last_order, l)
    return nothing
end

#####
##### Miscellaneous
#####

function store_info!(info, x, s::State, extrapolating)
    if s.store_info
        push!(info.x, copy(x))
        push!(info.σ, s.σ)
        push!(info.α, s.α)
        push!(info.p, s.p)
        push!(info.extrapolating, extrapolating)
    end
    return nothing
end

#####
##### Initialization funtions
#####

# While ACX tolerates a wide range of starting descent step sizes (α), some
# difficult problems benefit from a good initial α. A good starting α has the 
# largest possible value while respecting two conditions: i) The Armijo condition
# (if using_f == true) with constant set to 0.25 and ii) the gradient norm does 
# not increase too fast (|∇f(x₁)|₂/|∇f(x₀)|₂ ≤ 2).

function α_too_large!(
    f, gm!, s::State, ∇s::Vector{ArrT}, xs::Vector{ArrT}, x_in::ArrT, ∇_in::ArrT,
    lower::Union{AbstractArray,Nothing}, upper::Union{AbstractArray,Nothing}, using_f::Bool
)::Bool where {ArrT<:AbstractArray}

    ∇s[1] .= ∇_in # Because of box constraints, descent_x! may modify ∇s[1]

    descent!(xs[1], ∇s[1], s, x_in, lower, upper)
    ∇∇1 = abs(dot(∇s[1], ∇s[1]))

    if using_f
        obj_new = f(xs[1])
        s.f_calls += 1
        return _isbad(obj_new) || (obj_new > s.obj_now - 0.25s.α * ∇∇1)
    else
        gm!(∇s[2], xs[1])
        s.maps += 1
        descent!(xs[2], ∇s[2], s, xs[1], lower, upper)
        ∇∇2 = abs(dot(∇s[2], ∇s[2]))
        return _isbad(∇∇2) || ∇∇2 / ∇∇1 > 4
    end
end

# From an initial α = 1, we converge on the largest possible α within a factor of
# 2 that respects the Armijo condition and the gradient not increasing too fast.
@inline function initialize_α!(
    f, gm!, s::State, ∇s::Vector{ArrT}, xs::Vector{ArrT},
    x_in::ArrT, ∇_in::ArrT, lower::Union{AbstractArray,Nothing}, upper::Union{AbstractArray,Nothing}
) where {ArrT<:AbstractArray}
    tmp1 = xs[3] # Using alternative names to make clear these are used as temp storage to avoid allocation
    tmp2 = ∇s[3] # Using alternative names to make clear these are used as temp storage to avoid allocation
    max_α = 1e10
    min_mult = 2.0
    mult = min_mult * 64^2
    using_f = s.has_obj # Armijo condition first since computing f should be faster
    is_too_large = α_too_large!(f, gm!, s, ∇s, xs, x_in, ∇_in, lower, upper, using_f)
    for i ∈ 0:100
        s.α *= mult^(1 - 2is_too_large)
        was_too_large = is_too_large
        is_too_large = s.α > max_α || α_too_large!(f, gm!, s, ∇s, xs, x_in, ∇_in, lower, upper, using_f)
        if !is_too_large
            tmp1 .= ∇s[1]
            tmp2 .= ∇s[2]
        end
        if mult == min_mult && (is_too_large ⊻ was_too_large)
            if is_too_large # We are overshooting for the last time
                s.α /= min_mult  # The last good s.α
                ∇s[1] .= tmp1    # The last good ∇s[1], so no need to recompute it.
                ∇s[2] .= tmp2    # The last good ∇s[2], so no need to recompute it.
            end
            if using_f # Second condition to check: the gradient must decrease, or not increase too fast
                if s.α < max_α && !α_too_large!(f, gm!, s, ∇s, xs, x_in, ∇_in,
                    lower, upper, false)
                    break
                else
                    max_α = s.α # Updating max_α to make sure the 1st condition is still respected, just in case.
                    using_f = false
                    mult = min_mult * 64^2
                    is_too_large = true   # To make sure we restart reducing α.
                    was_too_large = false # To make sure we restart reducing α.
                end
            else
                break
            end
        end
        mult = max(mult / 64^(is_too_large ⊻ was_too_large), min_mult)
    end
    return nothing
end

#####
##### Main funtion
#####

# We use this internal functions _speedmapping and _f to remove the type 
# instability associated with f, g! and m! potentially being nothing.

function _f(x::AbstractArray, f)
    RealT = real(eltype(x))
    return RealT(f(x))
end

function _f(x::AbstractArray, f::Nothing)
    return zero(real(eltype(x))) # A bogus output for type stability
end

function _speedmapping(
    x_in::AbstractArray, f, gm!, s::State, orders::Vector{Int},
    σ_min::Real, stabilize::Bool,
    lower::Union{AbstractArray,Nothing}, upper::Union{AbstractArray,Nothing}
)

    t0 = time()

    # Inserting stabilization steps
    if stabilize
        orders = Int.(vec(hcat(ones(length(orders)), orders)'))
    end

    x₀ = copy(x_in)
    xs = [similar(x₀) for i ∈ 1:3]
    ∇s = [similar(x₀) for i ∈ 1:3] # Storing ∇s is equivalent to Δs, but saves comptations for gradient descent

    x_best = similar(x₀)

    if s.has_obj && (!s.is_map || s.check_obj)
        s.obj_now = s.obj_best = f(x₀) # Useful for initialize_α and tracking progress
        s.f_calls += 1
    end

    if !s.is_map
        gm!(x_best, x₀) # Here x_best acts purely as temp storage for the initial ∇ to avoid allocation. 
        s.maps = -1 # To avoid double counting since we'll save the 2 first gradient evaluations
        initialize_α!(f, gm!, s, ∇s, xs, x₀, x_best, lower, upper)
        if abs(dot(∇s[2], ∇s[1])) / abs(dot(∇s[2], ∇s[2])) < 1
            s.i_ord = length(orders) - 1
        end
    end

    info = (x=[copy(x₀)], σ=[s.σ], α=[s.α], p=[0], extrapolating=[false])

    RealT = typeof(s.σ)
    σs = zeros(RealT, 10)
    norm_∇s = zeros(RealT, 10)

    period_check_time = 1.0 # To avoid calling time() too often 
    t_start_loop = time_now = time()
    while !s.converged && s.maps < s.maps_limit && time_now - t0 ≤ s.time_limit
        s.k += 1
        s.i_ord += 1
        s.ix = 0 # Which x is currently beeing updated

        # Avoiding a stabilization step if the last extrapolation was close to 1
        s.i_ord += stabilize && s.i_ord % 2 == 1 && abs(s.σ - 1) < 0.01 && s.k > 1

        io = _mod1(s.i_ord, length(orders))
        s.p = orders[io]
        s.go_on = true

        mapping!(f, gm!, xs[1], ∇s[1], s, info, x₀, lower, upper)
        update_progress!(f, s, x₀, x_best)
        for i ∈ 2:s.p
            mapping!(f, gm!, xs[i], ∇s[i], s, info, xs[i-1], lower, upper)
        end

        if !s.converged && s.go_on
            if s.p > 1
                absΔᵃΔᵇ, absΔᵇΔᵇ = prodΔ(∇s, xs, s)
                σ_new = RealT(absΔᵃΔᵇ > 1e-100 && absΔᵇΔᵇ > 1e-100 ? absΔᵃΔᵇ / absΔᵇΔᵇ : 1.0)
                s.σ = max(σ_min, σ_new) * s.σ_mult_fail * s.σ_mult_loop
            end
            if s.has_constr
                extrapolate!(xs[1], x₀, ∇s, s)
                if lower ≠ nothing
                    bound!(max, xs[1], x₀, lower, s)
                end
                if upper ≠ nothing
                    bound!(min, xs[1], x₀, upper, s)
                end
                x₀ .= xs[1]
            else
                extrapolate!(x₀, x₀, ∇s, s)
            end
            store_info!(info, x₀, s, true)

            if s.p > 1
                if !s.is_map
                    update_α!(s, σ_new, absΔᵇΔᵇ)
                end
                check_∞_loop!(s, io == length(orders), σs, norm_∇s)
            end
        elseif !s.converged && !s.go_on
            backtrack!(s, x₀, x_best)
        end

        if s.k % period_check_time == 0 # Avoids calling time() too often
            time_now = time()
            period_check_time = min(ceil(0.01s.k / (time_now - t_start_loop)), 1000.0)
        end
    end

    if s.maps > s.maps_limit
        status = :max_eval
    elseif time() - t0 > s.time_limit
        status = :max_time
    else
        status = :first_order
    end

    minimizer = s.check_obj && s.obj_best < s.obj_now ? x_best : xs[s.ix]

    return (minimizer=minimizer, maps=s.maps, f_calls=s.f_calls,
        status=status, norm_∇=s.norm_∇, elapsed_time=time() - t0, info=info)
end

"""
    SpeedMapping
`speedmapping(x₀; m!, kwargs...)` accelerates the convergence of a mapping 
`m!(x_out, x_in)` to a fixed point of `m!` by the Alternating cyclic 
extrapolation algorithm (ACX). Since gradient descent is an example 
of such mapping, `speedmapping(x0; g!, kwargs...)` can also perform multivariate 
optimization based on the gradient function `g!(∇, x)`.

Reference:
N. Lepage-Saucier, _Alternating cyclic extrapolation methods for optimization 
algorithms_, arXiv:2104.04974 (2021). https://arxiv.org/abs/2104.04974

### Arguments

*   `x₀ :: AbstractArray`: The starting point; must be of eltype `Float`, Real 
    or Complex.

Main keyword arguments:
*   `m!(x_out, x_in)`: A map for which a fixed-point must be found.
*   `g!(∇, x)`: The gradient of a function to be minimized.
*   `f(x)`: The objective of the function to be minimized. It is useful to *i*)
    compute a good initial α (learning  rate) for the gradient descent, *ii*)
    optimize using autodiff or *iii*) track the progress of the algorithm. In
    case neither `m!` nor `g!` is provided, then `f` is used to compute `g`
    using `ForwardDiff`.
*   `lower, upper :: Union{AbstractArray, Nothing} = nothing`: Box constraints 
    for the optimization. NOTE: When appropriate, it should also be used with 
    `m!`. Even if `m!` always keeps `x_out` within bounds, an extrapolation 
    step could throw `x` out of bounds.
*   `tol :: Real = √eps(), Lp :: Real = 2` When using `m!`, the algorithm 
    stops when `norm(F(xₖ) - xₖ, Lp) ≤ tol`. When using `g!`, the algorithm 
    stops when `norm(∇f(xₖ), Lp) ≤ tol`. 
*   `maps_limit :: Real = 1e6`: Maximum number of `m!` calls or `g!` calls. 
*   `time_limit :: Real = 1000`: Maximum time in seconds.

Minor keyword arguments:
*   `orders :: Array{Int64} = [3, 3, 2]` determines `ACX`'s alternating order. 
    Must be between 1 and 3 (where 1 means no extrapolation). The two recommended
    orders are [3, 2] and [3, 3, 2], the latter being *potentially* better for 
    highly non-linear applications (see paper).
*   `check_obj :: Bool = false`: In case of `NaN` or `Inf` values, the algorithm
    restarts at the best past iterate. If `check_obj = true`, progress is 
    monitored with the value of the objective (requires `f`). 
    Otherwise, it is monitored with `norm(F(xₖ) - xₖ, Lp)`. Advantages of 
    `check_obj = true`: more precise and if the algorithm converges on a bad 
    local minimum, it can return the best of all past iterates. Advantages of 
    `check_obj = false`: for well-behaved convex problems, it avoids the 
    effort and time of providing `f` and calling it at every iteration.
*   `store_info :: Bool = false`: Stores `xₖ`, `σₖ` and `αₖ` (see paper).
*   `buffer :: Float64 = 0.01` If `xₖ` goes out of bounds, it is brought back in
    with a buffer. Ex. `xₖ = buffer * xₖ₋₁ + (1 - buffer) * upper`. Setting 
    `buffer = 0.001` may speed-up box-constrained optimization.

Keyword arguments to fine-tune fixed-point mapping acceleration (using `m!`):
*   `σ_min :: Real = 0.0`: Setting to `1` may avoid stalling (see paper).
*   `stabilize :: Bool = false`: performs a stabilization mapping before 
    extrapolating. Setting to `true` may improve the performance for 
    applications like accelerating the EM or MM algorithms (see paper).

# Example: Finding a dominant eigenvalue
```jldoctest
julia> using LinearAlgebra

julia> using SpeedMapping

julia> A = ones(10) * ones(10)' + Diagonal(1:10);

julia> function power_iteration!(x_out, x_in, A)
           mul!(x_out, A, x_in)
           x_out ./= maximum(abs.(x_out))
       end;

julia> res = speedmapping(ones(10); m! = (x_out, x_in) -> power_iteration!(x_out, x_in, A))
(minimizer = [0.4121491412218099, 0.44095060739689534, 0.47407986465655094, 0.5125916147320678, 0.5579135738427362, 0.6120273727167587, 0.6777660406970626, 0.7593262786058276, 0.8632012019116189, 1.0], maps = 16, f_calls = 0, status = :first_order, norm_∇ = 2.8042636614546266e-9, elapsed_time = 0.14999985694885254, info = (x = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], σ = [0.0], α = [1.0], p = [0], extrapolating = Bool[0]))

julia> V = res.minimizer;

julia> dominant_eigenvalue = V'A * V / V'V
16.3100056907922

```
# Example: Minimizing a multidimensional Rosenbrock
```
julia> f(x) = sum(100 * (x[i,1]^2 - x[i,2])^2 + (x[i,1] - 1)^2 for i ∈ 1:size(x,1));

julia> function g!(∇, x)
           ∇[:,1] .=  400(x[:,1].^2 .- x[:,2]) .* x[:,1] .+ 2(x[:,1] .- 1)
           ∇[:,2] .= -200(x[:,1].^2 .- x[:,2])
           return nothing
       end;

julia> x₀ = 1.0 * [-4 -3 -2 -1; 0 1 2 3]';
```

Optimizing, providing f and g!
```
julia> speedmapping(x₀; f, g!)
(minimizer = [0.9999999999827773 0.9999999999654845; 0.999999999973088 0.999999999946068; 0.9999999998866034 0.9999999997727524; 0.9999999951325508 0.9999999902456237], maps = 156, f_calls = 11, status = :first_order, norm_∇ = 4.35727535422407e-9, elapsed_time = 0.0, info = (x = [[-4.0 0.0; 
-3.0 1.0; -2.0 2.0; -1.0 3.0]], σ = [0.0], α = [0.0001220703125], p = [0], extrapolating = Bool[0]))
```

Optimizing without objective
```
julia> speedmapping(x₀; g!)
(minimizer = [1.000000000000002 1.000000000000004; 0.999999999999956 0.9999999999999117; 0.9999999999998761 0.9999999999997516; 0.999999999999863 
0.9999999999997254], maps = 148, f_calls = 0, status = :first_order, norm_∇ = 2.7446698204458767e-13, elapsed_time = 0.0, info = (x = [[-4.0 0.0; 
-3.0 1.0; -2.0 2.0; -1.0 3.0]], σ = [0.0], α = [0.000244140625], p = [0], extrapolating = Bool[0]))
```

Optimizing without gradient
```
julia> speedmapping(x₀; f)
(minimizer = [0.9999999999957527 0.9999999999914151; 0.9999999999933037 0.9999999999865801; 0.9999999999716946 0.9999999999432596; 0.9999999987753265 0.999999997545751], maps = 172, f_calls = 11, status = :first_order, norm_∇ = 1.0971818937506587e-9, elapsed_time = 0.9249999523162842, info = (x = [[-4.0 0.0; -3.0 1.0; -2.0 2.0; -1.0 3.0]], σ = [0.0], α = [0.0001220703125], p = [0], extrapolating = Bool[0]))
```

Optimizing with a box constraint
```
julia> speedmapping(x₀; f, g!, upper = [0.5ones(4) Inf * ones(4)])
(minimizer = [0.5 0.25; 0.5 0.24999999999999864; 0.49999999999999795 0.24999999999757333; 0.4999999999999807 0.24999999997737743], maps = 138, f_calls = 11, status = :first_order, norm_∇ = 7.140762381788846e-9, elapsed_time = 0.0, info = (x = [[-4.0 0.0; -3.0 1.0; -2.0 2.0; -1.0 3.0]], σ = [0.0], α = [0.0001220703125], p = [0], extrapolating = Bool[0]))
```
"""

function speedmapping(
    x_in::AbstractArray; f=nothing, (g!)=nothing, (m!)=nothing,
    orders::Vector{Int}=[3, 3, 2], σ_min::Real=zero(Real),
    stabilize::Bool=false, check_obj::Bool=false, tol::Real=√eps(),
    maps_limit::Int=Int(1e9), time_limit::Real=1000,
    lower=nothing, upper=nothing, buffer::Real=0.01,
    store_info::Bool=false, Lp::Real=2
)
    T = eltype(x_in)
    RealT = real(T) # Many quantities must be Real
    if RealT <: Integer || RealT <: Rational
        throw(ArgumentError("Starting point must be of floating-point type (real or complex)."))
    end

    has_lower = lower ≠ nothing
    has_upper = upper ≠ nothing

    s = State{T,RealT}(; check_obj, has_obj=f ≠ nothing, is_map=m! ≠ nothing,
        has_constr=has_lower || has_upper, maps_limit, store_info,
        time_limit, buffer, tol, Lp)

    if s.has_constr && !(T <: Real)
        println("Constraints without real variables? Email me to explain what you need.")
        throw(TypeError)
    end

    if has_lower
        if eachindex(lower) ≠ eachindex(x_in)
            throw(ArgumentError("x_in and lower must have same indices"))
        else
            if maximum(lower .- x_in) > 0
                throw(DomainError(x_in, "infeasible starting point"))
            end
        end
    end

    if has_upper
        if eachindex(upper) ≠ eachindex(x_in)
            throw(ArgumentError("x_in and lower must have same indices"))
        end
        if maximum(x_in .- upper) > 0
            throw(DomainError(x_in, "infeasible starting point"))
        end
    end

    if s.check_obj && !s.has_obj
        throw(ArgumentError("if check_obj == true, f must be provided."))
    end

    kwargs = (orders, σ_min, stabilize, lower, upper)

    if m! ≠ nothing
        if g! ≠ nothing
            throw(ArgumentError("must not provide both m! and g!"))
        end
        return _speedmapping(x_in, x -> _f(x, f), m!, s, kwargs...)
    elseif g! === nothing
        return _speedmapping(x_in, x -> _f(x, f), (∇, x) -> ForwardDiff.gradient!(∇, f, x),
            s, kwargs...)
    else
        return _speedmapping(x_in, x -> _f(x, f), g!, s, kwargs...)
    end
end