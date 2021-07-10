using AccurateArithmetic
using ForwardDiff
using LinearAlgebra


#this allows to bail out on other numeric types, but
#keep using the accurate arithmetic package when available

function accurate_dot(x::AbstractArray{T1},y::AbstractArray{T2}) where {T1<:Union{Float32,Float64},T2<:Union{Float32,Float64}} 
    return dot_oro(x,y)
end

function accurate_dot(x,y)
    return dot(x,y)
end


_isbad(x) = isnan(x) || isinf(x)
_mod1(x, m) = (x - 1) % m + 1

Base.@kwdef mutable struct State{T<:Real}
    check_obj    :: Bool
    autodiff     :: Bool
    Lp           :: T
    maps_limit   :: T
    time_limit   :: T
    buffer       :: T
    t0           :: T
    tol          :: T
 
    go_on        :: Bool = true
    converged    :: Bool = false
    maps         :: Int64 = 0 # Or the # of g! calls if optim == true.
    k            :: Int64 = 0
    f_calls      :: Int64 = 0
    σ            :: T = zero(T)
    α            :: T = one(T)
    obj_now      :: T = T(Inf)
    norm_∇       :: T = T(Inf)

    ix₀          :: Int64 = 1
    ix           :: Int64 = 1
    ix_best      :: Int64 = 1
    ix_new       :: Int64 = 1
    ord_best     :: Int64 = 0
    i_ord        :: Int64 = 0
    α_best       :: T = one(T)
    σ_mult_fail  :: T = one(T)
    σ_mult_loop  :: T = one(T)
    α_mult       :: T = one(T)
    norm_best    :: T = T(Inf)
    obj_best     :: T = T(Inf)
    α_boost      :: T = one(T)

    σs           :: Vector{T} = zeros(T,10)
    norm_∇s      :: Vector{T} = zeros(T,10)
    σs_i         :: Int64 = 1
end



function Base.eltype(::State{F}) where F
    return F
end
#####
##### Initialization funtions
#####

function check_arguments(
    f, g!, m!, x_in :: T, lower :: Union{T, Nothing}, upper :: Union{T, Nothing}, 
    s :: State
) where {T<:AbstractArray}

    if m! ≠ nothing && g! ≠ nothing
        throw(ArgumentError("must not provide both m! and g!")) 
    end
    if s.check_obj && f === nothing
        throw(ArgumentError("if check_obj == true, f must be provided.")) 
    end
    if (upper ≠ nothing && maximum(x_in .- upper) > 0) || 
       (lower ≠ nothing && maximum(lower .- x_in) > 0)
        throw(DomainError(x_in, "infeasible starting point")) 
    end
    
    if !((eltype(x_in) <: Real) | (eltype(x_in) <: Integer))
        throw(ArgumentError("starting point must be of type Float")) 
    end
    if s.autodiff
        #@info "minimizing f using gradient descent acceleration and ForwardDiff" 
    end
    return nothing
end

function α_too_large!(
    f, g_auto, g!, s :: State{F}, ∇temp :: T, ∇ :: T, x_in :: T, ∇∇ :: F
) :: Bool where {F,T<:AbstractArray}

    if f ≠ nothing
        obj_new = f(x_in - s.α * ∇)
        s.f_calls += 1
        return _isbad(obj_new) || (obj_new > s.obj_now - F(0.25)*s.α * ∇∇)
    else
        s.autodiff ? ∇temp .= g_auto(x_in .- s.α * ∇) : g!(∇temp, x_in .- s.α * ∇)
        s.maps += 1
        ∇∇temp = ∇temp ⋅ ∇temp
        return _isbad(∇∇temp) || ∇∇temp / ∇∇ > 2
    end
end

function initialize_α!(
    f, g_auto, g!, s :: State{F}, ∇ :: T, x_in :: T, temp :: T
) where {F,T<:AbstractArray}
    ∇∇ = F(∇ ⋅ ∇)
    if iszero(∇∇) 
        throw(DomainError(x_in, "∇f(x_in) = 0 (extremum or saddle point)")) 
    end

    is_too_large = α_too_large!(f, g_auto, g!, s, ∇, temp, x_in, ∇∇)
    min_mult = 2.0
    mult = min_mult * 64^2
    for i ∈ 0:30
        s.α *= mult^(1 - 2is_too_large)
        was_too_large = is_too_large
        is_too_large = α_too_large!(f, g_auto, g!, s, temp, ∇, x_in, ∇∇)
        if mult == min_mult && (is_too_large + was_too_large == 1) || s.α > 1e10
            s.α /= min_mult^(is_too_large && !was_too_large)
            break 
        end
        mult = max(mult / 64^(is_too_large + was_too_large == 1), min_mult)
    end
    return nothing
end

#####
##### Core functions
#####

# Makes sure x_try[i] stays within boundaries with a gap buffer * (bound[i] - x_old[i]).
function bound!(
    extr, x_try :: T, x_old :: T, bound :: T, buffer
) where {T<:AbstractArray}

    for i ∈ eachindex(x_try) 
        x_try[i] = extr(x_try[i], (1 - buffer) * bound[i] + buffer * x_old[i])
    end
    return nothing
end

function update_x!(
    g_auto, g!, m!, x_out :: T, ∇ :: T, s :: State, 
    x_in :: T, lower :: Union{T, Nothing}, upper :: Union{T, Nothing}
) where {T<:AbstractArray}

    if m! === nothing
        if s.maps > 0 # If maps == 0, ∇ was already computed but maps must still be updated.
            s.autodiff ? ∇ .= g_auto(x_in) : g!(∇, x_in) 
        end
        x_out .= x_in .- s.α * ∇

        if lower ≠ nothing; bound!(max, x_out, x_in, lower, s.buffer) end
        if upper ≠ nothing; bound!(min, x_out, x_in, upper, s.buffer) end
        if lower ≠ nothing || upper ≠ nothing; ∇ .= (x_in .- x_out) ./ s.α end
    else
        ∇ .= x_in
        m!(x_out, x_in)
        ∇ .-= x_out

        # If the algorithm has reached an infeasible value, it helps stability 
        # if we also slow down the change of x from mappings.
        if s.α < 1; x_out .+= (1 - s.α) * ∇ end
    end
    s.norm_∇ = norm(∇, s.Lp)
    return nothing
end

function check_convergence!(f, s :: State, x :: AbstractArray)
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

function mapping!(
    f, g_auto, g!, m!, x_out :: T, ∇ :: T, s :: State, info, 
    x_in :: T, lower :: Union{T, Nothing}, upper :: Union{T, Nothing}
) where {T<:AbstractArray}

    if s.go_on
        try
            update_x!(g_auto, g!, m!, x_out, ∇, s, x_in, lower, upper)
        catch
            if s.k == 1 # To initially catch any type or function errors.
                update_x!(g_auto, g!, m!, x_out, ∇, s, x_in, lower, upper) 
            end
            s.go_on = false
        end
        s.ix += 1
        s.maps += 1
        !_isbad(s.norm_∇) ? check_convergence!(f, s, x_out) : s.go_on = false
        store_info!(info, copy(x_out), s, false)
    end
    return nothing
end

# prodΔ computes the dot products Δᵃ ⋅ Δᵇ and Δᵇ ⋅ Δᵇ needed to compute σ 
# without creating the Δs themselves and minimizes cache misses.

function prodΔ(∇1 :: T, ∇2 :: T, temp :: T) where {T<:AbstractArray} # if p == 2
    temp .= ∇2 .- ∇1
    return accurate_dot(temp, ∇1), accurate_dot(temp, temp)
end

function prodΔ(∇1 :: T, ∇2 :: T, ∇3 :: T, temp :: T) where {T<:AbstractArray} # if p == 3
    temp .= ∇3 .- 2∇2 .+ ∇1
    return accurate_dot(temp, (∇2 .- ∇1)), accurate_dot(temp, temp)
end

function extrapolate!( # Not a real extrapolation, just a stabilizing step
    x_out :: T, x_in :: T, ∇ :: T, s :: State
) where {T<:AbstractArray}

    @. x_out = x_in - s.α * ∇
    return nothing
end

function extrapolate!(
    x_out :: T, x_in :: T, ∇1 :: T, ∇2 :: T, s :: State
) where {T<:AbstractArray}

    @. x_out = x_in - (s.α * (2s.σ - s.σ^2)) * ∇1 - (s.α * s.σ^2) * ∇2
    return nothing
end

function extrapolate!(
    x_out :: T, x_in :: T, ∇1 :: T, ∇2 :: T, ∇3 :: T, s :: State
) where {T<:AbstractArray}

    @. x_out = x_in - (s.α * (3s.σ - 3s.σ^2 + s.σ^3)) * ∇1 - 
        (s.α * (3s.σ^2 - 2s.σ^3)) * ∇2 - (s.α * s.σ^3) * ∇3
    return nothing
end

function update_α!(s :: State{F}, σ_new :: F, ΔᵇΔᵇ :: F) where F

    # Increasing / decreasing α to maintain σ ∈ [1, 2] as much as possible.
    s.α *= F(1.5)^((σ_new > 2) - (σ_new < 1)) 

    # Boosting α if ΔᵇΔᵇ gets really small
    if ΔᵇΔᵇ < 1e-100 
        s.α_boost *= 4 # Increasing more aggressively each time
        s.α = min(s.α * s.α_boost, one(F))
    end
    return nothing
end

#####
##### Functions to handle failures
#####

# update_progress! is used to know which iterate to fall back on when 
# backtracking and, optionally, which x was the smallest minimizer in case the 
# problem has multiple minima.
function update_progress!(f, s :: State{F}, x :: AbstractArray) where F

    if s.go_on && s.check_obj
        s.obj_now = f(x)
        s.f_calls += 1
        s.go_on = !_isbad(s.obj_now)
    end

    if s.go_on && (( s.check_obj && s.obj_now < s.obj_best) 
                || (!s.check_obj && s.norm_∇ < s.norm_best))
        s.ix_best = s.ix₀
        s.ix_new = _mod1(s.ix₀ + 1, 2)
        s.ord_best, s.α_best = (s.i_ord, s.α)
        s.σ_mult_fail = s.α_mult = one(F)
        s.check_obj ? s.obj_best = s.obj_now : s.norm_best = s.norm_∇
    end
    return nothing
end

function backtrack!(s :: State)

    s.ix₀ = s.ix_best
    s.i_ord = s.ord_best
    s.σ_mult_fail /= 2
    s.α_mult /= 2
    s.α = s.α_best * s.α_mult
    if s.check_obj; s.obj_now = s.obj_best end
    return nothing
end

# The future of the algorithm is fully determined by x₀, α and io. Since it is
# non-monotonic, the same (x₀, α, io) could appear twice with non-zero 
# probability, creating an infinite loop. Since the possibility is very small, 
# not much resource should be devoted to it. We verify this by comparing
# values of σ and norm(∇) over time. If (σ, norm(∇)) appear twice within a 
# certain number of iterations, we divide α and σ by 2.
function check_∞_loop!(s :: State, last_order :: Bool)

    if s.σ > s.σs[s.σs_i]
        s.σs[s.σs_i] = s.σ
        s.norm_∇s[s.σs_i] = s.norm_∇
    end

    l = length(s.σs)
    if last_order && s.σs[s.σs_i] > 0.001 && s.σ_mult_loop > 0.01 && s.σs_i == l
        loop = false
        for i ∈ 1:l - 1, j ∈ i+1:l
            if s.σs[i] > 1e-10 && abs(s.σs[i] - 1) > 1e-10 && 
                    abs(1 - s.σs[j]/s.σs[i]) < 1e-10 && 
                    abs(1 - s.norm_∇s[j]/s.norm_∇s[i]) < 1e-10
                loop = true
                s.σ_mult_loop /= 2
                s.α /= 2
                s.σs .= 0
                s.norm_∇s .= 0
                break
            end
        end
        if !loop; s.σ_mult_loop = 1.0 end
    end
    s.σs_i = _mod1(s.σs_i + last_order, l)
    return nothing
end

#####
##### Miscellaneous
#####

function store_info!(info, x, s :: State, extrapolating)

    if info ≠ nothing
        push!(info.x, x)
        push!(info.σ, s.σ)
        push!(info.α, s.α)
        push!(info.extrapolating, extrapolating)
    end
    return nothing
end

#####
##### Main funtion
#####

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
*   `orders :: Array{Int64} = [3, 3, 2]` determines `ACX`'s alternating order. 
    Must be between 1 and 3 (where 1 means no extrapolation). The two recommended
    orders are [3, 2] and [3, 3, 2], the latter being *potentially* better for 
    highly non-linear applications (see paper).
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
*   `σ_min :: Real = 0.0`: Setting to `1` may avoid stalling
    (see paper).
*   `stabilize :: Bool = false`: performs a stabilization mapping before 
    extrapolating. Setting to `true` may improve the performance for 
    applications like the EM or MM algorithm (see paper).

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
(minimizer = [0.41214914122181, 0.44095060739689546, 0.4740798646565511, 0.5125916147320679, 0.5579135738427364, 0.6120273727167589, 0.677766040697063, 0.7593262786058278, 0.8632012019116191, 1.0], maps = 16, f_calls = 0, converged = true, norm_∇ = 2.8042637093700587e-9)

julia> V = res.minimizer;

julia> dominant_eigenvalue = V'A * V / V'V
16.310005690792195

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
(minimizer = [0.9999999999962509 0.9999999999924867; 0.9999999999941136 0.9999999999882022; 0.9999999999752885 0.9999999999504781; 0.9999999989360835 0.9999999978679094], maps = 156, f_calls = 11, converged = true, norm_∇ = 9.524035730285234e-10)
```

Optimizing without objective
```
julia> speedmapping(x₀; g!)
(minimizer = [1.0000000000000542 1.0000000000001086; 0.9999999999999626 0.999999999999925; 0.9999999999997893 0.9999999999995778; 0.9999999999998465 0.9999999999996921], maps = 142, f_calls = 0, converged = true, norm_∇ = 3.524659991326121e-13)
```

Optimizing without gradient
```
julia> speedmapping(x₀; f)
(minimizer = [0.9999999999973896 0.9999999999947689; 0.9999999999958996 0.9999999999917817; 0.9999999999829567 0.9999999999658452; 0.9999999992743016 0.9999999985456991], maps = 156, f_calls = 11, converged = true, norm_∇ = 6.496402053553401e-10)
```

Optimizing with a box constraint
```
julia> upper = [0.5ones(4) Inf * ones(4)];

julia> speedmapping(x₀; f, g!, upper)
(minimizer = [0.5 0.25; 0.5 0.24999999999999864; 0.49999999999999795 0.2499999999975679; 0.4999999999999807 0.2499999999773891], maps = 138, f_calls = 11, converged = true, norm_∇ = 7.196533863189794e-9)
```
""" 
function speedmapping(
    x_in :: AbstractArray{F}; f = nothing, g! = nothing, m! = nothing, 
    orders :: Array{Int64} = [3,3,2], σ_min :: Real = zero(F), stabilize :: Bool = false,
    check_obj :: Bool = false, tol = F(1e-8), Lp :: Real = 2, 
    maps_limit :: Real = 1e6, time_limit :: Real = 1000, 
    lower :: Union{AbstractArray, Nothing} = nothing, 
    upper :: Union{AbstractArray, Nothing} = nothing, buffer  = F(0.01), 
    store_info :: Bool = false
) where F <: Real


    s = State{F}(; autodiff = f ≠ nothing && m! === nothing && g! === nothing, tol, 
        buffer, Lp, maps_limit, time_limit, t0 = time(), check_obj)
    g_auto = s.autodiff ? x -> ForwardDiff.gradient(f, x) : nothing

    type_x = F
    if lower !== nothing && eltype(lower) ≠ type_x; lower = type_x.(lower) end
    if upper !== nothing && eltype(upper) ≠ type_x; lower = type_x.(upper) end

    check_arguments(f, g!, m!, x_in, lower, upper, s)

    # Inserting stabilization steps
    if stabilize; orders = Int.(vec(hcat(ones(length(orders)),orders)')) end

    # Two x₀s to avoid copying at each improvement (maybe this is excessive optimization?)
    #using copy instead of similar because BigFloats have problems
    x₀ = [copy(x_in), similar(x_in)] 
    xs = [copy(x₀[1]) for i ∈ 1:maximum(orders)]
    ∇s = [copy(x₀[1]) for i ∈ 1:maximum(orders)] # Storing ∇s is equivalent to Δs
    temp = copy(x₀[1]) # temp storage

    if f !== nothing && (m! === nothing || check_obj)
        s.obj_now = s.obj_best = f(x_in) # Useful for initialize_α and tracking progress
        s.f_calls += 1
    end

    if m! === nothing
        s.autodiff ? ∇s[1] .= g_auto(x_in) : g!(∇s[1], x_in) # Will update s.maps later
        initialize_α!(f, g_auto, g!, s, ∇s[1], x_in, temp)
    end
    
    info = store_info ? (x = [x_in], σ = [s.σ], α = [s.α], extrapolating = [false]) : nothing

    while !s.converged && s.maps ≤ maps_limit && time() - s.t0 ≤ time_limit
        s.k += 1
        s.i_ord += 1
        s.ix = 0 # Which x is currently beeing updated

        # Avoiding a stabilization step if the last extrapolation was close to 1
        s.i_ord += stabilize && s.i_ord % 2 == 1 && abs(s.σ - 1) < 0.01 && s.k > 1

        io = _mod1(s.i_ord, length(orders))
        p = orders[io]
        s.go_on = true

        mapping!(f, g_auto, g!, m!, xs[1], ∇s[1], s, info, x₀[s.ix₀], lower, upper)
        update_progress!(f, s, x₀[s.ix₀])
        for i ∈ 2:p
            mapping!(f, g_auto, g!, m!, xs[i], ∇s[i], s, info, xs[i - 1], lower, upper)
        end

        if !s.converged && s.go_on
            if p > 1
                ΔᵃΔᵇ, ΔᵇΔᵇ = prodΔ(∇s[1:p]..., temp)
                σ_new = F(abs(ΔᵃΔᵇ) > 1e-100 && ΔᵇΔᵇ > 1e-100 ? abs(ΔᵃΔᵇ) / ΔᵇΔᵇ : 1.0)
                s.σ = max(σ_min, σ_new) * s.σ_mult_fail * s.σ_mult_loop
            end

            if lower === nothing && upper === nothing
                extrapolate!(x₀[s.ix_new], x₀[s.ix₀], ∇s[1:p]..., s)
            else
                extrapolate!(temp, x₀[s.ix₀], ∇s[1:p]..., s)
                if lower ≠ nothing; bound!(max, temp, x₀[s.ix₀], lower, s.buffer) end
                if upper ≠ nothing; bound!(min, temp, x₀[s.ix₀], upper, s.buffer) end
                x₀[s.ix_new] .= temp
            end
            store_info!(info, copy(x₀[s.ix_new]), s, true)
            s.ix₀ = s.ix_new

            if p > 1
                if m! === nothing; update_α!(s, σ_new, F(ΔᵇΔᵇ)) end
                check_∞_loop!(s, io == length(orders))
            end
        elseif !s.converged && !s.go_on
            backtrack!(s)
        end
    end
    
    if s.maps > maps_limit
        @warn "Maximum mappings exceeded."
    elseif time() - s.t0 > time_limit
        @warn "Exceeded time limit of $time_limit seconds."
    end

    minimizer = s.check_obj && s.obj_best < s.obj_now ? x₀[s.ix_best] : xs[s.ix]

    return (minimizer = minimizer, maps = s.maps, f_calls = s.f_calls, 
        converged = s.converged, norm_∇ = s.norm_∇)
end