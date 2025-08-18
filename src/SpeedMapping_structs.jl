struct AcxState{T, FT <: AbstractFloat}
    x::T
    step_length::FT
    resid_norm::FT # |Δx|
    i::Int64
end

mutable struct AcxCache{T <: AbstractArray}
    x_now :: T  # Mutable since we may swap pointers
    x_next :: T # Mutable since we may swap pointers
    x_best :: T # Mutable since we may swap pointers
    const err :: T
    const rs :: Tuple{T, T, T}
end

"""
Preallocates memory for the :acx algorithm based in the starting point.
"""
function AcxCache(x0 :: T) :: AcxCache{T} where T <: AbstractArray
    return AcxCache{T}(similar(x0), similar(x0), similar(x0), similar(x0), 
        (similar(x0), similar(x0), similar(x0)))
end

mutable struct AaIndices <: FieldVector{5, Int} # Indices and counters used inside functions
    i :: Int # Index to store the current iterate
    lags :: Int # Number of lags used in the qr decomposition
    qrdeletes :: Int # Tracks how many times qrdelete! was used. If qrdeletes > max_deletes, we rebuild the QR decomposition from scratch to minimize numerical inaccuracy.
    maps :: Int # Number of maps computed
    f_calls :: Int
end

mutable struct AaCache{
        VecT <: AbstractArray, ArrT <: AbstractArray, ArrQ <: AbstractArray, ArrR <: AbstractArray, 
        Arrα <: AbstractArray, ArrΔG <: AbstractArray, 
        ViewsX, ViewsQs, ViewsRs, Viewsθs, ViewsRs_sq, ViewsRs_tri
    }

    x :: VecT      # Mutable since we may swap pointers
    y :: VecT      # Mutable since we may swap pointers
    g :: VecT      # Mutable since we may swap pointers
    g_old :: VecT  # Mutable since we may swap pointers
    Xα :: VecT     # Mutable since we may swap pointers
    Yα :: VecT     # Mutable since we may swap pointers
    temp_x :: VecT # Mutable since we may swap pointers
    temp_y :: VecT # Mutable since we may swap pointers

    const X :: ArrT
    const Y :: ArrT
    const Q_cache :: ArrQ
    const R_cache :: ArrR
    const α :: Arrα
    const norm_sq_Δgs :: Arrα
    const inv_norm_sq_Δgs :: Arrα
    const ΔG :: ArrΔG

    const Xs :: ViewsX
    const Ys :: ViewsX
    const Qs :: ViewsQs 
    const Rs :: ViewsRs 
    const θs :: Viewsθs
    const Rs_sq :: ViewsRs_sq
    const Rs_tri :: ViewsRs_tri
    const ind :: AaIndices
end

"""
Preallocates memory for the :aa algorithm based in the starting point.

Keyword arguments:
; lags = 30
"""
function AaCache(x0 :: AbstractArray{FT}; lags = 30) where FT <: AbstractFloat
    Base.require_one_based_indexing(x0) # Just to avoid trouble for now... 
    n = length(x0)
    m = min(n, lags)
    X = zeros(FT, n, m + 1)
    Y = zeros(FT, n, m + 1)
    α = Vector{FT}(undef,m+1)
    θ = Vector{FT}(undef,m+1)
    norm_sq_Δgs = Vector{FT}(undef,m+1)
    inv_norm_sq_Δgs = Vector{FT}(undef,m+1)
    Q_cache = Array{FT}(undef,n, m)
    R_cache = zeros(FT,m, m)

    Ys = [view(Y,:,i) for i in 1:m+1]
    Xs = [view(X,:,i) for i in 1:m+1]
    Qs = [view(Q_cache,:,1:i) for i in 1:m]
    Rs = [view(R_cache,1:i-1,i) for i in 1:m]
    θs = [view(θ,1:i) for i in 1:m]
    Rs_sq = [view(R_cache,1:i,1:i) for i in 1:m]
    Rs_tri = [UpperTriangular(Rs_sq[i]) for i in 1:m]
    
    x = similar(x0,(n,)) # In case x0 is not a vector or is not 1 indexed.
    ΔG = [similar(x) for _ in 1:m+1]
    return AaCache{typeof(x), typeof(X), typeof(Q_cache), typeof(R_cache), typeof(α), 
        typeof(ΔG), typeof(Xs), typeof(Qs), typeof(Rs), typeof(θs), typeof(Rs_sq),
        typeof(Rs_tri)}(
        x,          # x
        similar(x), # y
        similar(x), # g
        similar(x), # g_old
        similar(x), # Xα
        similar(x), # Yα
        similar(x), # temp_x
        similar(x), # temp_y
        X, Y, Q_cache, R_cache, α, norm_sq_Δgs, inv_norm_sq_Δgs, ΔG, Xs, Ys, Qs, Rs, θs, Rs_sq, 
        Rs_tri, AaIndices(1,0,0,0,0))
end

struct AaState{T, FT<:AbstractFloat}
    x::T
    lags :: Int64 # The number of past iterates (y - x) used in the optimization
    relaxation_parameter :: FT # β
    residual_norm :: FT # |Δx|
end

"""
- ``minimizer :: typeof(x0)``: The solution
- ``residual_norm :: AbstractFloat``: The norm of the residual, which would be |xout - xin| for problem **1**, |residual| for problem **2**, and |∇f(x)| for problem **3** (only for non-binding components of the gradient).
- ``maps``: the number of maps, function evaluations or gradient evaluations
- ``f_calls``: The number of objective function evaluations
- ``iterations``: The number of iterations
- ``status :: Symbol``: Should equal ``:first_order`` if a solution has been found.
- ``algo ∈ (:acx, :aa)``
- ``acx_trace`` A vector of ``AcxState`` if `algo == :acx && store_trace == true`, `nothing` otherwise.
- ``aa_trace`` A vector of ``AaState`` if `algo == :aa && store_trace == true`, `nothing` otherwise.
- ``last_learning_rate :: AbstractFloat`` The last learning rate, only meaningful for problem **3**.
"""
struct SpeedMappingResult{T, FT <: AbstractFloat}
    minimizer :: T
    residual_norm :: FT # |resid|, or |∇f(x)| for optimization
    maps :: Int64
    f_calls :: Int64
    iterations :: Int64
    status :: Symbol
    algo :: Symbol
    acx_trace :: Union{Nothing,Vector{AcxState{T, FT}}}
    aa_trace :: Union{Nothing, Vector{AaState{T, FT}}}
    last_learning_rate :: FT # Only useful for the acx method (set to one(FT) for AA)
end

# Put back trace somehow

Base.show(io::IO, result::SpeedMappingResult) = begin # Ok we still initially throw exceptions, but dump them in the log if they create type instability.
    print("• minimizer: ")
    show(IOContext(stdout, :limit => true), "text/plain", result.minimizer)
    print("\n\nResult of SpeedMapping, ", result.algo, " algo")
    print("\n• maps: ",result.maps, "  ")
    result.f_calls > 0 && print("• f_calls: ", result.f_calls, "  ")
    print("• iterations: ", result.iterations, "  ")
    println("• residual_norm: = ", result.residual_norm)
    color = result.status == :first_order ? :light_green : :light_red
    printstyled("• status: ", result.status, color = color)
end