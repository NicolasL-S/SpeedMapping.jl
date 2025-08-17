"""
fast_condition(U :: AbstractMatrix, row_sums_U_inv :: AbstractVector, n :: Integer) 
It computes the upper bound of the infinity-norm condition number of an upper-triangular matrix U. 

It is based on Algorithm 8.18 on p. 147 of Higham, Nicholas (2022). Accuracy and Stability of 
Numerical Algorithms, SIAM, 2nd Ed.
"""
function fast_condition(U :: AbstractMatrix, row_sums_U_inv :: AbstractVector, n :: Integer)

    (length(row_sums_U_inv) >= n) || throw(ArgumentError("cache too small"))
    (size(U,1) >= n && size(U,2) >= n) || throw(ArgumentError("U too small"))
    FT = real(eltype(U))

    @inbounds begin
        max_row = abs(U[n,n])
        max_row_inv = row_sums_U_inv[n] = 1/abs(U[n,n])
        for i in n-1:-1:1
            r = abs(U[i,i])
            s = FT(1)
            for j in i+1:n
                r += abs(U[i,j])
                s += abs(U[i,j]) * row_sums_U_inv[j]
            end
            row_sums_U_inv[i] = s/abs(U[i,i])
            r > max_row && (max_row = r)
            row_sums_U_inv[i] > max_row_inv && (max_row_inv = row_sums_U_inv[i])
        end
    end
    return max_row_inv * max_row
end

function qradd!(c, i, k)
    FT = real(eltype(c.x))
    k > 1 && (c.temp_y .= c.ΔG[i] .- mul!(c.temp_x, c.Qs[k-1], mul!(c.Rs[k], c.Qs[k-1]',c.ΔG[i]))) # temp_x and temp_y serve as temporary cache
    ortho_column = k == 1 ? c.ΔG[i] : c.temp_y
    c.R_cache[k, k] = lpnorm(ortho_column, 2)
    c.Qs[k][:,end] .= (FT(1) / c.R_cache[k, k]) .* ortho_column
end

@inline function qrdelete!(Q::AbstractMatrix, R::AbstractMatrix, k::Int) # Copied from NLsolve, to avoid an additional dependency
    m = size(Q, 2)
    m == LinearAlgebra.checksquare(R) || throw(DimensionMismatch())
    1 ≤ k ≤ m || throw(ArgumentError("k must be between 1 and size(Q,2)"))

    for i in 2:k
        g = first(givens(R, i - 1, i, i))
        lmul!(g, R)
        rmul!(Q, g')
    end

    @inbounds for j in 1:(k-1)
        for i in 1:j
            R[i, j] = R[i, j + 1]
        end
    end
    
    return Q, R
end

@inline function clean_qradd!(c, ind, max_qrdeletes)
    if ind.qrdeletes < max_qrdeletes
        qradd!(c, ind.i, ind.lags)
    else # Rebuilding the QR decomposition to minimize numerical inaccuracy caused by too many givens rotations
        mp1 = size(c.Y, 2)
        start0 = ind.i - ind.lags + mp1 - 1
        for j in 1:ind.lags
            qradd!(c, (start0 + j) % mp1 + 1, j)
        end
        ind.qrdeletes = 0
    end
end

function safe_qradd!(c, ind, max_qrdeletes, condition_max)
    clean_qradd!(c, ind, max_qrdeletes)

    # If we insert a perfectly collinear column, the Q matrix gets NaN or inf entries. Then qrdelete! fails or propagates these bad values to the rest of the decomposition. We delete columns in the QR decomposition until the new column can be inserted without collinearity (or is the only one).
    while isbad(c.R_cache[ind.lags,ind.lags]) || c.R_cache[ind.lags,ind.lags] == 0 || isbad(c.Qs[ind.lags][end,end])
        ind.lags -= 1
        ind.lags == 0 && return nothing # Despite our best effort, there is probably something wrong with c.ΔG[ind.i]
        qrdelete!(c.Q_cache, c.R_cache, ind.lags)
        ind.qrdeletes += 1
        clean_qradd!(c, ind, max_qrdeletes)
    end

    # Eliminating old columns if bad conditionning
    while ind.lags > 1 && fast_condition(c.Rs_sq[ind.lags], c.α, ind.lags) > condition_max
        qrdelete!(c.Q_cache, c.R_cache, ind.lags)
        ind.qrdeletes += 1
        ind.lags -= 1
    end
end

function do_after_m!(g, xout, xin, ind, bounds, buffer)
    ind.maps += 1
    isbad(xout) && return false, eltype(g)(0)
    
    if bounds.l ≠ nothing || bounds.u ≠ nothing
        box_constraint!(xout, xin, bounds, buffer, true)
        g .= xout .- xin
    end
    return true, sumsq(g)
end

function safe_map!(m!, r!, c, ind, max_new_normsq_g, bounds, buffer, true_shape)

    box_constraint!(c.temp_x, c.x, bounds, buffer, true)

    FT = real(eltype(c.x))
    damp_step = FT(1)

    normsq_g = max_new_normsq_g # Just for initializing
    for tr in 1:1000
        success_mapping = true
        local good_map
        if m! !== nothing
            try
                m!(reshape(c.temp_y, true_shape), reshape(c.temp_x, true_shape))
                  c.g .= c.temp_y .- c.temp_x
                good_map, normsq_g = do_after_m!(c.g, c.temp_y, c.temp_x, ind, bounds, buffer)
            catch
                success_mapping = false
            end
        else
            try
                r!(reshape(c.g, true_shape), reshape(c.temp_x, true_shape))
                c.temp_y .= c.temp_x .+ c.g
                good_map, normsq_g = do_after_m!(c.g, c.temp_y, c.temp_x, ind, bounds, buffer)
            catch
                success_mapping = false
            end
        end
        success_mapping && good_map && normsq_g <= max_new_normsq_g && break
        
        damp_step *= FT(0.5)
        @. c.temp_x = c.x + damp_step * (c.temp_x - c.x)
    end
    
    c.x, c.temp_x = c.temp_x, c.x
    c.y, c.temp_y = c.temp_y, c.y
    return normsq_g
end

function check_monotonicity!(f, c, ind, max_new_obj_val, true_shape)
    obj_val = max_new_obj_val
    obj_val = f(reshape(c.temp_x, true_shape))
    ind.f_calls += 1
    if obj_val > max_new_obj_val
        c.temp_x .= c.y
        obj_val = f(reshape(c.temp_x, true_shape))
        ind.f_calls += 1
    end
    return obj_val
end

function compute_β_minimum_distance(y_new :: T, y :: T, x :: T, rel_default) where T
    num = den = zero(eltype(x))
    @inbounds @simd for i in eachindex(x)
        c1 = y[i] - x[i]
        num += c1 * (y_new[i] - x[i])
        den += c1 * c1
    end

    return den > 1e-50 ? num / den : rel_default
end

# TODO: change norms to sum square like for acx (minor importance for performance though)

function aa(f, r!, m!, c::AaCache, x0, condition_max, adarel, rel_default, 
        composite, params_F, params_I, bounds, max_time, store_trace::Bool
    )
    monotonic = f !== nothing

    abstol, pnorm, buffer, reltol_resid_grow, abstol_obj_grow = params_F
    iter_limit, maps_limit = params_I
    abstolsq = abstol^2

    FT = real(eltype(c.x))
    reltol_rr = min(reltol_resid_grow^2, prevfloat(typemax(FT)))
    true_shape = size(x0)
    T = typeof(x0)
    c.x .= vec(x0)
    max_qrdeletes = 10 # How many times we can use _qrdeletes without rebuilding the QR decomposition from scratch to mitigate inaccuracy

    mp1 = size(c.Y, 2)
    m = mp1 - 1

    ind = c.ind
    ind .= (
        1, # i: Index to store the current iterate
        0, # lags: Number of lags used in the qr decomposition
        0, # qrdeletes: count subsequent qrdelete!. If qrdeletes > max_deletes, we rebuild the QR decomposition from scratch to minimize numerical inaccuracy.
        0, # maps: number of maps computed
        0  # f_calls
    )

    if monotonic
        obj_val = f(x0)
        ind.f_calls += 1
    else
        obj_val = prevfloat(typemax(FT)) # A placeholder
    end
    
    β_md_AR1 = β = β_minimum_distance = last_β_minimum_distance = rel_default
    β_diffs_sq = one(FT)
    nβ_outside_unit = 0

    normsq_g = FT(Inf)
    iter = 0

    is_map = m! !== nothing
    if is_map
        m!(reshape(c.y,true_shape), reshape(c.x,true_shape))
        c.g .= c.y .- c.x
    else
        r!(reshape(c.g,true_shape), reshape(c.x,true_shape))
        c.y .= c.x .+ c.g
    end

    trace = store_trace ? [AaState{T, FT}(copy(c.x), 0, 0., lpnorm(c.g, pnorm))] : nothing

    success_mapping, normsq_g = do_after_m!(c.g, c.y, c.x, ind, bounds, buffer)

    success_mapping || return SpeedMappingResult{T, FT}(reshape(c.x, true_shape), sqrt(normsq_g), 
        ind.maps, ind.f_calls, 1, :failure, :aa, trace)
    
    while ind.maps <= maps_limit && (max_time == Inf || time() < max_time) && iter < iter_limit

        iter += 1
        c.Xs[ind.i] .= c.x
        c.Ys[ind.i] .= c.y

        aa_step = ind.lags > 0
        if aa_step
            c.ΔG[ind.i] .= c.g .- c.g_old
            dot(c.ΔG[ind.i],c.ΔG[ind.i]) < FT(1e-100) && (aa_step = false)
        end
        aa_step && safe_qradd!(c, ind, max_qrdeletes, condition_max) # Note, safe_qradd! may reduce ind.lags

        if aa_step && ind.lags > 0

            # Computing c.α
            ldiv!(c.Rs_tri[ind.lags], mul!(c.θs[ind.lags], c.Qs[ind.lags]', c.g))
            c.α .= FT(0)
            c.α[ind.i] = FT(1) - c.θs[ind.lags][ind.lags]
            s = ind.i - ind.lags + m - 1
            c.α[(s + 1) % mp1 + 1] = c.θs[ind.lags][1]
            @inbounds for i ∈ 2:ind.lags
                c.α[(s + i) % mp1 + 1] = c.θs[ind.lags][i] - c.θs[ind.lags][i - 1]
            end

            if adarel == :none
                β = rel_default

            elseif adarel == :minimum_distance
                max_nβ_outside_unit = 10 # 100
                β_md_AR1 = 0.1β_minimum_distance + 0.9β_md_AR1
                β_diffs_sq = 0.2*(β_minimum_distance - last_β_minimum_distance)^2 + 0.8β_diffs_sq
                root_β_diffs_sq = min(sqrt(β_diffs_sq),1)
                β = root_β_diffs_sq * rel_default + (1 - root_β_diffs_sq) * β_md_AR1
                
                nβ_outside_unit > max_nβ_outside_unit && (β = rel_default)

                β < 0 && (β = rel_default)
                nβ_outside_unit = 0 ≤ β ≤ rel_default ? 0 : nβ_outside_unit + 1

            else
                throw(error("Unknown adaptive relaxation"))
            end
            
            if β == 1 && adarel == :none
                mul!(c.temp_x, c.Y, c.α)
            else
                mul!(c.Xα, c.X, c.α)
                mul!(c.Yα, c.Y, c.α)
                @. c.temp_x = c.Xα + β * (c.Yα - c.Xα)
            end

        else
            c.temp_x .= c.Ys[ind.i]
            #c.temp_x, c.y = c.y, c.temp_x # put this back after checking
        end

        c.g, c.g_old = c.g_old, c.g

        monotonic && (obj_val = check_monotonicity!(f, c, ind, obj_val + abstol_obj_grow, true_shape))

        max_new_normsq_g = abstolsq + normsq_g * (aa_step && ind.lags > 0 ? reltol_rr : FT(1.01)) # Being extra safe if we used the map directly
        
        normsq_g = safe_map!(m!, r!, c, ind, max_new_normsq_g, bounds, buffer, true_shape)

        lpnormsq_g = pnorm == 2 ? normsq_g : lpnorm(c.g, pnorm)^2

        store_trace && push!(trace, AaState{T, FT}(copy(c.x), ind.lags, β, lpnormsq_g))

        lpnormsq_g < abstolsq && break
        
        if adarel == :minimum_distance && iter >= 2
            last_β_minimum_distance = β_minimum_distance
            β_minimum_distance = compute_β_minimum_distance(c.y, c.Yα, c.Xα, rel_default)
        end

        if composite ∈ (:aa1, :acx2)
            if is_map
                m!(reshape(c.temp_x, true_shape), reshape(c.y, true_shape))
            else
                r!(reshape(c.temp_x, true_shape), reshape(c.y,true_shape))
                c.temp_x .+= c.y
            end
            ind.maps += 1
            
            box_constraint!(c.temp_x, c.y, bounds, buffer, true)
            @. c.temp_y = c.temp_x - c.y - c.g
            #σ = accurate_dot(c.temp_y, c.g) / accurate_dot(c.temp_y,c.temp_y)
            σ = dot(c.temp_y, c.g) / dot(c.temp_y,c.temp_y)
            if composite == :aa1
                @. c.temp_x = c.y - σ * (c.temp_x - c.y)
            elseif composite == :acx2
                @. c.temp_x = c.x + FT(2) * abs(σ) * c.g + σ^2 * c.temp_y
            end

            monotonic && (obj_val = check_monotonicity!(f, c, ind, obj_val + abstol_obj_grow, true_shape))

            normsq_g = safe_map!(m!, r!, c, ind, normsq_g * reltol_rr, bounds, buffer, true_shape)
            normsq_g < abstolsq && break
        end

        (aa_step || ind.lags == 0) && (ind.i = ind.i % mp1 + 1)

        if ind.lags == m && ind.qrdeletes < max_qrdeletes # If ind.qrdeletes >= max_qrdeletes, we are rebuilding the QR decomposition anyway.
            qrdelete!(c.Q_cache, c.R_cache, m)
            ind.qrdeletes += 1
        end
        
        (aa_step || ind.lags == 0) && (ind.lags = min(ind.lags + 1, m))
    end

    status = assess_status(iter_limit - iter, maps_limit - ind.maps, max_time, normsq_g < abstolsq)
    
    return SpeedMappingResult{T, FT}(reshape(c.x, true_shape), sqrt(normsq_g), ind.maps, 
        ind.f_calls, iter, status, :aa, nothing, trace, one(FT))
end
