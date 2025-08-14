function box_constraints_and_update_resid_low!(
    extr, x_try::T, r::T, err::T, x_old::T, lower::T, params_F, ip, α
) where {T}

    b = params_F.buffer
    bc = 1 - b
    if ip
        @inbounds @simd for i in eachindex(x_try)
            l = x_old[i] * b + lower[i] * bc
            if x_try[i] <= l
                x_try[i] = l
                r[i] = (l - x_old[i]) / α
                x_old[i] - lower[i] < params_F.abstol && (err[i] = (lower[i] - x_old[i]) / α) # fix this when it is g (not a mapping)
            end
        end

        return x_try, r, err
    else
        l = x_old * b .+ lower * bc
        x_try = max.(x_try, l)
        r = (x_try .- x_old) ./ α
        too_close = (x_try .<= l) .&& (x_old .- lower .< params_F.abstol)
        err = too_close .* (lower .- x_old) ./ α .+ .!(too_close) .* err
        return x_try, r, err
    end
end

function box_constraints_and_update_resid_high!(
    extr, x_try::T, r::T, err::T, x_old::T, upper::T, params_F, ip, α
) where {T}
    b = params_F.buffer
    bc = 1 - b
    if ip
        @inbounds @simd for i in eachindex(x_try)
            u = x_old[i] * b + upper[i] * bc
            if x_try[i] >= u
                x_try[i] = u
                r[i] = (u - x_old[i]) / α
                upper[i] - x_old[i] < params_F.abstol && (err[i] = (upper[i] - x_old[i]) / α)
            end
        end
        return x_try, r, err
    else
        u = x_old * b .+ upper * bc
        x_try = min.(x_try, u)
        r = (x_try .- x_old) ./ α
        too_close = (x_try .>= u) .&& (upper .- x_old .< params_F.abstol)
        err = too_close .* (upper .- x_old) ./ α .+ .!(too_close) .* err
        return x_try, r, err
    end
end

@inline function do_after_map!(
    x_out::T, r::T, err::T, α::FT, x_in::T, bounds, params_F, is_map, ip, store_trace, trace, i, pnorm) where {T, FT}

    is_map ? (@bb @. r = x_out - x_in) : (@bb @. x_out = x_in + α * r)
    store_trace && push!(trace, AcxState{T, FT}(copy(x_in), α, lpnorm(r, pnorm), i))

    @bb @. err = r
    bounds.l ≠ nothing && ((x_out, r, err) = box_constraints_and_update_resid_low!(max, x_out, r, err, x_in, bounds.l, params_F, ip, α))
    bounds.u ≠ nothing && ((x_out, r, err) = box_constraints_and_update_resid_high!(min, x_out, r, err, x_in, bounds.u, params_F, ip, α))

    return x_out, r, cdot(err, FT)
end

function acx(
        c :: Union{AcxCache, Nothing}, f :: FN, g! :: FN, m! :: FN, g :: FN, m :: FN, x_in :: T, 
        orders, params_F, params_I, bounds, max_time, store_trace, initialize_learning_rate
    ) where {T}

    FT = real(eltype(T))
    abstol, pnorm, buffer, reltol_resid_grow, abstol_obj_grow, initial_learning_rate = params_F
    abstolsq = abstol^2

    ip = (m! !== nothing || g! !== nothing) # In place
    
    if ip
        (x_now, x_next, x_best, r1, r2, r3) = (c.x_now, c.x_next, c.x_best, c.rs[1], c.rs[2], c.rs[3])
        x_now .= x_in
    else
        x_now = x_in
    end

    bounds.l ≠ nothing && (@bb @. x_now = max(bounds.l, x_now))
    bounds.u ≠ nothing && (@bb @. x_now = min(bounds.u, x_now))
    
    !ip && (x_next = x_best = r1 = r2 = r3 = r_now = x_now)
    
    reltol_rr = min(reltol_resid_grow^2, prevfloat(typemax(FT)))
    
    is_map = (m! !== nothing || m !== nothing)

    maps = f_calls = iter = i_ord = 0

    iter_limit, maps_limit = params_I
    has_cons = bounds.l ≠ nothing || bounds.u ≠ nothing

    trace = store_trace ? AcxState{T, FT}[] : nothing

    rr_now = rr_best = norm_best = lpnormsq_best = typemax(FT) # Just an initialization, the value does not matter
    α = σ = one(FT)
    best_i = 1
    #=
    While ACX tolerates a wide range of starting descent step sizes (α), difficult problems benefit 
    from a good initial α. 

    - α should be small enough that the norm of the gradient does not increase too fast: 
        |∇f(x_next)| / |∇f(x_now)| < 4. 
    - α should be large enough that the absolute value of the step size σ for an order-2 
        extrapolation should be small enough: 
        |(∇f(x_next) - ∇f(x_now))⋅ ∇f(x_now)| / |∇f(x_next) - ∇f(x_now)|² < 100 
    - If f is provided, the Armijo condition is imposed. So the objective should decline, but the 
        next iterate should have value f(x_next) > f(x_now) - 0.8⋅|∇f(x_now)|².
    =#

    err = ip ? similar(x_in) : x_in # Put in parameters
    if !is_map && initialize_learning_rate # Initializing α
        check_obj = f !== nothing 
        if check_obj
            obj_in = obj_now = f(x_now)
            f_calls += 1
            max_new_obj_val = obj_now + abstol_obj_grow
        end
        ip ? g!(r1, x_now) : (r1 = g(x_now))
        maps += 1
        has_cons && (@bb @. r3 = r1) # To avoid recomputing r1 since it may be changed by box constraints if α is changed 
        log_α = log(initial_learning_rate)
        low, up = FT(-Inf), FT(Inf)
        tries = 0
        while tries < 100 && up - low > FT(0.001)
            α = -exp(log_α) # If we do minimization, gradient descent requires a negative α.
            has_cons && (@bb @. r1 = r3)
            x_next, r1, rr_best = do_after_map!(x_next, r1, err, α, x_now, bounds, params_F, is_map, 
                ip, store_trace && tries == 0, trace, 1, pnorm)
            isbad(rr_best) && throw(ArgumentError("Initial residual has NaN or Inf gradient"))
            lpnormsq_best = pnorm == 2 ? rr_best : lpnorm(r1, pnorm)^2
            lpnormsq_best < abstolsq && return SpeedMappingResult{T, FT}(x_now, sqrt(lpnormsq_best), 1, 0, 0, :first_order, :acx, trace, nothing, -α)
            if check_obj
                obj_now = f(x_next)
                f_calls += 1
                if !(obj_now < max_new_obj_val)
                    up = log_α
                    log_α = low == -Inf ? log_α - FT(5) : (log_α + low) * FT(0.5)
                elseif obj_now < obj_in + 0.8α * rr_best
                    low = log_α
                    log_α = up == Inf ? log_α + FT(5) : (log_α + up) * FT(0.5)
                else
                    check_obj = false
                end
            end
            
            if !check_obj # residual norm increases? / σ too large?
                ip ? g!(r2, x_next) : (r2 = g(x_next))
                x_best, r2, rr_now = do_after_map!(x_best, r2, err, α, x_next, bounds, params_F, 
                    is_map, ip, false, trace, 2, pnorm)
                maps += 1
                @bb @. x_best = r2 - r1 # Here x_best is just used as temp storage
                if isbad(rr_now) || rr_now > rr_best * FT(4)
                    up = log_α
                    log_α = low == -Inf ? log_α - FT(5) : (log_α + low) * FT(0.5)
                elseif !(cdot(x_best, r1, FT) / cdot(x_best, FT) < FT(100))
                    low = log_α
                    log_α = up == Inf ? log_α + FT(5) : (log_α + up) * FT(0.5)
                else
                    break
                end
            end
            tries += 1
        end
        if rr_best < rr_now
            @bb @. x_best = x_now
        else
            rr_best = rr_now
            best_i = 2 # best_i stores the index for the best residual: rs[best_i]
            @bb @. x_best = x_next
        end
        x_now, r2, rr_now = do_after_map!(x_now, r2, err, α, x_next, bounds, params_F, is_map, ip, 
            store_trace, trace, 2, pnorm)
        start_map = 3
    elseif !is_map
        α = -initial_learning_rate
    end

    start_map = !is_map && initialize_learning_rate ? 3 : 1 # Reusing the mappings already computed

    while maps < maps_limit && (max_time == Inf || time() < max_time) && iter < iter_limit
        iter += 1
        i_ord = i_ord % length(orders) + 1
        p = orders[i_ord]
        lpnormsq_best = pnorm == 2 ? rr_best : lpnorm(r1,pnorm)^2

        i = start_map # Start_map > 1 allows to avoid recomputing maps when we know they have already been computed.
        good_maps = true
        while i <= p
            ip && (r_now = i == 2 ? r2 : i == 3 ? r3 : r1)
            is_map ? ip ? m!(x_next, x_now) : (x_next = m(x_now)) : 
                     ip ? g!(r_now , x_now) : (r_now  = g(x_now))
            x_next, r_now, rr_now = do_after_map!(x_next, r_now, err, α, x_now, bounds, params_F, is_map, ip, store_trace, trace, i, pnorm)
            
            maps += 1
            if !is_map && !(rr_now < rr_best * reltol_rr)
                good_maps = false
                break
            elseif rr_now <= rr_best # Just a criterion used during the mapping. After the extrapolation, the new x automatically x_best.
                rr_best = rr_now
                lpnormsq_best = pnorm == 2 ? rr_best : lpnorm(r_now, pnorm)^2
                best_i = i
                x_best, x_now = x_now, x_best
                lpnormsq_best < abstolsq && break
            end
            x_now, x_next = x_next, x_now
            !ip && (i == 2 ? (r2 = r_now) : i == 3 ? (r3 = r_now) : (r1 = r_now))
            store_trace && push!(trace, AcxState{T,FT}(copy(x_now), α, sqrt(lpnormsq_best),i))
            i += 1
        end
        start_map = 2 # After the 1st iteration, we always do the 1st map after the extrapolation to perform checks and the following ones in the main loop.
        lpnormsq_best < abstolsq && break

        if good_maps # Computing σ
            p == 3 && (@bb @. r3 += FT(-2) * r2 + r1)
            @bb @. r2 -= r1
            σ = -abs(p == 2 ? accurate_cdot(r2, r1, FT) / accurate_cdot(r2, FT) : 
                accurate_cdot(r3, r2, FT) / accurate_cdot(r3, FT))
        end

        if isbad(σ) || !good_maps
            α *= good_maps ? FT(5) : FT(0.2)
            σ = FT(1.0)

            # We could avoid recomputing this if r1 is stored, or if best_i == 1 and there was no modification of r1 with box constraints. We could think about this...
            is_map ? ip ? m!(x_now, x_best) : (x_now = m(x_best)) : 
                        ip ? g!(r1, x_best) : (r1  = g(x_best))
            maps += 1
            x_now, r1, rr_best = do_after_map!(x_now, r1, err, α, x_best, bounds, params_F, is_map, ip, store_trace, trace, 1, pnorm)
        else

            # Extrapolating (from x_now)
            @bb @. x_now += p == 2 ? (FT(-2) * α * (FT(1) + σ)) * r1 + (α *(FT(-1) + σ^2)) * r2 : # Squared extrapolation
                (FT(-3) * α * (FT(1) + σ)) * r1 + (FT(-3) * α * (FT(1) - σ^2)) * r2 + (-α * (FT(1) + σ^3)) * r3  # Cubic extrapolation
            x_now = box_constraint!(x_now, x_best, bounds, buffer, ip)

            # Updating α
            if !is_map
                α *= abs(σ) < FT(1) ? FT(2/3) : 
                     abs(σ) < FT(2) ? FT(1.01)  :  # This is to make (almost) sure we don't get stuck for too long at the same position in an infinite loop. But verify it doesn't cause problems somewhat...
                                      FT(3/2)
            end

            damp_step = FT(1)
            for attempt in 1:100

                is_map ? ip ? m!(x_next, x_now) : (x_next = m(x_now)) :
                         ip ? g!(r1, x_now)     : (r1     = g(x_now))
                x_next, r1, rr_now = do_after_map!(x_next, r1, err, α, x_now, bounds, params_F, is_map, ip, false, trace, 1, pnorm)
                maps += 1
                (rr_now < rr_best * reltol_rr) && break
                damp_step *= FT(1/2)
                @bb @. x_now = x_best + damp_step * (x_now - x_best)
            end
            rr_best = rr_now
            (x_best, x_now, x_next) = (x_now, x_next, x_best) # x_best becomes x_now and x_now becomes x_next
            store_trace && push!(trace, AcxState{T,FT}(copy(x_now),σ,lpnorm(r1,pnorm),1))
        end
        best_i = 1
    end

    status = assess_status(iter_limit - iter, maps_limit - maps, max_time, lpnormsq_best < abstolsq)

    return SpeedMappingResult{T, FT}(x_best, FT(sqrt(lpnormsq_best)), maps, f_calls, iter, status, 
        :acx, trace, nothing, -α)
end