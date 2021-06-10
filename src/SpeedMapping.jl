module SpeedMapping

	using Base: Int64, Float64

	using LinearAlgebra, ForwardDiff
	mud(x, m) = mod.(x .- 1, m) .+ 1
	function compute_σ(∇s, σ_min)
		ΔᵇΔᵇ = ΔᵃΔᵇ = 0.0
		if length(∇s) == 2
			for i ∈ eachindex(∇s[1])
				Δᵇi = ∇s[2][i] - ∇s[1][i]
				ΔᵇΔᵇ += Δᵇi * Δᵇi
				ΔᵃΔᵇ += ∇s[1][i] * Δᵇi
			end
		else
			for i ∈ eachindex(∇s[1])
				Δᵃi = ∇s[2][i] - ∇s[1][i]
				Δᵇi = ∇s[3][i] - 2∇s[2][i] + ∇s[1][i]
				ΔᵇΔᵇ += Δᵇi * Δᵇi
				ΔᵃΔᵇ += Δᵃi * Δᵇi
			end
		end
		if ΔᵇΔᵇ <= 1e-50; return 1, 0 end
		return max(σ_min, abs(ΔᵃΔᵇ) / ΔᵇΔᵇ), sqrt(ΔᵇΔᵇ)
	end

	function extrapolate!(x_in, x_out, ∇s, s)
		σ² = s.σ * s.σ
		if length(∇s) == 1 # If stabilization mapping
			@. x_out = x_in - s.α * ∇s[1]
		elseif length(∇s) == 2
			@. x_out = x_in - (s.α * (2s.σ - σ²)) * ∇s[1] - (s.α * σ²) * ∇s[2]
		else 
			σ³ = σ² * s.σ
			@. x_out = x_in - (s.α * (3s.σ - 3σ² + σ³)) * ∇s[1] - 
				(s.α * (3σ² - 2σ³)) * ∇s[2] - (s.α * σ³) * ∇s[3]
		end
		return nothing
	end

	function too_large!(x₀, ∇, ∇∇, s)
		new_obj_val = s.f(x₀ - s.α * ∇)
		s.n_obj += 1
		return isnan(new_obj_val) || new_obj_val == -Inf || (new_obj_val > s.obj_now - 0.25s.α * ∇∇)
	end

	function optim_params!(init_descent :: Float64, ∇, x₀, s)
		if init_descent == 0.0 && !isnothing(s.f)
			s.α = 1.0
			∇∇ = ∇ ⋅ ∇
			is_too_large = too_large!(x₀, ∇, ∇∇, s)
			min_mult = 4.0
			mult = min_mult*64^2
			for i ∈ 0:30
				s.α *= mult^(1 - 2is_too_large)
				was_too_large = is_too_large
				is_too_large = too_large!(x₀, ∇, ∇∇, s)
				if mult == min_mult && (is_too_large + was_too_large == 1) || s.α > 10
					s.α /= min_mult^(is_too_large && !was_too_large)
					break 
				end
				mult = max(mult / 64^(is_too_large + was_too_large == 1), min_mult)
			end
		elseif init_descent > 0
			s.α = init_descent
		else
			s.α = 0.01
		end
		return (base = 1.5, ls = 1, hs = 2)
	end

	function update_x!(x_in, x_out, ∇, s, x_try)
		if s.optim
			if s.n_maps > 0 # If n_maps == 0, we have already computed ∇s[1], but still need to update n_maps.
				s.autodiff ? ∇ .= s.g(x_in) : s.g!(∇, x_in) 
			end
			if !s.constr 
				x_out .= x_in .- s.α * ∇
			else
				x_try .= x_in .- s.α * ∇
				if s.upper !== nothing; x_try .= min.(x_try, s.ω * s.upper .+ (1 - s.ω) * x_in) end
				if s.lower !== nothing; x_try .= max.(x_try, s.ω * s.lower .+ (1 - s.ω) * x_in) end
				∇ .= -(x_try .- x_in)/s.α
				s.norm_∇ = norm(x_try .- x_in, s.nnorm)/s.α
				x_out .= x_try
			end
		else
			∇ .= x_in
			s.map!(x_in, x_out)
			∇ .-= x_out
			if s.α < 1; x_out .= x_in .- s.α * ∇ end
		end
		if !s.optim || !s.constr; s.norm_∇ = norm(∇, s.nnorm) end
		return nothing
	end

	function map_acx!(x_in , x_out, ∇, s, x_try)
		if s.go_on
			try
				update_x!(x_in, x_out, ∇, s, x_try)
			catch
				if s.k == 1; update_x!(x_in, x_out, ∇, s, x_try) end # To initially catch any type errors, function errors, etc.
				s.go_on = false
			end
			s.n_maps += 1
			something_wrong = isnan(s.norm_∇) || isinf(s.norm_∇) || !s.go_on
			
			s.converged = !something_wrong && s.norm_∇ <= s.tol
			if s.converged && s.f !== nothing
				obj_fin = s.f(x_out)
				something_wrong = isnan(obj_fin) || isinf(obj_fin)
				s.converged = !something_wrong
			end
			s.go_on = !s.converged && !something_wrong
		end
		return nothing
	end

	function update_progress!(s, x, check_obj)
		if s.go_on
			if check_obj
				s.obj_now = s.f(x) # Maybe add a try-catch?
				s.n_obj += 1
				s.go_on = !(isnan(s.obj_now) || isinf(s.obj_now))
				if s.go_on; s.obj_best = min(s.obj_now, s.obj_best) end
			else
				s.norm_best = min(s.norm_best, s.norm_∇)
			end
			if s.go_on && ((check_obj && s.obj_best == s.obj_now) || (s.norm_best == s.norm_∇))
				s.ix_best = s.ix
				s.ix_new = mud(s.ix + 1, 2)
				s.ord_best, s.α_best = (s.i_ord, s.α)
				s.σ_mult = s.α_mult = 1.0
				if !s.optim; s.α = 1.0 end
			end
		end
		return nothing
	end

	function backtrack!(s, check_obj)
		s.ix = s.ix_best
		s.i_ord = s.ord_best
		s.σ_mult /= 10
		s.α_mult /= 2
		s.α = s.α_best * s.α_mult
		if check_obj; s.obj_now = s.obj_best end
	end

	function boost_α(s, normΔᵇ)
		if normΔᵇ < 1e-50 && s.optim
			s.α_boost *= 2
			s.α = min(s.α * s.α_boost, 1.0)
			s.n_boost += 1
		end
	end

	function check_arguments(f, map!, g!, nnorm, check_obj, init_descent, autodiff, verb)
		if map! !== nothing && g! !== nothing
			throw(ArgumentError("must not provide both map! and g!")) end
		if autodiff && verb > 0
			@warn "minimizing f using gradient descent and ForwardDiff" end
		if nnorm != 2 && nnorm != Inf
			throw(ArgumentError("nnorm must be either 2 or Inf.")) end
		if check_obj && f === nothing
			throw(ArgumentError("if check_obj == true, f must be provided.")) end
		if g! !== nothing && init_descent == 0.0 && f === nothing && verb > 0
			@warn "Warning, \U003B1 initialized to 0.01 automatically. For stability," *
				" provide an objective function or set \U003B1 manually using init_descent."
		end
		return nothing
	end

	mutable struct speedmapping_state
		map!
		g!
		g
		f

		ω          :: Float64
		lower
		upper
		constr     :: Bool

		autodiff   :: Bool
		optim      :: Bool
		nnorm      :: Float64
		max_maps   :: Float64
		time_limit :: Float64
		start_time :: Float64

		go_on      :: Bool
		converged  :: Bool
		n_maps     :: Int64
		k          :: Int64
		n_obj      :: Int64
		σ          :: Float64
		α          :: Float64
		tol        :: Float64
		obj_now    :: Float64
		norm_∇     :: Float64

		ix         :: Int64
		ix_new     :: Int64
		ix_best    :: Int64
		ord_best   :: Int64
		i_ord      :: Int64
		α_best     :: Float64
		σ_mult     :: Float64
		α_mult     :: Float64
		norm_best  :: Float64
		obj_best   :: Float64

		α_boost    :: Float64
		n_boost    :: Int64
	end

	function speedmapping(
		x_in; f = nothing, map! = nothing, g! = nothing, 
		ord :: Int64 = 3, σ_min :: Float64 = 0.0, stabilize :: Bool = false,
		tol :: Float64 = 1e-10, nnorm :: Float64 = 2.0, max_maps :: Real = 1_000_000, 
		time_limit :: Real = 1000.0,
		lower = nothing, upper = nothing, buffer :: Float64 = 0.01,
		init_descent :: Float64 = 0.0, check_obj :: Bool = false,
		verb :: Int64 = 1, store_info = nothing)

		autodiff = f !== nothing && map! === nothing && g! === nothing
		autodiff ? g = x -> ForwardDiff.gradient(f, x) : g = nothing

		check_arguments(f, map!, g!, nnorm, check_obj, init_descent, autodiff, verb)
			
		x₀ = [copy(x_in), similar(x_in)]
		x = similar(x₀[1])

		orders = [2,3,3][1:ord]
		if stabilize; orders = Int.(vec(hcat(ones(ord),orders)')) end

		∇s = [similar(x₀[1]) for i ∈ 1:maximum(orders)]

		x_try = similar(x₀[1])

		s = speedmapping_state(
			map!, g!, g, f, 
			1 - buffer, lower, upper, upper !== nothing || lower !== nothing,
			autodiff, g! !== nothing || autodiff, nnorm, max_maps, time_limit, time(), 
			true, false, 0, 0, 0, 0, 1.0, tol, Inf, 0,
			1, 1, 1, 0, 0, 1.0, 1.0, 1.0, Inf, Inf, 
			1, 0)

		if f !== nothing 
			s.obj_now = s.obj_best = f(x_in) 
		end

		if s.optim
			autodiff ? ∇s[1] .= g(x_in) : g!(∇s[1], x_in) # Don't worry, we will update n_maps later.
			base, ls, hs, = optim_params!(init_descent, ∇s[1], x_in, s) 
		end
		
		n_fails = 0
		
		while s.converged == false && s.n_maps < max_maps && time() - s.start_time < time_limit
			s.k += 1
			s.i_ord += 1
			p = orders[mud(s.i_ord, length(orders))]
			s.go_on = true

			map_acx!(x₀[s.ix], x, ∇s[1], s, x_try)
			update_progress!(s, x₀[s.ix], check_obj)
			for i ∈ 2:p
				map_acx!(x , x, ∇s[i], s, x_try)
			end

			if p > 1; σ_new, normΔᵇ = compute_σ(∇s[1:p], σ_min) end

			if s.k == 1 && p == 2 && length(orders) > 1 && σ_new > 1
				s.i_ord += 1
				p = 3
				map_acx!(x , x, ∇s[p], s, x_try)
				σ_new, normΔᵇ = compute_σ(∇s, σ_min)
			end

			n_fails += !s.go_on && !s.converged

			if !s.converged && s.go_on
				if p > 1
					boost_α(s, normΔᵇ)
					s.σ = σ_new * s.σ_mult + (1 - s.σ_mult)
				end

				if !s.constr
					extrapolate!(x₀[s.ix], x₀[s.ix_new], ∇s[1:p], s)
				else
					extrapolate!(x₀[s.ix], x_try, ∇s[1:p], s)
					if upper !== nothing; x₀[s.ix_new] .= min.(x_try, s.ω * upper .+ (1 - s.ω) * x₀[s.ix]) end # Essayer in place
					if lower !== nothing; x₀[s.ix_new] .= max.(x_try, s.ω * lower .+ (1 - s.ω) * x₀[s.ix]) end
				end
				s.ix = s.ix_new

				if s.optim && p > 1; s.α *= base^((σ_new > hs) - (σ_new < ls)) end

				if store_info !== nothing; store_info(x, x₀[s.ix], p, σ) end

			elseif !s.converged && !s.go_on
				backtrack!(s, check_obj)
			end
		end

		if check_obj && s.converged
			if f(x) > s.obj_best; x .= x₀[s.ix_best] end
			s.n_obj += 1
		end
		
		if     verb >= 1 && s.converged
			println("Converged. Mappings: $s.n_maps Obj. evals: $s.n_obj")
		elseif verb >= 1 && s.n_maps > max_maps
			println("Maximum mappings exceeded.")
		elseif verb >= 1 && time() - s.start_time < time_limit
			println("Exceeded time limit of $time_limit seconds.") 
		end

		return (solution = x, n_maps = s.n_maps, n_obj = s.n_obj, converged = s.converged, 
				n_boost = s.n_boost, n_fails = n_fails)
	end
end
