using LinearAlgebra
#using Base: Int64, Float64
const ∅ = nothing
const ∞ = Inf
# problem: speedmapping([0.0,0.0]; f, g!, upper = [Inf,0.25], buffer = 0.005, maps_limit = 100, ord = 3)
Base.@pure mud(x :: Int, m :: Int) :: Int64 = mod.(x - 1, m) + 1

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
	if ΔᵇΔᵇ ≤ 1e-50; return 1, 0 end
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
	return x_out
end

function too_large!(x₀, ∇, ∇∇, s)
	obj_new = s.f(x₀ - s.α * ∇)
	s.f_calls += 1
	return isnan(obj_new) || obj_new == -∞ || (obj_new > s.obj_now - 0.25s.α * ∇∇)
end

function optim_params!(init_descent :: Float64, ∇, x₀, s)
	∇∇ = ∇ ⋅ ∇
	if ∇∇ == 0; throw(DomainError("∇f(x_in) = 0 (extremum or saddle point)")) end
	if init_descent == 0.0 && s.f ≠ ∅
		is_too_large = too_large!(x₀, ∇, ∇∇, s)
		min_mult = 4.0
		mult = min_mult * 64^2
		for i ∈ 0:30
			s.α *= mult^(1 - 2is_too_large)
			was_too_large = is_too_large
			is_too_large = too_large!(x₀, ∇, ∇∇, s)
			if mult == min_mult && (is_too_large + was_too_large == 1) || s.α > 1_000_000
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

function check_bound!(x_old, x_try, x_out, s)
	if     s.upper == ∞ && s.lower == -∞
		x_out .= x_try
	elseif s.upper ≠  ∞ && s.lower == -∞
		x_out .= min.(x_try, s.ω * s.upper .+ (1 - s.ω) * x_old)
	elseif s.upper == ∞ && s.lower ≠  -∞
		x_out .= max.(x_try, s.ω * s.lower .+ (1 - s.ω) * x_old)
	else
		x_out .= max.(min.(x_try, s.ω * s.upper .+ (1 - s.ω) * x_old), 
		              s.ω * s.lower .+ (1 - s.ω) * x_old)
	end
	return x_out
end

function update_x!(x_in, x_out, ∇, s, x_try)
	if s.optim
		if s.map_calls > 0 # If map_calls == 0, ∇ is already computed, but map_calls must be updated.
			s.autodiff ? ∇ .= s.g(x_in) : s.g!(∇, x_in) 
		end
		if !s.constr 
			x_out .= x_in .- s.α * ∇
		else
			x_try .= x_in .- s.α * ∇
			∇ .= x_in
			check_bound!(x_in, x_try, x_out, s)
			∇ .-= x_out
			∇ ./= s.α
		end
	else
		∇ .= x_in
		s.map!(x_in, x_out)
		∇ .-= x_out
		if s.α < 1; x_out .+= (1 - s.α) * ∇ end 
	end
	s.norm_∇ = norm(∇, s.Lp)
	return ∅
end

function map_acx!(x_in , x_out, ∇, s, x_try, info)
	if s.go_on
		try
			update_x!(x_in, x_out, ∇, s, x_try)
		catch
			if s.k == 1; update_x!(x_in, x_out, ∇, s, x_try) end # To initially catch any type errors, function errors, etc.
			s.go_on = false
		end
		s.map_calls += 1
		something_wrong = isnan(s.norm_∇) || isinf(s.norm_∇) || !s.go_on
		
		s.converged = !something_wrong && s.norm_∇ <= s.tol
		if s.converged && s.f ≠ ∅
			obj_fin = s.f(x_out)
			something_wrong = isnan(obj_fin) || isinf(obj_fin)
			s.converged = !something_wrong
		end
		s.go_on = !s.converged && !something_wrong
		store_info!(info, copy(x_out), s, false)
	end
	return ∅
end

function update_progress!(s, x, check_obj)
	if s.go_on
		if check_obj
			try
				s.obj_now = s.f(x)
			catch
				s.obj_now = NaN
			end
			s.f_calls += 1
			s.go_on = !(isnan(s.obj_now) || isinf(s.obj_now))
		end
		if s.go_on && ((check_obj && s.obj_now < s.obj_best) || (s.norm_∇ < s.norm_best))
			s.ix_best = s.ix
			s.ix_new = mud(s.ix + 1, 2)
			s.ord_best, s.α_best = (s.i_ord, s.α)
			s.σ_mult_fail = s.α_mult = 1.0
			if !s.optim; s.α = 1.0 end
			check_obj ? s.obj_best = s.obj_now : s.norm_best = s.norm_∇
		end
	end
	return ∅
end

function backtrack!(s, check_obj)
	s.ix = s.ix_best
	s.i_ord = s.ord_best
	s.σ_mult_fail /= 2
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

function check_arguments(check_obj, init_descent, autodiff, verb, s, x_in)
	if s.map! ≠ ∅ && s.g! ≠ ∅
		throw(ArgumentError("must not provide both map! and g!")) end
	if autodiff && verb > 0
		@warn "minimizing f using gradient descent and ForwardDiff" end
	if check_obj && s.f === ∅
		throw(ArgumentError("if check_obj == true, f must be provided.")) end
	if s.g! ≠ ∅ && init_descent == 0.0 && s.f === ∅ && verb > 0
		@warn "\U003B1 initialized to 0.01 automatically. For stability, " *
			"provide an objective function or set \U003B1 manually using " * 
			"init_descent." end
	if (maximum(x_in .- s.upper) > 0) || maximum(s.lower .- x_in) > 0
		throw(DomainError("infeasible starting point")) end
	if !(eltype(x_in) <: AbstractFloat)
		throw(ArgumentError("starting point must be of type Float")) end
	return ∅
end

function store_info!(info, x, s, extrapolating)
	if info ≠ ∅
		push!(info.x, x)
		push!(info.σ, s.σ)
		push!(info.α, s.α)
		push!(info.extrapolating, extrapolating)
	end
end

function check_cycle!(σ_new, σs, σs_i, norm_∇s, s, last_order)
	if σ_new > σs[σs_i]
		σs[σs_i] = σ_new
		norm_∇s[σs_i] = s.norm_∇
	end
	if last_order && σs[σs_i] > 0.001 && s.σ_mult_cycle > 0.01
		cycle = false
		for i ∈ eachindex(σs)
			if i ≠ σs_i && abs(σs[i]      - σs[σs_i])      < 1e-8 && 
				            abs(norm_∇s[i] - norm_∇s[σs_i]) < 1e-8
				cycle = true
				s.n_cycles += 1
				s.σ_mult_cycle /= 2
				s.α /= 2
				σs .= 0
				norm_∇s .= 0
				break
			end
		end
		if !cycle; s.σ_mult_cycle = 1.0 end
	end
	return mud(σs_i + last_order, length(σs))
end

Base.@kwdef mutable struct state
	map!; g!; g; f

	ω            :: Float64
	lower; upper
	constr       :: Bool

	autodiff     :: Bool
	optim        :: Bool
	Lp           :: Real
	maps_limit   :: Float64
	time_limit   :: Float64
	t0           :: Float64
 
	go_on        :: Bool = true
	converged    :: Bool = false
	map_calls    :: Int64 = 0 # Or the # of g! calls if optim == true.
	k            :: Int64 = 0
	f_calls      :: Int64 = 0
	σ            :: Float64 = 0
	α            :: Float64 = 1.0
	tol          :: Float64
	obj_now      :: Float64 = ∞
	norm_∇       :: Float64 = ∞

	ix           :: Int64 = 1
	ix_new       :: Int64 = 1
	ix_best      :: Int64 = 1
	ord_best     :: Int64 = 0
	i_ord        :: Int64 = 0
	α_best       :: Float64 = 1.0
	σ_mult_fail  :: Float64 = 1.0
	σ_mult_cycle :: Float64 = 1.0
	α_mult       :: Float64 = 1.0
	norm_best    :: Float64 = ∞
	obj_best     :: Float64 = ∞

	α_boost      :: Float64 = 1.0
	n_boost      :: Int64 = 0
	n_cycles     :: Int64 = 0
end

function speedmapping(
	x_in :: AbstractArray; f = ∅, map! = ∅, g! = ∅, 
	ord :: Int64 = 3, σ_min :: Float64 = 0.0, stabilize :: Bool = false,
	tol :: Float64 = 1e-10, Lp :: Real = 2, maps_limit :: Real = 1e6, 
	time_limit :: Real = 1000,
	lower = -∞, upper = ∞, buffer :: Float64 = 0.01,
	init_descent :: Float64 = 0.0, check_obj :: Bool = false,
	verb :: Int64 = 1, store_info = false)

	autodiff = f ≠ ∅ && map! === ∅ && g! === ∅
	autodiff ? g = x -> ForwardDiff.gradient(f, x) : g = ∅

	s = state(; map!, g!, g, f, 
		ω = 1 - buffer, lower, upper, constr = upper ≠ ∞ || lower ≠ -∞, 
		autodiff, optim = g! ≠ ∅ || autodiff, Lp, maps_limit, time_limit, 
		t0 = time(), tol)

	check_arguments(check_obj, init_descent, autodiff, verb, s, x_in)

	orders = [3,3,2][1+3-ord:3]
	if stabilize; orders = Int.(vec(hcat(ones(ord),orders)')) end
	lo = length(orders)

	x₀ = [copy(x_in), similar(x_in)]
	x = similar(x₀[1])
	∇s = [similar(x₀[1]) for i ∈ 1:maximum(orders)]
	s.constr ? x_try = similar(x₀[1]) : x_try = ∅ # temp storage

	if f ≠ ∅; s.obj_now = s.obj_best = f(x_in) end

	if s.optim
		autodiff ? ∇s[1] .= g(x_in) : g!(∇s[1], x_in) # Will update map_calls later.
		base, ls, hs, = optim_params!(init_descent, ∇s[1], x_in, s) 
	end
	
	n_fails = 0
	
	store_info ? info = (x = [x_in], σ = [0.0], α = [0.0], extrapolating = [false]) : info = ∅

	lσs = 10
	σs = zeros(lσs)
	norm_∇s = zeros(lσs)
	σs_i = 1
	while s.converged == false && s.map_calls ≤ maps_limit && time() - s.t0 ≤ time_limit
		s.k += 1
		s.i_ord += 1
		io = mud(s.i_ord, lo)
		p = orders[io]
		s.go_on = true

		map_acx!(x₀[s.ix], x, ∇s[1], s, x_try, info)
		update_progress!(s, x₀[s.ix], check_obj)
		for i ∈ 2:p
			map_acx!(x , x, ∇s[i], s, x_try, info)
		end
		if p > 1; σ_new, normΔᵇ = compute_σ(∇s[1:p], σ_min) end

		if !s.converged && s.go_on
			if p > 1
				σs_i = check_cycle!(σ_new, σs, σs_i, norm_∇s, s, io == lo)
				boost_α(s, normΔᵇ)
				s.σ = (σ_new - 1) * s.σ_mult_fail * s.σ_mult_cycle + 1
			end

			if !s.constr
				extrapolate!(x₀[s.ix], x₀[s.ix_new], ∇s[1:p], s)
			else
				extrapolate!(x₀[s.ix], x_try, ∇s[1:p], s)
				check_bound!(x₀[s.ix], x_try, x₀[s.ix_new], s)
			end
			store_info!(info, copy(x₀[s.ix_new]), s, true)
			s.ix = s.ix_new

			if s.optim && p > 1; s.α *= base^((σ_new > hs) - (σ_new < ls)) end
	
		elseif !s.converged && !s.go_on
			backtrack!(s, check_obj)
		end
		n_fails += !s.go_on && !s.converged
	end

	if check_obj && s.converged
		if f(x) > s.obj_best; x .= x₀[s.ix_best] end
		s.f_calls += 1
	end
	
	if     verb ≥ 1 && s.converged
		println("Converged. Mappings: $s.map_calls Obj. evals: $s.f_calls")
	elseif verb ≥ 1 && s.map_calls > maps_limit
		println("Maximum mappings exceeded.")
	elseif verb ≥ 1 && time() - s.t0 > time_limit
		println("Exceeded time limit of $time_limit seconds.") 
	end

	return (minimizer = x, map_calls = s.map_calls, f_calls = s.f_calls, 
		converged = s.converged, n_boost = s.n_boost, n_fails = n_fails, info = info)
end