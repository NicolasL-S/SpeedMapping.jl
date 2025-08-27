using Format, Printf, CairoMakie, BenchmarkTools, LaTeXStrings

_isbad(x) = ismissing(x) || isnan(x) || isinf(x)
function one_problem_one_solver(wrapper, problem, abstol, compute_norm, time_limit, Fevals_limit, proper_benchmark)
	do_benchmark = Ref(true) # To easily interrupt the benchmark if something went wrong
	time_last_execution = Ref(0.)
	log = Ref("")
	Fevals = Ref(0)
	sol = similar(problem.x0)
	if proper_benchmark
		bench = @benchmark if $do_benchmark[]
			time_start = time()
			out_sol, out_Fevals, out_log = $wrapper($problem, $abstol, time_start, $Fevals_limit, $time_limit)
			$sol .= out_sol
			$Fevals[] = out_Fevals
			$log[] = out_log 
			$time_last_execution[] = time() - time_start
			$do_benchmark[] = $time_last_execution[] < $time_limit && $Fevals[] <= $Fevals_limit && 
				sum(_isbad.($sol)) == 0
		end
		_time = do_benchmark[] ? median(bench.times)/1e9 : time_last_execution[]
	else
		time_start = time()
		sol[:], Fevals[], log[] = wrapper(problem, abstol, time_start, Fevals_limit, time_limit)
		time_last_execution[] = time() - time_start
		_time = time_last_execution[]
	end
	time_last_execution[] > time_limit && (log[] = "Timed out")
	Fevals[] > Fevals_limit && (log[] = "Fevals > limit")
	out_norm = compute_norm(problem, sol) # compute_norm deals with the fact that i) some problems provide map!, some provide residuals r! or some provide gradients g!, and 2) we may want to compute 2-norms or Inf-norms
	residual_norm = out_norm[1]
	log_norm = length(out_norm) > 1 ? " " * out_norm[2] : "" # If I want to output a comment when computing the norm
	optional = hasproperty(problem, :obj) && problem.obj !== nothing ? (obj = problem.obj(sol),) : ()
	return Fevals[], _time, residual_norm ≤ abstol, residual_norm, log[]*log_norm, optional
end

padleft(in, L) = " "^max(0,(L - length(string(in)))) * first(string(in),L)
padright(in, L) = first(string(in),L) * " "^max(0,(L - length(string(in))))
function one_problem_many_solvers!(problems, pr_name, time_limit, abstol, 
		pr, F_name, Lname, solver_names, fixed_point_solvers, compute_norm, 
		proper_benchmark, tunits, results, draw, gen_Feval_limit
	)

	problem = isa(problems[pr_name], NamedTuple) ? problems[pr_name] : problems[pr_name]()

	Feval_limit = gen_Feval_limit(problem, time_limit[pr]) # To determine Feval limit based on how long a function takes to compute

	# Text output
	out_str = "\n" * pr_name * ": $(length(problem.x0)) parameters, abstol = $(abstol[pr]).\n" 
	sp = (10, 16, 6, 10, 32, 25)
	titlevec = ((F_name, sp[1]), ("time", sp[2]), ("conv", sp[3]), ("|resid|", sp[4]), ("log", sp[5]))
	out_str *= padright("Solver", Lname) * prod([padleft(item...) for item in titlevec])
	hasproperty(problem, :obj) && problem.obj !== nothing && (out_str *= padleft("obj", sp[6]))
	out_str *= "\n"

	for i in eachindex(solver_names)
		name = solver_names[i]
		wrapper = fixed_point_solvers[name]

		Feval, time, converged, res, rlog, optional = one_problem_one_solver(wrapper, 
			problem, abstol[pr], compute_norm, time_limit[pr], Feval_limit, proper_benchmark)
		converged && (results[pr, draw][name] = (Float64(Feval), time))

		# Text output
		t_str = cfmt("%10.2f ", time * (10^(3tunits[pr] - 3))) * ("s", "ms", "μs")[tunits[pr]]
		res_str = @sprintf "%.3e" res
		resvec = ((Feval, sp[1]), (t_str, sp[2]), (converged, sp[3]), (res_str, sp[4]), (first(rlog,sp[5] - 1), sp[5]))
		out_str *= padright(name, Lname) * prod([padleft(res...) for res in resvec])
		hasproperty(optional, :obj) && (out_str *= " "*padleft(optional.obj, sp[6] - 1))
		out_str *= "\n"
	end
	print(out_str) # All text output at once in case we are using Threads.@threads
end

"""
Benchmarks problems × solvers.

many_problems_many_solvers(problems, fixed_point_solvers, problem_names, solver_names, compute_norm; kwargs) :: results

where: 
- `problems :: Dictionary` is a dictionary of problems (can be tuples (e.g. (x0, obj, grad!)) or functions to generate problems)
- `fixed_point_solvers :: Dictionary` is a dictionary of solvers
- `problem_names :: Array{String}` contains the (ordered) names of the problems to benchmark
- `solver_names :: Array{String}` contains the (ordered) names of the solvers to benchmark
- `compute_norm :: Function` `compute_norm(problem, x)` is a function taking a problem and a point `x` and returning the norm at `x`. It is used to assess whether a problem has converged. It is necessary to specify this function because problems may have specific functions to compute residuals (`m!(xout,xin)`, `r!(resid, x)`, `g!(gradient, x)`. It also allows flexibility may compute the Euclidean norm, the infinity norm, etc. It may also optionally return a string as second argument as diagnostic.

Keyword arguments:
- `abstol = 1e-7`
- `time_limit = 10.`
- `tunits = 1` is the unit of time used to output computation times. Possible values are 1, 2, 3, corresponding to `s`, `ms`, and `μs`. `tunits` can be a scalar or a vector with same length as problem_names (to specify a different unit for each problem). 
- `F_name = "F evals"` specifies how to name gradient/maps or r evaluations in the output.
- `gen_Feval_limit :: Function = (problem, time_limit) -> 100000`. `gen_Feval_limit` is a function which, based on the specific problem and the desired maximum time to compute it, can compute an approximate number of maximum F evaluations.
- `results :: [Dict{String, Tuple{Float64, Float64}}] = nothing` A vector to save the benchmarks. Since these benchmarks may take hours, preallocating results may allow not to lose previous computation in case the program is interrupted. To combine with the keyword argument `problem_start`.
- `problem_indices = eachindex(problem_names)` Indices of a subset of `solver_names` to solve.
- `proper_benchmark = true` Whether to solve each problem × solver just once or to use BenchmarkTools to benchmark times accurately.
- `draws :: Integer = 1` For randomly generated problems, we may want to do many draws for each problem.
- `multithreaded = false` When `draws` > 1, it may make sense to use multithreaded = true. In that case, since compte times are less reliable, maybe set `proper_benchmark = false`.

Outputs a vector or matrix of results (n problems × n draws), which are Dictionary of "solver name" -> (nb of F evaluations, time) (Solvers that did not converge are not stored.)
"""
function many_problems_many_solvers(problems, fixed_point_solvers, problem_names, 
		solver_names, compute_norm; abstol = 1e-7, time_limit = 10., tunits = 1, 
		F_name = "F evals", gen_Feval_limit = (problem, time_limit) -> 100000, draws = 1, 
		proper_benchmark = true, results = nothing, problem_indices = eachindex(problem_names), multithreaded = false
	)
	length(tunits) == 1 && (tunits = tunits * ones(length(problem_names)))
	length(abstol) == 1 && (abstol = abstol * ones(length(problem_names)))
	length(time_limit) == 1 && (time_limit = time_limit * ones(length(problem_names)))

	Lname = maximum(length, solver_names) + 1 # Numb of character to display names
	if results == nothing
		results = [Dict{String, Tuple{Float64, Float64}}() for i in eachindex(problems), j in 1:draws]
	end
	
	for pr in problem_indices
		pr_name = problem_names[pr]
		if !multithreaded
			for draw = 1:draws # If the functions generate random data eash draw
				one_problem_many_solvers!(problems, pr_name, time_limit, 
					abstol, pr, F_name, Lname, solver_names, fixed_point_solvers, 
					compute_norm, proper_benchmark, tunits, results, draw, gen_Feval_limit)
			end
		else
			Threads.@threads for draw = 1:draws
				one_problem_many_solvers!(problems, pr_name, time_limit, 
					abstol, pr, F_name, Lname, solver_names, fixed_point_solvers, 
					compute_norm, proper_benchmark, tunits, results, draw, gen_Feval_limit)
			end
		end
	end
	return results
end

# Few problems, we can show all of them
function plot_res(results, problem_names, solver_names, title, path; size = (800, 500), 
	legend_rowgap = -5, xticklabelrotation = pi/3)

	# Assigning indices to solvers
	solver_ind = Dict{String, Int64}()
	for (i, name) in enumerate(solver_names)
		solver_ind[name] = i
	end

	# Displaying deviation from the fastest solver
	# Determining the order of solvers: showing first solvers that converged more often (and in less time to break ties)
	conv_solvers = zeros(Int64, length(solver_names)) # To determine the plotting order
	times_solvers = zeros(Float64, length(solver_names)) # To determine the plotting order
	results_dev = [Dict{String, Tuple{Float64, Float64}}() for i in eachindex(problem_names)]
	for pr in eachindex(problem_names)
		min_Feval = Inf
		min_time = Inf
		for (name, (Feval, time)) in results[pr]
			if name ∈ solver_names
				min_Feval = min(min_Feval, Feval)
				min_time = min(min_time, time)
			end
		end
		for (name, (Feval, time)) in results[pr]
			if name ∈ solver_names
				results_dev[pr][name] = (Feval/min_Feval, time/min_time)
				conv_solvers[solver_ind[name]] += 1
				times_solvers[solver_ind[name]] += time/min_time
			end
		end
	end
	times_solvers ./= max.(conv_solvers,1) # Average times
	order = sortperm(conv_solvers .- 0.9 * times_solvers ./ maximum(times_solvers))

	markers = [:circle, :rect, :diamond, :hexagon, :cross, :xcross, :utriangle, :dtriangle, 
		:ltriangle, :rtriangle, :pentagon, :star4, :star5, :star6, :star8, :vline, :hline, 
		'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
		'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
	stext, mtext, ltext = (9,11,13)
	
	x_vec = Vector{Int64}[]
	Feval_vec = Vector{Float64}[]
	time_vec = Vector{Float64}[]
	iorder = invperm(order)
	min_map = Inf
	max_map = 0.
	for pr in eachindex(results_dev)
		x_pr = Vector{Int64}(undef, length(results_dev[pr]))
		Feval_pr = Vector{Float64}(undef, length(results_dev[pr]))
		time_pr = Vector{Float64}(undef, length(results_dev[pr]))
		for (j, (name, (Feval, time))) in enumerate(results_dev[pr])
			x_pr[j] = iorder[solver_ind[name]]
			Feval_pr[j] = Feval
			min_map = min(Feval, min_map)
			max_map = max(Feval, max_map)
			time_pr[j] = time
		end
		push!(x_vec, x_pr)
		push!(Feval_vec, Feval_pr)
		push!(time_vec, time_pr)
	end

	# Makie backend

	f = Figure(size = size)

	ax = Axis(f[1:2, 1], yticks = (1:length(solver_names), 
		solver_names[order] .*" (".*string.(conv_solvers[order]).*"/".*string(length(problem_names)).*")"), 
		titlesize = ltext, yticklabelsize = stext, 
		xticklabelsize = stext, spinewidth = 0.5, ytickwidth = 0.5, xtickwidth = 0.5, 
		xlabel = "Time relative to the fastest", xlabelsize = stext, xscale = log10, 
		title = title, titlealign = :left, valign = :top,
		ylabel = "Solver (Fraction of problems that converged in time)", ylabelsize = stext)

	plots = Scatter{Tuple{Vector{Point{2, Float64}}}}[]

	for i in eachindex(results)
		push!(plots, scatter!(time_vec[i], x_vec[i], color = Feval_vec[i], colormap = :matter, colorscale = log10, 
		marker = markers[(i-1) % length(markers) + 1]))
	end
	
	Colorbar(f[1, 2], vertical = false, limits = (min_map, max_map), colormap = :matter, 
		scale = log10, flipaxis = false, ticklabelsize = stext, height = 5, width = 150, 
		tickwidth = 0.5, spinewidth = 0.5, label = "# of Feval rel. to the best", labelsize = stext,
		halign = :left)
	
	Legend(f[2, 2], plots, problem_names, labelsize = stext, framevisible = false, 
		"Problems (# parameters)", titlehalign = :left, titlesize = mtext, rowgap = legend_rowgap, 
		valign = :top)
	save(path, f, pt_per_unit = 1.5)
end

# Too many problems, using Dolan-Moré style performance benchmarks
function perf_profiles(results, title, path, solver_names; sizef = (640, 480), stat_num = 2, max_fact = 8)
	# Assigning indices to solvers
	solver_ind = Dict{String, Int64}()
	names_sorted = Vector{String}(undef,length(solver_names))
	for (i, name) in enumerate(solver_names)
		solver_ind[name] = i
		names_sorted[i] = name
	end
	stat_rel_all = Vector{Float64}[]
	nsolvers = length(solver_names)
	for r in results
		if length(r) > 0
			stat_rels = Inf*ones(nsolvers)
			min_stat = Inf
			for (_, stats) in r
				min_stat = min(min_stat, stats[stat_num])
			end
			for (name, stats) in r
				stat_rels[solver_ind[name]] = stats[stat_num]/min_stat
			end
			push!(stat_rel_all, stat_rels)
		end
	end
	stats_rel_allm = reduce(hcat, stat_rel_all)'

	n_ticks = 100
	fracts = range(1,max_fact,n_ticks)
	facts = zeros(n_ticks,nsolvers)
	for i in eachindex(fracts)
		facts[i,:] .= sum(x -> x <= fracts[i],stats_rel_allm, dims = 1)[:]
	end
	facts ./= size(stats_rel_allm,1)

	f = Figure(size = sizef)

	stext, mtext, ltext = (14,17,20)

	ax = Axis(f[1, 1], xticks = 1:max_fact, yticks = 0:0.1:1.1, title = title, xlabel = L"\pi", 
		ylabel = L"P(t_{algo_{i}}/t_{best}\leq\pi)", yticklabelsize = stext, ylabelsize = mtext,
		xticklabelsize = stext, xlabelsize = mtext)
	ylims!(ax, (-0.01, 1.01))
	ls = [lines!(collect(fracts), facts[:,i]) for i in axes(facts,2)]
	axislegend(ax, ls, names_sorted, valign = :bottom)
	save(path, f, pt_per_unit = 1)
	return nothing
end
