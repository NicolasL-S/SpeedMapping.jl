# Dire: The goal is to see how each algorithm performs on its own, but some speed gain may be obtained
# by tayloring the ad to each specific problem like in ... Interestingly, the NonlinearSolve, Default PolyAlg. did
# solve all 23 problems.

using BenchmarkTools, Optim, JLD2, FileIO, SpeedMapping, ArtificialLandscapes

path_out = ""

include("Benchmarking_utils.jl")

# Solver wrappers
optim_solvers = Dict{AbstractString, Function}()
optim_solvers_lower = Dict{AbstractString, Function}() # With lower bound

function Speedmapping_optim_wrapper(problem, abstol, maps_limit, time_limit, add_lower)
	x0, obj, grad! = problem
	if add_lower
		lower = x0 .- 1
		lower[1:(length(lower) ÷ 2)] .= -Inf
	else
		lower = nothing
	end
	res = speedmapping(x0; g! = grad!, f = obj, maps_limit, abstol, pnorm = Inf, time_limit, lower)
	return res.minimizer, res.maps, string(res.status)
end
optim_solvers["Speedmapping, acx"] = (problem, abstol, start_time, maps_limit, time_limit) -> 
	Speedmapping_optim_wrapper(problem, abstol, maps_limit, time_limit, false)

optim_solvers_lower["Speedmapping, acx"] = (problem, abstol, start_time, maps_limit, time_limit) -> 
	Speedmapping_optim_wrapper(problem, abstol, maps_limit, time_limit, true)

function Optim_wrapper(problem, abstol, maps_limit, time_limit, algo)
	x0, obj, grad! = problem
	res = optimize(obj, grad!, x0, algo, Optim.Options(x_abstol = NaN, x_reltol = NaN, 
		f_abstol = NaN, f_reltol = NaN, g_abstol = abstol, g_calls_limit = maps_limit, 
		time_limit = time_limit, iterations = 100_000_000))
	return res.minimizer, res.g_calls, string(res.termination_code)
end
								
optim_solvers["Optim, LBFGS"] = (problem, abstol, start_time, maps_limit, time_limit) -> 
	Optim_wrapper(problem, abstol, maps_limit, time_limit, LBFGS())

optim_solvers["Optim, ConjugateGradient"] = (problem, abstol, start_time, maps_limit, time_limit) -> 
	Optim_wrapper(problem, abstol, maps_limit, time_limit, ConjugateGradient())

function Optim_wrapper_lower(problem, abstol, maps_limit, time_limit, algo)
	x0, obj, grad! = problem
	upper = Inf*ones(length(x0))
	lower = x0 .- 1
	lower[1:(length(lower) ÷ 2)] .= -Inf
	res = optimize(obj, grad!, lower, upper, x0, Fminbox(algo), Optim.Options(x_abstol = NaN, 
		x_reltol = NaN, f_abstol = NaN, f_reltol = NaN, g_abstol = abstol, 
		g_calls_limit = maps_limit, time_limit = time_limit, iterations = 100_000_000))
	return res.minimizer, res.g_calls, string(res.termination_code)
end

optim_solvers_lower["Optim, LBFGS"] = (problem, abstol, start_time, maps_limit, time_limit) -> 
	Optim_wrapper_lower(problem, abstol, maps_limit, time_limit, LBFGS())

optim_solvers_lower["Optim, ConjugateGradient"] = (problem, abstol, start_time, maps_limit, time_limit) -> 
	Optim_wrapper_lower(problem, abstol, maps_limit, time_limit, ConjugateGradient())

optim_prob_sizes = Dict{String, Int}()
for (name, gen_problem) in landscapes
	optim_prob_sizes[name] = length(gen_problem().x0)
end

order_length = sortperm([l for (name, l) in optim_prob_sizes])
optim_problems_names = [name for (name, l) in optim_prob_sizes][order_length]
optim_solver_names = sort([name for (name, wrapper) in optim_solvers])

gen_Feval_limit(problem, time_limit) = 1_000_000 

function compute_norm(problem, solution)
	gout = similar(solution)
	if sum(_isbad.(solution)) == 0
		problem.grad!(gout, solution)
		last_res = norm(gout, Inf)
	else
		last_res = NaN
	end
end

function compute_norm_cons(problem, solution)
	gout = similar(solution)
	if sum(_isbad.(solution)) == 0
		problem.grad!(gout, solution)
		for i in (length(gout) ÷ 2 + 1):length(gout)
			abs(solution[i] - (problem.x0[i] - 1)) < 1e-7 && (gout[i] = 0)
		end
		norm(gout, Inf)
	else
		NaN
	end
end

res_optim_all = many_problems_many_solvers(landscapes, optim_solvers, optim_problems_names, 
	optim_solver_names,	compute_norm; tunits = 3, F_name = "Grad evals", gen_Feval_limit, 
	abstol = 1e-7, time_limit = 100., proper_benchmark = true)

res_maps_all = many_problems_many_solvers(fixed_point_problems, fixed_point_solvers, problems_names, 
	solver_names, compute_norm; tunits = time_units, F_name = "maps", gen_Feval_limit, 
	abstol = abstols, time_limit = time_limits, proper_benchmark = true)

JLD2.@save path_out*"res_optim.jld2" res_optim_all
title = "Performance profiles for non-linear, unconstrained optimization"
perf_profiles(res_optim_all, title, path_out*"perf_optim.svg", optim_solver_names; sizef = (640, 480), stat_num = 2, max_fact = 8)

res_all_cons = many_problems_many_solvers(landscapes, optim_solvers_lower, optim_problems_names, 
	optim_solver_names, compute_norm_cons; tunit = 3, abstol = 1e-7, time_limit = 100., 
	gen_maps_limit = gen_maps_limit)
	
JLD2.@save path_out*"res_optim_cons.jld2" res_all_cons
title = "Performance profiles for non-linear, box-constriained optimization"
perf_profiles(res_all_cons, title, path_out*"perf_optim_cons.svg", optim_solver_names; sizef = (640, 480), stat_num = 2, max_fact = 8)

#= Unconstrained output
CLIFF: 2 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient      149       35.08μs  true 3.712e-08        GradientNorm 0.1997866
Optim, LBFGS                  106       16.59μs  true 2.217e-09        GradientNorm 0.1997866
Speedmapping, acx             391       12.31μs  true 1.929e-11         first_order 0.1997866

BEALE: 2 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient       18        8.91μs  true 1.517e-08        GradientNorm 4.3098755
Optim, LBFGS                   33        8.97μs  true 1.358e-10        GradientNorm 3.2288345
Speedmapping, acx              58        3.31μs  true 2.916e-09         first_order 9.2250572

MISRA1BLS: 2 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient      184       57.79μs  true 2.477e-09        GradientNorm 0.0754646
Optim, LBFGS                  158       48.28μs  true 5.146e-08        GradientNorm 0.0754646
Speedmapping, acx         1000001   126770.97μs false 5.933e-02            max_eval 7.3148029

Hosaki: 2 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient       12        5.15μs  true 8.225e-08        GradientNorm -2.345811
Optim, LBFGS                   17        5.03μs  true 1.769e-11        GradientNorm -2.345811
Speedmapping, acx              15        1.14μs  true 7.278e-09         first_order -2.345811

MISRA1ALS: 2 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient      257      100.49μs  true 2.403e-09        GradientNorm 0.1245513
Optim, LBFGS                  189       68.47μs  true 2.753e-08        GradientNorm 0.1245513
Speedmapping, acx         1000001   108513.12μs false 6.669e-02            max_eval 19.514350

Six-hump camel: 2 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient        9        4.48μs  true 1.731e-08        GradientNorm -1.031628
Optim, LBFGS                   27        8.99μs  true 5.170e-13        GradientNorm -1.031628
Speedmapping, acx              26        1.38μs  true 4.941e-10         first_order -1.031628

Himmelblau: 2 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient       12        4.74μs  true 2.230e-10        GradientNorm 3.9234349
Optim, LBFGS                   22        6.47μs  true 2.986e-10        GradientNorm 8.8104899
Speedmapping, acx              18        1.11μs  true 8.851e-08         first_order 1.7844827

ROSENBR: 2 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient       63       20.86μs  true 2.320e-09        GradientNorm 1.2960162
Optim, LBFGS                   78       21.14μs  true 2.162e-08        GradientNorm 4.4351302
Speedmapping, acx              89        3.88μs  true 8.144e-09         first_order 1.0370723

Fletcher-Powell: 3 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient       70       41.91μs  true 9.457e-08        GradientNorm 3.3859550
Optim, LBFGS                   65       19.33μs  true 3.646e-10        GradientNorm 2.6185772
Speedmapping, acx              58        3.70μs  true 1.496e-09         first_order 5.5947326

Perm 2: 4 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient      116       70.59μs  true 7.727e-08        GradientNorm 0.0108838
Optim, LBFGS                   96       62.60μs  true 1.445e-08        GradientNorm 0.0108838
Speedmapping, acx             146       23.37μs  true 8.282e-08         first_order 0.0108838

Powell: 4 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient     4529     1850.10μs  true 9.959e-08        GradientNorm 9.2535349
Optim, LBFGS                   88       33.13μs  true 1.170e-08        GradientNorm 1.3862788
Speedmapping, acx             344       15.16μs  true 9.136e-08         first_order 6.0544046

PALMER5D: 4 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient       15        8.22μs  true 3.284e-09        GradientNorm 87.339399
Optim, LBFGS                   16        6.60μs  true 1.717e-09        GradientNorm 87.339399
Speedmapping, acx              63        6.67μs  true 6.846e-08         first_order 87.339399

LANCZOS2LS: 6 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient     1502     1831.19μs  true 8.262e-08        GradientNorm 8.4621513
Optim, LBFGS                  211      184.90μs  true 1.702e-08        GradientNorm 4.2982208
Speedmapping, acx            2359      822.32μs  true 6.739e-08         first_order 4.2985004

PALMER5C: 6 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient        8        5.56μs  true 1.046e-11        GradientNorm 2.1280866
Optim, LBFGS                   16        8.60μs  true 2.709e-14        GradientNorm 2.1280866
Speedmapping, acx              28        3.58μs  true 2.988e-08         first_order 2.1280866

LANCZOS1LS: 6 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient      803      927.79μs  true 8.388e-08        GradientNorm 9.6593453
Optim, LBFGS                  213      184.01μs  true 2.157e-08        GradientNorm 4.2906202
Speedmapping, acx            1715      593.27μs  true 9.954e-08         first_order 4.2913752

LANCZOS3LS: 6 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient      512      530.95μs  true 6.796e-08        GradientNorm 1.2691850
Optim, LBFGS                  218      184.60μs  true 1.232e-08        GradientNorm 4.3465532
Speedmapping, acx            1538      523.03μs  true 8.688e-08         first_order 4.3465539

BIGGS6: 6 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient      462      356.87μs  true 8.487e-08        GradientNorm 0.0056556
Optim, LBFGS                   88       50.49μs  true 1.495e-09        GradientNorm 0.0056556
Speedmapping, acx             476       97.65μs  true 9.643e-09         first_order 0.0056556

THURBERLS: 7 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient    29324    13737.27μs  true 9.899e-08        GradientNorm 5642.7082
Optim, LBFGS                  233       95.07μs  true 2.145e-08        GradientNorm 5642.7082
Speedmapping, acx            1446      125.38μs  true 6.588e-09         first_order 3.4166736

HAHN1LS: 7 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient        2        3.35μs  true 2.065e-09        GradientNorm 55530.953
Optim, LBFGS                    2        3.72μs  true 2.065e-10        GradientNorm 55530.953
Speedmapping, acx               4        2.27μs  true 2.065e-10         first_order 55530.953

PALMER1D: 7 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient   105782    58711.62μs  true 7.034e-08        GradientNorm 0.6526825
Optim, LBFGS                   37       14.45μs  true 5.637e-09        GradientNorm 0.6526825
Speedmapping, acx         1000000   154213.74μs false 1.026e+01            max_eval 4.9193253

GAUSS2LS: 8 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient   323630  4849739.60μs  true 3.269e-08        GradientNorm 1247.5282
Optim, LBFGS                  183     1637.45μs  true 9.300e-10        GradientNorm 1247.5282
Speedmapping, acx         1000001  3821178.91μs false 1.389e-04            max_eval 1247.5282

PALMER6C: 8 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient   619851   385923.60μs  true 8.207e-08        GradientNorm 0.0163874
Optim, LBFGS                   44       14.79μs  true 3.432e-08        GradientNorm 0.0163874
Speedmapping, acx         1000001   141055.11μs false 9.399e-04            max_eval 0.0950475

PALMER2C: 8 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient   670896   314269.27μs  true 4.829e-08        GradientNorm 0.0144213
Optim, LBFGS                   57       19.75μs  true 1.920e-09        GradientNorm 0.0144213
Speedmapping, acx         1000002   156586.17μs false 2.258e-02            max_eval 0.3045348

PALMER1C: 8 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient   630836   311231.71μs  true 9.854e-08        GradientNorm 0.0975979
Optim, LBFGS                   54       19.57μs  true 3.697e-08        GradientNorm 0.0975979
Speedmapping, acx         1000001   155941.01μs false 1.962e+00            max_eval 41.014589

PALMER4C: 8 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient   621338   418817.03μs  true 8.237e-08        GradientNorm 0.0503106
Optim, LBFGS                   42       15.61μs  true 4.718e-09        GradientNorm 0.0503106
Speedmapping, acx         1000001   162341.12μs false 5.787e-03            max_eval 0.3678101

PALMER3C: 8 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient   451326   214295.09μs  true 7.546e-08        GradientNorm 0.0195376
Optim, LBFGS                   43       14.77μs  true 6.783e-09        GradientNorm 0.0195376
Speedmapping, acx         1000000   129651.26μs false 4.393e-03            max_eval 0.1571557

VIBRBEAM: 8 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient   288922   232213.84μs  true 9.058e-08        GradientNorm 0.3322376
Optim, LBFGS                  267      208.66μs  true 1.945e-08        GradientNorm 1.7488666
Speedmapping, acx         1000000   461001.38μs false 8.612e+01            max_eval 24.255340

PALMER8C: 8 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient   182515    79322.30μs  true 8.400e-08        GradientNorm 0.1597680
Optim, LBFGS                   41       13.83μs  true 4.816e-09        GradientNorm 0.1597680
Speedmapping, acx         1000001   124192.00μs false 1.465e-03            max_eval 0.5637297

GAUSS3LS: 8 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient  1000000 13746333.24μs false 1.371e+01       GradientCalls 1245.3669
Optim, LBFGS                  213     1847.95μs  true 3.504e-10        GradientNorm 1244.4846
Speedmapping, acx            1030     3681.60μs  true 5.350e-08         first_order 11386.720

PALMER7C: 8 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient   686931   298317.26μs  true 9.012e-08        GradientNorm 0.6019856
Optim, LBFGS                   47       15.55μs  true 1.415e-08        GradientNorm 0.6019856
Speedmapping, acx         1000001   133152.96μs false 7.660e-03            max_eval 4.3413932

GAUSS1LS: 8 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient   739873 10591846.88μs  true 9.773e-08        GradientNorm 1315.8222
Optim, LBFGS                  181     1593.75μs  true 8.162e-08        GradientNorm 1315.8222
Speedmapping, acx             516     2006.52μs  true 3.460e-08         first_order 52889.248

STRTCHDV: 10 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient       46       71.89μs  true 6.924e-08        GradientNorm 1.2341096
Optim, LBFGS                   66       66.53μs  true 6.287e-08        GradientNorm 1.4651569
Speedmapping, acx              82       36.91μs  true 5.704e-08         first_order 1.1722089

HILBERTB: 10 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient        6        4.16μs  true 2.272e-09        GradientNorm 9.9477300
Optim, LBFGS                   12        6.13μs  true 2.272e-09        GradientNorm 9.9477300
Speedmapping, acx               8        1.44μs  true 6.584e-08         first_order 9.2422786

TRIGON1: 10 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient       63       53.51μs  true 8.440e-08        GradientNorm 1.8713990
Optim, LBFGS                   60       33.11μs  true 3.999e-09        GradientNorm 5.4585568
Speedmapping, acx             120       22.24μs  true 7.368e-08         first_order 4.6836831

TOINTQOR: 50 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient       49       38.47μs  true 7.421e-08        GradientNorm 1175.4722
Optim, LBFGS                   89       49.78μs  true 5.393e-08        GradientNorm 1175.4722
Speedmapping, acx              64       13.17μs  true 1.028e-08         first_order 1175.4722

CHNROSNB: 50 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient      324      315.18μs  true 9.395e-08        GradientNorm 6.1605286
Optim, LBFGS                  663      347.27μs  true 7.195e-08        GradientNorm 1.7987068
Speedmapping, acx            1044      145.24μs  true 6.242e-08         first_order 6.5256877

DECONVU: 63 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient     1560     2585.96μs  true 8.512e-08        GradientNorm 2.7875493
Optim, LBFGS                  984     1020.95μs  true 9.771e-08        GradientNorm 1.9037212
Speedmapping, acx           23974    12709.41μs  true 9.962e-08         first_order 1.5203465

LUKSAN13LS: 98 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient      219      231.94μs  true 5.765e-08        GradientNorm 25188.859
Optim, LBFGS                  197      169.85μs  true 5.222e-08        GradientNorm 25188.859
Speedmapping, acx             220       82.63μs  true 8.079e-08         first_order 25188.859

QING: 100 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient       87      117.01μs  true 9.265e-08        GradientNorm 2.1046666
Optim, LBFGS                  217      155.31μs  true 7.209e-08        GradientNorm 3.9116148
Speedmapping, acx             130       39.21μs  true 7.740e-08         first_order 6.7655489

Paraboloid Random Matrix: 100 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient      169      504.74μs  true 8.061e-08        GradientNorm 4.0449583
Optim, LBFGS                  283      533.87μs  true 9.265e-08        GradientNorm 8.8208202
Speedmapping, acx             210      184.46μs  true 8.770e-08         first_order 2.4644966

Extended Powell: 100 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient     8994    12618.16μs  true 9.726e-08        GradientNorm 5.2950521
Optim, LBFGS                 1449     1003.01μs  true 8.808e-08        GradientNorm 1.0691771
Speedmapping, acx             440      110.69μs  true 8.220e-08         first_order 6.6776778

Trigonometric: 100 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient       65      243.82μs  true 5.714e-08        GradientNorm 9.2048134
Optim, LBFGS                  123      315.30μs  true 9.552e-08        GradientNorm 9.2048149
Speedmapping, acx             114      167.22μs  true 4.389e-08         first_order 1.2026987

Dixon and Price: 100 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient      168      250.41μs  true 9.149e-08        GradientNorm 2.3701758
Optim, LBFGS                 2120     1664.63μs  true 6.772e-08        GradientNorm 0.6666666
Speedmapping, acx             298      104.43μs  true 5.400e-08         first_order 5.3437504

Paraboloid Diagonal: 100 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient     2175     3220.97μs  true 8.499e-08        GradientNorm 2.6472479
Optim, LBFGS                  260      251.52μs  true 9.046e-08        GradientNorm 6.0502738
Speedmapping, acx             223       93.35μs  true 6.989e-08         first_order 6.6221452

LUKSAN17LS: 100 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient     1208     8235.77μs  true 9.759e-08        GradientNorm 0.4931612
Optim, LBFGS                  699     3692.83μs  true 7.647e-08        GradientNorm 0.4931612
Speedmapping, acx             690     1950.62μs  true 5.417e-08         first_order 0.4931612

LUKSAN11LS: 100 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient      998     1491.47μs  true 9.980e-08        GradientNorm 1.8049443
Optim, LBFGS                 2481     2172.76μs  true 1.700e-08        GradientNorm 2.3034776
Speedmapping, acx            8359     2312.11μs  true 5.195e-12         first_order 6.7463141

Quadratic Diagonal: 100 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient       60      108.20μs  true 5.247e-08        GradientNorm 9.4898519
Optim, LBFGS                  174      166.28μs  true 5.247e-08        GradientNorm 9.4898516
Speedmapping, acx             124       51.50μs  true 4.535e-08         first_order 5.5008126

LUKSAN21LS: 100 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient     1899     3140.30μs  true 7.975e-08        GradientNorm 6.0188021
Optim, LBFGS                 1852     1653.18μs  true 8.233e-08        GradientNorm 6.0477643
Speedmapping, acx            1951      787.76μs  true 7.917e-08         first_order 2.8664114

LUKSAN16LS: 100 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient       65      518.03μs  true 7.514e-08        GradientNorm 3.5696970
Optim, LBFGS                  115      789.11μs  true 3.258e-08        GradientNorm 3.5696970
Speedmapping, acx              39      188.76μs  true 5.351e-08         first_order 3.5696970

LUKSAN15LS: 100 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient       65     1692.04μs  true 6.445e-08        GradientNorm 3.5696970
Optim, LBFGS                  107     2400.03μs  true 1.981e-08        GradientNorm 3.5696970
Speedmapping, acx              40      646.37μs  true 3.129e-08         first_order 3.5696970

VARDIM: 200 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient       75       98.87μs  true 5.093e-08        GradientNorm 1.6212745
Optim, LBFGS                   87       68.17μs  true 8.882e-16        GradientNorm 4.8810768
Speedmapping, acx             108       69.76μs  true 3.800e-08         first_order 1.9924766

BROWNAL: 200 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient       90      100.86μs  true 4.178e-09        GradientNorm 5.9431422
Optim, LBFGS                   51       37.94μs  true 3.331e-11        GradientNorm 1.4022938
Speedmapping, acx         1000001   473716.02μs false 1.060e-06            max_eval 1.1919990

ARGLINB: 200 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient        4100004203.80μs false 4.068e-02           Timed out 99.625468
Optim, LBFGS                   45       85.32μs false 4.280e-04    FailedLinesearch 99.625468
Speedmapping, acx         1000001   719660.04μs false 4.925e-04            max_eval 99.625468

ARGLINA: 200 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient        2        9.54μs  true 5.773e-15        GradientNorm     200.0
Optim, LBFGS                    3        8.43μs  true 0.000e+00        GradientNorm     200.0
Speedmapping, acx               4        4.42μs  true 0.000e+00         first_order     200.0

PENALTY2: 200 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient     1201     3573.31μs  true 9.835e-08        GradientNorm 4.7116277
Optim, LBFGS                  539     1417.94μs  true 7.478e-08        GradientNorm 4.7116277
Speedmapping, acx             324      374.62μs  true 7.119e-08         first_order 4.7116277

ARGTRIGLS: 200 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient      643     4089.33μs  true 5.980e-08        GradientNorm 1.7206304
Optim, LBFGS                 1925     7490.64μs  true 7.167e-08        GradientNorm 6.5351650
Speedmapping, acx            2620     4698.98μs  true 9.452e-08         first_order 2.4284270

ARGLINC: 200 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient     1693100004590.99μs false 2.403e-04           Timed out 101.12547
Optim, LBFGS              1000014  1352562.19μs false 2.403e-04       GradientCalls 101.12547
Speedmapping, acx         1000001   831238.99μs false 4.663e-04            max_eval 101.12547

PENALTY3: 200 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient      109      248.48μs  true 6.656e-08        GradientNorm 4.6270516
Optim, LBFGS                  206      311.53μs  true 5.245e-08        GradientNorm 4.1104417
Speedmapping, acx             185      132.40μs  true 5.519e-08         first_order 8.5484659

Large Polynomial: 250 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient        2       11.92μs  true 0.000e+00        GradientNorm       0.0
Optim, LBFGS                    3       10.51μs  true 0.000e+00        GradientNorm       0.0
Speedmapping, acx               4        5.82μs  true 0.000e+00         first_order       0.0

OSCIPATH: 500 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient       19       63.66μs  true 6.187e-08        GradientNorm 0.9999666
Optim, LBFGS                   38       80.02μs  true 5.961e-08        GradientNorm 0.9999666
Speedmapping, acx              15       24.22μs  true 1.571e-08         first_order 0.9999666

GENROSE: 500 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient     1126     4992.37μs  true 5.980e-08        GradientNorm 2.3527656
Optim, LBFGS                 2669     6395.43μs  true 6.474e-08        GradientNorm 4.4042959
Speedmapping, acx            3783     3734.83μs  true 1.195e-08         first_order 5.8365292

INTEQNELS: 502 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient        8       66.76μs  true 8.746e-09        GradientNorm 1.6540332
Optim, LBFGS                   14       73.22μs  true 7.600e-08        GradientNorm 1.0975417
Speedmapping, acx              10       33.09μs  true 5.466e-08         first_order 6.0028024

EXTROSNB: 1000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient    38267   333482.98μs  true 9.839e-08        GradientNorm 3.1704182
Optim, LBFGS               108272   507188.87μs  true 9.514e-08        GradientNorm 4.4595139
Speedmapping, acx         1000000  2130380.74μs false 3.660e-07            max_eval 1.8103875

PENALTY1: 1000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient       76      419.77μs  true 9.627e-08        GradientNorm 0.0096861
Optim, LBFGS                  189      712.24μs  true 7.396e-08        GradientNorm 0.0096861
Speedmapping, acx              98      204.41μs  true 9.531e-08         first_order 0.0096861

EG2: 1000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient       12      160.32μs  true 1.011e-13        GradientNorm -998.9473
Optim, LBFGS                   15      183.85μs  true 1.054e-11        GradientNorm -998.9473
Speedmapping, acx              10       81.57μs  true 3.422e-13         first_order -998.9473

FLETCHCR: 1000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient     4440    36073.42μs  true 9.848e-08        GradientNorm 6.0402170
Optim, LBFGS                11352    51857.31μs  true 8.919e-08        GradientNorm 2.6979499
Speedmapping, acx           46028    85721.43μs  true 5.683e-08         first_order 9.4846985

MSQRTBLS: 1024 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient     3006    66143.65μs  true 9.911e-08        GradientNorm 1.9787257
Optim, LBFGS                 8015   103980.62μs  true 9.512e-08        GradientNorm 1.4509481
Speedmapping, acx            8564    77155.32μs  true 9.527e-08         first_order 3.1477372

MSQRTALS: 1024 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient     3911    80304.13μs  true 9.181e-08        GradientNorm 2.4127664
Optim, LBFGS                10895   151764.46μs  true 9.105e-08        GradientNorm 2.1991833
Speedmapping, acx           12332   107393.06μs  true 9.864e-08         first_order 1.0731975

EDENSCH: 2000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient       51      582.97μs  true 9.625e-08        GradientNorm 12003.284
Optim, LBFGS                   71      588.42μs  true 3.654e-08        GradientNorm 12003.284
Speedmapping, acx              52      191.59μs  true 3.410e-08         first_order 12003.284

DIXMAANK: 3000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient     2780   236620.69μs  true 9.551e-08        GradientNorm 1.0000000
Optim, LBFGS                 8559   430593.33μs  true 8.498e-08        GradientNorm 1.0000000
Speedmapping, acx           24249   597319.52μs  true 9.478e-08         first_order 1.0000000

DIXMAANH: 3000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient      189    13126.42μs  true 9.546e-08        GradientNorm 1.0000000
Optim, LBFGS                  711    29554.33μs  true 9.977e-08        GradientNorm 1.0000000
Speedmapping, acx             378     7019.53μs  true 6.717e-08         first_order 1.0000000

DIXMAANI1: 3000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient     4798   395239.72μs  true 7.295e-08        GradientNorm 1.0000000
Optim, LBFGS                14227   706805.78μs  true 8.477e-08        GradientNorm 1.0000000
Speedmapping, acx           17029   389221.48μs  true 9.871e-08         first_order 1.0000000

DIXMAANB: 3000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient       11      465.80μs  true 1.562e-08        GradientNorm 1.0000000
Optim, LBFGS                   22      541.73μs  true 7.265e-08        GradientNorm 1.0000000
Speedmapping, acx              18      230.51μs  true 7.246e-09         first_order 1.0000000

DIXMAAND: 3000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient       14      578.58μs  true 6.504e-08        GradientNorm 1.0000000
Optim, LBFGS                   28      692.14μs  true 3.194e-08        GradientNorm 1.0000000
Speedmapping, acx              20      250.17μs  true 3.528e-08         first_order 1.0000000

DIXMAANN: 3000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient     3601   393245.89μs  true 9.877e-08        GradientNorm 1.0000000
Optim, LBFGS                11012   744051.29μs  true 9.153e-08        GradientNorm 1.0000000
Speedmapping, acx           15656   503478.95μs  true 9.983e-08         first_order 1.0000000

DIXMAANJ: 3000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient     2846   237985.79μs  true 8.435e-08        GradientNorm 1.0000000
Optim, LBFGS                 8592   433187.53μs  true 9.891e-08        GradientNorm 1.0000000
Speedmapping, acx           19455   459366.66μs  true 9.929e-08         first_order 1.0000000

DIXMAANM1: 3000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient     4748   527576.05μs  true 9.599e-08        GradientNorm 1.0000000
Optim, LBFGS                14275  1009480.68μs  true 9.228e-08        GradientNorm 1.0000000
Speedmapping, acx           32109  1073464.98μs  true 8.408e-08         first_order 1.0000000

DIXMAANF: 3000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient      199    13630.49μs  true 9.821e-08        GradientNorm 1.0000000
Optim, LBFGS                  702    28326.84μs  true 9.955e-08        GradientNorm 1.0000000
Speedmapping, acx             580    10611.77μs  true 6.192e-08         first_order 1.0000000

DIXMAANL: 3000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient     2755   235201.64μs  true 9.337e-08        GradientNorm 1.0000000
Optim, LBFGS                 8447   435613.39μs  true 9.822e-08        GradientNorm 1.0000000
Speedmapping, acx           22752   551191.16μs  true 9.803e-08         first_order 1.0000000

DIXMAANC: 3000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient       13      530.35μs  true 1.744e-08        GradientNorm 1.0000000
Optim, LBFGS                   23      562.66μs  true 4.768e-08        GradientNorm 1.0000000
Speedmapping, acx              20      250.47μs  true 2.212e-09         first_order 1.0000000

DIXMAANO: 3000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient     3533   387856.54μs  true 8.994e-08        GradientNorm 1.0000000
Optim, LBFGS                10734   743575.04μs  true 9.892e-08        GradientNorm 1.0000000
Speedmapping, acx           23679   779247.76μs  true 9.995e-08         first_order 1.0000000

DIXMAANP: 3000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient     3464   385954.64μs  true 8.065e-08        GradientNorm 1.0000000
Optim, LBFGS                10446   702253.50μs  true 9.947e-08        GradientNorm 1.0000000
Speedmapping, acx           22738   738473.88μs  true 1.000e-07         first_order 1.0000000

DIXMAANA1: 3000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient        8      329.89μs  true 5.255e-08        GradientNorm 1.0000000
Optim, LBFGS                   20      486.57μs  true 1.038e-12        GradientNorm       1.0
Speedmapping, acx              12      177.70μs  true 1.235e-09         first_order 1.0000000

DIXMAANE1: 3000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient      256    17718.42μs  true 9.810e-08        GradientNorm 1.0000000
Optim, LBFGS                  767    30830.11μs  true 9.317e-08        GradientNorm 1.0000000
Speedmapping, acx             436     7946.77μs  true 6.823e-08         first_order 1.0000000

DIXMAANG: 3000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient      199    13766.43μs  true 9.897e-08        GradientNorm 1.0000000
Optim, LBFGS                  699    28283.25μs  true 9.918e-08        GradientNorm 1.0000000
Speedmapping, acx             452     8252.99μs  true 8.878e-08         first_order 1.0000000

WOODS: 4000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient      174     5096.30μs  true 3.025e-09        GradientNorm 2.8402425
Optim, LBFGS                   62     1154.10μs  true 1.733e-10        GradientNorm 1.4220758
Speedmapping, acx             460     3160.49μs  true 2.072e-08         first_order 7.4502046

LIARWHD: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient       38     1154.69μs  true 4.232e-12        GradientNorm 5.4685323
Optim, LBFGS                   45     1131.65μs  true 8.882e-15        GradientNorm 9.8982322
Speedmapping, acx             127     1839.52μs  true 5.738e-08         first_order 1.0266710

BDQRTIC: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient      829    25352.75μs  true 5.804e-08        GradientNorm 20006.256
Optim, LBFGS                  151     4824.49μs  true 6.315e-08        GradientNorm 20006.256
Speedmapping, acx             242     4241.90μs  true 1.426e-08         first_order 20006.256

SCHMVETT: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient      131    19762.07μs  true 8.752e-08        GradientNorm  -14994.0
Optim, LBFGS                  146    17489.92μs  true 6.234e-08        GradientNorm  -14994.0
Speedmapping, acx              82     5230.41μs  true 9.962e-08         first_order  -14994.0

SROSENBR: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient       18      526.47μs  true 5.869e-10        GradientNorm 9.0591124
Optim, LBFGS                   26      489.55μs  true 3.119e-08        GradientNorm 1.5164287
Speedmapping, acx              31      298.31μs  true 7.017e-12         first_order 1.9234108

NCB20B: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient     7941   710277.53μs  true 8.622e-08        GradientNorm 7351.3005
Optim, LBFGS                 4617   375970.91μs  true 9.810e-08        GradientNorm 7351.3005
Speedmapping, acx            5692   326804.39μs  true 9.838e-08         first_order 7351.3005

QUARTC: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient       37     1571.39μs  true 9.662e-08        GradientNorm 3.4907174
Optim, LBFGS                  176     4488.27μs  true 8.911e-08        GradientNorm 6.2597909
Speedmapping, acx             108     1220.46μs  true 6.051e-08         first_order 1.7832465

BROYDN3DLS: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient       57     2436.21μs  true 9.721e-08        GradientNorm 1.5299856
Optim, LBFGS                  104     2807.44μs  true 6.469e-08        GradientNorm 3.1444820
Speedmapping, acx             178     2429.33μs  true 5.161e-08         first_order 1.8843361

POWELLSG: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient     4529   159330.35μs  true 9.959e-08        GradientNorm 1.1566908
Optim, LBFGS                   88     1707.77μs  true 1.170e-08        GradientNorm 1.7328484
Speedmapping, acx             330     3194.35μs  true 6.854e-08         first_order 1.0457420

TRIDIA: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient      825    40112.39μs  true 9.848e-08        GradientNorm 5.5027248
Optim, LBFGS                 2467    64520.39μs  true 9.899e-08        GradientNorm 4.7422105
Speedmapping, acx            3652    41424.25μs  true 7.871e-08         first_order 8.1892689

GENHUMPS: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient     6287  1224128.50μs  true 4.171e-10        GradientNorm 2.3602530
Optim, LBFGS                26597  2927749.56μs  true 2.011e-08        GradientNorm 1.3252402
Speedmapping, acx         1000001 62078738.93μs false 7.048e+00            max_eval 269099.67

INDEF: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient       52     9369.90μs false 1.841e+00    FailedLinesearch 4603.2873
Optim, LBFGS                   51     9203.47μs false 1.841e+00    FailedLinesearch 4603.2873
Speedmapping, acx         1000001 49609344.01μs false 1.000e+00            max_eval -6.789831

NONDQUAR: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient     8769   290050.12μs  true 5.666e-08        GradientNorm 3.6988198
Optim, LBFGS                33507   746346.24μs  true 9.169e-08        GradientNorm 4.1243133
Speedmapping, acx            6878    75816.42μs  true 9.527e-08         first_order 1.9172965

TQUARTIC: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient       33      656.79μs  true 2.141e-08        GradientNorm 6.6389107
Optim, LBFGS                   36      636.23μs  true 1.310e-14        GradientNorm 4.2906637
Speedmapping, acx          126663  1057932.76μs  true 9.999e-08         first_order 6.2496054

ARWHEAD: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient       26      471.52μs  true 1.813e-10        GradientNorm       0.0
Optim, LBFGS                   22      359.14μs  true 3.463e-09        GradientNorm       0.0
Speedmapping, acx              20      189.04μs  true 1.152e-09         first_order       0.0

DQRTIC: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient       37     1581.52μs  true 9.662e-08        GradientNorm 3.4907174
Optim, LBFGS                  176     4562.24μs  true 8.911e-08        GradientNorm 6.2597909
Speedmapping, acx             108     1212.88μs  true 6.051e-08         first_order 1.7832465

SINQUAD2: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient     1248   116558.49μs  true 7.238e-09        GradientNorm 2.3585287
Optim, LBFGS                  791    65009.72μs  true 8.550e-08        GradientNorm 2.1550919
Speedmapping, acx         1000001 49995467.90μs false 1.961e-06            max_eval 2.3740000

NONCVXU2: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient    18720  3011357.85μs  true 9.988e-08        GradientNorm 11585.316
Optim, LBFGS                23228  2975023.74μs  true 9.913e-08        GradientNorm 11584.233
Speedmapping, acx          209130 12914862.29μs  true 9.994e-08         first_order 11584.233

DQDRTIC: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient        7      246.25μs  true 2.122e-10        GradientNorm 3.7068202
Optim, LBFGS                   16      288.29μs  true 1.271e-11        GradientNorm 8.4974040
Speedmapping, acx              18      178.22μs  true 2.127e-10         first_order 2.0802578

FREUROTH: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient      198     8388.10μs  true 7.417e-08        GradientNorm 608159.18
Optim, LBFGS                   62     1723.14μs  true 5.731e-08        GradientNorm 608159.18
Speedmapping, acx              95     1787.19μs  true 7.246e-08         first_order 608159.18

TOINTGSS: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient        6      632.50μs  true 3.466e-09        GradientNorm 10.002000
Optim, LBFGS                   12      858.40μs  true 2.138e-09        GradientNorm 10.002000
Speedmapping, acx              63     2460.21μs  true 2.165e-09         first_order 10.002000

MOREBV: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient        5      230.80μs  true 7.292e-08        GradientNorm 1.0390636
Optim, LBFGS                    9      292.92μs  true 7.292e-08        GradientNorm 1.0390636
Speedmapping, acx               5      101.42μs  true 6.124e-08         first_order 1.0392835

CRAGGLVY: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient      196    19744.63μs  true 9.567e-08        GradientNorm 1688.2153
Optim, LBFGS                  305    23466.94μs  true 6.958e-08        GradientNorm 1688.2153
Speedmapping, acx             188     7783.99μs  true 7.257e-08         first_order 1688.2153

SPARSINE: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient   214340 53076992.54μs  true 9.775e-08        GradientNorm 1.3908981
Optim, LBFGS               560006100006145.00μs false 6.107e-06           Timed out 8.1706290
Speedmapping, acx          811465100000181.20μs false 6.405e-03           Timed out 0.3606919

NONCVXUN: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient   596031100036209.82μs false 1.402e-06           Timed out 11601.814
Optim, LBFGS               810256100000453.95μs false 6.778e-04           Timed out 11589.520
Speedmapping, acx         1000002 61219776.15μs false 1.507e-05            max_eval 11590.835

ENGVAL1: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient       60     1458.42μs  true 5.941e-08        GradientNorm 5548.6684
Optim, LBFGS                   54     1134.42μs  true 5.621e-08        GradientNorm 5548.6684
Speedmapping, acx              36      336.95μs  true 2.988e-08         first_order 5548.6684

NONDIA: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient       14      344.22μs  true 3.304e-09        GradientNorm 1.0164104
Optim, LBFGS                   29      419.61μs  true 1.111e-10        GradientNorm 6.1916336
Speedmapping, acx             838     6580.93μs  true 9.376e-08         first_order 7.6409948

NCB20: 5010 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient     6811   500103.76μs  true 7.451e-08        GradientNorm -1092.128
Optim, LBFGS                  994    72764.99μs  true 6.559e-08        GradientNorm -1179.943
Speedmapping, acx           11285   577333.51μs  true 9.963e-08         first_order -1462.668

FMINSURF: 5625 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient      591    42731.70μs  true 9.845e-08        GradientNorm 1.0000000
Optim, LBFGS                 1421    63447.18μs  true 9.284e-08        GradientNorm 1.0000000
Speedmapping, acx            1296    33476.71μs  true 9.894e-08         first_order 1.0000000

FMINSRF2: 5625 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient      652    45801.74μs  true 9.663e-08        GradientNorm 1.0000000
Optim, LBFGS                 1647    70934.06μs  true 9.487e-08        GradientNorm 1.0000000
Speedmapping, acx             986    25041.87μs  true 8.779e-08         first_order 1.0000240

CURLY20: 10000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient   914912100003295.18μs false 2.651e-04           Timed out -1.003162
Optim, LBFGS               217766 26771600.75μs  true 9.865e-08        GradientNorm -1.003162
Speedmapping, acx         1000001 79133285.05μs false 1.060e-04            max_eval -996865.5

CURLY10: 10000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient  1000002 97475128.89μs false 1.652e-04       GradientCalls -1.003162
Optim, LBFGS               145207 15089460.65μs  true 9.985e-08        GradientNorm -1.003162
Speedmapping, acx         1000002 71938434.84μs false 1.427e-05            max_eval -1.003162

POWER: 10000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient      443    39798.41μs  true 9.594e-08        GradientNorm 7.4850349
Optim, LBFGS                 1566    84529.11μs  true 9.617e-08        GradientNorm 7.5108579
Speedmapping, acx            1486    32340.62μs  true 6.974e-08         first_order 4.6809909

CURLY30: 10000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient   780182100000548.12μs false 4.677e-04           Timed out -1.003162
Optim, LBFGS               249498 32770187.36μs  true 9.991e-08        GradientNorm -1.003162
Speedmapping, acx         1000000 94486810.18μs false 1.090e-04            max_eval -997056.5

COSINE: 10000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient       24     4424.54μs  true 3.330e-08        GradientNorm   -9999.0
Optim, LBFGS                   35     5438.34μs  true 3.071e-08        GradientNorm   -9999.0
Speedmapping, acx         1000000 51755066.03μs false 9.490e+04            max_eval 8120.6312

DIXON3DQ: 10000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient    10002   863486.98μs  true 8.791e-08        GradientNorm 1.3293951
Optim, LBFGS                30004  1463390.28μs  true 3.806e-08        GradientNorm 1.3453068
Speedmapping, acx           97680  2126161.42μs  true 9.999e-08         first_order 0.0005025

SPARSQUR: 10000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log       obj
Optim, ConjugateGradient       26     3746.06μs  true 8.631e-08        GradientNorm 8.5331301
Optim, LBFGS                  125    12171.47μs  true 6.919e-08        GradientNorm 1.6108565
Speedmapping, acx              73     3556.76μs  true 3.212e-08         first_order 2.2878204

=#

#= Constrained output

CLIFF: 2 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient      278      137.02μs false 1.000e+00      NotImplemented      1.940410281452586e7
Optim, LBFGS                  422      121.08μs  true 3.767e-09        GradientNorm       0.1997866136777351
Speedmapping, acx             817       33.59μs  true 2.917e-11         first_order      0.19978661367769956

BEALE: 2 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       36       28.78μs  true 5.617e-08        GradientNorm   2.7979368618688495e-16
Optim, LBFGS                   59       24.99μs  true 5.550e-08        GradientNorm   3.2946658810854153e-16
Speedmapping, acx              58        3.13μs  true 1.346e-08         first_order   1.9632293933501833e-18

MISRA1BLS: 2 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient      319      157.06μs  true 7.530e-08        GradientNorm      0.07546468153337059
Optim, LBFGS                  244      119.73μs  true 5.033e-08        GradientNorm       0.0754646815334532
Speedmapping, acx         1000001   152868.99μs false 5.933e-02            max_eval        7.314802974183912

Hosaki: 2 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       12       11.97μs  true 8.817e-08        GradientNorm       -2.345811576101299
Optim, LBFGS                   18       10.61μs  true 1.361e-08        GradientNorm      -2.3458115761012923
Speedmapping, acx              15        1.29μs  true 7.278e-09         first_order       -2.345811576101292

MISRA1ALS: 2 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient      409      197.64μs  true 1.166e-08        GradientNorm      0.12455138894439205
Optim, LBFGS                  254      131.88μs  true 1.520e-08        GradientNorm      0.12455138894441055
Speedmapping, acx         1000001   112816.10μs false 6.669e-02            max_eval       19.514350123023213

Six-hump camel: 2 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       25       19.18μs  true 2.021e-08        GradientNorm      -1.0316284534898774
Optim, LBFGS                   77       30.28μs  true 1.048e-09        GradientNorm      -1.0316284534898774
Speedmapping, acx              21        1.23μs  true 6.448e-08         first_order     -0.21546382438371714

Himmelblau: 2 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       19       15.53μs  true 6.000e-08        GradientNorm    6.295138727997497e-17
Optim, LBFGS                   28       12.92μs  true 2.902e-08        GradientNorm   1.2304604871714436e-17
Speedmapping, acx              18        1.28μs  true 8.851e-08         first_order   1.7844827490094893e-16

ROSENBR: 2 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient      135       74.09μs  true 4.440e-10        GradientNorm     8.74002190227853e-20
Optim, LBFGS                  152       62.72μs  true 3.036e-10        GradientNorm    9.239124785605442e-20
Speedmapping, acx              57        2.83μs  true 2.054e-08         first_order    6.598156704218504e-16

Fletcher-Powell: 3 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient      195      163.31μs  true 9.874e-09        GradientNorm   1.3588906704111662e-18
Optim, LBFGS                   99       42.54μs  true 2.916e-08        GradientNorm     3.72036444113153e-18
Speedmapping, acx              58        3.84μs  true 1.496e-09         first_order    5.594732668231458e-21

Perm 2: 4 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient     1101      989.61μs  true 5.767e-08        GradientNorm    8.106750818101178e-15
Optim, LBFGS                  247      165.07μs  true 1.048e-08        GradientNorm    5.498234239891996e-16
Speedmapping, acx             146       27.19μs  true 8.282e-08         first_order     0.010883889811629459

Powell: 4 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient     1152      818.42μs  true 8.893e-08        GradientNorm   2.1557456924686847e-11
Optim, LBFGS                  202       90.98μs  true 1.439e-08        GradientNorm   1.0370396990723693e-12
Speedmapping, acx             344       17.23μs  true 9.136e-08         first_order    6.054404654446226e-11

PALMER5D: 4 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       50 98555422.90μs false 2.280e+02          Iterations        8401.013802882682
Optim, LBFGS                   87       46.43μs  true 2.421e-09        GradientNorm        87.33939952784851
Speedmapping, acx              92       10.55μs  true 1.664e-10         first_order        87.33939952784914

LANCZOS2LS: 6 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient  1115264103741072.18μs false 7.953e-07           Timed out      0.12688480173872244
Optim, LBFGS              3078832  2484471.08μs false 2.680e-02      NotImplemented      0.01264191253434643
Speedmapping, acx             778      276.21μs  true 3.694e-08         first_order     0.006916338261787107

PALMER5C: 6 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       82103302942.04μs false 7.655e-01           Timed out       2.3292843870116218
Optim, LBFGS                  161      107.64μs  true 8.375e-10        GradientNorm        2.251875793994389
Speedmapping, acx              31        4.48μs  true 1.750e-08         first_order         2.25187579329817

LANCZOS1LS: 6 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient  1079671106691106.08μs false 1.773e-05           Timed out       0.1268618260562242
Optim, LBFGS              3006236  2598122.12μs false 2.680e-02      NotImplemented     0.012619374861372214
Speedmapping, acx            1007      366.80μs  true 2.768e-10         first_order     0.006916594192213219

LANCZOS3LS: 6 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient  1003209  1059319.02μs false 5.404e-09        GradientNorm       0.1268756219345674
Optim, LBFGS              3005166  2457027.91μs false 2.680e-02      NotImplemented      0.01263949470016134
Speedmapping, acx            1143      418.83μs  true 4.581e-09         first_order     0.006912985164403775

BIGGS6: 6 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient  1009784   847270.97μs false 6.369e-08        GradientNorm    3.2331748972571936e-5
Optim, LBFGS              1007227   680983.78μs false 3.933e+01      NotImplemented     5.427587896000423e-5
Speedmapping, acx             476      108.65μs  true 9.643e-09         first_order     0.005655649925503534

THURBERLS: 7 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient   986595 55889311.71μs false 2.144e+01          Iterations        593200.8031492977
Optim, LBFGS                  187      116.81μs false 1.913e+04        GradientNorm         328680.388495147
Speedmapping, acx            2767      269.81μs  true 1.499e-08         first_order      3.415139027824454e7

HAHN1LS: 7 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       57      141.17μs  true 7.188e-14        GradientNorm        48580.65444216646
Optim, LBFGS                   54       75.74μs false 8.066e+05        GradientNorm      7.484020583837941e7
Speedmapping, acx               4        2.54μs  true 8.266e-13         first_order        55509.88868224071

PALMER1D: 7 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient     2852107803696.87μs false 8.620e+02           Timed out        512.8367731844031
Optim, LBFGS                  343      244.74μs  true 3.496e-12        GradientNorm        482.2790217404504
Speedmapping, acx             146       24.08μs  true 8.941e-08         first_order       482.27902168032426

GAUSS2LS: 8 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient  2008805 29805531.02μs false 4.058e-08        GradientNorm       1247.5282092309988
Optim, LBFGS                  223     1988.30μs  true 7.204e-09        GradientNorm       1247.5282092309988
Speedmapping, acx         1000001  3575538.87μs false 1.389e-04            max_eval       1247.5282092320424

PALMER6C: 8 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient    60880    57304.86μs  true 3.510e-09        GradientNorm       0.3614203110060443
Optim, LBFGS                  473      304.67μs  true 1.810e-10        GradientNorm      0.36142031100603766
Speedmapping, acx           24396     3289.54μs  true 8.330e-08         first_order      0.36142030973961137

PALMER2C: 8 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient    89196    72191.43μs  true 2.065e-10        GradientNorm       0.7813499006532737
Optim, LBFGS                  532      354.73μs  true 1.066e-11        GradientNorm       0.7813499006533251
Speedmapping, acx            6194      918.03μs  true 5.879e-08         first_order       0.7813499006120755

PALMER1C: 8 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient    44383105924993.04μs false 1.395e+02           Timed out       16.171269999926682
Optim, LBFGS                 1190      763.68μs  true 1.587e-10        GradientNorm       15.829657150677605
Speedmapping, acx          732186   115018.24μs  true 9.976e-08         first_order       15.829657150153416

PALMER4C: 8 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient    45919    39183.22μs  true 2.848e-09        GradientNorm       1.3343295743866512
Optim, LBFGS                  465      302.62μs  true 2.735e-09        GradientNorm       1.3343295743866674
Speedmapping, acx            5343      794.15μs  true 5.128e-08         first_order       1.3343295617239084

PALMER3C: 8 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient    51181    44648.30μs  true 9.533e-11        GradientNorm       0.6776698678449562
Optim, LBFGS                  500      322.97μs  true 3.252e-09        GradientNorm       0.6776698805167726
Speedmapping, acx            7684     1136.97μs  true 9.799e-08         first_order       0.6776698678325404

VIBRBEAM: 8 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient    25224111343076.94μs false 3.635e+05           Timed out       19.502570742676532
Optim, LBFGS               203930   348460.49μs false 1.926e-05      NotImplemented       3.3069024254501365
Speedmapping, acx         1000000   479312.34μs false 8.612e+01            max_eval        24.25534075035414

PALMER8C: 8 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient    40652108658743.14μs false 4.394e+00           Timed out       2.0089565564598195
Optim, LBFGS                  505      313.30μs  true 1.600e-10        GradientNorm       2.0080260185626626
Speedmapping, acx            2812      371.98μs  true 2.905e-08         first_order         2.00802601711683

GAUSS3LS: 8 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient  1216068 12207765.82μs false 5.677e-08        GradientNorm       1244.4846360131562
Optim, LBFGS                  233     2137.49μs  true 9.616e-08        GradientNorm        1244.484636013157
Speedmapping, acx             210      821.02μs  true 5.336e-08         first_order       61551.299321299804

PALMER7C: 8 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient    30351105415344.95μs false 6.901e+00           Timed out       10.573093917913937
Optim, LBFGS                  526      331.59μs  true 4.637e-10        GradientNorm        10.57204124399613
Speedmapping, acx           11083     1584.97μs  true 9.951e-08         first_order       10.572041238835329

GAUSS1LS: 8 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient  1169835 18467740.77μs false 9.483e-08        GradientNorm        1315.822243203378
Optim, LBFGS                  218     1975.59μs  true 2.119e-08        GradientNorm       1315.8222432033767
Speedmapping, acx             516     1987.92μs  true 3.460e-08         first_order        52889.24861821918

STRTCHDV: 10 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient     1106     2305.43μs  true 6.492e-08        GradientNorm   1.8908477884616686e-11
Optim, LBFGS                  975     1216.36μs  true 9.962e-08        GradientNorm    8.654884884877692e-11
Speedmapping, acx             981      461.19μs  true 8.987e-08         first_order       1.6966192150434525

HILBERTB: 10 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       15       23.54μs  true 1.709e-08        GradientNorm     8.83074507966866e-17
Optim, LBFGS                   31       24.82μs  true 1.770e-08        GradientNorm     7.01787669880342e-17
Speedmapping, acx               8        1.61μs  true 6.584e-08         first_order    9.242278604553933e-16

TRIGON1: 10 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       73      113.80μs  true 3.792e-08        GradientNorm    4.530755850023387e-16
Optim, LBFGS                  113      102.63μs  true 9.501e-08        GradientNorm    8.176381843902098e-17
Speedmapping, acx             120       22.95μs  true 7.368e-08         first_order   4.6836831973272524e-17

TOINTQOR: 50 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       70100000150.92μs false 2.261e+01           Timed out       1484.0904255354374
Optim, LBFGS                 1171     1436.45μs  true 3.091e-09        GradientNorm       1222.5124149971011
Speedmapping, acx              63       15.61μs  true 1.120e-08         first_order       1222.5124149967762

CHNROSNB: 50 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient      652     1258.88μs  true 9.466e-08        GradientNorm    8.708393871544616e-16
Optim, LBFGS                 1257     1572.30μs  true 5.797e-08        GradientNorm   5.2590559813442906e-17
Speedmapping, acx            1044      192.69μs  true 6.242e-08         first_order    6.525687738878384e-15

DECONVU: 63 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient  2007045  7641974.93μs false 8.849e-08        GradientNorm    3.229621300971975e-10
Optim, LBFGS              3081619  6524567.13μs false 9.260e-08        GradientNorm     3.626401807560488e-8
Speedmapping, acx           23974    13865.07μs  true 9.962e-08         first_order    1.5203465201255886e-9

LUKSAN13LS: 98 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient      686     1918.53μs  true 3.479e-08        GradientNorm        25188.85958964517
Optim, LBFGS                  719     1686.38μs  true 5.611e-08        GradientNorm        25188.85958964516
Speedmapping, acx             220      110.23μs  true 8.079e-08         first_order        25188.85958964517

QING: 100 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient      116      400.66μs false 1.413e-07      NotImplemented   2.7697537449374897e-16
Optim, LBFGS                  403      876.94μs  true 7.826e-08        GradientNorm    6.263218495370169e-16
Speedmapping, acx             107       44.48μs  true 3.441e-08         first_order    5.542564009954536e-18

Paraboloid Random Matrix: 100 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient  1488360  9520399.81μs false 6.916e-08        GradientNorm   2.6705868928450904e-15
Optim, LBFGS                15519    49754.04μs  true 9.127e-08        GradientNorm   2.6560987611748424e-15
Speedmapping, acx             210      202.13μs  true 8.770e-08         first_order    2.464496633437794e-14

Extended Powell: 100 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient     1503100006512.88μs false 4.710e-01           Timed out       0.8082713825776985
Optim, LBFGS                 7587    15993.45μs  true 1.545e-09        GradientNorm   2.9473378386853145e-12
Speedmapping, acx             636      189.45μs  true 9.480e-08         first_order     1.285845900495708e-9

Trigonometric: 100 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       83      511.91μs  true 7.362e-08        GradientNorm     9.204814830230329e-7
Optim, LBFGS                  214      737.52μs  true 8.856e-08        GradientNorm     9.204814956309874e-7
Speedmapping, acx             114      170.35μs  true 4.389e-08         first_order    1.2026987126958284e-6

Dixon and Price: 100 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient      269      968.17μs  true 6.780e-08        GradientNorm    5.475112319884422e-16
Optim, LBFGS                  882     1967.35μs  true 9.502e-08        GradientNorm    1.456280815492514e-16
Speedmapping, acx             276      117.72μs  true 7.821e-08         first_order   1.2094910421344376e-15

Paraboloid Diagonal: 100 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient 49000100101893015.15μs false 3.445e+00           Timed out        785.2281469167763
Optim, LBFGS               196355   361863.52μs  true 4.970e-08        GradientNorm   1.2525714758066394e-15
Speedmapping, acx             223      114.93μs  true 6.989e-08         first_order    6.622145287736198e-16

LUKSAN17LS: 100 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient  1000061107024383.07μs false 1.161e+04           Timed out       46154.727417217866
Optim, LBFGS              7802503 45559359.07μs false 1.660e-05      NotImplemented        46099.01255604742
Speedmapping, acx            4436    12032.97μs  true 9.490e-08         first_order         46098.9894332973

LUKSAN11LS: 100 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient      996     4254.41μs  true 5.079e-08        GradientNorm   3.9703653447328243e-16
Optim, LBFGS                 6711    14931.74μs  true 1.583e-08        GradientNorm   3.1010903109285467e-17
Speedmapping, acx            8359     2916.64μs  true 5.195e-12         first_order    6.746314117695596e-24

Quadratic Diagonal: 100 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       75      361.58μs  true 7.667e-08        GradientNorm    9.123465444869238e-16
Optim, LBFGS                  250      608.82μs false 1.034e-07      NotImplemented    5.310547428603952e-15
Speedmapping, acx             124       65.16μs  true 4.535e-08         first_order    5.500812608566012e-16

LUKSAN21LS: 100 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       73100000180.01μs false 4.701e-01           Timed out         97.7653781391842
Optim, LBFGS                14834    38479.77μs  true 1.690e-10        GradientNorm       60.408062593495195
Speedmapping, acx            4167     2041.67μs  true 9.782e-08         first_order       60.408062593528264

LUKSAN16LS: 100 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient     7742    73229.92μs  true 6.268e-08        GradientNorm        3.569697051173905
Optim, LBFGS                25276   207108.19μs  true 9.629e-08        GradientNorm        3.569697051173896
Speedmapping, acx              39      185.45μs  true 5.351e-08         first_order       3.5696970511738924

LUKSAN15LS: 100 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient  2002029 59261101.96μs false 8.753e-08        GradientNorm       3.5696970511739012
Optim, LBFGS              2006155 45985700.85μs false 9.739e-08        GradientNorm        3.569697051173901
Speedmapping, acx              40      638.99μs  true 3.129e-08         first_order        3.569697051173899

VARDIM: 200 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient  1000001  3234581.95μs false 6.830e+06      NotImplemented     9.401312700728958e13
Optim, LBFGS                 9946    41327.44μs  true 1.713e-09        GradientNorm    5.969249489915649e-18
Speedmapping, acx             108       90.04μs  true 3.800e-08         first_order    1.992476695730292e-15

BROWNAL: 200 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient    17925   125716.04μs  true 7.670e-08        GradientNorm   1.4426909875262811e-12
Optim, LBFGS                  245      742.19μs  true 5.409e-08        GradientNorm    7.344266756739795e-13
Speedmapping, acx         1000002   592557.19μs false 1.016e-06            max_eval    1.0965038436458557e-9

ARGLINB: 200 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient      124100000149.97μs false 1.842e+11           Timed out             9.90179315e9
Optim, LBFGS                  473     1326.66μs false 2.741e+11      NotImplemented            2.19221892e10
Speedmapping, acx              20       27.12μs  true 1.133e-08         first_order          99.625468164794

ARGLINA: 200 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       89      484.29μs  true 3.315e-09        GradientNorm       300.00000079999995
Optim, LBFGS                   54      253.65μs  true 1.050e-11        GradientNorm        300.0000008000002
Speedmapping, acx              10        9.63μs  true 0.000e+00         first_order                    300.0

PENALTY2: 200 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient    64715   353407.88μs  true 8.654e-08        GradientNorm     4.711627727546475e13
Optim, LBFGS                 3131    19108.12μs  true 7.981e-08        GradientNorm     4.711627727546475e13
Speedmapping, acx             324      427.16μs  true 7.119e-08         first_order     4.711627727546475e13

ARGTRIGLS: 200 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient     2820    23679.06μs  true 8.963e-08        GradientNorm     1.53835291467698e-16
Optim, LBFGS                 3672    25329.32μs  true 8.547e-08        GradientNorm   5.2392882763078725e-17
Speedmapping, acx            2620     4981.38μs  true 9.452e-08         first_order   2.4284270929110605e-17

ARGLINC: 200 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient  1000021  4266229.87μs false 1.664e+05      NotImplemented       101.13376187991838
Optim, LBFGS                  999     2544.47μs false 6.265e+09      NotImplemented          1.17467854375e7
Speedmapping, acx              20       26.82μs  true 8.877e-09         first_order        101.1254705144291

PENALTY3: 200 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient     9186    55529.22μs  true 2.524e-08        GradientNorm        39857.53556269422
Optim, LBFGS                16651    76508.12μs  true 3.902e-08        GradientNorm        39857.53556269423
Speedmapping, acx         1000001  1189699.89μs false 2.381e-06            max_eval       39857.535562694866

Large Polynomial: 250 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       10      177.53μs false 3.953e-03      NotImplemented   0.00024752696261260137
Optim, LBFGS                   83      294.45μs  true 3.958e-09        GradientNorm   2.4768628694261274e-16
Speedmapping, acx               4        7.64μs  true 0.000e+00         first_order                      0.0

OSCIPATH: 500 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient   158291  2083607.11μs  true 8.006e-08        GradientNorm       0.9999666655201664
Optim, LBFGS              1000027  5893643.14μs false 4.066e-08        GradientNorm       0.9999666655201666
Speedmapping, acx              15       32.86μs  true 1.571e-08         first_order       0.9999666655201663

GENROSE: 500 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient     1231    20881.79μs  true 7.966e-08        GradientNorm   2.4027274656446136e-16
Optim, LBFGS                 5182    38871.05μs  true 9.267e-08        GradientNorm    5.867289035138027e-16
Speedmapping, acx            3524     4798.23μs  true 8.779e-08         first_order   3.1239926741341477e-17

INTEQNELS: 502 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       14      390.99μs  true 1.753e-08        GradientNorm    2.206833546245142e-15
Optim, LBFGS                   29      444.27μs  true 2.335e-08        GradientNorm    8.904609959359295e-15
Speedmapping, acx              10       39.10μs  true 5.466e-08         first_order    6.002802406277538e-14

EXTROSNB: 1000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient    58754  1135948.68μs  true 7.320e-08        GradientNorm     3.600362157473285e-8
Optim, LBFGS               362034  5682821.52μs  true 8.687e-08        GradientNorm    6.570661168672569e-13
Speedmapping, acx         1000000  2959389.44μs false 3.660e-07            max_eval    1.8103875844569495e-7

PENALTY1: 1000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       62100000349.04μs false 1.165e+12           Timed out     8.494674163785717e16
Optim, LBFGS             ┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
┌ Warning: Failed to achieve finite new evaluation point, using alpha=0
└ @ LineSearches ~/.julia/packages/LineSearches/b4CwT/src/hagerzhang.jl:156
 2013848 26764140.84μs false 2.531e+04      NotImplemented     8.485088347241957e16
Speedmapping, acx              18       65.11μs  true 3.612e-15         first_order     8.485088347241954e16

EG2: 1000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient      728    17570.52μs  true 1.858e-08        GradientNorm       -998.9473933000194
Optim, LBFGS                  123     2660.99μs  true 1.080e-09        GradientNorm        -998.947393300942
Speedmapping, acx              10       95.22μs  true 3.422e-13         first_order       -998.9473933009451

FLETCHCR: 1000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient     4551   152915.03μs  true 6.559e-08        GradientNorm     4.48185615433388e-15
Optim, LBFGS                25579   362399.70μs  true 8.914e-08        GradientNorm    3.531774830762856e-17
Speedmapping, acx           46028   127475.01μs  true 5.683e-08         first_order    9.484698587935463e-17

MSQRTBLS: 1024 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       48100000453.95μs false 5.847e+01           Timed out       1327.5078793440907
Optim, LBFGS                24196   692164.26μs  true 9.542e-08        GradientNorm    4.799973668183446e-13
Speedmapping, acx            6677    72135.82μs  true 9.522e-08         first_order   1.4280765018333657e-10

MSQRTALS: 1024 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       47100000401.97μs false 5.551e+01           Timed out        1360.942190711669
Optim, LBFGS                17329   486502.93μs  true 9.058e-08        GradientNorm    1.300826257026342e-11
Speedmapping, acx           18633   196959.69μs  true 9.617e-08         first_order     1.099792308978135e-9

EDENSCH: 2000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient      219100000541.93μs false 1.722e+03           Timed out     1.9286954187882585e6
Optim, LBFGS              4051490 95085427.05μs false 2.215e-07      NotImplemented     1.9181646680198214e6
Speedmapping, acx             125      657.09μs  true 4.021e-09         first_order      1.918164668019821e6

DIXMAANK: 3000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       54100000793.22μs false 1.401e+01           Timed out       2494.2498267902997
Optim, LBFGS               848376103511175.87μs false 1.524e-11           Timed out       1688.3115750943375
Speedmapping, acx           11850   310405.95μs  true 9.685e-08         first_order       1688.3115747764418

DIXMAANH: 3000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       35100002264.02μs false 9.760e+00           Timed out        5513.134214188643
Optim, LBFGS              1003181 73709472.89μs false 7.478e-11        GradientNorm        2813.482821231536
Speedmapping, acx             727    15467.15μs  true 9.925e-08         first_order        2813.482820826226

DIXMAANI1: 3000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       71100000715.02μs false 3.145e+00           Timed out       1827.0514188742989
Optim, LBFGS              1001577107082585.10μs false 4.337e-12           Timed out        938.8109697042355
Speedmapping, acx            9404   247406.15μs  true 9.948e-08         first_order        938.8109696745503

DIXMAANB: 3000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       50100000689.03μs false 4.262e+00           Timed out        4104.985505712922
Optim, LBFGS                  540    32987.79μs  true 5.333e-11        GradientNorm       1906.0522099723614
Speedmapping, acx              20      314.48μs  true 1.502e-08         first_order       1906.0522098641827

DIXMAAND: 3000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       89100000740.05μs false 1.101e+01           Timed out        4189.484372575945
Optim, LBFGS                 1035    61314.87μs  true 3.605e-10        GradientNorm       3174.8431721521792
Speedmapping, acx              26      395.16μs  true 7.777e-09         first_order        3174.843171739925

DIXMAANN: 3000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient      104100000730.04μs false 8.750e+00           Timed out       3051.1512071451034
Optim, LBFGS               778817104895540.00μs false 1.411e-12           Timed out       1175.6418026675492
Speedmapping, acx           10632   380518.58μs  true 9.231e-08         first_order       1175.6418026389726

DIXMAANJ: 3000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient  1000039194939682.96μs false 3.513e+00           Timed out       1319.4136070850393
Optim, LBFGS               902511102583917.86μs false 1.088e-11           Timed out        1282.353827352575
Speedmapping, acx           10135   265781.80μs  true 9.969e-08         first_order       1282.3538272752692

DIXMAANM1: 3000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       33100000763.89μs false 2.609e+00           Timed out       2103.7143956864456
Optim, LBFGS               837101106124414.92μs false 2.223e-12           Timed out        912.7397834178765
Speedmapping, acx           11303   409624.53μs  true 9.634e-08         first_order        912.7397834155505

DIXMAANF: 3000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient      121100000761.03μs false 6.144e+00           Timed out        2257.927349802423
Optim, LBFGS              1001959 73141204.12μs false 1.041e-17        GradientNorm       1532.1113824587078
Speedmapping, acx             378     8098.06μs  true 9.737e-08         first_order        1532.111382356693

DIXMAANL: 3000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       68100001804.83μs false 9.739e+00           Timed out         4140.99822117542
Optim, LBFGS              1025391 85382449.87μs false 6.874e-12        GradientNorm       2565.1488865016468
Speedmapping, acx           14630   393214.52μs  true 9.932e-08         first_order       2565.1488816686997

DIXMAANC: 3000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       75100000687.84μs false 6.242e+00           Timed out        4138.259670874727
Optim, LBFGS                  429    28207.79μs  true 1.116e-10        GradientNorm       2309.3121299991835
Speedmapping, acx              23      350.70μs  true 2.672e-08         first_order       2309.3121297945586

DIXMAANO: 3000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient      635100000714.06μs false 6.626e+00           Timed out       1966.5829656309245
Optim, LBFGS               784271103581022.02μs false 1.183e-12           Timed out       1474.8648968519772
Speedmapping, acx           10028   365088.17μs  true 9.982e-08         first_order       1474.8648967831275

DIXMAANP: 3000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient      400100000684.02μs false 9.493e+00           Timed out       3327.9274035245717
Optim, LBFGS               643739105637388.94μs false 2.301e-12           Timed out         2121.09790556935
Speedmapping, acx           10603   383831.64μs  true 9.990e-08         first_order       2121.0979054051286

DIXMAANA1: 3000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient      164100010975.84μs false 2.252e+00           Timed out        1562.905230101539
Optim, LBFGS                  287    17179.56μs  true 2.228e-12        GradientNorm       1559.8107639494096
Speedmapping, acx              12      228.24μs  true 5.115e-11         first_order        1559.810763888852

DIXMAANE1: 3000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       58100001974.11μs false 4.512e+00           Timed out        2796.183484950909
Optim, LBFGS              1002539 73956702.95μs false 4.857e-17        GradientNorm        1188.259725058829
Speedmapping, acx             368     7883.28μs  true 7.524e-08         first_order       1188.2597250047443

DIXMAANG: 3000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       36100000725.03μs false 5.545e+00           Timed out        3772.629442448541
Optim, LBFGS              1002151 73596446.99μs false 3.469e-17        GradientNorm       1937.7606991563853
Speedmapping, acx             498    10665.35μs  true 9.925e-08         first_order       1937.7606989583612

WOODS: 4000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient      520    49560.82μs false 4.459e+00      NotImplemented       4013.3706337572344
Optim, LBFGS                  781    50060.18μs  true 1.266e-08        GradientNorm    6.646187363714368e-16
Speedmapping, acx             460     4958.88μs  true 2.072e-08         first_order    7.450204628522614e-13

LIARWHD: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient    42830100046803.00μs false 5.249e-09           Timed out        37913.44422407352
Optim, LBFGS              2103080136385351.18μs false 7.614e-07           Timed out       37913.444219725076
Speedmapping, acx              71     1403.28μs  true 1.470e-10         first_order       37913.444219725076

BDQRTIC: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       27100001400.95μs false 4.068e+02           Timed out        500663.2031745837
Optim, LBFGS                 1204   133220.55μs  true 4.342e-10        GradientNorm        20006.25687843367
Speedmapping, acx             106     2445.95μs  true 1.568e-09         first_order       20006.256878433676

SCHMVETT: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient      242    53797.35μs  true 8.885e-08        GradientNorm                 -14994.0
Optim, LBFGS                  252    45493.26μs  true 8.305e-08        GradientNorm                 -14994.0
Speedmapping, acx              82     5793.16μs  true 9.962e-08         first_order                 -14994.0

SROSENBR: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient      116    18169.52μs false 2.602e-01      NotImplemented       154.64224617516498
Optim, LBFGS                  157    12714.91μs  true 7.978e-10        GradientNorm    3.098116052170267e-16
Speedmapping, acx              31      465.18μs  true 7.017e-12         first_order   1.9234108169266209e-19

NCB20B: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient     5887  1035805.03μs  true 9.525e-08        GradientNorm        7351.300597853552
Optim, LBFGS                 3777   553479.59μs  true 8.582e-08        GradientNorm        7351.300595116978
Speedmapping, acx            5692   370390.41μs  true 9.838e-08         first_order        7351.300594096428

QUARTC: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient   224047 29044161.25μs false 9.873e+04      NotImplemented     1.1179038249853103e9
Optim, LBFGS                 5256   450362.94μs  true 3.976e-08        GradientNorm     3.141408506141678e-8
Speedmapping, acx             108     1752.83μs  true 6.051e-08         first_order     1.783246523878415e-7

BROYDN3DLS: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       78    11634.15μs false 8.520e-03      NotImplemented       1.5327408706544858
Optim, LBFGS                  349    34538.53μs  true 7.350e-08        GradientNorm       0.8809135203572963
Speedmapping, acx             172     3174.82μs  true 9.880e-08         first_order       1.8843361523758562

POWELLSG: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient  2018326106531672.95μs false 1.009e-07           Timed out        4497.865187626896
Optim, LBFGS                  737    56588.07μs  true 3.228e-10        GradientNorm        4497.865187852172
Speedmapping, acx             676     9953.07μs  true 9.933e-08         first_order       4497.8651876829645

TRIDIA: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       38100001121.04μs false 1.560e+05           Timed out     2.4323612030675337e6
Optim, LBFGS                37410  3966964.24μs  true 4.312e-09        GradientNorm   1.2504247601928753e-11
Speedmapping, acx            5882    96146.86μs  true 8.273e-09         first_order     6.73584965686089e-19

GENHUMPS: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient     4835  1657934.01μs false 3.358e-04      NotImplemented    0.0002960074663561432
Optim, LBFGS                21409  3475339.98μs  true 2.864e-08        GradientNorm    1.850448640171182e-12
Speedmapping, acx         1000001 67525753.02μs false 7.048e+00            max_eval       269099.67888903775

INDEF: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       24100001113.18μs false 2.000e+00           Timed out      -2272.6320664204513
Optim, LBFGS               978956103085310.94μs false 7.491e+01           Timed out      -139857.86111806927
Speedmapping, acx         1000000 49495750.42μs false 2.000e+00            max_eval    -1.8391954131900787e8

NONDQUAR: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient   871144172458072.90μs false 9.688e+00           Timed out        4.397610552921495
Optim, LBFGS               260429 18950423.92μs  true 9.654e-08        GradientNorm      6.28428446843377e-8
Speedmapping, acx            8606   137977.17μs  true 9.976e-08         first_order    2.0815560884731196e-6

TQUARTIC: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient     3945   688437.99μs false 4.022e-07      NotImplemented     2.242472157810534e-7
Optim, LBFGS                   74     5412.70μs  true 3.789e-10        GradientNorm    2.243685555776915e-13
Speedmapping, acx            2471    33159.43μs  true 1.831e-11         first_order     0.009998765601424093

ARWHEAD: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       76100003941.06μs false 2.320e-02           Timed out      0.05587805368122645
Optim, LBFGS               447965 20635119.64μs  true 1.218e-12        GradientNorm                      0.0
Speedmapping, acx              15      219.54μs  true 6.128e-13         first_order                      0.0

DQRTIC: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient   224047 29003306.73μs false 9.873e+04      NotImplemented     1.1179038249853103e9
Optim, LBFGS                 5256   450869.11μs  true 3.976e-08        GradientNorm     3.141408506141678e-8
Speedmapping, acx             108     1744.79μs  true 6.051e-08         first_order     1.783246523878415e-7

SINQUAD2: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient    12710  2546195.73μs  true 1.912e-10        GradientNorm     9.878859047624925e-5
Optim, LBFGS                  257    35506.86μs  true 9.036e-10        GradientNorm     9.878859164060274e-5
Speedmapping, acx          201360 10953432.09μs  true 9.981e-08         first_order     9.892309821391777e-5

NONCVXU2: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       17100001286.03μs false 8.878e+04           Timed out    3.2010763090947363e11
Optim, LBFGS               486045100001528.02μs false 8.946e+04           Timed out    1.5585380295302594e11
Speedmapping, acx            1359   103017.24μs  true 5.144e-08         first_order    1.5585358200213586e11

DQDRTIC: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient  2000868111354150.06μs false 9.574e-09           Timed out     2.0095920000060273e6
Optim, LBFGS              1963011120638165.00μs false 9.292e-06           Timed out     2.0095920000000054e6
Speedmapping, acx              26      389.22μs  true 8.017e-12         first_order               2.009592e6

FREUROTH: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       28100001065.97μs false 7.329e+02           Timed out     1.2023707309903316e6
Optim, LBFGS                 7497   734946.07μs  true 2.792e-09        GradientNorm     1.0340021342136848e6
Speedmapping, acx              79     1890.99μs  true 4.230e-08         first_order     1.0340021303137005e6

TOINTGSS: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       41100001050.00μs false 4.796e+00           Timed out       10947.667295869023
Optim, LBFGS               832016100042081.83μs false 2.272e-10           Timed out       10012.085379815657
Speedmapping, acx              54     2293.73μs  true 2.403e-08         first_order       10012.087350627793

MOREBV: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient        5     1526.19μs  true 7.292e-08        GradientNorm   1.0390636363116897e-11
Optim, LBFGS                    7     1563.58μs  true 7.292e-08        GradientNorm   1.0390636361256028e-11
Speedmapping, acx               5      141.84μs  true 6.124e-08         first_order   1.0392835862589593e-11

CRAGGLVY: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient      273100043465.85μs false 5.074e+01           Timed out        7254.314072619096
Optim, LBFGS               691810100321439.03μs false 6.248e-08           Timed out        7116.315347873886
Speedmapping, acx             196     8971.76μs  true 4.460e-08         first_order        7116.315340044162

SPARSINE: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient   563751114094418.05μs false 1.521e-04           Timed out     7.322854644419103e-8
Optim, LBFGS               289531 57853067.66μs  true 9.923e-08        GradientNorm    4.898985944462599e-13
Speedmapping, acx          710852100000142.10μs false 1.027e-02           Timed out      0.11630781072559998

NONCVXUN: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       18100001189.95μs false 1.097e+05           Timed out    3.3044714256111743e11
Optim, LBFGS               728746134469953.06μs false 4.990e-04           Timed out    1.6635775199925623e11
Speedmapping, acx            2876   224276.80μs  true 5.626e-08         first_order    1.6635775209547278e11

ENGVAL1: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient      157100015971.18μs false 1.210e+01           Timed out       10275.859048827235
Optim, LBFGS              1001528 77177646.16μs false 4.062e-10        GradientNorm       10273.234281536443
Speedmapping, acx              23      328.03μs  true 2.241e-08         first_order       10273.234280916571

NONDIA: 5000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient    39260  3297340.35μs false 7.211e-01      NotImplemented        181.0367287216745
Optim, LBFGS                 4030   310064.08μs false       NaN      NotImplemented                      NaN
Speedmapping, acx             838    10197.34μs  true 9.376e-08         first_order    7.640994851423039e-11

NCB20: 5010 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient      161100001183.03μs false 1.361e+01           Timed out      -1083.2872998872535
Optim, LBFGS                 7805  1098363.67μs  true 7.722e-10        GradientNorm      -1454.1038369328999
Speedmapping, acx            9426   543966.60μs  true 9.950e-08         first_order      -1462.6682991069683

FMINSURF: 5625 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       76100010977.03μs false 3.796e-02           Timed out       24.644036510109558
Optim, LBFGS               993711105608663.08μs false 7.291e-11           Timed out       13.051536213295675
Speedmapping, acx           10387   328871.26μs  true 9.178e-08         first_order       13.051536227371225

FMINSRF2: 5625 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient      109100001202.82μs false 3.781e-02           Timed out       24.487831027556844
Optim, LBFGS              2000103141521264.08μs false 1.825e-02           Timed out        21.51253889586533
Speedmapping, acx            8130   254098.06μs  true 9.944e-08         first_order        1.025527052319696

CURLY20: 10000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient   396446100002455.00μs false 6.192e-03           Timed out    -1.0031625410771775e6
Optim, LBFGS               388326 96543788.56μs  true 7.161e-08        GradientNorm    -1.0031369717384004e6
Speedmapping, acx         1000002 89407838.11μs false 3.647e-04            max_eval       -998391.9117119187

CURLY10: 10000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient   410585100002027.99μs false 2.739e-03           Timed out     -1.003162566496468e6
Optim, LBFGS               263225 61484343.37μs  true 7.196e-08        GradientNorm    -1.0031629024131879e6
Speedmapping, acx         1000001 80741528.03μs false 1.425e-05            max_eval    -1.0031629010683455e6

POWER: 10000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient   398610100001941.92μs false 1.125e+10           Timed out    2.5004854019097905e12
Optim, LBFGS               408976100002401.11μs false 1.153e+10           Timed out      2.50869609715225e12
Speedmapping, acx            1089    34107.65μs  true 5.514e-08         first_order    3.766579485528895e-11

CURLY30: 10000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient   364321100002337.93μs false 1.069e-02           Timed out    -1.0031625343885289e6
Optim, LBFGS               460572122928358.08μs false 9.370e-08           Timed out    -1.0031432962932267e6
Speedmapping, acx          948920100000048.16μs false 1.097e-04           Timed out    -1.0031628988579394e6

COSINE: 10000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient   357404100002119.06μs false 7.065e-04           Timed out       -9998.999903124237
Optim, LBFGS               360657100002516.98μs false 2.040e-01           Timed out        -9998.99999536596
Speedmapping, acx         1000002 83159379.01μs false 1.758e+00            max_eval      -5556.4650529965265

DIXON3DQ: 10000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient    30002 10720535.23μs false 5.360e-07      NotImplemented    0.0012284994322227286
Optim, LBFGS               119177 21995017.84μs  true 9.768e-08        GradientNorm    1.2830296117718597e-8
Speedmapping, acx           97680  2931265.23μs  true 9.999e-08         first_order    0.0005025348031645479

SPARSQUR: 10000 parameters, abstol = 1.0e-7.

Solver                    f evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient     3363   818910.83μs  true 9.393e-08        GradientNorm     5.449200283582466e-9
Optim, LBFGS                 2082   453615.62μs  true 5.955e-08        GradientNorm     9.34136768121753e-10
Speedmapping, acx              58     3512.33μs  true 4.907e-08         first_order    4.026025213352197e-10
=#