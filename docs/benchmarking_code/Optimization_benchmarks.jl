# Dire: The goal is to see how each algorithm performs on its own, but some speed gain may be obtained
# by tayloring the ad to each specific problem like in ... Interestingly, the NonlinearSolve, Default PolyAlg. did
# solve all 23 problems.

absolute_path_to_docs = "" # Update

using BenchmarkTools, Optim, JLD2, FileIO, SpeedMapping, ArtificialLandscapes, LinearAlgebra, Logging, LBFGSB

path_plots = absolute_path_to_docs*"assets/"
path_out = absolute_path_to_docs*"benchmarking_code/Output/"

ArtificialLandscapes.check_gradient_indices(gradient, x) = nothing # Very bad! But necessary for LBFGSB to work


include(absolute_path_to_docs * "benchmarking_code/Benchmarking_utils.jl")

Logging.disable_logging(Logging.Warn)

# Solver wrappers
optim_solvers = Dict{AbstractString, Function}()
optim_solvers_constr = Dict{AbstractString, Function}() # With upper bound

function Speedmapping_optim_wrapper(problem, abstol, maps_limit, time_limit, add_upper)
	x0, obj, grad! = problem
	upper = add_upper ? x0 .+ 0.5 : nothing
	res = speedmapping(x0; g! = grad!, f = obj, maps_limit, abstol, pnorm = Inf, time_limit, upper)
	return res.minimizer, res.maps, string(res.status)
end

optim_solvers["Speedmapping, acx"] = (problem, abstol, start_time, maps_limit, time_limit) -> 
	Speedmapping_optim_wrapper(problem, abstol, maps_limit, time_limit, false)

optim_solvers_constr["Speedmapping, acx"] = (problem, abstol, start_time, maps_limit, time_limit) -> 
	Speedmapping_optim_wrapper(problem, abstol, maps_limit, time_limit, true)

function Optim_wrapper(problem, abstol, maps_limit, time_limit, algo)
	x0, obj, grad! = problem
	res = optimize(obj, grad!, x0, algo, Optim.Options(x_abstol = NaN, x_reltol = NaN, 
		f_abstol = NaN, f_reltol = NaN, g_abstol = abstol, g_calls_limit = maps_limit, 
		time_limit = time_limit, iterations = maps_limit))
	return res.minimizer, res.g_calls, string(res.termination_code)
end

optim_solvers["Optim, LBFGS"] = (problem, abstol, start_time, maps_limit, time_limit) -> 
	Optim_wrapper(problem, abstol, maps_limit, time_limit, LBFGS())

optim_solvers["Optim, ConjugateGradient"] = (problem, abstol, start_time, maps_limit, time_limit) -> 
	Optim_wrapper(problem, abstol, maps_limit, time_limit, ConjugateGradient())

# This is needed because optim does not seem to track the total number of gradient evaluations, 
# and LBFGSB does not track evaluation time
function _grad!(grad!, gradient, x, gevals, maps_limit, timesup)
	gevals[] += 1
	if time() < timesup && gevals[] <= maps_limit
		grad!(gradient, x)
	else
		gradient .= 0 # To trick the solver into thinking it has reached the minimum
	end
	return gradient
end

function Optim_wrapper_constr(problem, abstol, maps_limit, time_limit, algo)
	x0, obj, grad! = problem
	upper = x0 .+ 0.5
	lower = -Inf .* ones(length(x0))
	gevals = Ref(0)
	timesup = time() + time_limit
	try
		res = optimize(obj, (gradient, x) -> _grad!(grad!, gradient, x, gevals, maps_limit, timesup), 
			lower, upper, x0, Fminbox(algo), Optim.Options(x_abstol = NaN, 
			x_reltol = NaN, f_abstol = NaN, f_reltol = NaN, g_abstol = abstol, 
			g_calls_limit = maps_limit, time_limit = time_limit, iterations = maps_limit))
		return res.minimizer, gevals[], string(res.termination_code)
	catch e # To catch errors in line search
		return NaN .* ones(length(x0)), 0, sprint(showerror, typeof(e))
	end
end

optim_solvers_constr["Optim, LBFGS"] = (problem, abstol, start_time, maps_limit, time_limit) -> 
	Optim_wrapper_constr(problem, abstol, maps_limit, time_limit, LBFGS())

optim_solvers_constr["Optim, ConjugateGradient"] = (problem, abstol, start_time, maps_limit, time_limit) -> 
	Optim_wrapper_constr(problem, abstol, maps_limit, time_limit, ConjugateGradient())

optimizer = L_BFGS_B(10000, 10) # 10000 is the maximum problem size, 10 is the number of past lags used for LBFGS (same as Optim's default)

global oldstd = stdout
function LBFGSB_wrapper(problem, abstol, maps_limit, time_limit, optimizer)
	x0, obj, grad! = problem
	n = length(x0)
	bounds = Matrix{Float64}(undef, 3, n)
	bounds[1,:] .= 3 # 3 means only upper bound
	bounds[2,:] .= -Inf # The lower bounds
	bounds[3,:] .= x0 .+ 0.5 # The upper bounds
	gevals = Ref(0)
	timesup = time() + time_limit
	redirect_stdout(devnull)
	try
		fout, xout = optimizer(obj, (gradient, x) -> _grad!(grad!, gradient, x, gevals, maps_limit, timesup), x0, 
			bounds, m = 10, factr = 0., pgtol = abstol, iprint=-1, maxfun = maps_limit, maxiter = maps_limit)
		redirect_stdout(oldstd)
		return xout, gevals[], ""
	catch e # To catch errors in line search
		redirect_stdout(oldstd)
		return NaN .* ones(length(x0)), gevals[], sprint(showerror, typeof(e))
	end
end

optim_solvers_constr["LBFGSB"] = (problem, abstol, start_time, maps_limit, time_limit) -> 
	LBFGSB_wrapper(problem, abstol, maps_limit, time_limit, optimizer)

optim_prob_sizes = Dict{String, Int}()
for (name, gen_problem) in landscapes
	optim_prob_sizes[name] = length(gen_problem().x0)
end

order_length = sortperm([l for (name, l) in optim_prob_sizes])
optim_problems_names = [name for (name, l) in optim_prob_sizes][order_length]
optim_solver_names = sort([name for (name, wrapper) in optim_solvers])
optim_solver_constr_names = sort([name for (name, wrapper) in optim_solvers_constr])

function compute_norm(problem, solution)
	gout = similar(solution)
	if sum(_isbad.(solution)) == 0
		problem.grad!(gout, solution)
		last_res = norm(gout, Inf)
	else
		last_res = NaN
	end
end

"""
This function computes the norm of the last gradient. But since there is an upper bound of x0 + 0.5,
when x_i is close to the bound and hase a negative gradient, we conclude that the bound is binding
and set it to zero before computing the norm. The function also outputs the number of binding 
constraints.
"""
function compute_norm_constr(problem, solution)
	gout = similar(solution)
	if sum(_isbad.(solution)) == 0
		problem.grad!(gout, solution)
		n_binding = 0
		for i in eachindex(gout)
			if abs(solution[i] - (problem.x0[i] + 0.5)) < 1e-7 && gout[i] < 0
				gout[i] = 0
				n_binding += 1
			end
		end
		return norm(gout, Inf), n_binding > 0 ? "Binding ($(n_binding))" : "Non-binding"
	else
		NaN, ""
	end
end

res_optim_all = many_problems_many_solvers(landscapes, optim_solvers, optim_problems_names, 
	optim_solver_names,	compute_norm; tunits = 3, F_name = "Grad evals", abstol = 1e-7, 
	time_limit = 100., proper_benchmark = true)

JLD2.@save path_out*"res_optim.jld2" res_optim_all
title = "Performance profiles for non-linear, unconstrained optimization"
perf_profiles(res_optim_all, title, path_plots*"optimization_performance.svg", optim_solver_names; sizef = (640, 480), stat_num = 2, max_fact = 8)

redirect_stdout(oldstd) # To make sure that desired text output is visible
res_all_constr = [Dict{String, Tuple{Float64, Float64}}() for i in eachindex(optim_problems_names)] # Preallocating in case something goes wrong and we want to restart midway ()
res_all_constr = many_problems_many_solvers(landscapes, optim_solvers_constr, optim_problems_names, 
	optim_solver_constr_names, compute_norm_constr; tunits = 3, F_name = "Grad evals", abstol = 1e-7, 
	time_limit = 100., proper_benchmark = true, results = res_all_constr, problem_start = 1)

JLD2.@save path_out*"res_optim_constr.jld2" res_all_constr
title = "Performance profiles for non-linear, box-constriained optimization"
perf_profiles(res_all_constr, title, path_plots*"optimization_constr_performance.svg", 
	optim_solver_constr_names; sizef = (640, 480), stat_num = 2, max_fact = 16)



#= Unconstrained output
CLIFF: 2 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient        149       32.76μs  true 3.712e-08        GradientNorm      0.19978661367773198
Optim, LBFGS                    106       16.55μs  true 2.217e-09        GradientNorm      0.19978661367769956
Speedmapping, acx               552       14.81μs  true 1.175e-09         first_order      0.19978661367769956

BEALE: 2 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient         18        9.08μs  true 1.517e-08        GradientNorm    4.309875521967506e-18
Optim, LBFGS                     33       10.79μs  true 1.358e-10        GradientNorm   3.2288345955802687e-22
Speedmapping, acx                58        2.08μs  true 1.515e-08         first_order   2.4873805095533573e-18

MISRA1BLS: 2 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient        184       54.99μs  true 2.477e-09        GradientNorm      0.07546468153344021
Optim, LBFGS                    158       40.71μs  true 5.146e-08        GradientNorm      0.07546468153341979
Speedmapping, acx            100002     9976.86μs false 5.933e-02      Fevals > limit       7.3170150660192546

Hosaki: 2 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient         12        5.69μs  true 8.225e-08        GradientNorm       -2.345811576101281
Optim, LBFGS                     17        6.28μs  true 1.769e-11        GradientNorm      -2.3458115761012914
Speedmapping, acx                15        1.00μs  true 7.278e-09         first_order       -2.345811576101292

MISRA1ALS: 2 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient        257       84.03μs  true 2.403e-09        GradientNorm      0.12455138894439657
Optim, LBFGS                    189       65.82μs  true 2.753e-08        GradientNorm      0.12455138894440893
Speedmapping, acx            100000     9053.49μs false 6.668e-02            max_eval       19.515847902427694

Six-hump camel: 2 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient          9        4.91μs  true 1.731e-08        GradientNorm      -1.0316284534898774
Optim, LBFGS                     27        8.89μs  true 5.170e-13        GradientNorm      -1.0316284534898774
Speedmapping, acx                26        2.00μs  true 4.941e-10         first_order      -1.0316284534898772

Himmelblau: 2 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient         12        5.63μs  true 2.230e-10        GradientNorm   3.9234349347561203e-22
Optim, LBFGS                     22        7.42μs  true 2.986e-10        GradientNorm     8.81048998674334e-22
Speedmapping, acx                18        1.06μs  true 8.851e-08         first_order   1.7844827490094893e-16

ROSENBR: 2 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient         63       26.77μs  true 2.320e-09        GradientNorm   1.2960162126384444e-18
Optim, LBFGS                     78       23.68μs  true 2.162e-08        GradientNorm    4.435130255645974e-19
Speedmapping, acx                72        2.36μs  true 9.582e-08         first_order   1.4356892863641826e-14

Fletcher-Powell: 3 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient         70       44.70μs  true 9.457e-08        GradientNorm    3.385955053393419e-15
Optim, LBFGS                     65       18.77μs  true 3.646e-10        GradientNorm   2.6185772486779363e-22
Speedmapping, acx                58        2.74μs  true 1.500e-09         first_order    5.626333277455105e-21

Perm 2: 4 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient        116       73.39μs  true 7.727e-08        GradientNorm     0.010883889811628935
Optim, LBFGS                     96       48.99μs  true 1.445e-08        GradientNorm     0.010883889811626983
Speedmapping, acx               146       23.32μs  true 8.265e-08         first_order     0.010883889811629365

Powell: 4 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       4529     1729.90μs  true 9.959e-08        GradientNorm    9.253534937630595e-13
Optim, LBFGS                     88       24.93μs  true 1.170e-08        GradientNorm   1.3862788161000863e-13
Speedmapping, acx               295       11.48μs  true 8.817e-08         first_order   1.1633480105133248e-10

PALMER5D: 4 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient         15        6.72μs  true 3.284e-09        GradientNorm        87.33939952784888
Optim, LBFGS                     16        6.76μs  true 1.717e-09        GradientNorm        87.33939952784839
Speedmapping, acx                45        4.38μs  true 3.961e-09         first_order        87.33939952784901

LANCZOS2LS: 6 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       1502     1681.89μs  true 8.262e-08        GradientNorm     8.462151327901949e-8
Optim, LBFGS                    211      174.00μs  true 1.702e-08        GradientNorm     4.298220881281585e-6
Speedmapping, acx               802      257.20μs  true 7.559e-08         first_order    4.2985607561031815e-6

PALMER5C: 6 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient          8        5.39μs  true 1.046e-11        GradientNorm       2.1280866660550926
Optim, LBFGS                     16        8.52μs  true 2.709e-14        GradientNorm       2.1280866660551125
Speedmapping, acx                28        3.32μs  true 2.988e-08         first_order       2.1280866660551143

LANCZOS1LS: 6 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient        803      866.60μs  true 8.388e-08        GradientNorm      9.65934530197198e-8
Optim, LBFGS                    213      185.08μs  true 2.157e-08        GradientNorm     4.290620207523742e-6
Speedmapping, acx              1402      557.75μs  true 8.871e-08         first_order     4.290642085545953e-6

LANCZOS3LS: 6 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient        512      595.20μs  true 6.796e-08        GradientNorm    1.2691850872962035e-7
Optim, LBFGS                    218      206.32μs  true 1.232e-08        GradientNorm     4.346553278382654e-6
Speedmapping, acx              1535      498.22μs  true 6.674e-08         first_order     4.346854756861299e-6

BIGGS6: 6 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient        462      365.99μs  true 8.487e-08        GradientNorm     0.005655649926147886
Optim, LBFGS                     88       51.37μs  true 1.495e-09        GradientNorm    0.0056556499254999375
Speedmapping, acx               503       96.99μs  true 2.576e-08         first_order     0.005655649927930996

THURBERLS: 7 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient      29324    14197.15μs  true 9.899e-08        GradientNorm         5642.70823966697
Optim, LBFGS                    233       94.09μs  true 2.145e-08        GradientNorm        5642.708239666975
Speedmapping, acx              1180       84.32μs  true 7.383e-08         first_order      3.424145183791597e7

HAHN1LS: 7 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient          2        3.40μs  true 2.065e-09        GradientNorm       55530.953492718145
Optim, LBFGS                      2        4.13μs  true 2.065e-10        GradientNorm       55530.953492718145
Speedmapping, acx                 4        2.15μs  true 2.065e-10         first_order       55530.953492718145

PALMER1D: 7 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient     100000    57221.06μs false 1.540e-04       GradientCalls       0.6526825943754183
Optim, LBFGS                     37       14.33μs  true 5.637e-09        GradientNorm       0.6526825943738854
Speedmapping, acx            100001    13653.04μs false 1.952e-01      Fevals > limit       19.328761238836886

GAUSS2LS: 8 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient     100000  1617916.99μs false 4.631e+01       GradientCalls       1264.6648393658168
Optim, LBFGS                    183     1582.29μs  true 9.300e-10        GradientNorm       1247.5282092309992
Speedmapping, acx            100001   474396.94μs false 5.816e-04      Fevals > limit        1247.528209249839

PALMER6C: 8 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient     100000    86375.56μs false 5.729e-02       GradientCalls       0.4470212755416697
Optim, LBFGS                     44       18.39μs  true 3.432e-08        GradientNorm     0.016387421618639826
Speedmapping, acx            100000    12719.41μs false 1.071e-01            max_eval      0.16854738934010122

PALMER2C: 8 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient     100000    70198.55μs false 2.001e-02       GradientCalls     0.015767187667362924
Optim, LBFGS                     57       20.61μs  true 1.920e-09        GradientNorm      0.01442139119280385
Speedmapping, acx            100001    12785.91μs false 7.813e-02      Fevals > limit       2.2224909387748033

PALMER1C: 8 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient     100000    81547.39μs false 2.531e+01       GradientCalls       0.6791417372916205
Optim, LBFGS                     54       19.43μs  true 3.697e-08        GradientNorm      0.09759799126342045
Speedmapping, acx            100001    12422.08μs false 1.504e+00      Fevals > limit       119.24106442807648

PALMER4C: 8 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient     100000    95283.33μs false 6.384e+00       GradientCalls        1.504720948476262
Optim, LBFGS                     42       14.58μs  true 4.718e-09        GradientNorm      0.05031069582074478
Speedmapping, acx            100002    20270.11μs false 3.083e-02      Fevals > limit       0.7860174678873362

PALMER3C: 8 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient     100000    81088.72μs false 4.852e-02       GradientCalls      0.01988341056398906
Optim, LBFGS                     43       15.86μs  true 6.783e-09        GradientNorm     0.019537638513101023
Speedmapping, acx            100000    12706.72μs false 2.723e-02            max_eval      0.47497349397802585

VIBRBEAM: 8 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient     100005    99724.05μs false 2.757e+00      Fevals > limit       0.3322380326249857
Optim, LBFGS                    267      215.08μs  true 1.945e-08        GradientNorm       1.7488666547254572
Speedmapping, acx            100000    64879.87μs false 1.171e+02            max_eval       36.843256394319674

PALMER8C: 8 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient     100001    78891.04μs false 3.789e-05      Fevals > limit       0.1597700520336323
Optim, LBFGS                     41       14.54μs  true 4.816e-09        GradientNorm      0.15976806347027606
Speedmapping, acx            100000    10687.69μs false 5.741e-03            max_eval       0.6387173932394259

GAUSS3LS: 8 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient     100000  1577167.24μs false 2.331e+02       GradientCalls         1314.81109352708
Optim, LBFGS                    213     1866.39μs  true 3.504e-10        GradientNorm        1244.484636013157
Speedmapping, acx             23200    93060.69μs  true 9.446e-08         first_order       11386.720893857404

PALMER7C: 8 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient     100000    70122.02μs false 1.183e+00       GradientCalls       0.6237200196100231
Optim, LBFGS                     47       15.67μs  true 1.415e-08        GradientNorm       0.6019856723140617
Speedmapping, acx            100001    11139.87μs false 1.651e-02      Fevals > limit        4.736179326907035

GAUSS1LS: 8 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient     100000  1413765.82μs false 2.140e+02       GradientCalls       1346.9302128859188
Optim, LBFGS                    181     1560.17μs  true 8.162e-08        GradientNorm       1315.8222432033774
Speedmapping, acx               232      863.31μs  true 3.870e-09         first_order        52889.24861822945

STRTCHDV: 10 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient         46       72.17μs  true 6.924e-08        GradientNorm    1.234109671927607e-10
Optim, LBFGS                     66       68.88μs  true 6.287e-08        GradientNorm   1.4651569572307154e-11
Speedmapping, acx                82       37.59μs  true 5.704e-08         first_order   1.1722200474526637e-10

HILBERTB: 10 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient          6        4.21μs  true 2.272e-09        GradientNorm    9.947730090945665e-19
Optim, LBFGS                     12        6.47μs  true 2.272e-09        GradientNorm    9.947730090981393e-19
Speedmapping, acx                 8        1.45μs  true 6.584e-08         first_order    9.242278604553933e-16

TRIGON1: 10 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient         63       64.15μs  true 8.440e-08        GradientNorm   1.8713990914681344e-16
Optim, LBFGS                     60       35.68μs  true 3.999e-09        GradientNorm    5.458556857344474e-19
Speedmapping, acx               120       22.70μs  true 7.374e-08         first_order    4.684158834218382e-17

TOINTQOR: 50 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient         59       46.07μs  true 7.451e-08        GradientNorm        1175.472222146169
Optim, LBFGS                     89       49.76μs  true 5.393e-08        GradientNorm       1175.4722221461693
Speedmapping, acx                63       10.09μs  true 1.127e-08         first_order       1175.4722221461693

CHNROSNB: 50 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient        324      300.60μs  true 9.395e-08        GradientNorm    6.160528641801317e-16
Optim, LBFGS                    663      352.98μs  true 7.195e-08        GradientNorm   1.7987068991354112e-16
Speedmapping, acx              1151      155.88μs  true 9.807e-08         first_order     8.81098045138212e-16

DECONVU: 63 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       1560     2540.91μs  true 8.512e-08        GradientNorm    2.787549304857856e-10
Optim, LBFGS                    984     1025.22μs  true 9.771e-08        GradientNorm   1.9037212602361224e-10
Speedmapping, acx             14390     7417.76μs  true 9.832e-08         first_order    1.4399268304258726e-9

LUKSAN13LS: 98 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient        219      231.84μs  true 5.765e-08        GradientNorm       25188.859589645173
Optim, LBFGS                    197      172.45μs  true 5.222e-08        GradientNorm        25188.85958964516
Speedmapping, acx               239       76.14μs  true 7.091e-08         first_order        25188.85958964517

QING: 100 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient         87      118.62μs  true 9.265e-08        GradientNorm    2.104666668791853e-16
Optim, LBFGS                    217      154.06μs  true 7.209e-08        GradientNorm    3.911614815313697e-16
Speedmapping, acx               131       32.29μs  true 7.739e-08         first_order    6.764967580974411e-17

Extended Powell: 100 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       8994    12226.78μs  true 9.726e-08        GradientNorm    5.295052135637918e-12
Optim, LBFGS                   1449     1040.94μs  true 8.808e-08        GradientNorm   1.0691771728067718e-10
Speedmapping, acx               359       79.17μs  true 5.903e-08         first_order     5.45794455581233e-10

Trigonometric: 100 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient         65      237.79μs  true 5.714e-08        GradientNorm     9.204813466115113e-7
Optim, LBFGS                    123      305.84μs  true 9.552e-08        GradientNorm     9.204814954278251e-7
Speedmapping, acx               119      162.15μs  true 1.969e-08         first_order     1.202698700231473e-6

Dixon and Price: 100 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient        168      252.40μs  true 9.149e-08        GradientNorm    2.370175850959668e-16
Optim, LBFGS                   2120     1628.83μs  true 6.772e-08        GradientNorm       0.6666666666666675
Speedmapping, acx               311       87.03μs  true 9.065e-08         first_order   4.1028176218331614e-16

Paraboloid Diagonal: 100 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       2175     2388.22μs  true 8.499e-08        GradientNorm    2.647247956045749e-15
Optim, LBFGS                    260      186.67μs  true 9.046e-08        GradientNorm    6.050273864023003e-15
Speedmapping, acx               231       56.22μs  true 6.028e-08         first_order    4.231972505465712e-16

LUKSAN17LS: 100 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       1208     8408.33μs  true 9.759e-08        GradientNorm       0.4931612903225897
Optim, LBFGS                    699     3803.41μs  true 7.647e-08        GradientNorm       0.4931612903225881
Speedmapping, acx               724     2087.82μs  true 7.909e-08         first_order        0.493161290322595

LUKSAN11LS: 100 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient        998     1604.69μs  true 9.980e-08        GradientNorm   1.8049443361108847e-15
Optim, LBFGS                   2481     2686.13μs  true 1.700e-08        GradientNorm    2.303477682844003e-18
Speedmapping, acx              7876     2673.03μs  true 1.711e-08         first_order   1.5295449296827437e-18

Quadratic Diagonal: 100 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient         60       91.79μs  true 5.247e-08        GradientNorm    9.489851919121134e-16
Optim, LBFGS                    174      132.25μs  true 5.247e-08        GradientNorm    9.489851667227087e-16
Speedmapping, acx               124       27.63μs  true 4.532e-08         first_order    5.493119114651368e-16

LUKSAN21LS: 100 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       1899     3208.89μs  true 7.975e-08        GradientNorm    6.018802172199134e-13
Optim, LBFGS                   1852     1631.00μs  true 8.233e-08        GradientNorm    6.047764343298193e-14
Speedmapping, acx              1788      729.67μs  true 6.120e-08         first_order    8.578302241030856e-14

LUKSAN16LS: 100 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient         65      507.17μs  true 7.514e-08        GradientNorm        3.569697051173904
Optim, LBFGS                    115      762.35μs  true 3.258e-08        GradientNorm       3.5696970511739012
Speedmapping, acx                39      181.33μs  true 5.350e-08         first_order        3.569697051173897

LUKSAN15LS: 100 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient         65     1677.33μs  true 6.445e-08        GradientNorm        3.569697051173904
Optim, LBFGS                    107     2395.29μs  true 1.981e-08        GradientNorm        3.569697051173902
Speedmapping, acx                40      639.37μs  true 3.129e-08         first_order          3.5696970511739

VARDIM: 200 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient         75       99.02μs  true 5.093e-08        GradientNorm    1.621274591698643e-20
Optim, LBFGS                     87       80.77μs  true 8.882e-16        GradientNorm   4.8810768510550105e-30
Speedmapping, acx               100       59.81μs  true 1.308e-08         first_order    2.488047565828395e-21

BROWNAL: 200 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient         90       99.51μs  true 4.178e-09        GradientNorm    5.943142237882882e-16
Optim, LBFGS                     51       45.13μs  true 3.331e-11        GradientNorm   1.4022938869590369e-24
Speedmapping, acx            100001    39390.09μs false 1.136e-06      Fevals > limit    1.3670923614369151e-9

ARGLINB: 200 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient          4100003837.82μs false 4.068e-02           Timed out        99.62546816479403
Optim, LBFGS                     45       98.87μs false 4.280e-04    FailedLinesearch        99.62546816479401
Speedmapping, acx            100001    64981.94μs false 4.736e-04      Fevals > limit          99.625468164794

ARGLINA: 200 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient          2        9.51μs  true 5.773e-15        GradientNorm                    200.0
Optim, LBFGS                      3       15.76μs  true 0.000e+00        GradientNorm                    200.0
Speedmapping, acx                 4        4.24μs  true 0.000e+00         first_order                    200.0

PENALTY2: 200 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       1201     3366.64μs  true 9.835e-08        GradientNorm     4.711627727546475e13
Optim, LBFGS                    539     1386.66μs  true 7.478e-08        GradientNorm     4.711627727546475e13
Speedmapping, acx               343      408.17μs  true 2.339e-08         first_order     4.711627727546475e13

ARGTRIGLS: 200 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient        643     4235.14μs  true 5.980e-08        GradientNorm   1.7206304759555208e-16
Optim, LBFGS                   1925     7856.44μs  true 7.167e-08        GradientNorm    6.535165081315374e-17
Speedmapping, acx              1914     3393.92μs  true 7.871e-08         first_order    8.358151997065141e-17

ARGLINC: 200 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       1693100005306.96μs false 2.403e-04           Timed out       101.12547051442911
Optim, LBFGS                 100024   124197.96μs false 2.403e-04      Fevals > limit       101.12547051442911
Speedmapping, acx            100001    69016.93μs false 3.959e-04      Fevals > limit       101.12547051442908

PENALTY3: 200 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient        109      258.78μs  true 6.656e-08        GradientNorm    4.627051627677082e-19
Optim, LBFGS                    206      316.29μs  true 5.245e-08        GradientNorm   4.1104417865830587e-19
Speedmapping, acx               179      136.25μs  true 8.147e-09         first_order     1.697129067354959e-9

Large-Scale Quadratic: 250 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient          2       12.87μs  true 0.000e+00        GradientNorm                      0.0
Optim, LBFGS                      3       21.56μs  true 0.000e+00        GradientNorm                      0.0
Speedmapping, acx                 4        5.85μs  true 0.000e+00         first_order                      0.0

OSCIPATH: 500 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient         19       63.41μs  true 6.187e-08        GradientNorm       0.9999666655201666
Optim, LBFGS                     38       80.14μs  true 5.961e-08        GradientNorm       0.9999666655201664
Speedmapping, acx                15       16.51μs  true 1.571e-08         first_order       0.9999666655201663

GENROSE: 500 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       1126     5076.67μs  true 5.980e-08        GradientNorm   2.3527656788743175e-17
Optim, LBFGS                   2669     6539.87μs  true 6.474e-08        GradientNorm    4.404295964413134e-17
Speedmapping, acx              3679     3461.64μs  true 4.944e-08         first_order   5.4654824087189934e-17

INTEQNELS: 502 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient          8       66.09μs  true 8.746e-09        GradientNorm   1.6540332786000442e-15
Optim, LBFGS                     14       75.86μs  true 7.600e-08        GradientNorm   1.0975417944411849e-13
Speedmapping, acx                10       33.27μs  true 5.466e-08         first_order    6.002802406342395e-14

EXTROSNB: 1000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient      38267   333693.55μs  true 9.839e-08        GradientNorm     3.170418262028877e-8
Optim, LBFGS                 100000   450883.79μs false 3.155e-07       GradientCalls     5.66405611202721e-12
Speedmapping, acx            100002   240025.04μs false 7.321e-07      Fevals > limit    3.6843695052118047e-7

PENALTY1: 1000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient         76      425.36μs  true 9.627e-08        GradientNorm     0.009686176339125207
Optim, LBFGS                    189      717.58μs  true 7.396e-08        GradientNorm     0.009686175434613111
Speedmapping, acx                98      206.65μs  true 2.307e-08         first_order     0.009686175495025356

EG2: 1000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient         12      162.43μs  true 1.011e-13        GradientNorm       -998.9473933009451
Optim, LBFGS                     15      183.50μs  true 1.054e-11        GradientNorm       -998.9473933009451
Speedmapping, acx                10       82.59μs  true 3.422e-13         first_order       -998.9473933009451

FLETCHCR: 1000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       4440    36571.95μs  true 9.848e-08        GradientNorm    6.040217013039128e-15
Optim, LBFGS                  11352    51305.38μs  true 8.919e-08        GradientNorm    2.697949909329217e-16
Speedmapping, acx             45700    78560.64μs  true 4.030e-08         first_order   2.1340529440319674e-15

MSQRTBLS: 1024 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       3006    63555.17μs  true 9.911e-08        GradientNorm   1.9787257587323126e-12
Optim, LBFGS                   8015   109935.92μs  true 9.512e-08        GradientNorm     1.45094815360375e-12
Speedmapping, acx              5885    53661.91μs  true 5.444e-08         first_order    6.027391052204702e-11

MSQRTALS: 1024 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       3911    83800.35μs  true 9.181e-08        GradientNorm     2.41276642878585e-12
Optim, LBFGS                  10895   153935.79μs  true 9.105e-08        GradientNorm   2.1991833537974746e-12
Speedmapping, acx             13554   123679.03μs  true 9.949e-08         first_order    1.1252088716011202e-9

EDENSCH: 2000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient         51      580.72μs  true 9.625e-08        GradientNorm       12003.284592020766
Optim, LBFGS                     71      584.53μs  true 3.654e-08        GradientNorm       12003.284592020764
Speedmapping, acx                52      201.14μs  true 3.410e-08         first_order       12003.284592020764

DIXMAANK: 3000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       2780   233545.43μs  true 9.551e-08        GradientNorm       1.0000000002725384
Optim, LBFGS                   8559   423780.41μs  true 8.498e-08        GradientNorm        1.000000000215962
Speedmapping, acx             11567   272348.77μs  true 9.585e-08         first_order       1.0000000206813466

DIXMAANH: 3000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient        189    13169.68μs  true 9.546e-08        GradientNorm       1.0000000000054472
Optim, LBFGS                    711    28675.78μs  true 9.977e-08        GradientNorm       1.0000000000088836
Speedmapping, acx               404     7232.70μs  true 8.937e-08         first_order       1.0000000000010638

DIXMAANI1: 3000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       4798   395937.61μs  true 7.295e-08        GradientNorm       1.0000000006594718
Optim, LBFGS                  14227   754112.76μs  true 8.477e-08        GradientNorm       1.0000000000089686
Speedmapping, acx             17393   409630.17μs  true 9.756e-08         first_order         1.00000002150479

DIXMAANB: 3000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient         11      465.19μs  true 1.562e-08        GradientNorm       1.0000000000000056
Optim, LBFGS                     22      545.53μs  true 7.265e-08        GradientNorm       1.0000000000000036
Speedmapping, acx                18      232.41μs  true 7.246e-09         first_order       1.0000000000000027

DIXMAAND: 3000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient         14      572.61μs  true 6.504e-08        GradientNorm       1.0000000000000029
Optim, LBFGS                     28      691.34μs  true 3.194e-08        GradientNorm       1.0000000000000018
Speedmapping, acx                20      250.32μs  true 3.528e-08         first_order       1.0000000000000253

DIXMAANN: 3000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       3601   399996.14μs  true 9.877e-08        GradientNorm        1.000000000506924
Optim, LBFGS                  11012   730986.72μs  true 9.153e-08        GradientNorm        1.000000000098451
Speedmapping, acx             16249   520220.67μs  true 9.922e-08         first_order       1.0000000221488081

DIXMAANJ: 3000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       2846   235943.80μs  true 8.435e-08        GradientNorm        1.000000000210818
Optim, LBFGS                   8592   431467.14μs  true 9.891e-08        GradientNorm       1.0000000001107434
Speedmapping, acx             17283   408070.15μs  true 9.643e-08         first_order       1.0000000209214888

DIXMAANM1: 3000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       4748   523029.22μs  true 9.599e-08        GradientNorm       1.0000000000412754
Optim, LBFGS                  14275   954724.53μs  true 9.228e-08        GradientNorm       1.0000000000033318
Speedmapping, acx             13365   426740.22μs  true 9.990e-08         first_order       1.0000000210168385

DIXMAANF: 3000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient        199    13885.10μs  true 9.821e-08        GradientNorm        1.000000000010803
Optim, LBFGS                    702    28271.12μs  true 9.955e-08        GradientNorm         1.00000000000579
Speedmapping, acx               471     8482.97μs  true 9.100e-08         first_order       1.0000000000008114

DIXMAANL: 3000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       2755   227986.99μs  true 9.337e-08        GradientNorm       1.0000000002232792
Optim, LBFGS                   8447   415940.93μs  true 9.822e-08        GradientNorm        1.000000000230228
Speedmapping, acx              9423   214594.83μs  true 8.919e-08         first_order       1.0000000178987185

DIXMAANC: 3000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient         13      532.45μs  true 1.744e-08        GradientNorm       1.0000000000000036
Optim, LBFGS                     23      564.46μs  true 4.768e-08        GradientNorm       1.0000000000000306
Speedmapping, acx                20      250.33μs  true 2.212e-09         first_order       1.0000000000000002

DIXMAANO: 3000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       3533   391155.95μs  true 8.994e-08        GradientNorm       1.0000000004242626
Optim, LBFGS                  10734   717473.81μs  true 9.892e-08        GradientNorm        1.000000000093563
Speedmapping, acx             11952   403173.56μs  true 8.005e-08         first_order       1.0000000109697404

DIXMAANP: 3000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       3464   387639.95μs  true 8.065e-08        GradientNorm       1.0000000004208596
Optim, LBFGS                  10446   748916.06μs  true 9.947e-08        GradientNorm       1.0000000001095768
Speedmapping, acx             15702   516449.13μs  true 8.325e-08         first_order       1.0000000148915698

DIXMAANA1: 3000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient          8      334.23μs  true 5.255e-08        GradientNorm       1.0000000000012823
Optim, LBFGS                     20      487.90μs  true 1.038e-12        GradientNorm                      1.0
Speedmapping, acx                12      177.48μs  true 1.235e-09         first_order       1.0000000000000007

DIXMAANE1: 3000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient        256    17695.97μs  true 9.810e-08        GradientNorm       1.0000000000043696
Optim, LBFGS                    767    30791.80μs  true 9.317e-08        GradientNorm         1.00000000000546
Speedmapping, acx               455     8053.29μs  true 2.301e-08         first_order       1.0000000000000604

DIXMAANG: 3000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient        199    13679.55μs  true 9.897e-08        GradientNorm        1.000000000006234
Optim, LBFGS                    699    28052.78μs  true 9.918e-08        GradientNorm        1.000000000006425
Speedmapping, acx               488     8652.92μs  true 9.930e-08         first_order       1.0000000000050482

WOODS: 4000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient        174     5052.35μs  true 3.025e-09        GradientNorm   2.8402425208649075e-16
Optim, LBFGS                     62     1252.22μs  true 1.733e-10        GradientNorm   1.4220758290748901e-19
Speedmapping, acx               500     3534.78μs  true 1.458e-08         first_order    1.868313346760708e-16

LIARWHD: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient         38     1151.29μs  true 4.232e-12        GradientNorm    5.468532322709365e-25
Optim, LBFGS                     45     1124.18μs  true 8.882e-15        GradientNorm    9.898232208260646e-28
Speedmapping, acx               127     1846.53μs  true 5.724e-08         first_order   1.0216009627627096e-12

BDQRTIC: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient        829    25788.41μs  true 5.804e-08        GradientNorm       20006.256878433673
Optim, LBFGS                    151     4660.53μs  true 6.315e-08        GradientNorm       20006.256878433676
Speedmapping, acx               250     4423.81μs  true 6.930e-08         first_order       20006.256878433673

SCHMVETT: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient        131    19925.40μs  true 8.752e-08        GradientNorm                 -14994.0
Optim, LBFGS                    146    17406.98μs  true 6.234e-08        GradientNorm                 -14994.0
Speedmapping, acx                82     5152.75μs  true 9.962e-08         first_order                 -14994.0

SROSENBR: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient         18      530.11μs  true 5.869e-10        GradientNorm    9.059112427803899e-19
Optim, LBFGS                     26      494.01μs  true 3.119e-08        GradientNorm    1.516428751618877e-15
Speedmapping, acx                31      306.68μs  true 7.017e-12         first_order   1.9234108169266209e-19

NCB20B: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       7941   746355.79μs  true 8.622e-08        GradientNorm        7351.300597853566
Optim, LBFGS                   4617   395788.53μs  true 9.810e-08        GradientNorm        7351.300597853562
Speedmapping, acx              6765   382950.18μs  true 9.437e-08         first_order        7351.300594032889

QUARTC: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient         37     1558.22μs  true 9.662e-08        GradientNorm    3.4907174167938507e-7
Optim, LBFGS                    176     4391.74μs  true 8.911e-08        GradientNorm    6.259790949464199e-10
Speedmapping, acx               108     1248.05μs  true 7.394e-08         first_order    2.3161980281156326e-7

BROYDN3DLS: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient         57     2387.27μs  true 9.721e-08        GradientNorm       1.5299856363042652
Optim, LBFGS                    104     2726.04μs  true 6.469e-08        GradientNorm   3.1444820485428944e-16
Speedmapping, acx               183     2468.17μs  true 2.459e-08         first_order       1.8843361523758428
Speedmapping, acx               183     2468.17μs  true 2.459e-08         first_order       1.8843361523758428

POWELLSG: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       4529   157812.13μs  true 9.959e-08        GradientNorm    1.1566908242938565e-9
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       4529   157812.13μs  true 9.959e-08        GradientNorm    1.1566908242938565e-9
Optim, LBFGS                     88     1708.46μs  true 1.170e-08        GradientNorm   1.7328484548664498e-10
Speedmapping, acx               290     2833.36μs  true 9.542e-08         first_order    1.3581133882322476e-7


TRIDIA: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient        825    39522.60μs  true 9.848e-08        GradientNorm   5.5027248627020967e-17
Optim, LBFGS                   2467    64673.52μs  true 9.899e-08        GradientNorm     4.74221059319102e-17
Speedmapping, acx              4964    51940.24μs  true 7.188e-08         first_order   1.5415716420764567e-16

GENHUMPS: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       6287  1204474.16μs  true 4.171e-10        GradientNorm    2.360253068924752e-18
Optim, LBFGS                  26597  2867193.49μs  true 2.011e-08        GradientNorm   1.3252402556366641e-14
Speedmapping, acx            100001  5867790.94μs false 3.436e+00      Fevals > limit         42465.3484667014

INDEF: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient         52     9220.22μs false 1.841e+00    FailedLinesearch       4603.2873795320465
Optim, LBFGS                     51     9145.59μs false 1.841e+00    FailedLinesearch       4603.2873795320465
Speedmapping, acx            100001  4445693.02μs false 1.000e+00      Fevals > limit    -2.8766537949899514e6

NONDQUAR: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       8769   304771.81μs  true 5.666e-08        GradientNorm     3.698819809375723e-7
Optim, LBFGS                  33507   739501.51μs  true 9.169e-08        GradientNorm      4.12431330236748e-8
Speedmapping, acx              6284    67713.10μs  true 9.902e-08         first_order     1.983490993049976e-6

TQUARTIC: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient         33      670.73μs  true 2.141e-08        GradientNorm    6.638910721515401e-19
Optim, LBFGS                     36      651.97μs  true 1.310e-14        GradientNorm   4.2906637673036595e-29
Speedmapping, acx              6907    59118.49μs  true 9.998e-08         first_order     6.246858628052582e-8

ARWHEAD: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient         26      473.21μs  true 1.813e-10        GradientNorm                      0.0
Optim, LBFGS                     22      374.36μs  true 3.463e-09        GradientNorm                      0.0
Speedmapping, acx                20      194.07μs  true 1.152e-09         first_order                      0.0

DQRTIC: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient         37     1554.60μs  true 9.662e-08        GradientNorm    3.4907174167938507e-7
Optim, LBFGS                    176     4317.09μs  true 8.911e-08        GradientNorm    6.259790949464199e-10
Speedmapping, acx               108     1259.40μs  true 7.394e-08         first_order    2.3161980281156326e-7

SINQUAD2: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       1248   110837.13μs  true 7.238e-09        GradientNorm    2.3585287568990307e-8
Optim, LBFGS                    791    66677.79μs  true 8.550e-08        GradientNorm    2.155091992268146e-15
Speedmapping, acx            100001  5030457.97μs false 2.461e-07      Fevals > limit     4.731395141815426e-5

NONCVXU2: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient      18720  2981283.33μs  true 9.988e-08        GradientNorm       11585.316017278245
Optim, LBFGS                  23228  3045301.24μs  true 9.913e-08        GradientNorm       11584.233364477728
Speedmapping, acx            100001  6097685.10μs false 4.826e-07      Fevals > limit       11585.104203895089

DQDRTIC: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient          7      250.15μs  true 2.122e-10        GradientNorm   3.7068202802538254e-22
Optim, LBFGS                     16      309.79μs  true 1.271e-11        GradientNorm     8.49740404736296e-25
Speedmapping, acx                18      188.27μs  true 2.127e-10         first_order   2.0802588095422822e-22

FREUROTH: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient        198     9072.89μs  true 7.417e-08        GradientNorm        608159.1890463276
Optim, LBFGS                     62     1744.10μs  true 5.731e-08        GradientNorm        608159.1890463271
Speedmapping, acx                95     1818.38μs  true 2.806e-08         first_order        608159.1890463276

TOINTGSS: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient          6      631.69μs  true 3.466e-09        GradientNorm       10.002000800319914
Optim, LBFGS                     12      855.69μs  true 2.138e-09        GradientNorm       10.002000800319914
Speedmapping, acx                63     2473.67μs  true 2.165e-09         first_order       10.002000800319914

MOREBV: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient          5      236.77μs  true 7.292e-08        GradientNorm   1.0390636358892595e-11
Optim, LBFGS                      9      306.92μs  true 7.292e-08        GradientNorm   1.0390636358892632e-11
Speedmapping, acx                 5      103.74μs  true 6.124e-08         first_order   1.0392835862589593e-11

CRAGGLVY: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient        196    19862.72μs  true 9.567e-08        GradientNorm        1688.215309714459
Optim, LBFGS                    305    23327.43μs  true 6.958e-08        GradientNorm       1688.2153097144592
Speedmapping, acx               197     7961.21μs  true 9.878e-08         first_order        1688.215309714459

SPARSINE: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient     100000 27985659.26μs false 1.235e-03       GradientCalls     2.253121477617587e-7
Optim, LBFGS                 100000 18796464.11μs false 1.181e-02       GradientCalls     0.001067678403754292
Speedmapping, acx            100001 11810212.85μs false 2.110e-02      Fevals > limit       0.7881167950352594

NONCVXUN: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient     100000 19836432.79μs false 6.255e-04       GradientCalls       11601.872938242754
Optim, LBFGS                 100000 12754757.64μs false 2.464e-03       GradientCalls       11592.670458388875
Speedmapping, acx            100000  5986092.36μs false 3.489e-05            max_eval       11601.953005858082

ENGVAL1: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient         60     1430.83μs  true 5.941e-08        GradientNorm        5548.668419415788
Optim, LBFGS                     54     1126.09μs  true 5.621e-08        GradientNorm        5548.668419415788
Speedmapping, acx                36      338.21μs  true 2.988e-08         first_order        5548.668419415788

NONDIA: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient         14      339.23μs  true 3.304e-09        GradientNorm   1.0164104562835172e-13
Optim, LBFGS                     29      423.19μs  true 1.111e-10        GradientNorm    6.191633659611637e-27
Speedmapping, acx              7575    61509.19μs  true 9.544e-08         first_order   1.4086961799976045e-10

NCB20: 5010 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient       6811   495858.23μs  true 7.451e-08        GradientNorm      -1092.1282194521664
Optim, LBFGS                    994    71810.45μs  true 6.559e-08        GradientNorm      -1179.9435222805587
Speedmapping, acx              3707   203402.61μs  true 9.977e-08         first_order      -1462.6682995664705

FMINSURF: 5625 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient        591    42923.01μs  true 9.845e-08        GradientNorm        1.000000000029774
Optim, LBFGS                   1421    63386.48μs  true 9.284e-08        GradientNorm       1.0000000000452787
Speedmapping, acx              1106    27885.58μs  true 8.838e-08         first_order       1.0000000030982978

FMINSRF2: 5625 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient        652    46153.89μs  true 9.663e-08        GradientNorm       1.0000000000260174
Optim, LBFGS                   1647    71313.03μs  true 9.487e-08        GradientNorm       1.0000000000125107
Speedmapping, acx               922    22958.73μs  true 9.412e-08         first_order       1.0000240798224609

CURLY20: 10000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient     100000 12004205.34μs false 3.604e-04       GradientCalls    -1.0031628964755975e6
Optim, LBFGS                 100001 11118072.99μs false 1.102e-03      Fevals > limit     -1.003162902334507e6
Speedmapping, acx            100001  7532691.96μs false 1.220e-03      Fevals > limit       -996864.5608599131

CURLY10: 10000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient     100001 10922296.05μs false 8.921e-05      Fevals > limit    -1.0031629001543493e6
Optim, LBFGS                 100000 10241091.11μs false 5.139e-05       GradientCalls    -1.0031629024068014e6
Speedmapping, acx            100001  6869515.90μs false 1.285e-04      Fevals > limit    -1.0031628459506547e6

POWER: 10000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient        443    39341.21μs  true 9.594e-08        GradientNorm    7.485034908869639e-11
Optim, LBFGS                   1566    82108.14μs  true 9.617e-08        GradientNorm    7.510857984786999e-11
Speedmapping, acx              1577    32680.18μs  true 8.088e-08         first_order    5.905735078107576e-11

CURLY30: 10000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient     100001 14132980.11μs false 5.775e-04      Fevals > limit    -1.0031628952424236e6
Optim, LBFGS                 100002 12980638.98μs false 1.361e-03      Fevals > limit    -1.0031629022904477e6
Speedmapping, acx            100002  9187541.96μs false 1.098e-03      Fevals > limit       -997056.3651336201

COSINE: 10000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient         24     4369.71μs  true 3.330e-08        GradientNorm                  -9999.0
Optim, LBFGS                     35     5310.23μs  true 3.071e-08        GradientNorm                  -9999.0
Speedmapping, acx            100002  8509210.82μs false 1.703e-04      Fevals > limit        -9994.40199192553

DIXON3DQ: 10000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient      10002   835019.55μs  true 8.791e-08        GradientNorm   1.3293951876221227e-12
Optim, LBFGS                  30004  1619652.27μs  true 3.806e-08        GradientNorm   1.3453068856809803e-13
Speedmapping, acx             53034  1109277.24μs  true 9.993e-08         first_order    0.0005042620860168958

SPARSQUR: 10000 parameters, abstol = 1.0e-7.
Solver                   Grad evals          time  conv   |resid|                 log                      obj
Optim, ConjugateGradient         26     3730.77μs  true 8.631e-08        GradientNorm    8.533130180391535e-10
Optim, LBFGS                    125    12260.35μs  true 6.919e-08        GradientNorm    1.610856536906379e-11
Speedmapping, acx                74     3651.54μs  true 3.212e-08         first_order   2.2878208232927505e-10

=#

#= Constrained output

CLIFF: 2 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           35        132.26μs  true 3.968e-09                     Binding (1)       0.2011186945926685
Optim, ConjugateGradient        836        319.20μs  true 8.957e-09        GradientNorm Binding (1)       0.2011186945926734
Optim, LBFGS                    271         93.02μs  true 8.920e-13        GradientNorm Binding (1)       0.2011186945975201
Speedmapping, acx             11590        373.85μs  true 2.457e-09         first_order Binding (1)      0.20111869459266846

BEALE: 2 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           10         66.06μs  true 1.247e-10                     Binding (1)                 1.828125
Optim, ConjugateGradient          0      42066.10μs false       NaN                 AssertionError                       NaN
Optim, LBFGS                 100075      33593.89μs false 3.750e+00      Fevals > limit Non-binding       1.8281268739045795
Speedmapping, acx            100000       3426.46μs false 2.822e-04            max_eval Non-binding       0.4727624236765404

MISRA1BLS: 2 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           92        341.53μs false 3.279e-07                     Non-binding       0.0754646815763385
Optim, ConjugateGradient        192        120.64μs  true 8.461e-11        GradientNorm Non-binding        6761.787892857144
Optim, LBFGS                 100585      28070.93μs false 2.125e+17      Fevals > limit Non-binding        58.23834279127181
Speedmapping, acx            100002      10252.00μs false 5.933e-02      Fevals > limit Non-binding       7.3170150660192546

Hosaki: 2 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            8         56.57μs  true 4.407e-10                     Non-binding       -2.345811576101292
Optim, ConjugateGradient         25         19.03μs  true 7.152e-08        GradientNorm Non-binding       -2.345811576101286
Optim, LBFGS                     25         11.63μs  true 2.191e-09        GradientNorm Non-binding      -2.3458115761012914
Speedmapping, acx                18          1.14μs  true 7.268e-09         first_order Non-binding       -2.345811576101292

MISRA1ALS: 2 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           25         58.66μs false 6.668e-02                     Non-binding       19.515934144011414
Optim, ConjugateGradient        396        207.57μs  true 7.716e-08        GradientNorm Non-binding      0.12455138894440401
Optim, LBFGS                    260        119.91μs  true 7.697e-08        GradientNorm Non-binding      0.12455138894439793
Speedmapping, acx            100000       9443.52μs false 6.668e-02            max_eval Non-binding       19.515847902427694

Six-hump camel: 2 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           14         71.44μs  true 1.032e-08                     Non-binding      -1.0316284534898774
Optim, ConjugateGradient         15         16.10μs  true 5.861e-08        GradientNorm Non-binding       -1.031628453489877
Optim, LBFGS                     31         14.39μs  true 7.921e-10        GradientNorm Non-binding      -1.0316284534898774
Speedmapping, acx                26          1.37μs  true 4.941e-10         first_order Non-binding      -1.0316284534898772

Himmelblau: 2 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            7         53.72μs  true 2.678e-09                     Binding (1)        6.566362580202338
Optim, ConjugateGradient     100175      39236.07μs false 2.390e+01      Fevals > limit Non-binding       6.5663822795360725
Optim, LBFGS                 100125      36538.84μs false 2.390e+01      Fevals > limit Non-binding       6.5663757131542795
Speedmapping, acx                31          1.64μs  true 2.792e-09         first_order Binding (1)        6.566362580202338

ROSENBR: 2 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           14         71.68μs  true 0.000e+00                     Binding (1)       2.8899999999999997
Optim, ConjugateGradient        202        106.44μs  true 4.186e-09        GradientNorm Binding (1)          2.8900000000759
Optim, LBFGS                    130         70.69μs  true 8.054e-11        GradientNorm Binding (1)       2.8900000000759003
Speedmapping, acx                15          1.06μs  true 1.140e-11         first_order Binding (1)       2.8899999999999997

Fletcher-Powell: 3 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            4         43.07μs  true 0.000e+00                     Binding (3)       1065.0786437626905
Optim, ConjugateGradient         38      90346.84μs false 7.835e+02          Iterations Binding (1)         1538.47196260091
Optim, LBFGS                      0      57107.21μs false       NaN                 AssertionError                       NaN
Speedmapping, acx                 4          0.56μs  true 0.000e+00         first_order Binding (3)       1065.0786437626905

Perm 2: 4 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            2         42.94μs  true 0.000e+00                     Binding (4)       2954.4095775555993
Optim, ConjugateGradient        105        103.40μs  true 0.000e+00        GradientNorm Binding (4)       2954.4096017523198
Optim, LBFGS                    142        109.56μs  true 0.000e+00        GradientNorm Binding (4)       2954.4096017523198
Speedmapping, acx               103         14.43μs  true 0.000e+00         first_order Binding (4)       2954.4095775555993

Powell: 4 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           16         93.56μs  true 3.031e-09                     Binding (1)        16.60283817426926
Optim, ConjugateGradient         38      96385.72μs false 8.087e+01          Iterations Binding (1)        35.98654008187128
Optim, LBFGS                      0      53663.02μs false       NaN                 AssertionError                       NaN
Speedmapping, acx                39          2.30μs  true 7.781e-09         first_order Binding (1)       16.602838174269262

PALMER5D: 4 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            1     148755.07μs false       NaN              DimensionMismatch                       NaN
Optim, ConjugateGradient          0      55345.06μs false       NaN                 AssertionError                       NaN
Optim, LBFGS                    226        153.08μs  true 0.000e+00        GradientNorm Binding (4)       20842.640588235965
Speedmapping, acx               103          7.55μs  true 0.000e+00         first_order Binding (4)       20842.640586552014

LANCZOS2LS: 6 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                          124        681.78μs  true 2.976e-08                     Non-binding    3.6093925243379834e-9
Optim, ConjugateGradient         69     106338.12μs false 1.219e+01          Iterations Binding (1)       39.269941246059645
Optim, LBFGS                 100192      96652.03μs false 1.592e+22      Fevals > limit Non-binding     0.014627758863282447
Speedmapping, acx              1650        548.69μs  true 7.175e-08         first_order Non-binding     4.560242392550631e-9

PALMER5C: 6 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            1     117350.10μs false       NaN              DimensionMismatch                       NaN
Optim, ConjugateGradient         40     105205.43μs false 4.943e+02          Iterations Binding (1)       24722.808992271785
Optim, LBFGS                 101121      35301.92μs false 1.649e-07      Fevals > limit Binding (4)        18693.74315180639
Speedmapping, acx                18          2.64μs  true 2.387e-10         first_order Binding (4)       18693.743151806393

LANCZOS1LS: 6 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                          120        652.58μs  true 4.575e-08                     Non-binding    3.3612519818593638e-9
Optim, ConjugateGradient         66     111356.11μs false 1.316e+01          Iterations Binding (1)        43.46135908647855
Optim, LBFGS                 100001      97802.16μs false 2.178e+16      Fevals > limit Non-binding      0.01482964212542771
Speedmapping, acx              3190       1066.65μs  true 2.976e-08         first_order Non-binding    3.6388288178459535e-9

LANCZOS3LS: 6 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                          125        669.65μs  true 2.751e-08                     Non-binding    1.7985960409766386e-8
Optim, ConjugateGradient         81     107379.22μs false 2.191e+01          Iterations Binding (1)         28.9093161316659
Optim, LBFGS                 100001      80912.11μs false 5.878e+17      Fevals > limit Non-binding     0.014767757748946666
Speedmapping, acx              1564        535.69μs  true 9.782e-08         first_order Non-binding     2.028150767924237e-8

BIGGS6: 6 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           22        134.12μs  true 3.000e-08                     Binding (3)      0.26207727502009803
Optim, ConjugateGradient        125     115551.07μs false 4.598e-02          Iterations Binding (2)      0.26251293614253374
Optim, LBFGS                    361        278.50μs  true 2.354e-12        GradientNorm Binding (3)       0.2620772750213176
Speedmapping, acx                42          9.93μs  true 1.996e-10         first_order Binding (3)       0.2620772750200983

THURBERLS: 7 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                          106        516.86μs false 3.425e-03                     Binding (2)        863971.4796332164
Optim, ConjugateGradient     200020     259964.94μs false 1.368e+10      Fevals > limit Non-binding        977417.8496138378
Optim, LBFGS                    178        120.56μs false 5.944e+03        GradientNorm Non-binding        836682.1586928166
Speedmapping, acx              2540        221.08μs  true 3.694e-08         first_order Non-binding      3.415935058766809e7

HAHN1LS: 7 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                         1703       8250.70μs false 2.422e+06                     Non-binding        7.843564997696817
Optim, ConjugateGradient         72        141.14μs  true 1.678e-13        GradientNorm Non-binding        49036.25159562894
Optim, LBFGS                     73         86.18μs false 1.659e+07        GradientNorm Non-binding         52924.2189663368
Speedmapping, acx               625        135.35μs  true 8.119e-08         first_order Non-binding       55350.766351989754

PALMER1D: 7 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            1     119497.06μs false       NaN              DimensionMismatch                       NaN
Optim, ConjugateGradient     163379     117610.93μs false 5.673e+04      Fevals > limit Binding (1)       37924.005023632075
Optim, LBFGS                 102892      49073.93μs false 3.342e+03      Fevals > limit Binding (4)       37873.637641005844
Speedmapping, acx             16149       2412.42μs  true 9.663e-08         first_order Binding (4)        37871.93324826802

GAUSS2LS: 8 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                          132       1656.98μs false 3.033e-03                     Binding (6)        6715.834802895376
Optim, ConjugateGradient      80255    1276399.11μs false 7.548e-03          Iterations Binding (6)        6715.834802895754
Optim, LBFGS                      0     947681.90μs false       NaN                 AssertionError                       NaN
Speedmapping, acx            100000     353469.54μs false 1.512e-05            max_eval Binding (6)        6715.834802895378

PALMER6C: 8 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            1     113458.16μs false       NaN              DimensionMismatch                       NaN
Optim, ConjugateGradient       3996     118320.67μs false 2.641e+00          Iterations Binding (1)       172.73468769290218
Optim, LBFGS                   7326       3792.94μs  true 2.847e-09        GradientNorm Binding (5)       172.35915721618892
Speedmapping, acx               236         30.64μs  true 3.585e-08         first_order Binding (5)        172.3591572156609

PALMER2C: 8 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            1     110080.00μs false       NaN              DimensionMismatch                       NaN
Optim, ConjugateGradient      21677     140803.85μs false 2.244e+02          Iterations Binding (1)       1230.3374476261508
Optim, LBFGS                   9676       5402.56μs  true 4.121e-09        GradientNorm Binding (4)        1222.490615242839
Speedmapping, acx            100001      12831.93μs false 2.496e-05      Fevals > limit Binding (4)        1222.490615230047

PALMER1C: 8 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            1     119463.92μs false       NaN              DimensionMismatch                       NaN
Optim, ConjugateGradient      56815     163159.30μs false 4.341e+01          Iterations Binding (2)        37847.85348717804
Optim, LBFGS                 102788      45243.98μs false 3.283e+04      Fevals > limit Binding (2)        37849.73544483939
Speedmapping, acx            100000      14245.18μs false 2.839e-02            max_eval Binding (4)       37847.828164026796

PALMER4C: 8 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            1     114316.94μs false       NaN              DimensionMismatch                       NaN
Optim, ConjugateGradient       9049     119427.79μs false 8.232e+01          Iterations Binding (1)       202.53192263996823
Optim, LBFGS                 100820      53142.07μs false 4.008e+00      Fevals > limit Binding (5)       199.69552869517176
Speedmapping, acx               416         58.37μs  true 6.611e-08         first_order Binding (5)       199.69369803679118

PALMER3C: 8 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            1     118958.00μs false       NaN              DimensionMismatch                       NaN
Optim, ConjugateGradient      55295     172797.66μs false 7.539e-06          Iterations Binding (4)       217.26371118170908
Optim, LBFGS                 103516      45333.15μs false 2.674e+02      Fevals > limit Binding (4)        217.2767474079502
Speedmapping, acx             86641      11574.70μs  true 8.055e-08         first_order Binding (4)        217.2637111817049

VIBRBEAM: 8 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                          420       1998.86μs false 1.190e-02                     Non-binding      0.15644608774809152
Optim, ConjugateGradient     200023     321507.93μs false 1.231e+08      Fevals > limit Non-binding       19123.175496660853
Optim, LBFGS                   2372       3178.90μs  true 7.957e-08        GradientNorm Non-binding        7.590116364524644
Speedmapping, acx            100000      51472.14μs false 9.591e+01            max_eval Non-binding        36.29302592790894

PALMER8C: 8 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            1     120801.93μs false       NaN              DimensionMismatch                       NaN
Optim, ConjugateGradient       3670       2493.75μs  true 1.222e-09        GradientNorm Binding (6)       115.48202650788595
Optim, LBFGS                 100991      37918.09μs false 9.757e-01      Fevals > limit Binding (6)        115.4837762797095
Speedmapping, acx                69          9.31μs  true 2.688e-09         first_order Binding (6)       115.48202650716308

GAUSS3LS: 8 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                          146       1816.63μs false 3.334e-02                     Binding (3)        7644.736622856321
Optim, ConjugateGradient      97251    1424804.79μs false 4.636e-01          Iterations Binding (3)        7644.736622856429
Optim, LBFGS                 105840     919105.05μs false 9.227e+04      Fevals > limit Binding (3)        7645.582651298191
Speedmapping, acx            100001     354323.86μs false 1.058e-03      Fevals > limit Binding (3)        7644.736622875823

PALMER7C: 8 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            1     114319.80μs false       NaN              DimensionMismatch                       NaN
Optim, ConjugateGradient          0      87222.81μs false       NaN                 AssertionError                       NaN
Optim, LBFGS                 101011      51620.96μs false 2.281e+00      Fevals > limit Binding (6)       170.32892185912672
Speedmapping, acx                31          4.80μs  true 2.618e-09         first_order Binding (6)        170.3263489948777

GAUSS1LS: 8 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           86        960.59μs false 3.365e-06                     Binding (6)        5066.896864959217
Optim, ConjugateGradient     100042    1347879.17μs false 6.341e-01      Fevals > limit Binding (6)       5066.8968650233055
Optim, LBFGS                 102393     953850.98μs false 9.008e+01      Fevals > limit Binding (6)        5067.094534061257
Speedmapping, acx            100001     346323.97μs false 2.078e-04      Fevals > limit Binding (6)        5066.896864959842

STRTCHDV: 10 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           31        231.80μs  true 8.099e-08                     Non-binding    3.913082390558157e-11
Optim, ConjugateGradient        166        359.63μs  true 5.785e-08        GradientNorm Non-binding    7.333164295041699e-11
Optim, LBFGS                    483        608.32μs  true 7.613e-08        GradientNorm Non-binding    3.279612699924784e-11
Speedmapping, acx               276        126.84μs  true 4.896e-08         first_order Binding (1)      0.10566696638598741

HILBERTB: 10 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            3         47.20μs  true 0.000e+00                    Binding (10)        354.2982126984642
Optim, ConjugateGradient         38     136711.57μs false 3.047e+01          Iterations Binding (1)        370.9622301157076
Optim, LBFGS                      0      74693.92μs false       NaN                 AssertionError                       NaN
Speedmapping, acx               103          8.96μs  true 0.000e+00        first_order Binding (10)        354.2982126984642

TRIGON1: 10 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            1     113399.03μs false       NaN              DimensionMismatch                       NaN
Optim, ConjugateGradient        103        156.70μs  true 2.223e-08        GradientNorm Non-binding   1.2517526782176597e-17
Optim, LBFGS                    113        104.89μs  true 6.732e-08        GradientNorm Non-binding    4.015279841164168e-17
Speedmapping, acx               120         23.08μs  true 7.374e-08         first_order Non-binding    4.684158834218382e-17

TOINTQOR: 50 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            1     109176.16μs false       NaN              DimensionMismatch                       NaN
Optim, ConjugateGradient         42     262109.09μs false 4.367e+01          Iterations Binding (1)       1966.2689502884173
Optim, LBFGS                   2438       3237.84μs  true 4.297e-10       GradientNorm Binding (26)       1431.2275561544468
Speedmapping, acx                47         10.44μs  true 4.920e-08        first_order Binding (26)       1431.2275561541453

CHNROSNB: 50 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            3         46.78μs false 2.200e+02                    Binding (49)                1218.9775
Optim, ConjugateGradient         43     251522.93μs false 3.817e+02          Iterations Binding (1)       3903.4808777226726
Optim, LBFGS                      0     122744.08μs false       NaN                 AssertionError                       NaN
Speedmapping, acx               103         12.66μs  true 0.000e+00        first_order Binding (50)                1156.4775

DECONVU: 63 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                         9650     106932.30μs  true 9.533e-08                    Binding (14)     0.009550386405618996
Optim, ConjugateGradient         44     305528.15μs false 1.550e+01          Iterations Binding (1)       49.112697562616106
Optim, LBFGS                 200023     313158.04μs false 8.407e-01      Fevals > limit Non-binding      0.02293862833355709
Speedmapping, acx            100002      58539.15μs false 3.053e-05     Fevals > limit Binding (13)     0.009679022809227429

LUKSAN13LS: 98 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           14        110.97μs  true 1.058e-10                    Binding (97)        40940.60127510612
Optim, ConjugateGradient         45     460547.15μs false 3.490e+02          Iterations Binding (1)        43696.12621110176
Optim, LBFGS                    739       1487.56μs  true 2.464e-09       GradientNorm Binding (97)       40940.601275145214
Speedmapping, acx                23         12.56μs  true 5.329e-15        first_order Binding (97)        40940.60127510612

QING: 100 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            9        221.78μs  true 1.437e-10                    Binding (98)               316129.625
Optim, ConjugateGradient         27     438781.68μs false 5.788e+02          Iterations Binding (1)        320508.4025457767
Optim, LBFGS                      0     219715.83μs false       NaN                 AssertionError                       NaN
Speedmapping, acx                23          9.12μs  true 5.426e-13        first_order Binding (98)               316129.625

Extended Powell: 100 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                          547       5747.35μs false 4.985e-07                    Binding (20)       103.58609558626107
Optim, ConjugateGradient         64     468746.82μs false 1.331e+02          Iterations Binding (5)       1804.3064041267485
Optim, LBFGS                 109048     179409.03μs false 3.993e-01     Fevals > limit Binding (20)       103.60155467347558
Speedmapping, acx               884        276.51μs  true 9.774e-08        first_order Binding (20)       103.58609558642691

Trigonometric: 100 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            1     147006.03μs false       NaN              DimensionMismatch                       NaN
Optim, ConjugateGradient         89        619.75μs  true 6.767e-08        GradientNorm Non-binding     9.204814699941642e-7
Optim, LBFGS                    206        756.58μs  true 7.100e-08        GradientNorm Non-binding     9.204814629711125e-7
Speedmapping, acx               119        175.35μs  true 1.969e-08         first_order Non-binding     1.202698700231473e-6

Dixon and Price: 100 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                         1600     195255.71μs  true 7.781e-08                     Non-binding       0.6666666666666672
Optim, ConjugateGradient        569       2200.11μs  true 7.611e-08        GradientNorm Non-binding   1.4454005697687317e-16
Optim, LBFGS                   2719       6171.25μs  true 8.189e-08        GradientNorm Non-binding        0.666666666666672
Speedmapping, acx               311        114.29μs  true 9.065e-08         first_order Non-binding   4.1028176218331614e-16

Paraboloid Diagonal: 100 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            1     111682.18μs false       NaN              DimensionMismatch                       NaN
Optim, ConjugateGradient      28008     496162.45μs false 3.001e+02          Iterations Binding (1)        22899.54294738802
Optim, LBFGS                      0     185173.03μs false       NaN                 AssertionError                       NaN
Speedmapping, acx               103         23.24μs  true 0.000e+00       first_order Binding (100)                22720.625

LUKSAN17LS: 100 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                          349       5093.27μs false 5.476e-07                     Non-binding       0.4931612903225823
Optim, ConjugateGradient     103018     809730.05μs false 7.663e-02      Fevals > limit Non-binding       0.4931924017309112
Optim, LBFGS                   8652      58554.99μs  true 9.764e-08        GradientNorm Non-binding      0.49316129032258305
Speedmapping, acx               826       2345.50μs  true 5.469e-08         first_order Non-binding       0.4931612903225869

LUKSAN11LS: 100 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           88        409.89μs false 3.993e+01                     Binding (1)        391.3713786179732
Optim, ConjugateGradient     100740     183311.94μs false 1.329e+01      Fevals > limit Non-binding         387.387099887469
Optim, LBFGS                      0     189541.10μs false       NaN                 AssertionError                       NaN
Speedmapping, acx                47         16.01μs  true 3.834e-08         first_order Binding (1)         387.381676508541

Quadratic Diagonal: 100 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            1     108763.93μs false       NaN              DimensionMismatch                       NaN
Optim, ConjugateGradient         50     425831.76μs false 5.030e+01          Iterations Binding (1)       1139.2593855194255
Optim, LBFGS                 100371     194765.09μs false 5.000e+01      Fevals > limit Binding (1)        631.2541840888878
Speedmapping, acx               103         22.39μs  true 0.000e+00       first_order Binding (100)                   631.25

LUKSAN21LS: 100 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           43         66.99μs false 2.831e+00                     Non-binding        99.95203696384378
Optim, ConjugateGradient         73     425256.34μs false 2.486e-01          Iterations Non-binding        96.37347117946817
Optim, LBFGS                   6396      18969.99μs  true 6.811e-08        GradientNorm Non-binding   1.3269939322510062e-13
Speedmapping, acx              2111       1026.07μs  true 9.613e-08         first_order Non-binding    2.190132492584259e-12

LUKSAN16LS: 100 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           89       1014.33μs false 2.720e-07                     Non-binding        3.569697051173897
Optim, ConjugateGradient       1311      12926.56μs  true 4.635e-08        GradientNorm Non-binding          3.5696970511739
Optim, LBFGS                   2030      17320.62μs  true 4.875e-09        GradientNorm Non-binding        3.569697051173893
Speedmapping, acx                39        180.91μs  true 5.350e-08         first_order Non-binding        3.569697051173897

LUKSAN15LS: 100 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           71       1936.47μs false 1.981e-06                     Non-binding       3.5696970511738964
Optim, ConjugateGradient     100001    3813766.96μs false 7.572e+01      Fevals > limit Non-binding        3.573309476166831
Optim, LBFGS                 100059    2860501.05μs false 3.514e+07      Fevals > limit Non-binding        4.286832066343399
Speedmapping, acx                40        637.02μs  true 3.129e-08         first_order Non-binding          3.5696970511739

VARDIM: 200 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            7        114.77μs  true 0.000e+00                   Binding (200)    1.3105836968930898e14
Optim, ConjugateGradient     200002     889158.96μs false 4.251e+13      Fevals > limit Non-binding     1.998018722639582e14
Optim, LBFGS                 100070     390053.03μs false 4.461e+13      Fevals > limit Non-binding    2.1303490310653138e14
Speedmapping, acx               103         57.18μs  true 0.000e+00       first_order Binding (200)    1.3105836968930898e14

BROWNAL: 200 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            3         56.13μs false 1.990e+02                   Binding (199)                    49.75
Optim, ConjugateGradient         75        501.89μs  true 0.000e+00      GradientNorm Binding (200)     2.01325103797075e-12
Optim, LBFGS                     64        330.60μs  true 0.000e+00      GradientNorm Binding (200)   2.0101019665229743e-12
Speedmapping, acx               103         44.04μs  true 0.000e+00         first_order Non-binding                      0.0

ARGLINB: 200 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            6         90.64μs false 3.556e-04                     Non-binding          99.625468164794
Optim, ConjugateGradient          0     577487.95μs false       NaN                 AssertionError                       NaN
Optim, LBFGS                   2900      14441.79μs false 9.223e-04      NotImplemented Non-binding          99.625468164794
Speedmapping, acx            100001      84265.95μs false 4.736e-04      Fevals > limit Non-binding          99.625468164794

ARGLINA: 200 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            1     108929.87μs false       NaN              DimensionMismatch                       NaN
Optim, ConjugateGradient         15        157.25μs  true 8.000e-10        GradientNorm Non-binding       199.99999999999997
Optim, LBFGS                     18        131.27μs  true 8.000e-10        GradientNorm Non-binding       199.99999999999997
Speedmapping, acx                 4          5.48μs  true 0.000e+00         first_order Non-binding                    200.0

PENALTY2: 200 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           28        505.48μs false 1.822e+03                     Binding (7)     4.711627729677791e13
Optim, ConjugateGradient       2915     775361.92μs false 1.798e+03          Iterations Binding (3)    4.7116277295873125e13
Optim, LBFGS                  13293      73199.51μs false 3.882e-06      NotImplemented Binding (8)      4.71162772958727e13
Speedmapping, acx                92        134.61μs  true 6.638e-10         first_order Binding (8)      4.71162772958727e13

ARGTRIGLS: 200 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                         1174      27321.35μs  true 9.847e-08                     Non-binding   1.6662241032374368e-16
Optim, ConjugateGradient       5592      56976.94μs  true 9.598e-08        GradientNorm Non-binding   2.1408404387096865e-16
Optim, LBFGS                   7918      61338.89μs  true 9.968e-08        GradientNorm Non-binding   1.8903997762693878e-16
Speedmapping, acx              1914       3771.69μs  true 7.871e-08         first_order Non-binding    8.358151997065141e-17

ARGLINC: 200 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           45        100.78μs false 2.451e-03                     Non-binding        101.1254705144291
Optim, ConjugateGradient     200028    1230062.96μs false 1.631e+10      Fevals > limit Non-binding      7.961493340391916e7
Optim, LBFGS                  16509     140066.86μs false       NaN                 NotImplemented                       NaN
Speedmapping, acx            100001      83976.03μs false 3.959e-04      Fevals > limit Non-binding       101.12547051442908

PENALTY3: 200 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            1     137739.18μs false       NaN              DimensionMismatch                       NaN
Optim, ConjugateGradient         59     876180.13μs false 6.623e+05          Iterations Binding (1)      8.413168194526021e8
Optim, LBFGS                 113893     435284.14μs false 7.969e-01    Fevals > limit Binding (199)      2.567342131971555e6
Speedmapping, acx                70         72.59μs  true 5.097e-09       first_order Binding (199)      2.567342131970553e6

Large-Scale Quadratic: 250 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            4         51.50μs  true 0.000e+00                   Binding (250)              5.2083125e6
Optim, ConjugateGradient         23    1067091.73μs false 4.970e+02          Iterations Binding (1)      5.218688088772731e6
Optim, LBFGS                      0     519834.04μs false       NaN                 AssertionError                       NaN
Speedmapping, acx               103         67.85μs  true 0.000e+00       first_order Binding (250)              5.2083125e6

OSCIPATH: 500 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           14        261.60μs  true 8.059e-08                     Non-binding       0.9999666655201662
Optim, ConjugateGradient      92085    1672647.33μs  true 3.573e-08        GradientNorm Non-binding       0.9999666655201664
Optim, LBFGS                  58950     482453.43μs  true 2.697e-08        GradientNorm Non-binding       0.9999666655201664
Speedmapping, acx                15         24.73μs  true 1.571e-08         first_order Non-binding       0.9999666655201663

GENROSE: 500 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                          779      23682.71μs false 5.319e-03                    Binding (40)         273.151944124752
Optim, ConjugateGradient         37    1839735.02μs false 1.128e+02          Iterations Non-binding        389.1162415230237
Optim, LBFGS                 151698    1294243.81μs false 4.457e+00     Fevals > limit Binding (39)        272.6410634554596
Speedmapping, acx              2061       2690.26μs  true 3.897e-08        first_order Binding (40)       273.15193881705994

INTEQNELS: 502 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            7        144.23μs  true 2.179e-08                     Non-binding    9.481003229328983e-15
Optim, ConjugateGradient         15        442.88μs  true 2.162e-08        GradientNorm Non-binding   1.7472926924069516e-14
Optim, LBFGS                     27        452.51μs  true 7.427e-08        GradientNorm Non-binding   1.3714530480217093e-13
Speedmapping, acx                10         40.90μs  true 5.466e-08         first_order Non-binding    6.002802406342395e-14

EXTROSNB: 1000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            4        110.01μs  true 0.000e+00                  Binding (1000)                  56196.0
Optim, ConjugateGradient          0    1612959.86μs false       NaN                 AssertionError                       NaN
Optim, LBFGS                    719      11217.25μs  true 0.000e+00     GradientNorm Binding (1000)        56196.00059940202
Speedmapping, acx               103        200.92μs  true 0.000e+00      first_order Binding (1000)                  56196.0

PENALTY1: 1000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            1     111298.08μs false       NaN              DimensionMismatch                       NaN
Optim, ConjugateGradient        646      16160.14μs  true 6.123e-08        GradientNorm Non-binding     0.009686175433475214
Optim, LBFGS                   2709      52637.19μs  true 4.235e-08        GradientNorm Non-binding     0.009686175432902356
Speedmapping, acx                98        297.28μs  true 2.307e-08         first_order Non-binding     0.009686175495025356

EG2: 1000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           45        536.05μs false 1.457e-06                     Non-binding       -998.9473933009804
Optim, ConjugateGradient        441      13503.57μs  true 8.569e-08        GradientNorm Non-binding       -998.9473932734693
Optim, LBFGS                     71       2019.90μs  true 9.909e-10        GradientNorm Non-binding        -998.994811152719
Speedmapping, acx                10         94.40μs  true 3.422e-13         first_order Non-binding       -998.9473933009451

FLETCHCR: 1000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           76        956.47μs false 2.001e-02                     Binding (1)        987.5927193044184
Optim, ConjugateGradient        557      11189.90μs  true 5.626e-09        GradientNorm Binding (1)        987.5927183041797
Optim, LBFGS                 100241    1449645.04μs false 7.005e-01      Fevals > limit Binding (1)        987.5966686817297
Speedmapping, acx                48        126.34μs  true 7.056e-08         first_order Binding (1)        987.5927183031806

MSQRTBLS: 1024 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            1     151120.19μs false       NaN              DimensionMismatch                       NaN
Optim, ConjugateGradient         40    3289745.66μs false 4.776e+01          Iterations Binding (1)        4679.737065381235
Optim, LBFGS                 102780    3070293.19μs false 3.325e-01    Fevals > limit Binding (285)       129.41457564621334
Speedmapping, acx               548       5443.04μs  true 3.450e-08       first_order Binding (290)       129.40873940228792

MSQRTALS: 1024 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            1     131385.09μs false       NaN              DimensionMismatch                       NaN
Optim, ConjugateGradient         39    3357177.68μs false 4.772e+01          Iterations Binding (1)        4676.787103916349
Optim, LBFGS                 100003    2854204.18μs false 1.171e-02    Fevals > limit Binding (279)       129.19493082545196
Speedmapping, acx               580       5520.44μs  true 4.010e-08       first_order Binding (281)        129.1949295723665

EDENSCH: 2000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           69       3454.43μs false 1.806e+01                     Non-binding       12089.741233424558
Optim, ConjugateGradient        135       7519.95μs  true 8.531e-08        GradientNorm Non-binding       12003.284592020764
Optim, LBFGS                    384      15934.32μs  true 9.457e-08        GradientNorm Non-binding       12003.284592020764
Speedmapping, acx                52        297.62μs  true 3.410e-08         first_order Non-binding       12003.284592020764

DIXMAANK: 3000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                         4551    1365134.19μs  true 9.094e-08                     Non-binding       1.0000000012143864
Optim, ConjugateGradient     200032   20264936.92μs false 6.570e+00      Fevals > limit Non-binding        2.539826126158864
Optim, LBFGS                 100001   11363371.13μs false 2.926e+00      Fevals > limit Non-binding       2.5879463401759093
Speedmapping, acx             11567     288933.04μs  true 9.585e-08         first_order Non-binding       1.0000000206813466

DIXMAANH: 3000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                          268      63405.39μs  true 9.257e-08                     Non-binding       1.0000000000053277
Optim, ConjugateGradient       1968     296929.37μs  true 9.969e-08        GradientNorm Non-binding       1.0000000000085751
Optim, LBFGS                   3454     276901.03μs  true 9.881e-08        GradientNorm Non-binding       1.0000000000068963
Speedmapping, acx               404       8286.00μs  true 8.937e-08         first_order Non-binding       1.0000000000010638

DIXMAANI1: 3000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                         8156    2554893.01μs  true 8.388e-08                     Non-binding       1.0000000007346284
Optim, ConjugateGradient      25815    4581716.34μs  true 4.479e-08        GradientNorm Non-binding       1.0000000001027693
Optim, LBFGS                  76583    6487475.87μs  true 6.847e-08        GradientNorm Non-binding        1.000000000050536
Speedmapping, acx             17393     445590.86μs  true 9.756e-08         first_order Non-binding         1.00000002150479

DIXMAANB: 3000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           13       1537.03μs  true 8.248e-08                     Non-binding        1.000000000000113
Optim, ConjugateGradient         21       2997.38μs  true 7.350e-09        GradientNorm Non-binding       1.0000000000000355
Optim, LBFGS                    238      18687.55μs  true 8.580e-09        GradientNorm Non-binding        1.000000000000036
Speedmapping, acx                18        283.20μs  true 7.246e-09         first_order Non-binding       1.0000000000000027

DIXMAAND: 3000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           17       2432.07μs  true 3.183e-08                     Non-binding       1.0000000000000009
Optim, ConjugateGradient         30       4092.45μs  true 1.476e-08        GradientNorm Non-binding       1.0000000000000508
Optim, LBFGS                    313      24789.39μs  true 4.332e-08        GradientNorm Non-binding         1.00000000000051
Speedmapping, acx                20        311.33μs  true 3.528e-08         first_order Non-binding       1.0000000000000253

DIXMAANN: 3000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                         6014    1993831.34μs  true 9.434e-08                     Non-binding       1.0000000007400678
Optim, ConjugateGradient      31845    6547454.79μs  true 5.283e-08        GradientNorm Non-binding        1.000000000414493
Optim, LBFGS                  87235    9459777.63μs  true 9.668e-08        GradientNorm Non-binding       1.0000000001578409
Speedmapping, acx             16249     603399.80μs  true 9.922e-08         first_order Non-binding       1.0000000221488081

DIXMAANJ: 3000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                         4275    1255268.13μs  true 9.484e-08                     Non-binding       1.0000000000726048
Optim, ConjugateGradient     100001   16572584.15μs false 6.657e-03      Fevals > limit Non-binding       1.5520268915014712
Optim, LBFGS                 200063   15451477.05μs false 9.055e+01      Fevals > limit Non-binding       1.6278076890707744
Speedmapping, acx             17283     438916.46μs  true 9.643e-08         first_order Non-binding       1.0000000209214888

DIXMAANM1: 3000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                         8454    2822086.81μs  true 8.954e-08                     Non-binding       1.0000000010809482
Optim, ConjugateGradient      21043    4438386.37μs  true 2.909e-08        GradientNorm Non-binding        1.000000000061764
Optim, LBFGS                  70867    7477946.44μs  true 8.624e-08        GradientNorm Non-binding        1.000000000031789
Speedmapping, acx             13365     470704.31μs  true 9.990e-08         first_order Non-binding       1.0000000210168385

DIXMAANF: 3000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                          267      63089.77μs  true 8.793e-08                     Non-binding       1.0000000000070013
Optim, ConjugateGradient        948     158018.67μs  true 9.789e-08        GradientNorm Non-binding       1.0000000000057616
Optim, LBFGS                   2421     185412.28μs  true 9.872e-08        GradientNorm Non-binding       1.0000000000059706
Speedmapping, acx               471       9630.41μs  true 9.100e-08         first_order Non-binding       1.0000000000008114

DIXMAANL: 3000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                         4831    1427886.59μs  true 9.459e-08                     Non-binding         1.00000000214062
Optim, ConjugateGradient     200040   21188147.07μs false 1.959e+01      Fevals > limit Non-binding        5.271301281341621
Optim, LBFGS                 100061   10907445.19μs false 1.328e+02      Fevals > limit Non-binding        5.228813050804674
Speedmapping, acx              9423     233959.81μs  true 8.919e-08         first_order Non-binding       1.0000000178987185

DIXMAANC: 3000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           15       2020.76μs  true 9.547e-08                     Non-binding       1.0000000000000941
Optim, ConjugateGradient         24       3551.00μs  true 1.222e-08        GradientNorm Non-binding         1.00000000000009
Optim, LBFGS                    186      15288.85μs  true 6.348e-08        GradientNorm Non-binding       1.0000000000026932
Speedmapping, acx                20        308.22μs  true 2.212e-09         first_order Non-binding       1.0000000000000002

DIXMAANO: 3000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                         6456    2080187.46μs  true 9.740e-08                     Non-binding       1.0000000004274987
Optim, ConjugateGradient      34696    7141111.91μs  true 7.267e-08        GradientNorm Non-binding       1.0000000003230145
Optim, LBFGS                  63359    6817183.54μs  true 9.284e-08        GradientNorm Non-binding       1.0000000000796532
Speedmapping, acx             11952     431071.80μs  true 8.005e-08         first_order Non-binding       1.0000000109697404

DIXMAANP: 3000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                         4455    1524636.21μs  true 7.647e-08                     Non-binding       1.0000000003116163
Optim, ConjugateGradient      34773    7439102.47μs  true 6.072e-08        GradientNorm Non-binding       1.0000000050502564
Optim, LBFGS                  81401    8620363.46μs  true 7.288e-08        GradientNorm Non-binding       1.0000000010952081
Speedmapping, acx             15702     551105.05μs  true 8.325e-08         first_order Non-binding       1.0000000148915698

DIXMAANA1: 3000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           14       1758.73μs  true 1.167e-08                     Non-binding       1.0000000000000397
Optim, ConjugateGradient         21       2557.64μs  true 4.059e-09        GradientNorm Non-binding       1.0000000000000109
Optim, LBFGS                    102       6782.22μs  true 4.064e-09        GradientNorm Non-binding       1.0000000000000109
Speedmapping, acx                12        215.83μs  true 1.235e-09         first_order Non-binding       1.0000000000000007

DIXMAANE1: 3000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                          359      84122.24μs  true 8.534e-08                     Non-binding        1.000000000010723
Optim, ConjugateGradient        741     113899.10μs  true 9.300e-08        GradientNorm Non-binding        1.000000000010418
Optim, LBFGS                   2570     206495.94μs  true 9.391e-08        GradientNorm Non-binding       1.0000000000112728
Speedmapping, acx               455       9291.48μs  true 2.301e-08         first_order Non-binding       1.0000000000000604

DIXMAANG: 3000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                          273      64553.01μs  true 6.233e-08                     Non-binding       1.0000000000050226
Optim, ConjugateGradient       1417     215204.50μs  true 9.980e-08        GradientNorm Non-binding        1.000000000005036
Optim, LBFGS                   2513     190343.19μs  true 9.934e-08        GradientNorm Non-binding       1.0000000000053793
Speedmapping, acx               488       9960.51μs  true 9.930e-08         first_order Non-binding       1.0000000000050482

WOODS: 4000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            5        458.83μs  true 0.000e+00                  Binding (4000)               8.771375e6
Optim, ConjugateGradient         69   11929497.53μs false 6.446e+03       Iterations Binding (1000)       9.24468247088609e6
Optim, LBFGS                 100002    6801445.01μs  true 0.000e+00   Fevals > limit Binding (4000)       8.77137501338797e6
Speedmapping, acx               103        869.31μs  true 0.000e+00      first_order Binding (4000)               8.771375e6

LIARWHD: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           25       5542.78μs  true 5.554e-12                     Non-binding   1.6846527613747236e-20
Optim, ConjugateGradient         89      11892.90μs  true 8.743e-08        GradientNorm Non-binding   1.9333372972708314e-17
Optim, LBFGS                    142      13384.81μs  true 8.158e-08        GradientNorm Non-binding   1.9358502439389863e-17
Speedmapping, acx                79       1492.10μs  true 2.728e-09         first_order Non-binding    2.308688090999018e-15

BDQRTIC: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           52       3702.08μs false 4.141e+04                     Non-binding       25894.516692378318
Optim, ConjugateGradient       3523     587691.97μs  true 9.171e-08        GradientNorm Non-binding       20006.256878433676
Optim, LBFGS                   1052     131707.99μs  true 5.062e-08        GradientNorm Non-binding       20006.256878433676
Speedmapping, acx               258       5602.68μs  true 7.572e-08         first_order Non-binding       20006.256878433673

SCHMVETT: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           51      18693.59μs false 1.698e-06                     Non-binding                 -14994.0
Optim, ConjugateGradient          1       1236.96μs false 1.056e+00      NotImplemented Non-binding      -14294.607674120512
Optim, LBFGS                     47       7150.88μs false 1.056e+00      NotImplemented Non-binding      -14294.607674120512
Speedmapping, acx                92       6323.69μs  true 1.265e-08         first_order Binding (2)      -14988.130491927808

SROSENBR: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           20       5067.89μs  true 1.402e-09                     Non-binding   1.3907111129893354e-17
Optim, ConjugateGradient         60       8451.93μs false 1.497e-04      NotImplemented Non-binding    0.0001032967950105693
Optim, LBFGS                     97       8683.64μs false 1.497e-04      NotImplemented Non-binding    0.0001032968067624215
Speedmapping, acx                31        455.13μs  true 7.017e-12         first_order Non-binding   1.9234108169266209e-19

NCB20B: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            1     112364.05μs false       NaN              DimensionMismatch                       NaN
Optim, ConjugateGradient          1       1137.33μs false 4.000e+00      NotImplemented Non-binding                  10000.0
Optim, LBFGS                     50       5959.32μs false 4.000e+00      NotImplemented Non-binding                  10000.0
Speedmapping, acx              5787     350282.16μs  true 9.657e-08         first_order Non-binding        7351.300590425842

QUARTC: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           13        691.23μs false 7.046e+01                  Binding (4995)      6.23750978791737e17
Optim, ConjugateGradient         39   15102262.50μs false 4.993e+11          Iterations Binding (1)     6.238845771517582e17
Optim, LBFGS                 200040   12912750.01μs false 4.993e+11      Fevals > limit Non-binding     6.247206367415864e17
Speedmapping, acx                47        683.65μs  true 6.455e-08      first_order Binding (4998)     6.237509787917368e17

BROYDN3DLS: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           48       1224.22μs false 4.800e+02                     Non-binding        534.3423620617153
Optim, ConjugateGradient          1       1108.41μs false 3.800e+01      NotImplemented Non-binding                   5011.0
Optim, LBFGS                     44       3371.10μs false 3.800e+01      NotImplemented Non-binding                   5011.0
Speedmapping, acx                60       1052.85μs  true 6.769e-09         first_order Binding (1)      0.13251784724572144

POWELLSG: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           17       3323.42μs  true 3.433e-09                  Binding (1250)       20753.547717836584
Optim, ConjugateGradient         37   14963045.48μs false 9.761e+01       Iterations Binding (1250)        54601.05506145657
Optim, LBFGS                      0    7077933.79μs false       NaN                 AssertionError                       NaN
Speedmapping, acx                39        577.27μs  true 7.781e-09      first_order Binding (1250)       20753.547717836584

TRIDIA: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                         1355     474310.90μs false 3.000e+04                     Non-binding       14998.999733227101
Optim, ConjugateGradient       8290    1177195.90μs  true 9.189e-08        GradientNorm Non-binding    4.556819811210321e-16
Optim, LBFGS                   8817     942063.11μs  true 9.292e-08        GradientNorm Non-binding   1.6861519634452205e-16
Speedmapping, acx              3468      50477.42μs  true 7.607e-08         first_order Non-binding    4.564831859919715e-16

GENHUMPS: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            5        920.44μs false 4.322e+01                  Binding (4999)     1.2784453573177227e8
Optim, ConjugateGradient          1       1307.69μs false 8.778e+01      NotImplemented Non-binding     1.2809812932201523e8
Optim, LBFGS                     43       7570.59μs false 8.778e+01      NotImplemented Non-binding     1.2809812932201523e8
Speedmapping, acx               103       6047.14μs  true 0.000e+00      first_order Binding (5000)     1.2784451036771561e8

INDEF: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                          114      22846.73μs false 3.103e+01                     Non-binding    -3.419330103471207e53
Optim, ConjugateGradient         74   14874746.61μs false 3.781e+02          Iterations Non-binding       -68150.16383799749
Optim, LBFGS                     51      11851.76μs false 1.841e+00      NotImplemented Non-binding       4603.2873795320465
Speedmapping, acx            100001    4772150.99μs false 1.000e+00      Fevals > limit Non-binding    -1.1484702595895752e8

NONDQUAR: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           33       8635.14μs false 6.830e+00                    Binding (10)        5.905847623334981
Optim, ConjugateGradient        103   15042651.09μs false 6.957e+02          Iterations Binding (1)        59.29387225349399
Optim, LBFGS                 146384   10031064.99μs false 9.597e-03      Fevals > limit Binding (3)       1.7550603870159869
Speedmapping, acx               980      14387.84μs  true 9.940e-08         first_order Binding (3)       1.7537342298888468

TQUARTIC: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           18       3608.97μs  true 0.000e+00                     Binding (1)      0.16000000000000003
Optim, ConjugateGradient         82   15225545.88μs false 7.692e-01          Iterations Non-binding      0.16008551512018246
Optim, LBFGS                      0    7653983.83μs false       NaN                 AssertionError                       NaN
Speedmapping, acx                12        204.57μs  true 0.000e+00         first_order Binding (1)      0.16000000000000003

ARWHEAD: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            8        775.74μs false 2.931e+04                     Non-binding       12090.520514143154
Optim, ConjugateGradient         24       3602.10μs false 1.197e-02      NotImplemented Non-binding      0.02990238130769285
Optim, LBFGS                     77       5071.50μs false 1.197e-02      NotImplemented Non-binding      0.02990238130587386
Speedmapping, acx                20        286.58μs  true 1.152e-09         first_order Non-binding                      0.0

DQRTIC: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            1     191787.00μs false       NaN                    BoundsError                       NaN
Optim, ConjugateGradient         39   14796313.33μs false 4.993e+11          Iterations Binding (1)     6.238845771517582e17
Optim, LBFGS                 200040   12843541.15μs false 4.993e+11      Fevals > limit Non-binding     6.247206367415864e17
Speedmapping, acx                47        711.32μs  true 6.455e-08      first_order Binding (4998)     6.237509787917368e17

SINQUAD2: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           38      10645.33μs false 9.735e-02                     Non-binding      0.12006644320662163
Optim, ConjugateGradient        358   14813468.24μs false 4.330e+00          Iterations Non-binding      0.06002900592518502
Optim, LBFGS                    327      43334.11μs  true 3.014e-10        GradientNorm Binding (1)      0.02560000030123596
Speedmapping, acx               663      29226.06μs  true 0.000e+00         first_order Binding (1)      0.02560000000000001

NONCVXU2: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                        23654   11270042.05μs false 2.033e-06                     Binding (3)       11584.459429152212
Optim, ConjugateGradient         25   15441573.32μs false 8.670e+04          Iterations Binding (1)    3.0332403834989026e11
Optim, LBFGS                 185791   28006667.14μs false 1.640e+01      Fevals > limit Non-binding       11600.402446100985
Speedmapping, acx            100000    6763908.60μs false 5.528e-07            max_eval Binding (5)       11587.352259425323

DQDRTIC: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           17       2231.39μs false 1.200e+03                     Non-binding                   2700.0
Optim, ConjugateGradient        152      19846.67μs  true 1.162e-08        GradientNorm Non-binding    4.670180275514243e-18
Optim, LBFGS                     65       6404.22μs  true 2.566e-08        GradientNorm Non-binding   2.7584992243654477e-16
Speedmapping, acx                18        293.34μs  true 2.127e-10         first_order Non-binding   2.0802588095422822e-22

FREUROTH: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           19       3084.18μs false 1.756e-05                     Binding (2)        608215.6586295282
Optim, ConjugateGradient         26   14840119.38μs false 8.327e+02          Iterations Binding (1)      3.706890615875321e6
Optim, LBFGS                   1007     112680.82μs  true 3.194e-09        GradientNorm Binding (2)        608215.6586295291
Speedmapping, acx                34        810.35μs  true 2.533e-10         first_order Binding (2)        608215.6586295282

TOINTGSS: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           22       5425.23μs  true 4.214e-08                     Non-binding       10.002000800319914
Optim, ConjugateGradient        619     140340.50μs  true 4.274e-09        GradientNorm Non-binding       10.002000800319914
Optim, LBFGS                 100081   10698859.93μs false       NaN      Fevals > limit Non-binding       10.014693075181954
Speedmapping, acx                63       2764.51μs  true 2.165e-09         first_order Non-binding       10.002000800319914

MOREBV: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           21        447.08μs false 1.599e-07                     Non-binding   1.0395425222561866e-11
Optim, ConjugateGradient          5       1592.46μs  true 7.292e-08        GradientNorm Non-binding    1.039063635837199e-11
Optim, LBFGS                      7       1609.52μs  true 7.293e-08        GradientNorm Non-binding   1.0390635962359223e-11
Speedmapping, acx                 5        133.08μs  true 6.124e-08         first_order Non-binding   1.0392835862589593e-11

CRAGGLVY: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            1     113507.03μs false       NaN                    BoundsError                       NaN
Optim, ConjugateGradient        487      92512.15μs  true 9.396e-08        GradientNorm Non-binding       1688.2153097144587
Optim, LBFGS                   2349     380403.26μs  true 5.839e-08        GradientNorm Non-binding        1688.215309714459
Speedmapping, acx               199       8798.20μs  true 9.019e-08         first_order Non-binding       1688.2153097144587

SPARSINE: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            1     124467.13μs false       NaN              DimensionMismatch                       NaN
Optim, ConjugateGradient     200023   51143567.09μs false 2.966e+02      Fevals > limit Non-binding       31.225698804863164
Optim, LBFGS                 200005   41165068.86μs false 1.135e+02      Fevals > limit Non-binding       1.1811824079068627
Speedmapping, acx            100001   12521044.02μs false 2.612e-02      Fevals > limit Non-binding       1.1953084405219383

NONCVXUN: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                       100000   48846719.45μs false 2.898e-04                    Binding (26)       11600.756512827182
Optim, ConjugateGradient         25   14944287.64μs false 1.056e+05          Iterations Binding (1)      3.08956047944921e11
Optim, LBFGS                 143621   23712069.03μs false 5.938e+00      Fevals > limit Non-binding       11601.770445102133
Speedmapping, acx            100000    6303827.89μs false 6.651e-05           max_eval Binding (11)       11597.819814527775

ENGVAL1: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           18       3889.74μs false 3.238e+01                     Non-binding        5566.119152598236
Optim, ConjugateGradient        113      15272.73μs  true 4.133e-08        GradientNorm Non-binding        5548.668419415788
Optim, LBFGS                    706      78497.73μs  true 2.025e-08        GradientNorm Non-binding        5548.668419415788
Speedmapping, acx                36        501.38μs  true 2.988e-08         first_order Non-binding        5548.668419415788

NONDIA: 5000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            5        565.06μs  true 0.000e+00                  Binding (4999)                 281196.0
Optim, ConjugateGradient         73   14925876.14μs false 1.529e+02          Iterations Binding (1)        285491.9594652238
Optim, LBFGS                 103447    7761070.97μs false 1.500e+02      Fevals > limit Binding (1)       281225.54670734383
Speedmapping, acx               103       1022.93μs  true 0.000e+00      first_order Binding (4999)                 281196.0

NCB20: 5010 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                          367      63054.92μs false 2.529e-06                  Binding (4745)        2598.622145783581
Optim, ConjugateGradient          1       1114.13μs false 4.000e+00      NotImplemented Non-binding                10002.002
Optim, LBFGS                     50       5551.44μs false 4.000e+00      NotImplemented Non-binding                10002.002
Speedmapping, acx             16397     880455.35μs  true 7.889e-08      first_order Binding (4724)       2596.3511729724796

FMINSURF: 5625 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            1     111469.03μs false       NaN              DimensionMismatch                       NaN
Optim, ConjugateGradient         46   16775620.11μs false 1.862e-02          Iterations Binding (1)        27.18976799615933
Optim, LBFGS                 100003    7658319.95μs false 3.340e-02     Fevals > limit Binding (13)         16.3057634665211
Speedmapping, acx              1428      42247.16μs  true 9.144e-08         first_order Non-binding       1.0000000309976502

FMINSRF2: 5625 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                            1     107022.05μs false       NaN              DimensionMismatch                       NaN
Optim, ConjugateGradient         44   16541649.97μs false 2.223e-02          Iterations Binding (1)       27.087712575686144
Optim, LBFGS                 100005    7404631.85μs false 3.761e-02      Fevals > limit Binding (5)        14.47961692762552
Speedmapping, acx              1242      36496.52μs  true 7.641e-08         first_order Non-binding       1.0000170728474718

CURLY20: 10000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                        26769   21589272.69μs false 3.025e-04                    Binding (24)    -1.0028749355656555e6
Optim, ConjugateGradient          1       2177.79μs false 3.860e+00      NotImplemented Non-binding       -1.343675753380224
Optim, LBFGS                 100002   16316082.00μs false 2.018e+02      Fevals > limit Binding (5)         -999383.18335827
Speedmapping, acx            100001    8391939.16μs false 2.312e-03    Fevals > limit Binding (620)         -996582.46447461

CURLY10: 10000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                        22944   18139706.23μs false 2.959e-04                     Binding (9)    -1.0028749355627537e6
Optim, ConjugateGradient          1       2110.73μs false 1.583e+00      NotImplemented Non-binding      -0.6306184152244703
Optim, LBFGS                 200040   37158045.05μs false 2.060e+02      Fevals > limit Non-binding        -998264.858153581
Speedmapping, acx            100001    7604547.98μs false 4.588e-05   Fevals > limit Binding (1822)    -1.0028749292163793e6

POWER: 10000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                          496     353710.50μs  true 9.936e-08                     Non-binding   1.1990263926898825e-10
Optim, ConjugateGradient      53639   10854425.74μs  true 9.731e-08        GradientNorm Non-binding   1.0398483603196139e-10
Optim, LBFGS                 200071   29456787.11μs false 2.091e+08      Fevals > limit Non-binding     4.900825353369438e10
Speedmapping, acx              1521      44075.27μs  true 9.768e-08         first_order Non-binding    7.090795505715685e-11

CURLY30: 10000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                        28976   23784209.16μs false 6.798e-04                    Binding (19)    -1.0028749355297603e6
Optim, ConjugateGradient          1       2165.77μs false 6.932e+00      NotImplemented Non-binding      -2.1896375904938865
Optim, LBFGS                 154003   38960886.00μs false 2.937e+02     Fevals > limit Binding (11)       -998905.5397286557
Speedmapping, acx            100002    9893599.03μs false 1.836e-02    Fevals > limit Binding (411)       -997685.3303455279

COSINE: 10000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           31       6657.15μs  true 2.952e-08                  Binding (9998)        705.1284798824845
Optim, ConjugateGradient          1       2197.92μs false 9.589e-01      NotImplemented Non-binding         8774.94803634173
Optim, LBFGS                 100030   29197112.08μs false 1.221e+00      Fevals > limit Non-binding       -9998.992663165442
Speedmapping, acx              3653     263728.44μs  true 5.424e-08      first_order Binding (9996)        703.3592562265328

DIXON3DQ: 10000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                        36466   24975612.02μs  true 9.357e-08                     Binding (2)        4.500000072678273
Optim, ConjugateGradient         62   30650256.60μs false 3.001e+00          Iterations Binding (1)        4.633916495603215
Optim, LBFGS                 200036   30101104.97μs false 3.000e+00      Fevals > limit Non-binding        4.508890376223858
Speedmapping, acx             19671     636877.98μs  true 9.805e-08         first_order Binding (2)        4.500057382383752

SPARSQUR: 10000 parameters, abstol = 1.0e-7.
Solver                   Grad evals            time  conv   |resid|                             log                      obj
LBFGSB                           45      29387.99μs  true 6.607e-08                     Non-binding    9.072076389272298e-12
Optim, ConjugateGradient         65   31339171.47μs false 1.393e+05          Iterations Non-binding      7.448518316444763e6
Optim, LBFGS                   1150     306175.94μs  true 8.711e-08        GradientNorm Non-binding     1.605404678062682e-9
Speedmapping, acx                58       3508.70μs  true 3.329e-08         first_order Non-binding   2.3997276830899155e-10
=#