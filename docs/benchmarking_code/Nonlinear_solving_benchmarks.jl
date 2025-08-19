# Dire: The goal is to see how each algorithm performs on its own, but some speed gain may be obtained
# by tayloring the ad to each specific problem like in ... Interestingly, the NonlinearSolve, Default PolyAlg. did
# solve all 23 problems.

absolute_path_to_docs = "" # Update

using BenchmarkTools, NonlinearProblemLibrary, NonlinearSolve, NLsolve, JLD2, FileIO, DiffEqBase, LinearAlgebra #, SpeedMapping

path_plots = absolute_path_to_docs*"assets/"
path_out = absolute_path_to_docs*"benchmarking_code/Output/"
include(absolute_path_to_docs * "benchmarking_code/Benchmarking_utils.jl")

# Creating a problem dictionary
NlPL = NonlinearProblemLibrary

# Problem 23 from NonlinearProblemLibrary allocates. This version does not.
@inbounds function p23_f!(out, x, μ)
    c = 0.9
    out .= x
    n = length(μ)
    for i in eachindex(μ)
        s = 0.0
        for j in eachindex(x)
            s += (μ[i] * x[j]) / (μ[i] + μ[j])
        end
        out[i] = x[i] - 1.0 / (1.0 - c * s / (2 * n))
    end
    nothing
end

gen_p23(;n = 10) = (du, u) -> p23_f!(du, u, [(2 * i) / (2 * n) for i in 1:n])

rs! = (NlPL.problems[1:22]..., gen_p23())

nlproblems = Dict()
nlproblem_names = Vector{String}(undef, length(rs!))
nlproblem_names_len = Vector{String}(undef, length(rs!)) # For plotting results
sizes = Vector{Int64}(undef, length(rs!))
for i in eachindex(NlPL.dicts)
	nlproblem_names[i] = NlPL.dicts[i]["title"]
	sizes[i] = length(NlPL.dicts[i]["start"])
	nlproblem_names_len[i] = NlPL.dicts[i]["title"]*" ($(sizes[i]))"
	nlproblems[NlPL.dicts[i]["title"]] = (x0 = NlPL.dicts[i]["start"], r! = rs![i], obj = nothing, 
        abstol = 1e-7, time_limit = 10)
end

# This wrapper around r!
function _r!(r!, res, x, p, maps)
	maps[] += 1
	r!(res, x)
	return res
end

# Solver wrappers
nlsolvers = Dict{AbstractString, Function}()

function Speedmapping_nls_wrapper(problem, abstol, maps_limit)
	x0, r! = problem
	maps = Ref(0)
	res = speedmapping(x0; r! = (resid, x) -> _r!(r!, resid, x, 0, maps), algo = :aa, abstol, 
		lags = 30, maps_limit, condition_max = 1e6)
	return res.minimizer, maps[], string(res.status)
end

nlsolvers["Speedmapping, aa"] = (problem, abstol, start_time, maps_limit, time_limit) -> 
	Speedmapping_nls_wrapper(problem, abstol, maps_limit)

# NonlinearSolve
function NonlinearSolve_wrapper(problem, abstol, maps_limit, solver)
	x0, r! = problem
	maps = Ref(0)
	prob = NonlinearProblem((du, u, p) -> _r!(r!, du, u, p, maps), x0, 0)
	res = NonlinearSolve.solve(prob, solver; reltol = 0.0, maxiters = maps_limit, abstol = abstol,
		termination_condition = AbsNormTerminationMode(norm))
	return res.u, maps[], string(res.retcode)
end

import LineSearches
const RUS = RadiusUpdateSchemes;
HagerZhang() = LineSearchesJL(; method = LineSearches.HagerZhang())
MoreThuente() = LineSearchesJL(; method = LineSearches.MoreThuente())

# Same specifications as https://docs.sciml.ai/SciMLBenchmarksOutput/stable/NonlinearProblem/nonlinear_solver_23_tests/
# but no set_ad_chunksize

for (solver, solver_fct) in 	
   (("Default PolyAlg.",          FastShortcutNonlinearPolyalg(; u0_len = 10, linsolve = \)),
    ("Newton Raphson",            NewtonRaphson(; linsolve = \)),
    ("NR (HagerZhang)",           NewtonRaphson(; linsolve = \, linesearch = HagerZhang())),
    ("NR (MoreThuente)",          NewtonRaphson(; linsolve = \, linesearch = MoreThuente())),
    ("NR (BackTracking)",         NewtonRaphson(; linsolve = \, linesearch = NonlinearSolve.BackTracking())),
    ("Trust Region",              TrustRegion(; linsolve = \)),
    ("TR (NLsolve Update)",       TrustRegion(; linsolve = \, radius_update_scheme = RUS.NLsolve)),
    ("TR (Nocedal Wright)",       TrustRegion(; linsolve = \, radius_update_scheme = RUS.NocedalWright)),
    ("TR (Hei)",                  TrustRegion(; linsolve = \, radius_update_scheme = RUS.Hei)),
    ("TR (Yuan)",                 TrustRegion(; linsolve = \, radius_update_scheme = RUS.Yuan)),
    ("TR (Bastin)",               TrustRegion(; linsolve = \, radius_update_scheme = RUS.Bastin)),
    ("TR (Fan)",                  TrustRegion(; linsolve = \, radius_update_scheme = RUS.Fan)),
    ("Levenberg-Marquardt",       LevenbergMarquardt(; linsolve = QRFactorization())),
    ("LM with Cholesky",          LevenbergMarquardt(; linsolve = CholeskyFactorization())),
    ("LM (α_geodesic=0.5)",       LevenbergMarquardt(; linsolve = QRFactorization(), α_geodesic=0.5)),
    ("LM (α_geodesic=0.5) Chol.", LevenbergMarquardt(; linsolve = CholeskyFactorization(), α_geodesic=0.5)),
    ("LM (no Accln.)",            LevenbergMarquardt(; linsolve = QRFactorization(), disable_geodesic = Val(true))),
    ("LM (no Accln.) Chol.",      LevenbergMarquardt(; linsolve = CholeskyFactorization(), disable_geodesic = Val(true))),
    ("Pseudo Transient",          PseudoTransient(; linsolve = \, alpha_initial=10.0)))

	nlsolvers["NonlinearSolve, " * solver] = (problem, abstol, start_time, maps_limit, time_limit) -> 
		NonlinearSolve_wrapper(problem, abstol, maps_limit, solver_fct)
end

function NLsolve_wrapper(problem, abstol, maps_limit, method)
	x0, r! = problem
	maps = Ref(0)
	try
		res = NLsolve.nlsolve((du, u) -> _r!(r!, du, u, 0, maps), x0; method = method, 
			xtol = abstol, ftol = 0.0, iterations = maps_limit, m = 30, droptol = 1e6) # m = 30 and droptol = 1e6 to be the same as the Anderson acceleration in SpeedMapping
		return res.zero, maps[], string("x_converged = $(res.x_converged)")
	catch e
		return NaN*ones(length(x0)), maps[], sprint(showerror, typeof(e))
	end
end

for method in ("trust_region", "newton", "anderson")
	nlsolvers["NLSolve, " * method] = (problem, abstol, start_time, maps_limit, time_limit) -> 
		NLsolve_wrapper(problem, abstol, maps_limit, Symbol(method))
end

nlsolver_names = sort([name for (name, wrapper) in nlsolvers])

function compute_norm(problem, x)
	xout = similar(x)
	if sum(_isbad.(x)) == 0
		problem.r!(xout, x)
		return norm(xout, 2)
	else
		return NaN
	end
end

gen_Feval_limit(problem, time_limit) = 1_000_000

nl_res_all = many_problems_many_solvers(nlproblems, nlsolvers, nlproblem_names, 
	nlsolver_names, compute_norm; tunits = 3, F_name = "F evals", gen_Feval_limit, abstol = 1e-7,
	time_limit = 10, proper_benchmark = true)

JLD2.@save path_out*"nl_res_all.jld2" nl_res_all
# nl_res_all = JLD2.load_object(path_out*"nl_res_all.jld2") # To load

title = "Performance of various Julia solvers for nonlinear problems"
plot_res(nl_res_all, nlproblem_names_len, nlsolver_names, title, path_plots*"nonlinear_benchmarks.svg"; 
	size = (800, 600), legend_rowgap = -5)

#=
Generalized Rosenbrock function: 10 parameters, abstol = 1.0e-7.
Solver                                       F evals          time  conv   |resid|                 log
NLSolve, anderson                                 19   488289.83μs false       NaN NLsolve.IsFiniteExc
NLSolve, newton                                   84       21.75μs false 1.813e+72  x_converged = true
NLSolve, trust_region                           5048      360.87μs  true 0.000e+00  x_converged = true
NonlinearSolve, Default PolyAlg.                  26       21.68μs  true 0.000e+00             Success
NonlinearSolve, LM (no Accln.)                    56      227.98μs  true 5.555e-09             Success
NonlinearSolve, LM (no Accln.) Chol.              56       49.77μs  true 5.555e-09             Success
NonlinearSolve, LM (α_geodesic=0.5)              287     2166.95μs  true 2.158e-08             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.        287      270.62μs  true 2.158e-08             Success
NonlinearSolve, LM with Cholesky                 204      192.61μs  true 5.623e-09             Success
NonlinearSolve, Levenberg-Marquardt              204     1475.21μs  true 5.623e-09             Success
NonlinearSolve, NR (BackTracking)               1287      312.26μs  true 1.554e-14             Success
NonlinearSolve, NR (HagerZhang)                 2014      271.91μs  true 1.276e-10             Success
NonlinearSolve, NR (MoreThuente)                6109      907.22μs  true 9.142e-08             Success
NonlinearSolve, Newton Raphson                    26       19.84μs  true 0.000e+00             Success
NonlinearSolve, Pseudo Transient             2000003   796752.93μs false 2.638e+10      Fevals > limit
NonlinearSolve, TR (Bastin)                     8557      472.94μs  true 2.884e-08             Success
NonlinearSolve, TR (Fan)                        1683      700.95μs  true 1.455e-08             Success
NonlinearSolve, TR (Hei)                         736      298.99μs  true 3.006e-14             Success
NonlinearSolve, TR (NLsolve Update)             1126      452.50μs  true 7.850e-15             Success
NonlinearSolve, TR (Nocedal Wright)              769      305.69μs  true 3.671e-13             Success
NonlinearSolve, TR (Yuan)                      57896     3606.09μs  true 9.129e-08             Success
NonlinearSolve, Trust Region                     769      307.85μs  true 3.671e-13             Success
Speedmapping, aa                                 346      193.62μs  true 1.640e-08         first_order

Powell singular function: 4 parameters, abstol = 1.0e-7.
Solver                                       F evals          time  conv   |resid|                 log
NLSolve, anderson                                 16     7374.05μs false       NaN NLsolve.IsFiniteExc
NLSolve, newton                                  234       16.70μs  true 1.127e-14  x_converged = true
NLSolve, trust_region                            234       17.32μs  true 1.127e-14  x_converged = true
NonlinearSolve, Default PolyAlg.                  32       32.12μs  true 4.727e-08             Success
NonlinearSolve, LM (no Accln.)                    44       83.97μs  true 5.768e-08             Success
NonlinearSolve, LM (no Accln.) Chol.              44       40.38μs  true 5.768e-08             Success
NonlinearSolve, LM (α_geodesic=0.5)               79      186.55μs  true 6.328e-08             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.         79       55.96μs  true 6.328e-08             Success
NonlinearSolve, LM with Cholesky                  61       46.06μs  true 9.054e-08             Success
NonlinearSolve, Levenberg-Marquardt               61      137.69μs  true 9.054e-08             Success
NonlinearSolve, NR (BackTracking)                 88       38.76μs  true 4.727e-08             Success
NonlinearSolve, NR (HagerZhang)                  149       63.55μs  true 5.159e-08             Success
NonlinearSolve, NR (MoreThuente)                 158       55.47μs  true 4.727e-08             Success
NonlinearSolve, Newton Raphson                    32       32.62μs  true 4.727e-08             Success
NonlinearSolve, Pseudo Transient                  32       32.94μs  true 4.799e-08             Success
NonlinearSolve, TR (Bastin)                      298       47.18μs  true 7.328e-08             Success
NonlinearSolve, TR (Fan)                          32       34.99μs  true 4.602e-08             Success
NonlinearSolve, TR (Hei)                          32       35.14μs  true 4.527e-08             Success
NonlinearSolve, TR (NLsolve Update)               32       35.14μs  true 4.727e-08             Success
NonlinearSolve, TR (Nocedal Wright)               32       34.79μs  true 4.586e-08             Success
NonlinearSolve, TR (Yuan)                        211       47.84μs  true 4.546e-08             Success
NonlinearSolve, Trust Region                      32       35.19μs  true 4.586e-08             Success
Speedmapping, aa                                  35       11.05μs  true 8.920e-08         first_order

Powell badly scaled function: 2 parameters, abstol = 1.0e-7.
Solver                                       F evals          time  conv   |resid|                 log
NLSolve, anderson                                 22       25.18μs  true 2.141e-08  x_converged = true
NLSolve, newton                                   70        9.80μs  true 0.000e+00  x_converged = true
NLSolve, trust_region                             76       10.57μs  true 0.000e+00  x_converged = true
NonlinearSolve, Default PolyAlg.                  28       14.60μs  true 1.573e-11             Success
NonlinearSolve, LM (no Accln.)                    44       39.49μs  true 3.128e-08             Success
NonlinearSolve, LM (no Accln.) Chol.              44       19.16μs  true 3.128e-08             Success
NonlinearSolve, LM (α_geodesic=0.5)               70       77.97μs  true 1.974e-08             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.         70       27.33μs  true 1.974e-08             Success
NonlinearSolve, LM with Cholesky                  61       24.54μs  true 2.939e-08             Success
NonlinearSolve, Levenberg-Marquardt               61       63.70μs  true 2.939e-08             Success
NonlinearSolve, NR (BackTracking)                353       65.18μs  true 6.227e-12             Success
NonlinearSolve, NR (HagerZhang)                  661       97.41μs  true 1.550e-08             Success
NonlinearSolve, NR (MoreThuente)                 816      128.72μs  true 8.931e-08             Success
NonlinearSolve, Newton Raphson                    28       14.80μs  true 1.573e-11             Success
NonlinearSolve, Pseudo Transient                  56       18.16μs  true 8.264e-08             Success
NonlinearSolve, TR (Bastin)                      747       66.73μs  true 1.325e-11             Success
NonlinearSolve, TR (Fan)                         432      144.09μs  true 1.349e-11             Success
NonlinearSolve, TR (Hei)                         158       50.68μs  true 1.368e-12             Success
NonlinearSolve, TR (NLsolve Update)              219       67.41μs  true 1.656e-12             Success
NonlinearSolve, TR (Nocedal Wright)              106       36.48μs  true 3.164e-12             Success
NonlinearSolve, TR (Yuan)                      36134     2905.49μs  true 6.106e-13             Success
NonlinearSolve, Trust Region                     106       32.66μs  true 3.164e-12             Success
Speedmapping, aa                                  31        6.95μs  true 1.619e-08         first_order

Wood function: 4 parameters, abstol = 1.0e-7.
Solver                                       F evals          time  conv   |resid|                 log
NLSolve, anderson                                  8     3443.96μs false       NaN NLsolve.IsFiniteExc
NLSolve, newton                                  135        9.17μs  true 7.178e-14  x_converged = true
NLSolve, trust_region                            148       14.20μs  true 2.450e-14  x_converged = true
NonlinearSolve, Default PolyAlg.                  32       16.59μs  true 8.994e-14             Success
NonlinearSolve, LM (no Accln.)                    52       77.53μs  true 8.953e-08             Success
NonlinearSolve, LM (no Accln.) Chol.              52       24.65μs  true 8.953e-08             Success
NonlinearSolve, LM (α_geodesic=0.5)               76      141.23μs  true 1.084e-08             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.         76       33.04μs  true 1.084e-08             Success
NonlinearSolve, LM with Cholesky                  76       33.42μs  true 1.084e-08             Success
NonlinearSolve, Levenberg-Marquardt               76      140.47μs  true 1.084e-08             Success
NonlinearSolve, NR (BackTracking)                112       24.95μs  true 4.502e-13             Success
NonlinearSolve, NR (HagerZhang)                  236       38.07μs  true 1.019e-10             Success
NonlinearSolve, NR (MoreThuente)                 659      101.64μs  true 1.709e-08             Success
NonlinearSolve, Newton Raphson                    32       16.11μs  true 8.994e-14             Success
NonlinearSolve, Pseudo Transient                  32       16.61μs  true 2.755e-13             Success
NonlinearSolve, TR (Bastin)                      511       48.39μs  true 4.022e-13             Success
NonlinearSolve, TR (Fan)                          73       34.89μs  true 2.128e-10             Success
NonlinearSolve, TR (Hei)                          79       34.62μs  true 9.155e-10             Success
NonlinearSolve, TR (NLsolve Update)               83       34.78μs  true 5.940e-13             Success
NonlinearSolve, TR (Nocedal Wright)               44       22.70μs  true 3.143e-08             Success
NonlinearSolve, TR (Yuan)                       1437      130.83μs  true 4.452e-10             Success
NonlinearSolve, Trust Region                      48       25.02μs  true 1.371e-08             Success
Speedmapping, aa                                 351       94.69μs  true 4.900e-08         first_order

Helical valley function: 3 parameters, abstol = 1.0e-7.
Solver                                       F evals          time  conv   |resid|                 log
NLSolve, anderson                                  3     5596.88μs false       NaN   SingularException
NLSolve, newton                                   77        6.77μs  true 1.326e-14  x_converged = true
NLSolve, trust_region                             64        8.90μs  true 1.166e-14  x_converged = true
NonlinearSolve, Default PolyAlg.                  24       14.98μs  true 1.325e-14             Success
NonlinearSolve, LM (no Accln.)                    26       34.52μs  true 9.278e-09             Success
NonlinearSolve, LM (no Accln.) Chol.              26       16.94μs  true 9.278e-09             Success
NonlinearSolve, LM (α_geodesic=0.5)               41       65.66μs  true 1.065e-09             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.         41       23.44μs  true 1.065e-09             Success
NonlinearSolve, LM with Cholesky                  37       21.84μs  true 1.141e-08             Success
NonlinearSolve, Levenberg-Marquardt               37       57.68μs  true 1.141e-08             Success
NonlinearSolve, NR (BackTracking)                 60       18.39μs  true 4.089e-11             Success
NonlinearSolve, NR (HagerZhang)                  113       22.96μs  true 1.354e-10             Success
NonlinearSolve, NR (MoreThuente)                 557       89.41μs  true 7.721e-08             Success
NonlinearSolve, Newton Raphson                    24       14.67μs  true 1.325e-14             Success
NonlinearSolve, Pseudo Transient                  24       15.32μs  true 5.155e-14             Success
NonlinearSolve, TR (Bastin)                      157       20.54μs  true 9.841e-11             Success
NonlinearSolve, TR (Fan)                          20       15.77μs  true 3.087e-10             Success
NonlinearSolve, TR (Hei)                          23       17.20μs  true 2.774e-15             Success
NonlinearSolve, TR (NLsolve Update)               21       16.34μs  true 7.569e-08             Success
NonlinearSolve, TR (Nocedal Wright)               22       15.90μs  true 5.127e-11             Success
NonlinearSolve, TR (Yuan)                        107       20.31μs  true 4.549e-08             Success
NonlinearSolve, Trust Region                      22       15.65μs  true 5.127e-11             Success
Speedmapping, aa                                  81       19.82μs  true 1.461e-09         first_order

Watson function: 2 parameters, abstol = 1.0e-7.
Solver                                       F evals          time  conv   |resid|                 log
NLSolve, anderson                                 10     3099.20μs false       NaN NLsolve.IsFiniteExc
NLSolve, newton                                   55       10.37μs  true 3.975e-15  x_converged = true
NLSolve, trust_region                             77       20.70μs  true 2.688e-15  x_converged = true
NonlinearSolve, Default PolyAlg.                  22       17.54μs  true 4.214e-13             Success
NonlinearSolve, LM (no Accln.)                    22       27.49μs  true 7.069e-08             Success
NonlinearSolve, LM (no Accln.) Chol.              22       18.72μs  true 7.069e-08             Success
NonlinearSolve, LM (α_geodesic=0.5)               40       55.23μs  true 3.386e-08             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.         40       26.86μs  true 3.386e-08             Success
NonlinearSolve, LM with Cholesky                  39       26.68μs  true 6.032e-10             Success
NonlinearSolve, Levenberg-Marquardt               39       52.88μs  true 6.032e-10             Success
NonlinearSolve, NR (BackTracking)                 54       22.89μs  true 9.452e-11             Success
NonlinearSolve, NR (HagerZhang)                  105       32.22μs  true 3.093e-10             Success
NonlinearSolve, NR (MoreThuente)                 683      175.04μs  true 5.611e-15             Success
NonlinearSolve, Newton Raphson                    22       16.98μs  true 4.214e-13             Success
NonlinearSolve, Pseudo Transient                  20       16.84μs  true 7.951e-09             Success
NonlinearSolve, TR (Bastin)                       82       25.30μs  true 3.202e-15             Success
NonlinearSolve, TR (Fan)                          43       28.05μs  true 8.412e-12             Success
NonlinearSolve, TR (Hei)                          14       15.92μs  true 7.826e-13             Success
NonlinearSolve, TR (NLsolve Update)               25       20.50μs  true 6.171e-14             Success
NonlinearSolve, TR (Nocedal Wright)               17       17.82μs  true 2.854e-13             Success
NonlinearSolve, TR (Yuan)                        289       68.55μs  true 4.820e-15             Success
NonlinearSolve, Trust Region                      18       17.64μs  true 1.495e-12             Success
Speedmapping, aa                                  30        8.21μs  true 1.360e-09         first_order

Chebyquad function: 2 parameters, abstol = 1.0e-7.
Solver                                       F evals          time  conv   |resid|                 log
NLSolve, anderson                                 13     3188.85μs false       NaN NLsolve.IsFiniteExc
NLSolve, newton                                   35        4.15μs  true 5.551e-17  x_converged = true
NLSolve, trust_region                             32        5.36μs  true 9.437e-16  x_converged = true
NonlinearSolve, Default PolyAlg.                  14       12.81μs  true 3.685e-12             Success
NonlinearSolve, LM (no Accln.)                    18       21.14μs  true 1.169e-08             Success
NonlinearSolve, LM (no Accln.) Chol.              18       15.14μs  true 1.169e-08             Success
NonlinearSolve, LM (α_geodesic=0.5)               30       40.13μs  true 2.873e-08             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.         30       19.28μs  true 2.873e-08             Success
NonlinearSolve, LM with Cholesky                  25       18.19μs  true 6.729e-09             Success
NonlinearSolve, Levenberg-Marquardt               25       32.12μs  true 6.729e-09             Success
NonlinearSolve, NR (BackTracking)                 35       15.27μs  true 1.951e-09             Success
NonlinearSolve, NR (HagerZhang)                   74       18.97μs  true 9.645e-10             Success
NonlinearSolve, NR (MoreThuente)                 381       62.25μs  true 6.962e-08             Success
NonlinearSolve, Newton Raphson                    14       13.07μs  true 3.685e-12             Success
NonlinearSolve, Pseudo Transient                  14       13.08μs  true 1.059e-11             Success
NonlinearSolve, TR (Bastin)                       83       17.04μs  true 9.381e-15             Success
NonlinearSolve, TR (Fan)                          16       14.37μs  true 8.729e-13             Success
NonlinearSolve, TR (Hei)                          17       14.56μs  true 8.687e-14             Success
NonlinearSolve, TR (NLsolve Update)               15       14.37μs  true 1.874e-08             Success
NonlinearSolve, TR (Nocedal Wright)               18       15.04μs  true 4.269e-14             Success
NonlinearSolve, TR (Yuan)                         44       15.42μs  true 1.373e-11             Success
NonlinearSolve, Trust Region                      18       14.81μs  true 4.269e-14             Success
Speedmapping, aa                                  13        3.16μs  true 5.005e-08         first_order

Brown almost linear function: 10 parameters, abstol = 1.0e-7.
Solver                                       F evals          time  conv   |resid|                 log
NLSolve, anderson                                  9     3178.83μs false       NaN NLsolve.IsFiniteExc
NLSolve, newton                                 1932      132.32μs  true 1.070e-14  x_converged = true
NLSolve, trust_region                            126       12.67μs  true 2.220e-16  x_converged = true
NonlinearSolve, Default PolyAlg.                 184      128.89μs  true 9.545e-12             Success
NonlinearSolve, LM (no Accln.)                    28      115.93μs  true 3.202e-08             Success
NonlinearSolve, LM (no Accln.) Chol.              28       29.68μs  true 3.202e-08             Success
NonlinearSolve, LM (α_geodesic=0.5)               49      290.37μs  true 3.162e-09             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.         49       52.66μs  true 3.162e-09             Success
NonlinearSolve, LM with Cholesky                  45       49.14μs  true 4.805e-08             Success
NonlinearSolve, Levenberg-Marquardt               45      260.01μs  true 4.804e-08             Success
NonlinearSolve, NR (BackTracking)               1056   307887.08μs false       NaN InternalLineSearchF
NonlinearSolve, NR (HagerZhang)                  181       36.31μs false 1.801e+67 InternalLineSearchF
NonlinearSolve, NR (MoreThuente)                 552      146.77μs  true 4.918e-13             Success
NonlinearSolve, Newton Raphson                   184      124.65μs  true 9.545e-12             Success
NonlinearSolve, Pseudo Transient             2000003  1503834.01μs false 1.359e+19      Fevals > limit
NonlinearSolve, TR (Bastin)                      319       43.07μs  true 5.718e-09             Success
NonlinearSolve, TR (Fan)                          14       22.03μs  true 1.614e-10             Success
NonlinearSolve, TR (Hei)                          18       23.54μs  true 2.285e-10             Success
NonlinearSolve, TR (NLsolve Update)               10       17.98μs  true 1.653e-10             Success
NonlinearSolve, TR (Nocedal Wright)               12       19.92μs  true 1.241e-08             Success
NonlinearSolve, TR (Yuan)                        751       92.38μs false 1.653e+01 ShrinkThresholdExce
NonlinearSolve, Trust Region                      12       19.37μs  true 1.241e-08             Success
Speedmapping, aa                                  18        7.03μs  true 3.685e-09         first_order

Discrete boundary value function: 10 parameters, abstol = 1.0e-7.
Solver                                       F evals          time  conv   |resid|                 log
NLSolve, anderson                                 15       46.39μs  true 1.290e-09  x_converged = true
NLSolve, newton                                  105        8.57μs  true 5.004e-17  x_converged = true
NLSolve, trust_region                            105       10.03μs  true 5.004e-17  x_converged = true
NonlinearSolve, Default PolyAlg.                   8       15.44μs  true 3.110e-08             Success
NonlinearSolve, LM (no Accln.)                    26      103.24μs  true 2.153e-08             Success
NonlinearSolve, LM (no Accln.) Chol.              26       28.11μs  true 2.153e-08             Success
NonlinearSolve, LM (α_geodesic=0.5)               37      195.09μs  true 2.192e-08             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.         37       39.66μs  true 2.192e-08             Success
NonlinearSolve, LM with Cholesky                  37       40.34μs  true 2.192e-08             Success
NonlinearSolve, Levenberg-Marquardt               37      197.00μs  true 2.192e-08             Success
NonlinearSolve, NR (BackTracking)                 16       16.02μs  true 3.110e-08             Success
NonlinearSolve, NR (HagerZhang)                   29       18.08μs  true 3.071e-09             Success
NonlinearSolve, NR (MoreThuente)                  26       19.21μs  true 3.110e-08             Success
NonlinearSolve, Newton Raphson                     8       15.30μs  true 3.110e-08             Success
NonlinearSolve, Pseudo Transient                  14       18.97μs  true 9.636e-11             Success
NonlinearSolve, TR (Bastin)                       94       23.08μs  true 3.110e-08             Success
NonlinearSolve, TR (Fan)                          12       19.72μs  true 1.526e-08             Success
NonlinearSolve, TR (Hei)                           8       15.84μs  true 3.110e-08             Success
NonlinearSolve, TR (NLsolve Update)                8       16.74μs  true 3.110e-08             Success
NonlinearSolve, TR (Nocedal Wright)               14       21.21μs  true 1.784e-09             Success
NonlinearSolve, TR (Yuan)                        117       26.34μs  true 4.589e-13             Success
NonlinearSolve, Trust Region                      14       20.66μs  true 1.784e-09             Success
Speedmapping, aa                                  19        8.88μs  true 6.548e-09         first_order

Discrete integral equation function: 10 parameters, abstol = 1.0e-7.
Solver                                       F evals          time  conv   |resid|                 log
NLSolve, anderson                                 19     3252.98μs false       NaN NLsolve.IsFiniteExc
NLSolve, newton                                  105       15.59μs  true 8.002e-17  x_converged = true
NLSolve, trust_region                            105       17.21μs  true 8.002e-17  x_converged = true
NonlinearSolve, Default PolyAlg.                  10       17.95μs  true 2.781e-15             Success
NonlinearSolve, LM (no Accln.)                    16       65.66μs  true 2.034e-09             Success
NonlinearSolve, LM (no Accln.) Chol.              16       25.04μs  true 2.034e-09             Success
NonlinearSolve, LM (α_geodesic=0.5)               22      119.08μs  true 2.063e-09             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.         22       32.56μs  true 2.063e-09             Success
NonlinearSolve, LM with Cholesky                  22       32.35μs  true 2.063e-09             Success
NonlinearSolve, Levenberg-Marquardt               22      120.75μs  true 2.063e-09             Success
NonlinearSolve, NR (BackTracking)                 22       21.21μs  true 2.781e-15             Success
NonlinearSolve, NR (HagerZhang)                   29       21.20μs  true 1.021e-08             Success
NonlinearSolve, NR (MoreThuente)                  37       25.33μs  true 2.781e-15             Success
NonlinearSolve, Newton Raphson                    10       17.20μs  true 2.781e-15             Success
NonlinearSolve, Pseudo Transient                  10       17.10μs  true 1.606e-09             Success
NonlinearSolve, TR (Bastin)                      139       38.69μs  true 2.781e-15             Success
NonlinearSolve, TR (Fan)                          12       21.15μs  true 9.374e-16             Success
NonlinearSolve, TR (Hei)                          10       18.54μs  true 2.781e-15             Success
NonlinearSolve, TR (NLsolve Update)               10       19.42μs  true 2.781e-15             Success
NonlinearSolve, TR (Nocedal Wright)               14       23.44μs  true 6.079e-10             Success
NonlinearSolve, TR (Yuan)                         94       31.24μs  true 2.781e-15             Success
NonlinearSolve, Trust Region                      14       23.36μs  true 6.079e-10             Success
Speedmapping, aa                                  14        7.67μs  true 3.448e-08         first_order

Trigonometric function: 10 parameters, abstol = 1.0e-7.
Solver                                       F evals          time  conv   |resid|                 log
NLSolve, anderson                                 72      401.24μs false 5.241e+01  x_converged = true
NLSolve, newton                                  189       25.81μs  true 1.593e-15  x_converged = true
NLSolve, trust_region                            195       41.70μs  true 2.692e-15  x_converged = true
NonlinearSolve, Default PolyAlg.                  18       22.91μs  true 7.266e-12             Success
NonlinearSolve, LM (no Accln.)                  1868     8018.46μs  true 5.107e-13             Success
NonlinearSolve, LM (no Accln.) Chol.            3234     2356.37μs  true 5.581e-09             Success
NonlinearSolve, LM (α_geodesic=0.5)          1773713 16172842.98μs false 5.348e-03      Fevals > limit
NonlinearSolve, LM (α_geodesic=0.5) Chol.    1773714  1997269.87μs false 5.308e-03      Fevals > limit
NonlinearSolve, LM with Cholesky             1773714  1658739.09μs false 5.773e-03      Fevals > limit
NonlinearSolve, Levenberg-Marquardt          1773714 15689632.89μs false 5.773e-03      Fevals > limit
NonlinearSolve, NR (BackTracking)                 57       33.68μs  true 4.347e-08             Success
NonlinearSolve, NR (HagerZhang)                  142       49.77μs  true 5.801e-09             Success
NonlinearSolve, NR (MoreThuente)                 360      123.60μs  true 3.122e-08             Success
NonlinearSolve, Newton Raphson                    18       22.84μs  true 7.266e-12             Success
NonlinearSolve, Pseudo Transient             2000003  2025787.12μs false 6.884e+01      Fevals > limit
NonlinearSolve, TR (Bastin)                      321       68.61μs  true 8.918e-10             Success
NonlinearSolve, TR (Fan)                         414      382.99μs false 5.287e-03 ShrinkThresholdExce
NonlinearSolve, TR (Hei)                         256      247.20μs false 5.287e-03 ShrinkThresholdExce
NonlinearSolve, TR (NLsolve Update)              325      298.93μs false 5.287e-03 ShrinkThresholdExce
NonlinearSolve, TR (Nocedal Wright)              267      260.32μs false 5.287e-03 ShrinkThresholdExce
NonlinearSolve, TR (Yuan)                       5757      962.78μs false 5.287e-03 ShrinkThresholdExce
NonlinearSolve, Trust Region                     257      246.31μs false 5.287e-03 ShrinkThresholdExce
Speedmapping, aa                                  49       29.80μs  true 6.432e-10         first_order

Variably dimensioned function: 10 parameters, abstol = 1.0e-7.
Solver                                       F evals          time  conv   |resid|                 log
NLSolve, anderson                                  4     3663.06μs false       NaN NLsolve.IsFiniteExc
NLSolve, newton                                  336       26.60μs  true 0.000e+00  x_converged = true
NLSolve, trust_region                            336       30.89μs  true 0.000e+00  x_converged = true
NonlinearSolve, Default PolyAlg.                  32       33.70μs  true 2.597e-12             Success
NonlinearSolve, LM (no Accln.)                    68      286.87μs  true 6.993e-09             Success
NonlinearSolve, LM (no Accln.) Chol.              68       57.07μs  true 6.993e-09             Success
NonlinearSolve, LM (α_geodesic=0.5)              127      819.34μs  true 9.671e-10             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.        127      120.23μs  true 9.671e-10             Success
NonlinearSolve, LM with Cholesky                 105       96.93μs  true 3.223e-08             Success
NonlinearSolve, Levenberg-Marquardt              105      608.87μs  true 3.223e-08             Success
NonlinearSolve, NR (BackTracking)                 88       43.49μs  true 2.597e-12             Success
NonlinearSolve, NR (HagerZhang)                   43       22.35μs  true 8.529e-09             Success
NonlinearSolve, NR (MoreThuente)                 158       55.25μs  true 2.597e-12             Success
NonlinearSolve, Newton Raphson                    32       32.25μs  true 2.597e-12             Success
NonlinearSolve, Pseudo Transient                  32       33.08μs  true 2.597e-12             Success
NonlinearSolve, TR (Bastin)                      634       75.56μs  true 2.597e-12             Success
NonlinearSolve, TR (Fan)                          32       46.97μs  true 2.597e-12             Success
NonlinearSolve, TR (Hei)                          32       36.63μs  true 2.597e-12             Success
NonlinearSolve, TR (NLsolve Update)               32       36.86μs  true 2.597e-12             Success
NonlinearSolve, TR (Nocedal Wright)               32       42.67μs  true 2.597e-12             Success
NonlinearSolve, TR (Yuan)                        347       59.99μs  true 2.597e-12             Success
NonlinearSolve, Trust Region                      32       37.73μs  true 2.597e-12             Success
Speedmapping, aa                                  32       16.01μs  true 1.694e-08         first_order

Broyden tridiagonal function: 10 parameters, abstol = 1.0e-7.
Solver                                       F evals          time  conv   |resid|                 log
NLSolve, anderson                                 30     3466.13μs false       NaN NLsolve.IsFiniteExc
NLSolve, newton                                  126        9.91μs  true 1.574e-15  x_converged = true
NLSolve, trust_region                            126       11.84μs  true 1.574e-15  x_converged = true
NonlinearSolve, Default PolyAlg.                  12       18.68μs  true 1.062e-09             Success
NonlinearSolve, LM (no Accln.)                    20       87.27μs  true 8.403e-09             Success
NonlinearSolve, LM (no Accln.) Chol.              20       25.52μs  true 8.403e-09             Success
NonlinearSolve, LM (α_geodesic=0.5)               28      150.46μs  true 6.112e-09             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.         28       35.06μs  true 6.112e-09             Success
NonlinearSolve, LM with Cholesky                  28       37.42μs  true 6.112e-09             Success
NonlinearSolve, Levenberg-Marquardt               28      176.32μs  true 6.112e-09             Success
NonlinearSolve, NR (BackTracking)                 28       20.36μs  true 1.062e-09             Success
NonlinearSolve, NR (HagerZhang)                  326      105.08μs false 5.335e+09 InternalLineSearchF
NonlinearSolve, NR (MoreThuente)                  48       29.59μs  true 1.062e-09             Success
NonlinearSolve, Newton Raphson                    12       19.88μs  true 1.062e-09             Success
NonlinearSolve, Pseudo Transient                  12       17.77μs  true 2.970e-09             Success
NonlinearSolve, TR (Bastin)                      184       33.72μs  true 1.062e-09             Success
NonlinearSolve, TR (Fan)                          14       21.24μs  true 4.623e-12             Success
NonlinearSolve, TR (Hei)                          12       19.99μs  true 1.062e-09             Success
NonlinearSolve, TR (NLsolve Update)               12       20.96μs  true 1.062e-09             Success
NonlinearSolve, TR (Nocedal Wright)               14       20.92μs  true 6.750e-12             Success
NonlinearSolve, TR (Yuan)                        117       27.29μs  true 1.062e-09             Success
NonlinearSolve, Trust Region                      14       21.04μs  true 6.750e-12             Success
Speedmapping, aa                                  25       11.62μs  true 4.758e-08         first_order

Broyden banded function: 10 parameters, abstol = 1.0e-7.
Solver                                       F evals          time  conv   |resid|                 log
NLSolve, anderson                                 12     4432.92μs false       NaN NLsolve.IsFiniteExc
NLSolve, newton                                  147       14.27μs  true 9.619e-16  x_converged = true
NLSolve, trust_region                            147       16.24μs  true 9.619e-16  x_converged = true
NonlinearSolve, Default PolyAlg.                  14       23.36μs  true 1.548e-08             Success
NonlinearSolve, LM (no Accln.)                    22       94.99μs  true 3.386e-10             Success
NonlinearSolve, LM (no Accln.) Chol.              22       31.37μs  true 3.386e-10             Success
NonlinearSolve, LM (α_geodesic=0.5)               28      153.76μs  true 8.820e-08             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.         28       40.16μs  true 8.820e-08             Success
NonlinearSolve, LM with Cholesky                  28       39.88μs  true 8.820e-08             Success
NonlinearSolve, Levenberg-Marquardt               28      156.33μs  true 8.820e-08             Success
NonlinearSolve, NR (BackTracking)                 34       30.83μs  true 1.548e-08             Success
NonlinearSolve, NR (HagerZhang)              5000906  3944584.13μs false 2.028e+00      Fevals > limit
NonlinearSolve, NR (MoreThuente)                  59       30.53μs  true 1.548e-08             Success
NonlinearSolve, Newton Raphson                    14       21.69μs  true 1.548e-08             Success
NonlinearSolve, Pseudo Transient                  14       22.10μs  true 1.902e-08             Success
NonlinearSolve, TR (Bastin)                      229       48.26μs  true 1.548e-08             Success
NonlinearSolve, TR (Fan)                          14       24.94μs  true 1.548e-08             Success
NonlinearSolve, TR (Hei)                          14       24.29μs  true 1.548e-08             Success
NonlinearSolve, TR (NLsolve Update)               14       24.73μs  true 1.548e-08             Success
NonlinearSolve, TR (Nocedal Wright)               14       24.08μs  true 1.548e-08             Success
NonlinearSolve, TR (Yuan)                        140       36.54μs  true 1.548e-08             Success
NonlinearSolve, Trust Region                      14       24.23μs  true 1.548e-08             Success
Speedmapping, aa                                  35       20.29μs  true 5.255e-08         first_order

Hammarling 2 by 2 matrix square root problem: 4 parameters, abstol = 1.0e-7.
Solver                                       F evals          time  conv   |resid|                 log
NLSolve, anderson                                 13     6145.95μs false       NaN NLsolve.IsFiniteExc
NLSolve, newton                                  117        7.87μs  true 0.000e+00  x_converged = true
NLSolve, trust_region                            117        9.96μs  true 0.000e+00  x_converged = true
NonlinearSolve, Default PolyAlg.                  24       15.73μs  true 4.711e-08             Success
NonlinearSolve, LM (no Accln.)                    90      131.38μs  true 2.391e-09             Success
NonlinearSolve, LM (no Accln.) Chol.              90       33.91μs  true 2.189e-09             Success
NonlinearSolve, LM (α_geodesic=0.5)              135      264.21μs  true 1.173e-08             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.        135       54.26μs  true 1.150e-08             Success
NonlinearSolve, LM with Cholesky                 130       50.85μs  true 1.557e-09             Success
NonlinearSolve, Levenberg-Marquardt              130      248.11μs  true 1.679e-09             Success
NonlinearSolve, NR (BackTracking)                 64       18.23μs  true 4.711e-08             Success
NonlinearSolve, NR (HagerZhang)                  778      105.84μs  true 5.005e-15             Success
NonlinearSolve, NR (MoreThuente)                 359       64.49μs  true 9.121e-08             Success
NonlinearSolve, Newton Raphson                    24       15.49μs  true 4.711e-08             Success
NonlinearSolve, Pseudo Transient                  32       16.81μs  true 8.379e-11             Success
NonlinearSolve, TR (Bastin)                     1117       84.68μs  true 1.563e-12             Success
NonlinearSolve, TR (Fan)                       22205     7148.15μs  true 9.998e-08             Success
NonlinearSolve, TR (Hei)                       20335     6559.43μs  true 9.996e-08             Success
NonlinearSolve, TR (NLsolve Update)            29263     9214.19μs  true 9.999e-08             Success
NonlinearSolve, TR (Nocedal Wright)            13222     3682.96μs  true 9.999e-08             Success
NonlinearSolve, TR (Yuan)                    9663214  1148109.20μs false 9.999e-08      Fevals > limit
NonlinearSolve, Trust Region                   13802     3788.89μs  true 9.997e-08             Success
Speedmapping, aa                                  38        9.10μs  true 3.996e-09         first_order

Hammarling 3 by 3 matrix square root problem: 9 parameters, abstol = 1.0e-7.
Solver                                       F evals          time  conv   |resid|                 log
NLSolve, anderson                                 13     6001.00μs false       NaN NLsolve.IsFiniteExc
NLSolve, newton                                  247       14.35μs  true 0.000e+00  x_converged = true
NLSolve, trust_region                            247       18.75μs  true 0.000e+00  x_converged = true
NonlinearSolve, Default PolyAlg.                  24       24.55μs  true 4.711e-08             Success
NonlinearSolve, LM (no Accln.)                    90      333.71μs  true 2.391e-09             Success
NonlinearSolve, LM (no Accln.) Chol.              90       64.71μs  true 1.824e-09             Success
NonlinearSolve, LM (α_geodesic=0.5)              134      630.21μs  true 7.459e-11             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.        134       92.13μs  true 5.414e-11             Success
NonlinearSolve, LM with Cholesky                 130       86.94μs  true 1.019e-09             Success
NonlinearSolve, Levenberg-Marquardt              130      597.83μs  true 1.679e-09             Success
NonlinearSolve, NR (BackTracking)                 64       28.38μs  true 4.711e-08             Success
NonlinearSolve, NR (HagerZhang)                 1120      233.07μs  true 8.634e-08             Success
NonlinearSolve, NR (MoreThuente)                 359       95.91μs  true 8.911e-08             Success
NonlinearSolve, Newton Raphson                    24       23.55μs  true 4.711e-08             Success
NonlinearSolve, Pseudo Transient                  32       29.39μs  true 9.637e-14             Success
NonlinearSolve, TR (Bastin)                     2177      222.26μs  true 1.739e-12             Success
NonlinearSolve, TR (Fan)                       15707    15327.64μs  true 9.996e-08             Success
NonlinearSolve, TR (Hei)                       16222    15656.08μs  true 9.997e-08             Success
NonlinearSolve, TR (NLsolve Update)            23388    22515.46μs  true 1.000e-07             Success
NonlinearSolve, TR (Nocedal Wright)            13178    12386.08μs  true 9.998e-08             Success
NonlinearSolve, TR (Yuan)                    2263266   528148.89μs false 1.000e-07      Fevals > limit
NonlinearSolve, Trust Region                   13383    12573.58μs  true 9.999e-08             Success
Speedmapping, aa                                  31       10.70μs  true 3.373e-08         first_order

Dennis and Schnabel 2 by 2 example: 2 parameters, abstol = 1.0e-7.
Solver                                       F evals          time  conv   |resid|                 log
NLSolve, anderson                                 13    18772.84μs false       NaN NLsolve.IsFiniteExc
NLSolve, newton                                   35        4.14μs  true 0.000e+00  x_converged = true
NLSolve, trust_region                             35        4.69μs  true 0.000e+00  x_converged = true
NonlinearSolve, Default PolyAlg.                  14       13.77μs  true 1.098e-11             Success
NonlinearSolve, LM (no Accln.)                    20       22.66μs  true 1.118e-08             Success
NonlinearSolve, LM (no Accln.) Chol.              20       15.91μs  true 1.118e-08             Success
NonlinearSolve, LM (α_geodesic=0.5)               34       46.04μs  true 6.526e-08             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.         34       33.07μs  true 6.526e-08             Success
NonlinearSolve, LM with Cholesky                  33       21.01μs  true 1.571e-09             Success
NonlinearSolve, Levenberg-Marquardt               33       43.60μs  true 1.571e-09             Success
NonlinearSolve, NR (BackTracking)                 34       14.73μs  true 1.098e-11             Success
NonlinearSolve, NR (HagerZhang)                   85       26.88μs  true 2.328e-08             Success
NonlinearSolve, NR (MoreThuente)                  59       22.97μs  true 1.098e-11             Success
NonlinearSolve, Newton Raphson                    14       13.67μs  true 1.098e-11             Success
NonlinearSolve, Pseudo Transient                  14       13.05μs  true 3.974e-12             Success
NonlinearSolve, TR (Bastin)                       69       16.41μs  true 5.771e-08             Success
NonlinearSolve, TR (Fan)                          14       14.00μs  true 2.377e-11             Success
NonlinearSolve, TR (Hei)                          14       15.14μs  true 2.819e-08             Success
NonlinearSolve, TR (NLsolve Update)               14       15.01μs  true 1.098e-11             Success
NonlinearSolve, TR (Nocedal Wright)               14       13.83μs  true 8.366e-11             Success
NonlinearSolve, TR (Yuan)                         44       15.34μs  true 1.098e-11             Success
NonlinearSolve, Trust Region                      14       14.54μs  true 8.366e-11             Success
Speedmapping, aa                                  16        3.53μs  true 4.316e-11         first_order

Sample problem 18: 2 parameters, abstol = 1.0e-7.
Solver                                       F evals          time  conv   |resid|                 log
NLSolve, anderson                                 18       20.33μs  true 1.211e-18  x_converged = true
NLSolve, newton                                   45        4.74μs  true 0.000e+00 x_converged = false
NLSolve, trust_region                             40        5.85μs  true 0.000e+00 x_converged = false
NonlinearSolve, Default PolyAlg.                  14       14.17μs  true 8.199e-09             Success
NonlinearSolve, LM (no Accln.)                    20       22.67μs  true 4.107e-10             Success
NonlinearSolve, LM (no Accln.) Chol.              20       15.34μs  true 4.107e-10             Success
NonlinearSolve, LM (α_geodesic=0.5)               30       41.03μs  true 6.813e-08             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.         30       19.57μs  true 6.834e-08             Success
NonlinearSolve, LM with Cholesky                  29       20.23μs  true 2.198e-09             Success
NonlinearSolve, Levenberg-Marquardt               29       37.14μs  true 2.198e-09             Success
NonlinearSolve, NR (BackTracking)                 34       15.21μs  true 8.199e-09             Success
NonlinearSolve, NR (HagerZhang)                   40       16.18μs  true 2.033e-08             Success
NonlinearSolve, NR (MoreThuente)                 205       40.13μs  true 5.270e-12             Success
NonlinearSolve, Newton Raphson                    14       13.55μs  true 8.199e-09             Success
NonlinearSolve, Pseudo Transient                  16       13.34μs  true 9.503e-12             Success
NonlinearSolve, TR (Bastin)                       82       24.98μs  true 6.181e-10             Success
NonlinearSolve, TR (Fan)                          14       14.42μs  true 1.546e-12             Success
NonlinearSolve, TR (Hei)                          12       14.39μs  true 4.487e-09             Success
NonlinearSolve, TR (NLsolve Update)               10       15.39μs  true 7.809e-09             Success
NonlinearSolve, TR (Nocedal Wright)               18       15.40μs  true 1.614e-08             Success
NonlinearSolve, TR (Yuan)                         44       15.25μs  true 8.199e-09             Success
NonlinearSolve, Trust Region                      18       15.05μs  true 1.614e-08             Success
Speedmapping, aa                                  16        4.15μs  true 1.213e-19         first_order

Sample problem 19: 2 parameters, abstol = 1.0e-7.
Solver                                       F evals          time  conv   |resid|                 log
NLSolve, anderson                                  8     4781.96μs false       NaN NLsolve.IsFiniteExc
NLSolve, newton                                  225       18.38μs  true 3.257e-18  x_converged = true
NLSolve, trust_region                            225       19.84μs  true 3.257e-18  x_converged = true
NonlinearSolve, Default PolyAlg.                  38       16.06μs  true 7.985e-08             Success
NonlinearSolve, LM (no Accln.)                    56       47.06μs  true 4.966e-08             Success
NonlinearSolve, LM (no Accln.) Chol.              56       20.68μs  true 4.966e-08             Success
NonlinearSolve, LM (α_geodesic=0.5)              111      126.81μs  true 7.132e-08             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.        111       37.69μs  true 7.132e-08             Success
NonlinearSolve, LM with Cholesky                  79       27.78μs  true 8.444e-08             Success
NonlinearSolve, Levenberg-Marquardt               79       79.30μs  true 8.444e-08             Success
NonlinearSolve, NR (BackTracking)                106       20.81μs  true 7.985e-08             Success
NonlinearSolve, NR (HagerZhang)                   18       13.79μs  true 7.487e-23             Success
NonlinearSolve, NR (MoreThuente)                 191       35.63μs  true 7.985e-08             Success
NonlinearSolve, Newton Raphson                    38       15.82μs  true 7.985e-08             Success
NonlinearSolve, Pseudo Transient                  38       15.50μs  true 8.051e-08             Success
NonlinearSolve, TR (Bastin)                      238       24.15μs  true 3.980e-08             Success
NonlinearSolve, TR (Fan)                          38       18.06μs  true 7.985e-08             Success
NonlinearSolve, TR (Hei)                          40       18.29μs  true 3.565e-08             Success
NonlinearSolve, TR (NLsolve Update)               38       17.88μs  true 7.985e-08             Success
NonlinearSolve, TR (Nocedal Wright)               38       18.55μs  true 7.985e-08             Success
NonlinearSolve, TR (Yuan)                        142       20.96μs  true 6.436e-08             Success
NonlinearSolve, Trust Region                      38       18.11μs  true 7.985e-08             Success
Speedmapping, aa                                  31        7.58μs  true 9.339e-08         first_order

Scalar problem f(x) = x(x - 5)^2: 1 parameters, abstol = 1.0e-7.
Solver                                       F evals          time  conv   |resid|                 log
NLSolve, anderson                                 11        9.66μs  true 3.581e-11  x_converged = true
NLSolve, newton                                   21        2.53μs  true 1.940e-15  x_converged = true
NLSolve, trust_region                              6        1.97μs  true 0.000e+00 x_converged = false
NonlinearSolve, Default PolyAlg.                  16       13.06μs  true 1.939e-15             Success
NonlinearSolve, LM (no Accln.)                     6       12.96μs  true 1.110e-14             Success
NonlinearSolve, LM (no Accln.) Chol.               6       13.08μs  true 5.551e-15             Success
NonlinearSolve, LM (α_geodesic=0.5)               30       30.94μs  true 1.801e-09             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.         30       19.96μs  true 1.801e-09             Success
NonlinearSolve, LM with Cholesky                  30       20.02μs  true 1.801e-09             Success
NonlinearSolve, Levenberg-Marquardt               30       26.27μs  true 1.801e-09             Success
NonlinearSolve, NR (BackTracking)                 41       13.98μs  true 2.913e-14             Success
NonlinearSolve, NR (HagerZhang)                   63       16.48μs  true 8.712e-08             Success
NonlinearSolve, NR (MoreThuente)                 263       38.18μs  true 9.807e-08             Success
NonlinearSolve, Newton Raphson                    16       12.39μs  true 1.939e-15             Success
NonlinearSolve, Pseudo Transient                  16       12.47μs  true 1.633e-15             Success
NonlinearSolve, TR (Bastin)                       13       15.19μs  true 0.000e+00             Success
NonlinearSolve, TR (Fan)                          15       13.51μs  true 9.075e-08             Success
NonlinearSolve, TR (Hei)                           6       12.49μs  true 0.000e+00             Success
NonlinearSolve, TR (NLsolve Update)                6       12.73μs  true 0.000e+00             Success
NonlinearSolve, TR (Nocedal Wright)               14       13.10μs  true 1.109e-11             Success
NonlinearSolve, TR (Yuan)                         57       14.84μs  true 6.672e-13             Success
NonlinearSolve, Trust Region                      14       12.61μs  true 1.109e-11             Success
Speedmapping, aa                                  28        4.46μs  true 9.061e-08         first_order

Freudenstein-Roth function: 2 parameters, abstol = 1.0e-7.
Solver                                       F evals          time  conv   |resid|                 log
NLSolve, anderson                                 46     6483.08μs false       NaN NLsolve.IsFiniteExc
NLSolve, newton                                  365       24.28μs  true 1.589e-14  x_converged = true
NLSolve, trust_region                            168       42.10μs false 6.999e+00  x_converged = true
NonlinearSolve, Default PolyAlg.                  88       22.06μs  true 5.639e-10             Success
NonlinearSolve, LM (no Accln.)                   144      107.21μs  true 6.107e-09             Success
NonlinearSolve, LM (no Accln.) Chol.             156       36.22μs  true 6.720e-09             Success
NonlinearSolve, LM (α_geodesic=0.5)          1773714  3394726.99μs false 7.054e+00      Fevals > limit
NonlinearSolve, LM (α_geodesic=0.5) Chol.    1773714   910011.77μs false 7.263e+00      Fevals > limit
NonlinearSolve, LM with Cholesky             1773714   552028.18μs false 7.112e+00      Fevals > limit
NonlinearSolve, Levenberg-Marquardt          1773714  2960766.79μs false 7.378e+00      Fevals > limit
NonlinearSolve, NR (BackTracking)           65999661 20240458.97μs false 7.611e+00      Fevals > limit
NonlinearSolve, NR (HagerZhang)                   58       17.35μs false 1.188e+09 InternalLineSearchF
NonlinearSolve, NR (MoreThuente)                  53       19.69μs false 9.130e+12 InternalLineSearchF
NonlinearSolve, Newton Raphson                    88       21.42μs  true 5.639e-10             Success
NonlinearSolve, Pseudo Transient                 114       24.98μs  true 4.727e-08             Success
NonlinearSolve, TR (Bastin)                      504       58.65μs false 6.999e+00 ShrinkThresholdExce
NonlinearSolve, TR (Fan)                         109       42.56μs false 6.999e+00 ShrinkThresholdExce
NonlinearSolve, TR (Hei)                         109       39.08μs false 6.999e+00 ShrinkThresholdExce
NonlinearSolve, TR (NLsolve Update)              115       41.01μs false 6.999e+00 ShrinkThresholdExce
NonlinearSolve, TR (Nocedal Wright)              102       37.51μs false 6.999e+00 ShrinkThresholdExce
NonlinearSolve, TR (Yuan)                        511       57.09μs false 6.999e+00 ShrinkThresholdExce
NonlinearSolve, Trust Region                     102       36.34μs false 6.999e+00 ShrinkThresholdExce
Speedmapping, aa                                  35        7.32μs  true 1.579e-10         first_order

Boggs function: 2 parameters, abstol = 1.0e-7.
Solver                                       F evals          time  conv   |resid|                 log
NLSolve, anderson                                 15       16.95μs  true 1.251e-08  x_converged = true
NLSolve, newton                                   25        3.60μs  true 6.661e-16  x_converged = true
NLSolve, trust_region                             41        5.97μs  true 4.950e-19  x_converged = true
NonlinearSolve, Default PolyAlg.                  10       13.06μs  true 0.000e+00             Success
NonlinearSolve, LM (no Accln.)                    22       25.09μs  true 6.630e-10             Success
NonlinearSolve, LM (no Accln.) Chol.              22       15.78μs  true 6.630e-10             Success
NonlinearSolve, LM (α_geodesic=0.5)               32       39.76μs  true 1.131e-08             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.         32       19.97μs  true 1.131e-08             Success
NonlinearSolve, LM with Cholesky                  28       19.02μs  true 8.650e-08             Success
NonlinearSolve, Levenberg-Marquardt               28       34.94μs  true 8.650e-08             Success
NonlinearSolve, NR (BackTracking)                 41       16.02μs  true 8.470e-08             Success
NonlinearSolve, NR (HagerZhang)                  102       23.29μs  true 7.087e-10             Success
NonlinearSolve, NR (MoreThuente)                 480       85.30μs  true 8.234e-08             Success
NonlinearSolve, Newton Raphson                    10       12.48μs  true 0.000e+00             Success
NonlinearSolve, Pseudo Transient                  18       13.94μs  true 3.187e-14             Success
NonlinearSolve, TR (Bastin)                       82       16.93μs  true 6.477e-12             Success
NonlinearSolve, TR (Fan)                          18       15.60μs  true 6.326e-11             Success
NonlinearSolve, TR (Hei)                          16       15.57μs  true 6.477e-12             Success
NonlinearSolve, TR (NLsolve Update)               16       14.35μs  true 6.477e-12             Success
NonlinearSolve, TR (Nocedal Wright)               18       14.75μs  true 1.336e-11             Success
NonlinearSolve, TR (Yuan)                         57       16.31μs  true 1.119e-08             Success
NonlinearSolve, Trust Region                      18       14.56μs  true 1.336e-11             Success
Speedmapping, aa                                  50       10.64μs  true 9.268e-11         first_order

Chandrasekhar function: 10 parameters, abstol = 1.0e-7.
Solver                                       F evals          time  conv   |resid|                 log
NLSolve, anderson                               1031     9367.94μs false       NaN NLsolve.IsFiniteExc
NLSolve, newton                                  126       26.63μs  true 4.441e-16  x_converged = true
NLSolve, trust_region                            126       28.12μs  true 4.441e-16  x_converged = true
NonlinearSolve, Default PolyAlg.                  12       21.62μs  true 4.955e-14             Success
NonlinearSolve, LM (no Accln.)                    18       81.93μs  true 9.729e-08             Success
NonlinearSolve, LM (no Accln.) Chol.              18       33.42μs  true 9.729e-08             Success
NonlinearSolve, LM (α_geodesic=0.5)               25      145.64μs  true 7.468e-08             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.         25       35.00μs  true 7.468e-08             Success
NonlinearSolve, LM with Cholesky                  25       35.66μs  true 7.468e-08             Success
NonlinearSolve, Levenberg-Marquardt               25      138.00μs  true 7.468e-08             Success
NonlinearSolve, NR (BackTracking)                 28       25.39μs  true 4.955e-14             Success
NonlinearSolve, NR (HagerZhang)                   43       29.85μs  true 2.396e-08             Success
NonlinearSolve, NR (MoreThuente)                  48       33.83μs  true 4.955e-14             Success
NonlinearSolve, Newton Raphson                    12       21.57μs  true 4.955e-14             Success
NonlinearSolve, Pseudo Transient                  12       21.91μs  true 3.441e-09             Success
NonlinearSolve, TR (Bastin)                      184       60.73μs  true 2.316e-08             Success
NonlinearSolve, TR (Fan)                          14       25.85μs  true 5.481e-09             Success
NonlinearSolve, TR (Hei)                          12       23.82μs  true 2.316e-08             Success
NonlinearSolve, TR (NLsolve Update)               12       23.24μs  true 4.955e-14             Success
NonlinearSolve, TR (Nocedal Wright)               18       30.07μs  true 1.558e-10             Success
NonlinearSolve, TR (Yuan)                        117       47.93μs  true 3.364e-10             Success
NonlinearSolve, Trust Region                      18       30.52μs  true 1.558e-10             Success
Speedmapping, aa                                  14        9.09μs  true 1.278e-08         first_order
=#
