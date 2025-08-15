# Dire: The goal is to see how each algorithm performs on its own, but some speed gain may be obtained
# by tayloring the ad to each specific problem like in ... Interestingly, the NonlinearSolve, Default PolyAlg. did
# solve all 23 problems.

using BenchmarkTools, NonlinearProblemLibrary, NonlinearSolve, NLsolve, JLD2, FileIO, DiffEqBase, SpeedMapping
path_out = ""
include("Benchmarking_utils.jl")

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

title = "Performance of various Julia solvers for nonlinear problems"
plot_res(nl_res_all, nlproblem_names_len, nlsolver_names, title, path_out*"nl_res_all.svg"; 
	size = (800, 500), legend_rowgap = -5)

#=
Generalized Rosenbrock function: 10 parameters, abstol = 1.0e-7.
Solver                                   f evals          time  conv     resid                 log
NLSolve, anderson                             19     4076.00μs false       NaN NLsolve.IsFiniteExc
NLSolve, newton                               84       20.90μs false 1.813e+72  x_converged = true
NLSolve, trust_region                       5048      352.97μs  true 0.000e+00  x_converged = true
NonlinearSolve, Default PolyAlg.              26       36.82μs  true 0.000e+00             Success
NonlinearSolve, LM (no Accln.)                56      236.67μs  true 5.555e-09             Success
NonlinearSolve, LM (no Accln.) Chol.          56       63.04μs  true 5.555e-09             Success
NonlinearSolve, LM (α_geodesic=0.5)          287     2179.16μs  true 2.158e-08             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.    287      256.45μs  true 2.158e-08             Success
NonlinearSolve, LM with Cholesky             204      189.74μs  true 5.623e-09             Success
NonlinearSolve, Levenberg-Marquardt          204     1485.24μs  true 5.623e-09             Success
NonlinearSolve, NR (BackTracking)           1287      382.35μs  true 1.554e-14             Success
NonlinearSolve, NR (HagerZhang)             2014      265.17μs  true 1.276e-10             Success
NonlinearSolve, NR (MoreThuente)            6109      910.84μs  true 9.142e-08             Success
NonlinearSolve, Newton Raphson                26       36.42μs  true 0.000e+00             Success
NonlinearSolve, Pseudo Transient          200003   224741.94μs false 7.568e+08            MaxIters
NonlinearSolve, TR (Bastin)                 8557      472.45μs  true 2.884e-08             Success
NonlinearSolve, TR (Fan)                    1683      735.30μs  true 1.455e-08             Success
NonlinearSolve, TR (Hei)                     736      324.51μs  true 3.006e-14             Success
NonlinearSolve, TR (NLsolve Update)         1126      482.80μs  true 7.850e-15             Success
NonlinearSolve, TR (Nocedal Wright)          769      327.06μs  true 3.671e-13             Success
NonlinearSolve, TR (Yuan)                  57896     3669.86μs  true 9.129e-08             Success
NonlinearSolve, Trust Region                 769      321.96μs  true 3.671e-13             Success
Speedmapping, aa                             346      195.29μs  true 1.640e-08         first_order

Powell singular function: 4 parameters, abstol = 1.0e-7.
Solver                                   f evals          time  conv     resid                 log
NLSolve, anderson                             16     2488.85μs false       NaN NLsolve.IsFiniteExc
NLSolve, newton                              234       12.88μs  true 1.127e-14  x_converged = true
NLSolve, trust_region                        234       16.55μs  true 1.127e-14  x_converged = true
NonlinearSolve, Default PolyAlg.              32       34.96μs  true 4.727e-08             Success
NonlinearSolve, LM (no Accln.)                44       85.50μs  true 5.768e-08             Success
NonlinearSolve, LM (no Accln.) Chol.          44       39.83μs  true 5.768e-08             Success
NonlinearSolve, LM (α_geodesic=0.5)           79      186.82μs  true 6.328e-08             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.     79       54.78μs  true 6.328e-08             Success
NonlinearSolve, LM with Cholesky              61       46.98μs  true 9.054e-08             Success
NonlinearSolve, Levenberg-Marquardt           61      133.59μs  true 9.054e-08             Success
NonlinearSolve, NR (BackTracking)             88       38.78μs  true 4.727e-08             Success
NonlinearSolve, NR (HagerZhang)              149       45.81μs  true 5.159e-08             Success
NonlinearSolve, NR (MoreThuente)             158       50.40μs  true 4.727e-08             Success
NonlinearSolve, Newton Raphson                32       33.96μs  true 4.727e-08             Success
NonlinearSolve, Pseudo Transient              32       34.59μs  true 4.799e-08             Success
NonlinearSolve, TR (Bastin)                  298       46.97μs  true 7.328e-08             Success
NonlinearSolve, TR (Fan)                      32       36.52μs  true 4.602e-08             Success
NonlinearSolve, TR (Hei)                      32       36.95μs  true 4.527e-08             Success
NonlinearSolve, TR (NLsolve Update)           32       37.10μs  true 4.727e-08             Success
NonlinearSolve, TR (Nocedal Wright)           32       37.15μs  true 4.586e-08             Success
NonlinearSolve, TR (Yuan)                    211       48.64μs  true 4.546e-08             Success
NonlinearSolve, Trust Region                  32       36.63μs  true 4.586e-08             Success
Speedmapping, aa                              35       11.41μs  true 8.920e-08         first_order

Powell badly scaled function: 2 parameters, abstol = 1.0e-7.
Solver                                   f evals          time  conv     resid                 log
NLSolve, anderson                             22       25.29μs  true 2.141e-08  x_converged = true
NLSolve, newton                               70        8.79μs  true 0.000e+00  x_converged = true
NLSolve, trust_region                         76        9.75μs  true 0.000e+00  x_converged = true
NonlinearSolve, Default PolyAlg.              28       32.91μs  true 1.573e-11             Success
NonlinearSolve, LM (no Accln.)                44       56.76μs  true 3.128e-08             Success
NonlinearSolve, LM (no Accln.) Chol.          44       37.20μs  true 3.128e-08             Success
NonlinearSolve, LM (α_geodesic=0.5)           70       99.65μs  true 1.974e-08             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.     70       45.94μs  true 1.974e-08             Success
NonlinearSolve, LM with Cholesky              61       43.90μs  true 2.939e-08             Success
NonlinearSolve, Levenberg-Marquardt           61       84.58μs  true 2.939e-08             Success
NonlinearSolve, NR (BackTracking)            353       86.66μs  true 6.227e-12             Success
NonlinearSolve, NR (HagerZhang)              661      122.51μs  true 1.550e-08             Success
NonlinearSolve, NR (MoreThuente)             816      153.03μs  true 8.931e-08             Success
NonlinearSolve, Newton Raphson                28       31.77μs  true 1.573e-11             Success
NonlinearSolve, Pseudo Transient              56       36.03μs  true 8.264e-08             Success
NonlinearSolve, TR (Bastin)                  747       81.96μs  true 1.325e-11             Success
NonlinearSolve, TR (Fan)                     432      157.71μs  true 1.349e-11             Success
NonlinearSolve, TR (Hei)                     158       69.20μs  true 1.368e-12             Success
NonlinearSolve, TR (NLsolve Update)          219       85.25μs  true 1.656e-12             Success
NonlinearSolve, TR (Nocedal Wright)          106       52.71μs  true 3.164e-12             Success
NonlinearSolve, TR (Yuan)                  36134     2974.25μs  true 6.106e-13             Success
NonlinearSolve, Trust Region                 106       52.40μs  true 3.164e-12             Success
Speedmapping, aa                              31        7.70μs  true 1.619e-08         first_order

Wood function: 4 parameters, abstol = 1.0e-7.
Solver                                   f evals          time  conv     resid                 log
NLSolve, anderson                              8     3570.08μs false       NaN NLsolve.IsFiniteExc
NLSolve, newton                              135        8.87μs  true 7.178e-14  x_converged = true
NLSolve, trust_region                        148       14.67μs  true 2.450e-14  x_converged = true
NonlinearSolve, Default PolyAlg.              32       34.69μs  true 8.994e-14             Success
NonlinearSolve, LM (no Accln.)                52       95.43μs  true 8.953e-08             Success
NonlinearSolve, LM (no Accln.) Chol.          52       42.80μs  true 8.953e-08             Success
NonlinearSolve, LM (α_geodesic=0.5)           76      161.21μs  true 1.084e-08             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.     76       52.40μs  true 1.084e-08             Success
NonlinearSolve, LM with Cholesky              76       51.69μs  true 1.084e-08             Success
NonlinearSolve, Levenberg-Marquardt           76      162.48μs  true 1.084e-08             Success
NonlinearSolve, NR (BackTracking)            112       44.78μs  true 4.502e-13             Success
NonlinearSolve, NR (HagerZhang)              236       56.95μs  true 1.019e-10             Success
NonlinearSolve, NR (MoreThuente)             659      124.85μs  true 1.709e-08             Success
NonlinearSolve, Newton Raphson                32       34.26μs  true 8.994e-14             Success
NonlinearSolve, Pseudo Transient              32       33.32μs  true 2.755e-13             Success
NonlinearSolve, TR (Bastin)                  511       64.34μs  true 4.022e-13             Success
NonlinearSolve, TR (Fan)                      73       56.22μs  true 2.128e-10             Success
NonlinearSolve, TR (Hei)                      79       55.62μs  true 9.155e-10             Success
NonlinearSolve, TR (NLsolve Update)           83       55.06μs  true 5.940e-13             Success
NonlinearSolve, TR (Nocedal Wright)           44       41.99μs  true 3.143e-08             Success
NonlinearSolve, TR (Yuan)                   1437      140.25μs  true 4.452e-10             Success
NonlinearSolve, Trust Region                  48       45.63μs  true 1.371e-08             Success
Speedmapping, aa                             351      100.07μs  true 4.900e-08         first_order

Helical valley function: 3 parameters, abstol = 1.0e-7.
Solver                                   f evals          time  conv     resid                 log
NLSolve, anderson                              3     4437.21μs false       NaN   SingularException
NLSolve, newton                               77        6.61μs  true 1.326e-14  x_converged = true
NLSolve, trust_region                         64        8.71μs  true 1.166e-14  x_converged = true
NonlinearSolve, Default PolyAlg.              24       32.88μs  true 1.325e-14             Success
NonlinearSolve, LM (no Accln.)                26       54.02μs  true 9.278e-09             Success
NonlinearSolve, LM (no Accln.) Chol.          26       35.54μs  true 9.278e-09             Success
NonlinearSolve, LM (α_geodesic=0.5)           41       85.36μs  true 1.065e-09             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.     41       42.07μs  true 1.065e-09             Success
NonlinearSolve, LM with Cholesky              37       39.66μs  true 1.141e-08             Success
NonlinearSolve, Levenberg-Marquardt           37       78.48μs  true 1.141e-08             Success
NonlinearSolve, NR (BackTracking)             60       36.40μs  true 4.089e-11             Success
NonlinearSolve, NR (HagerZhang)              113       41.70μs  true 1.354e-10             Success
NonlinearSolve, NR (MoreThuente)             557      104.78μs  true 7.721e-08             Success
NonlinearSolve, Newton Raphson                24       33.43μs  true 1.325e-14             Success
NonlinearSolve, Pseudo Transient              24       32.25μs  true 5.155e-14             Success
NonlinearSolve, TR (Bastin)                  157       39.90μs  true 9.841e-11             Success
NonlinearSolve, TR (Fan)                      20       33.72μs  true 3.087e-10             Success
NonlinearSolve, TR (Hei)                      23       34.95μs  true 2.774e-15             Success
NonlinearSolve, TR (NLsolve Update)           21       34.12μs  true 7.569e-08             Success
NonlinearSolve, TR (Nocedal Wright)           22       33.37μs  true 5.127e-11             Success
NonlinearSolve, TR (Yuan)                    107       37.53μs  true 4.549e-08             Success
NonlinearSolve, Trust Region                  22       34.58μs  true 5.127e-11             Success
Speedmapping, aa                              81       21.19μs  true 1.461e-09         first_order

Watson function: 2 parameters, abstol = 1.0e-7.
Solver                                   f evals          time  conv     resid                 log
NLSolve, anderson                             10     3974.20μs false       NaN NLsolve.IsFiniteExc
NLSolve, newton                               55       11.76μs  true 3.975e-15  x_converged = true
NLSolve, trust_region                         77       19.89μs  true 2.688e-15  x_converged = true
NonlinearSolve, Default PolyAlg.              22       35.06μs  true 4.214e-13             Success
NonlinearSolve, LM (no Accln.)                22       45.21μs  true 7.069e-08             Success
NonlinearSolve, LM (no Accln.) Chol.          22       36.48μs  true 7.069e-08             Success
NonlinearSolve, LM (α_geodesic=0.5)           40       74.48μs  true 3.386e-08             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.     40       45.52μs  true 3.386e-08             Success
NonlinearSolve, LM with Cholesky              39       45.59μs  true 6.032e-10             Success
NonlinearSolve, Levenberg-Marquardt           39       71.40μs  true 6.032e-10             Success
NonlinearSolve, NR (BackTracking)             54       41.69μs  true 9.452e-11             Success
NonlinearSolve, NR (HagerZhang)              105       52.06μs  true 3.093e-10             Success
NonlinearSolve, NR (MoreThuente)             683      191.23μs  true 5.611e-15             Success
NonlinearSolve, Newton Raphson                22       34.95μs  true 4.214e-13             Success
NonlinearSolve, Pseudo Transient              20       34.05μs  true 7.951e-09             Success
NonlinearSolve, TR (Bastin)                   82       44.43μs  true 3.202e-15             Success
NonlinearSolve, TR (Fan)                      43       47.07μs  true 8.412e-12             Success
NonlinearSolve, TR (Hei)                      14       34.94μs  true 7.826e-13             Success
NonlinearSolve, TR (NLsolve Update)           25       38.66μs  true 6.171e-14             Success
NonlinearSolve, TR (Nocedal Wright)           17       34.87μs  true 2.854e-13             Success
NonlinearSolve, TR (Yuan)                    289       90.25μs  true 4.820e-15             Success
NonlinearSolve, Trust Region                  18       35.62μs  true 1.495e-12             Success
Speedmapping, aa                              30        8.57μs  true 1.360e-09         first_order

Chebyquad function: 2 parameters, abstol = 1.0e-7.
Solver                                   f evals          time  conv     resid                 log
NLSolve, anderson                             13     3957.99μs false       NaN NLsolve.IsFiniteExc
NLSolve, newton                               35        4.41μs  true 5.551e-17  x_converged = true
NLSolve, trust_region                         32        5.71μs  true 9.437e-16  x_converged = true
NonlinearSolve, Default PolyAlg.              14       30.67μs  true 3.685e-12             Success
NonlinearSolve, LM (no Accln.)                18       39.34μs  true 1.169e-08             Success
NonlinearSolve, LM (no Accln.) Chol.          18       33.33μs  true 1.169e-08             Success
NonlinearSolve, LM (α_geodesic=0.5)           30       57.98μs  true 2.873e-08             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.     30       38.48μs  true 2.873e-08             Success
NonlinearSolve, LM with Cholesky              25       37.12μs  true 6.729e-09             Success
NonlinearSolve, Levenberg-Marquardt           25       50.13μs  true 6.729e-09             Success
NonlinearSolve, NR (BackTracking)             35       33.40μs  true 1.951e-09             Success
NonlinearSolve, NR (HagerZhang)               74       36.92μs  true 9.645e-10             Success
NonlinearSolve, NR (MoreThuente)             381       80.83μs  true 6.962e-08             Success
NonlinearSolve, Newton Raphson                14       30.66μs  true 3.685e-12             Success
NonlinearSolve, Pseudo Transient              14       30.76μs  true 1.059e-11             Success
NonlinearSolve, TR (Bastin)                   83       35.40μs  true 9.381e-15             Success
NonlinearSolve, TR (Fan)                      16       32.04μs  true 8.729e-13             Success
NonlinearSolve, TR (Hei)                      17       32.11μs  true 8.687e-14             Success
NonlinearSolve, TR (NLsolve Update)           15       32.40μs  true 1.874e-08             Success
NonlinearSolve, TR (Nocedal Wright)           18       32.33μs  true 4.269e-14             Success
NonlinearSolve, TR (Yuan)                     44       32.38μs  true 1.373e-11             Success
NonlinearSolve, Trust Region                  18       32.38μs  true 4.269e-14             Success
Speedmapping, aa                              13        3.57μs  true 5.005e-08         first_order

Brown almost linear function: 10 parameters, abstol = 1.0e-7.
Solver                                   f evals          time  conv     resid                 log
NLSolve, anderson                              9     4884.00μs false       NaN NLsolve.IsFiniteExc
NLSolve, newton                             1932      130.84μs  true 1.070e-14  x_converged = true
NLSolve, trust_region                        126       12.40μs  true 2.220e-16  x_converged = true
NonlinearSolve, Default PolyAlg.             184      142.96μs  true 9.545e-12             Success
NonlinearSolve, LM (no Accln.)                28      130.59μs  true 3.202e-08             Success
NonlinearSolve, LM (no Accln.) Chol.          28       46.17μs  true 3.202e-08             Success
NonlinearSolve, LM (α_geodesic=0.5)           49      298.11μs  true 3.162e-09             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.     49       63.28μs  true 3.162e-09             Success
NonlinearSolve, LM with Cholesky              45       59.85μs  true 4.805e-08             Success
NonlinearSolve, Levenberg-Marquardt           45      266.81μs  true 4.804e-08             Success
NonlinearSolve, NR (BackTracking)           1056   307903.05μs false       NaN InternalLineSearchF
NonlinearSolve, NR (HagerZhang)              181       52.89μs false 1.801e+67 InternalLineSearchF
NonlinearSolve, NR (MoreThuente)             552      163.02μs  true 4.918e-13             Success
NonlinearSolve, Newton Raphson               184      142.28μs  true 9.545e-12             Success
NonlinearSolve, Pseudo Transient          200003   266855.00μs false 1.358e+18            MaxIters
NonlinearSolve, TR (Bastin)                  319       63.69μs  true 5.718e-09             Success
NonlinearSolve, TR (Fan)                      14       38.85μs  true 1.614e-10             Success
NonlinearSolve, TR (Hei)                      18       43.46μs  true 2.285e-10             Success
NonlinearSolve, TR (NLsolve Update)           10       36.63μs  true 1.653e-10             Success
NonlinearSolve, TR (Nocedal Wright)           12       37.86μs  true 1.241e-08             Success
NonlinearSolve, TR (Yuan)                    751      111.74μs false 1.653e+01 ShrinkThresholdExce
NonlinearSolve, Trust Region                  12       37.20μs  true 1.241e-08             Success
Speedmapping, aa                              18        7.97μs  true 3.685e-09         first_order

Discrete boundary value function: 10 parameters, abstol = 1.0e-7.
Solver                                   f evals          time  conv     resid                 log
NLSolve, anderson                             15       48.18μs  true 1.290e-09  x_converged = true
NLSolve, newton                              105        8.97μs  true 5.004e-17  x_converged = true
NLSolve, trust_region                        105       10.39μs  true 5.004e-17  x_converged = true
NonlinearSolve, Default PolyAlg.               8       33.70μs  true 3.110e-08             Success
NonlinearSolve, LM (no Accln.)                26      129.82μs  true 2.153e-08             Success
NonlinearSolve, LM (no Accln.) Chol.          26       45.04μs  true 2.153e-08             Success
NonlinearSolve, LM (α_geodesic=0.5)           37      212.85μs  true 2.192e-08             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.     37       53.00μs  true 2.192e-08             Success
NonlinearSolve, LM with Cholesky              37       52.40μs  true 2.192e-08             Success
NonlinearSolve, Levenberg-Marquardt           37      216.80μs  true 2.192e-08             Success
NonlinearSolve, NR (BackTracking)             16       35.96μs  true 3.110e-08             Success
NonlinearSolve, NR (HagerZhang)               29       35.84μs  true 3.071e-09             Success
NonlinearSolve, NR (MoreThuente)              26       36.98μs  true 3.110e-08             Success
NonlinearSolve, Newton Raphson                 8       32.37μs  true 3.110e-08             Success
NonlinearSolve, Pseudo Transient              14       37.05μs  true 9.636e-11             Success
NonlinearSolve, TR (Bastin)                   94       41.53μs  true 3.110e-08             Success
NonlinearSolve, TR (Fan)                      12       36.96μs  true 1.526e-08             Success
NonlinearSolve, TR (Hei)                       8       34.88μs  true 3.110e-08             Success
NonlinearSolve, TR (NLsolve Update)            8       34.09μs  true 3.110e-08             Success
NonlinearSolve, TR (Nocedal Wright)           14       38.89μs  true 1.784e-09             Success
NonlinearSolve, TR (Yuan)                    117       46.89μs  true 4.589e-13             Success
NonlinearSolve, Trust Region                  14       39.80μs  true 1.784e-09             Success
Speedmapping, aa                              19        9.79μs  true 6.548e-09         first_order

Discrete integral equation function: 10 parameters, abstol = 1.0e-7.
Solver                                   f evals          time  conv     resid                 log
NLSolve, anderson                             19     5151.03μs false       NaN NLsolve.IsFiniteExc
NLSolve, newton                              105       16.24μs  true 8.002e-17  x_converged = true
NLSolve, trust_region                        105       18.14μs  true 8.002e-17  x_converged = true
NonlinearSolve, Default PolyAlg.              10       38.36μs  true 2.781e-15             Success
NonlinearSolve, LM (no Accln.)                16       88.63μs  true 2.034e-09             Success
NonlinearSolve, LM (no Accln.) Chol.          16       43.49μs  true 2.034e-09             Success
NonlinearSolve, LM (α_geodesic=0.5)           22      138.66μs  true 2.063e-09             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.     22       49.85μs  true 2.063e-09             Success
NonlinearSolve, LM with Cholesky              22       49.16μs  true 2.063e-09             Success
NonlinearSolve, Levenberg-Marquardt           22      139.15μs  true 2.063e-09             Success
NonlinearSolve, NR (BackTracking)             22       40.59μs  true 2.781e-15             Success
NonlinearSolve, NR (HagerZhang)               29       39.31μs  true 1.021e-08             Success
NonlinearSolve, NR (MoreThuente)              37       41.88μs  true 2.781e-15             Success
NonlinearSolve, Newton Raphson                10       36.10μs  true 2.781e-15             Success
NonlinearSolve, Pseudo Transient              10       35.71μs  true 1.606e-09             Success
NonlinearSolve, TR (Bastin)                  139       58.86μs  true 2.781e-15             Success
NonlinearSolve, TR (Fan)                      12       39.66μs  true 9.374e-16             Success
NonlinearSolve, TR (Hei)                      10       38.53μs  true 2.781e-15             Success
NonlinearSolve, TR (NLsolve Update)           10       37.59μs  true 2.781e-15             Success
NonlinearSolve, TR (Nocedal Wright)           14       41.32μs  true 6.079e-10             Success
NonlinearSolve, TR (Yuan)                     94       52.59μs  true 2.781e-15             Success
NonlinearSolve, Trust Region                  14       41.50μs  true 6.079e-10             Success
Speedmapping, aa                              14        8.36μs  true 3.448e-08         first_order

Trigonometric function: 10 parameters, abstol = 1.0e-7.
Solver                                   f evals          time  conv     resid                 log
NLSolve, anderson                             72      401.36μs false 5.241e+01  x_converged = true
NLSolve, newton                              189       24.80μs  true 1.593e-15  x_converged = true
NLSolve, trust_region                        195       43.53μs  true 2.692e-15  x_converged = true
NonlinearSolve, Default PolyAlg.              18       41.89μs  true 7.266e-12             Success
NonlinearSolve, LM (no Accln.)              1868     8012.18μs  true 5.107e-13             Success
NonlinearSolve, LM (no Accln.) Chol.        3234     2230.14μs  true 5.581e-09             Success
NonlinearSolve, LM (α_geodesic=0.5)       177378  2244040.97μs false 5.384e-03            MaxIters
NonlinearSolve, LM (α_geodesic=0.5) Chol. 177378   523592.00μs false 5.384e-03            MaxIters
NonlinearSolve, LM with Cholesky          177378   182215.93μs false 5.500e-03            MaxIters
NonlinearSolve, Levenberg-Marquardt       177378  1720284.22μs false 5.500e-03            MaxIters
NonlinearSolve, NR (BackTracking)             57       57.71μs  true 4.347e-08             Success
NonlinearSolve, NR (HagerZhang)              142       72.99μs  true 5.801e-09             Success
NonlinearSolve, NR (MoreThuente)             360      139.63μs  true 3.122e-08             Success
NonlinearSolve, Newton Raphson                18       41.17μs  true 7.266e-12             Success
NonlinearSolve, Pseudo Transient          200003   365064.14μs false 3.819e+01            MaxIters
NonlinearSolve, TR (Bastin)                  321       89.03μs  true 8.918e-10             Success
NonlinearSolve, TR (Fan)                     414      374.72μs false 5.287e-03 ShrinkThresholdExce
NonlinearSolve, TR (Hei)                     256      249.05μs false 5.287e-03 ShrinkThresholdExce
NonlinearSolve, TR (NLsolve Update)          325      293.20μs false 5.287e-03 ShrinkThresholdExce
NonlinearSolve, TR (Nocedal Wright)          267      254.16μs false 5.287e-03 ShrinkThresholdExce
NonlinearSolve, TR (Yuan)                   5757      864.52μs false 5.287e-03 ShrinkThresholdExce
NonlinearSolve, Trust Region                 257      238.75μs false 5.287e-03 ShrinkThresholdExce
Speedmapping, aa                              49       29.75μs  true 6.432e-10         first_order

Variably dimensioned function: 10 parameters, abstol = 1.0e-7.
Solver                                   f evals          time  conv     resid                 log
NLSolve, anderson                              4     5361.08μs false       NaN NLsolve.IsFiniteExc
NLSolve, newton                              336       25.12μs  true 0.000e+00  x_converged = true
NLSolve, trust_region                        336       30.34μs  true 0.000e+00  x_converged = true
NonlinearSolve, Default PolyAlg.              32       52.44μs  true 2.597e-12             Success
NonlinearSolve, LM (no Accln.)                68      299.41μs  true 6.993e-09             Success
NonlinearSolve, LM (no Accln.) Chol.          68       73.44μs  true 6.993e-09             Success
NonlinearSolve, LM (α_geodesic=0.5)          127      762.22μs  true 9.671e-10             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.    127      120.46μs  true 9.671e-10             Success
NonlinearSolve, LM with Cholesky             105      101.14μs  true 3.223e-08             Success
NonlinearSolve, Levenberg-Marquardt          105      582.31μs  true 3.223e-08             Success
NonlinearSolve, NR (BackTracking)             88       66.06μs  true 2.597e-12             Success
NonlinearSolve, NR (HagerZhang)               43       39.17μs  true 8.529e-09             Success
NonlinearSolve, NR (MoreThuente)             158       74.69μs  true 2.597e-12             Success
NonlinearSolve, Newton Raphson                32       51.41μs  true 2.597e-12             Success
NonlinearSolve, Pseudo Transient              32       51.71μs  true 2.597e-12             Success
NonlinearSolve, TR (Bastin)                  634       90.36μs  true 2.597e-12             Success
NonlinearSolve, TR (Fan)                      32       56.26μs  true 2.597e-12             Success
NonlinearSolve, TR (Hei)                      32       56.59μs  true 2.597e-12             Success
NonlinearSolve, TR (NLsolve Update)           32       54.14μs  true 2.597e-12             Success
NonlinearSolve, TR (Nocedal Wright)           32       56.96μs  true 2.597e-12             Success
NonlinearSolve, TR (Yuan)                    347       76.06μs  true 2.597e-12             Success
NonlinearSolve, Trust Region                  32       54.22μs  true 2.597e-12             Success
Speedmapping, aa                              32       10.43μs  true 1.694e-08         first_order

Broyden tridiagonal function: 10 parameters, abstol = 1.0e-7.
Solver                                   f evals          time  conv     resid                 log
NLSolve, anderson                             30     7011.18μs false       NaN NLsolve.IsFiniteExc
NLSolve, newton                              126       10.24μs  true 1.574e-15  x_converged = true
NLSolve, trust_region                        126       12.59μs  true 1.574e-15  x_converged = true
NonlinearSolve, Default PolyAlg.              12       36.40μs  true 1.062e-09             Success
NonlinearSolve, LM (no Accln.)                20       99.68μs  true 8.403e-09             Success
NonlinearSolve, LM (no Accln.) Chol.          20       42.66μs  true 8.403e-09             Success
NonlinearSolve, LM (α_geodesic=0.5)           28      165.95μs  true 6.112e-09             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.     28       49.08μs  true 6.112e-09             Success
NonlinearSolve, LM with Cholesky              28       48.65μs  true 6.112e-09             Success
NonlinearSolve, Levenberg-Marquardt           28      164.73μs  true 6.112e-09             Success
NonlinearSolve, NR (BackTracking)             28       40.23μs  true 1.062e-09             Success
NonlinearSolve, NR (HagerZhang)              326       95.10μs false 5.335e+09 InternalLineSearchF
NonlinearSolve, NR (MoreThuente)              48       42.09μs  true 1.062e-09             Success
NonlinearSolve, Newton Raphson                12       35.83μs  true 1.062e-09             Success
NonlinearSolve, Pseudo Transient              12       36.48μs  true 2.970e-09             Success
NonlinearSolve, TR (Bastin)                  184       51.73μs  true 1.062e-09             Success
NonlinearSolve, TR (Fan)                      14       39.22μs  true 4.623e-12             Success
NonlinearSolve, TR (Hei)                      12       38.43μs  true 1.062e-09             Success
NonlinearSolve, TR (NLsolve Update)           12       37.27μs  true 1.062e-09             Success
NonlinearSolve, TR (Nocedal Wright)           14       39.36μs  true 6.750e-12             Success
NonlinearSolve, TR (Yuan)                    117       45.57μs  true 1.062e-09             Success
NonlinearSolve, Trust Region                  14       39.34μs  true 6.750e-12             Success
Speedmapping, aa                              25       12.09μs  true 4.758e-08         first_order

Broyden banded function: 10 parameters, abstol = 1.0e-7.
Solver                                   f evals          time  conv     resid                 log
NLSolve, anderson                             12     3752.95μs false       NaN NLsolve.IsFiniteExc
NLSolve, newton                              147       13.51μs  true 9.619e-16  x_converged = true
NLSolve, trust_region                        147       15.97μs  true 9.619e-16  x_converged = true
NonlinearSolve, Default PolyAlg.              14       40.16μs  true 1.548e-08             Success
NonlinearSolve, LM (no Accln.)                22      114.43μs  true 3.386e-10             Success
NonlinearSolve, LM (no Accln.) Chol.          22       48.68μs  true 3.386e-10             Success
NonlinearSolve, LM (α_geodesic=0.5)           28      170.28μs  true 8.820e-08             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.     28       53.48μs  true 8.820e-08             Success
NonlinearSolve, LM with Cholesky              28       54.25μs  true 8.820e-08             Success
NonlinearSolve, Levenberg-Marquardt           28      170.16μs  true 8.820e-08             Success
NonlinearSolve, NR (BackTracking)             34       46.05μs  true 1.548e-08             Success
NonlinearSolve, NR (HagerZhang)           500906   925622.94μs false 2.028e+00            MaxIters
NonlinearSolve, NR (MoreThuente)              59       49.67μs  true 1.548e-08             Success
NonlinearSolve, Newton Raphson                14       41.66μs  true 1.548e-08             Success
NonlinearSolve, Pseudo Transient              14       40.59μs  true 1.902e-08             Success
NonlinearSolve, TR (Bastin)                  229       64.33μs  true 1.548e-08             Success
NonlinearSolve, TR (Fan)                      14       42.16μs  true 1.548e-08             Success
NonlinearSolve, TR (Hei)                      14       41.65μs  true 1.548e-08             Success
NonlinearSolve, TR (NLsolve Update)           14       41.98μs  true 1.548e-08             Success
NonlinearSolve, TR (Nocedal Wright)           14       42.15μs  true 1.548e-08             Success
NonlinearSolve, TR (Yuan)                    140       55.42μs  true 1.548e-08             Success
NonlinearSolve, Trust Region                  14       42.53μs  true 1.548e-08             Success
Speedmapping, aa                              35       20.08μs  true 5.255e-08         first_order

Hammarling 2 by 2 matrix square root problem: 4 parameters, abstol = 1.0e-7.
Solver                                   f evals          time  conv     resid                 log
NLSolve, anderson                             13    10143.04μs false       NaN NLsolve.IsFiniteExc
NLSolve, newton                              117        8.35μs  true 0.000e+00  x_converged = true
NLSolve, trust_region                        117        9.32μs  true 0.000e+00  x_converged = true
NonlinearSolve, Default PolyAlg.              24       32.69μs  true 4.711e-08             Success
NonlinearSolve, LM (no Accln.)                90      147.85μs  true 2.391e-09             Success
NonlinearSolve, LM (no Accln.) Chol.          90       51.77μs  true 2.189e-09             Success
NonlinearSolve, LM (α_geodesic=0.5)          135      275.97μs  true 1.173e-08             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.    135       68.09μs  true 1.150e-08             Success
NonlinearSolve, LM with Cholesky             130       66.18μs  true 1.557e-09             Success
NonlinearSolve, Levenberg-Marquardt          130      261.85μs  true 1.679e-09             Success
NonlinearSolve, NR (BackTracking)             64       35.63μs  true 4.711e-08             Success
NonlinearSolve, NR (HagerZhang)              778      121.69μs  true 5.005e-15             Success
NonlinearSolve, NR (MoreThuente)             359       80.81μs  true 9.121e-08             Success
NonlinearSolve, Newton Raphson                24       32.29μs  true 4.711e-08             Success
NonlinearSolve, Pseudo Transient              32       34.53μs  true 8.379e-11             Success
NonlinearSolve, TR (Bastin)                 1117      103.56μs  true 1.563e-12             Success
NonlinearSolve, TR (Fan)                   22205     6810.00μs  true 9.998e-08             Success
NonlinearSolve, TR (Hei)                   20335     6318.35μs  true 9.996e-08             Success
NonlinearSolve, TR (NLsolve Update)        29263     9109.58μs  true 9.999e-08             Success
NonlinearSolve, TR (Nocedal Wright)        13222     3512.11μs  true 9.999e-08             Success
NonlinearSolve, TR (Yuan)                 105003   343760.97μs false 1.865e-06            MaxIters
NonlinearSolve, Trust Region               13802     3659.35μs  true 9.997e-08             Success
Speedmapping, aa                              38        9.59μs  true 3.996e-09         first_order

Hammarling 3 by 3 matrix square root problem: 9 parameters, abstol = 1.0e-7.
Solver                                   f evals          time  conv     resid                 log
NLSolve, anderson                             13     3767.01μs false       NaN NLsolve.IsFiniteExc
NLSolve, newton                              247       14.46μs  true 0.000e+00  x_converged = true
NLSolve, trust_region                        247       18.64μs  true 0.000e+00  x_converged = true
NonlinearSolve, Default PolyAlg.              24       41.69μs  true 4.711e-08             Success
NonlinearSolve, LM (no Accln.)                90      337.90μs  true 2.391e-09             Success
NonlinearSolve, LM (no Accln.) Chol.          90       81.91μs  true 1.824e-09             Success
NonlinearSolve, LM (α_geodesic=0.5)          134      631.03μs  true 7.459e-11             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.    134      108.79μs  true 5.414e-11             Success
NonlinearSolve, LM with Cholesky             130      104.58μs  true 1.019e-09             Success
NonlinearSolve, Levenberg-Marquardt          130      601.77μs  true 1.679e-09             Success
NonlinearSolve, NR (BackTracking)             64       45.71μs  true 4.711e-08             Success
NonlinearSolve, NR (HagerZhang)             1120      227.79μs  true 8.634e-08             Success
NonlinearSolve, NR (MoreThuente)             359      110.32μs  true 8.911e-08             Success
NonlinearSolve, Newton Raphson                24       40.97μs  true 4.711e-08             Success
NonlinearSolve, Pseudo Transient              32       45.53μs  true 9.637e-14             Success
NonlinearSolve, TR (Bastin)                 2177      178.50μs  true 1.739e-12             Success
NonlinearSolve, TR (Fan)                   15707    11203.79μs  true 9.996e-08             Success
NonlinearSolve, TR (Hei)                   16222    11267.66μs  true 9.997e-08             Success
NonlinearSolve, TR (NLsolve Update)        23388    16631.94μs  true 1.000e-07             Success
NonlinearSolve, TR (Nocedal Wright)        13178     8606.21μs  true 9.998e-08             Success
NonlinearSolve, TR (Yuan)                 205061   485858.92μs false 1.274e-07            MaxIters
NonlinearSolve, Trust Region               13383     8662.13μs  true 9.999e-08             Success
Speedmapping, aa                              31       11.14μs  true 3.373e-08         first_order

Dennis and Schnabel 2 by 2 example: 2 parameters, abstol = 1.0e-7.
Solver                                   f evals          time  conv     resid                 log
NLSolve, anderson                             13     5317.93μs false       NaN NLsolve.IsFiniteExc
NLSolve, newton                               35        4.22μs  true 0.000e+00  x_converged = true
NLSolve, trust_region                         35        5.10μs  true 0.000e+00  x_converged = true
NonlinearSolve, Default PolyAlg.              14       29.80μs  true 1.098e-11             Success
NonlinearSolve, LM (no Accln.)                20       39.73μs  true 1.118e-08             Success
NonlinearSolve, LM (no Accln.) Chol.          20       32.62μs  true 1.118e-08             Success
NonlinearSolve, LM (α_geodesic=0.5)           34       62.65μs  true 6.526e-08             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.     34       38.33μs  true 6.526e-08             Success
NonlinearSolve, LM with Cholesky              33       37.98μs  true 1.571e-09             Success
NonlinearSolve, Levenberg-Marquardt           33       59.48μs  true 1.571e-09             Success
NonlinearSolve, NR (BackTracking)             34       32.14μs  true 1.098e-11             Success
NonlinearSolve, NR (HagerZhang)               85       39.51μs  true 2.328e-08             Success
NonlinearSolve, NR (MoreThuente)              59       36.70μs  true 1.098e-11             Success
NonlinearSolve, Newton Raphson                14       30.18μs  true 1.098e-11             Success
NonlinearSolve, Pseudo Transient              14       30.45μs  true 3.974e-12             Success
NonlinearSolve, TR (Bastin)                   69       33.51μs  true 5.771e-08             Success
NonlinearSolve, TR (Fan)                      14       32.15μs  true 2.377e-11             Success
NonlinearSolve, TR (Hei)                      14       31.05μs  true 2.819e-08             Success
NonlinearSolve, TR (NLsolve Update)           14       31.43μs  true 1.098e-11             Success
NonlinearSolve, TR (Nocedal Wright)           14       31.57μs  true 8.366e-11             Success
NonlinearSolve, TR (Yuan)                     44       31.84μs  true 1.098e-11             Success
NonlinearSolve, Trust Region                  14       31.80μs  true 8.366e-11             Success
Speedmapping, aa                              16        3.98μs  true 4.316e-11         first_order

Sample problem 18: 2 parameters, abstol = 1.0e-7.
Solver                                   f evals          time  conv     resid                 log
NLSolve, anderson                             18       20.92μs  true 1.211e-18  x_converged = true
NLSolve, newton                               45        4.84μs  true 0.000e+00 x_converged = false
NLSolve, trust_region                         40        5.98μs  true 0.000e+00 x_converged = false
NonlinearSolve, Default PolyAlg.              14       30.06μs  true 8.199e-09             Success
NonlinearSolve, LM (no Accln.)                20       39.82μs  true 4.107e-10             Success
NonlinearSolve, LM (no Accln.) Chol.          20       32.15μs  true 4.107e-10             Success
NonlinearSolve, LM (α_geodesic=0.5)           30       57.30μs  true 6.813e-08             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.     30       37.49μs  true 6.834e-08             Success
NonlinearSolve, LM with Cholesky              29       36.52μs  true 2.198e-09             Success
NonlinearSolve, Levenberg-Marquardt           29       54.59μs  true 2.198e-09             Success
NonlinearSolve, NR (BackTracking)             34       31.71μs  true 8.199e-09             Success
NonlinearSolve, NR (HagerZhang)               40       33.16μs  true 2.033e-08             Success
NonlinearSolve, NR (MoreThuente)             205       55.46μs  true 5.270e-12             Success
NonlinearSolve, Newton Raphson                14       29.64μs  true 8.199e-09             Success
NonlinearSolve, Pseudo Transient              16       29.76μs  true 9.503e-12             Success
NonlinearSolve, TR (Bastin)                   82       34.73μs  true 6.181e-10             Success
NonlinearSolve, TR (Fan)                      14       31.18μs  true 1.546e-12             Success
NonlinearSolve, TR (Hei)                      12       31.59μs  true 4.487e-09             Success
NonlinearSolve, TR (NLsolve Update)           10       31.01μs  true 7.809e-09             Success
NonlinearSolve, TR (Nocedal Wright)           18       32.89μs  true 1.614e-08             Success
NonlinearSolve, TR (Yuan)                     44       33.05μs  true 8.199e-09             Success
NonlinearSolve, Trust Region                  18       32.25μs  true 1.614e-08             Success
Speedmapping, aa                              16        4.57μs  true 1.213e-19         first_order

Sample problem 19: 2 parameters, abstol = 1.0e-7.
Solver                                   f evals          time  conv     resid                 log
NLSolve, anderson                              8     5023.96μs false       NaN NLsolve.IsFiniteExc
NLSolve, newton                              225       14.88μs  true 3.257e-18  x_converged = true
NLSolve, trust_region                        225       19.76μs  true 3.257e-18  x_converged = true
NonlinearSolve, Default PolyAlg.              38       32.94μs  true 7.985e-08             Success
NonlinearSolve, LM (no Accln.)                56       65.62μs  true 4.966e-08             Success
NonlinearSolve, LM (no Accln.) Chol.          56       38.52μs  true 4.966e-08             Success
NonlinearSolve, LM (α_geodesic=0.5)          111      148.10μs  true 7.132e-08             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.    111       57.35μs  true 7.132e-08             Success
NonlinearSolve, LM with Cholesky              79       46.06μs  true 8.444e-08             Success
NonlinearSolve, Levenberg-Marquardt           79       97.43μs  true 8.444e-08             Success
NonlinearSolve, NR (BackTracking)            106       39.72μs  true 7.985e-08             Success
NonlinearSolve, NR (HagerZhang)               18       31.30μs  true 7.487e-23             Success
NonlinearSolve, NR (MoreThuente)             191       59.32μs  true 7.985e-08             Success
NonlinearSolve, Newton Raphson                38       32.72μs  true 7.985e-08             Success
NonlinearSolve, Pseudo Transient              38       33.18μs  true 8.051e-08             Success
NonlinearSolve, TR (Bastin)                  238       40.46μs  true 3.980e-08             Success
NonlinearSolve, TR (Fan)                      38       36.47μs  true 7.985e-08             Success
NonlinearSolve, TR (Hei)                      40       35.76μs  true 3.565e-08             Success
NonlinearSolve, TR (NLsolve Update)           38       35.82μs  true 7.985e-08             Success
NonlinearSolve, TR (Nocedal Wright)           38       35.92μs  true 7.985e-08             Success
NonlinearSolve, TR (Yuan)                    142       39.09μs  true 6.436e-08             Success
NonlinearSolve, Trust Region                  38       36.05μs  true 7.985e-08             Success
Speedmapping, aa                              31        7.36μs  true 9.339e-08         first_order

Scalar problem f(x) = x(x - 5)^2: 1 parameters, abstol = 1.0e-7.
Solver                                   f evals          time  conv     resid                 log
NLSolve, anderson                             11       10.11μs  true 3.581e-11  x_converged = true
NLSolve, newton                               21        2.83μs  true 1.940e-15  x_converged = true
NLSolve, trust_region                          6        2.39μs  true 0.000e+00 x_converged = false
NonlinearSolve, Default PolyAlg.              16       29.47μs  true 1.939e-15             Success
NonlinearSolve, LM (no Accln.)                 6       30.05μs  true 1.110e-14             Success
NonlinearSolve, LM (no Accln.) Chol.           6       29.12μs  true 5.551e-15             Success
NonlinearSolve, LM (α_geodesic=0.5)           30       43.96μs  true 1.801e-09             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.     30       36.72μs  true 1.801e-09             Success
NonlinearSolve, LM with Cholesky              30       37.33μs  true 1.801e-09             Success
NonlinearSolve, Levenberg-Marquardt           30       43.64μs  true 1.801e-09             Success
NonlinearSolve, NR (BackTracking)             41       32.15μs  true 2.913e-14             Success
NonlinearSolve, NR (HagerZhang)               63       35.45μs  true 8.712e-08             Success
NonlinearSolve, NR (MoreThuente)             263       64.83μs  true 9.807e-08             Success
NonlinearSolve, Newton Raphson                16       29.09μs  true 1.939e-15             Success
NonlinearSolve, Pseudo Transient              16       28.30μs  true 1.633e-15             Success
NonlinearSolve, TR (Bastin)                   13       29.81μs  true 0.000e+00             Success
NonlinearSolve, TR (Fan)                      15       30.05μs  true 9.075e-08             Success
NonlinearSolve, TR (Hei)                       6       28.75μs  true 0.000e+00             Success
NonlinearSolve, TR (NLsolve Update)            6       29.26μs  true 0.000e+00             Success
NonlinearSolve, TR (Nocedal Wright)           14       29.93μs  true 1.109e-11             Success
NonlinearSolve, TR (Yuan)                     57       31.72μs  true 6.672e-13             Success
NonlinearSolve, Trust Region                  14       29.77μs  true 1.109e-11             Success
Speedmapping, aa                              28        4.82μs  true 9.061e-08         first_order

Freudenstein-Roth function: 2 parameters, abstol = 1.0e-7.
Solver                                   f evals          time  conv     resid                 log
NLSolve, anderson                             46     3795.86μs false       NaN NLsolve.IsFiniteExc
NLSolve, newton                              365       24.54μs  true 1.589e-14  x_converged = true
NLSolve, trust_region                        168       43.05μs false 6.999e+00  x_converged = true
NonlinearSolve, Default PolyAlg.              88       37.66μs  true 5.639e-10             Success
NonlinearSolve, LM (no Accln.)               144      124.39μs  true 6.107e-09             Success
NonlinearSolve, LM (no Accln.) Chol.         156       53.31μs  true 6.720e-09             Success
NonlinearSolve, LM (α_geodesic=0.5)       177378   632036.92μs false 8.518e+00            MaxIters
NonlinearSolve, LM (α_geodesic=0.5) Chol. 177378   405198.81μs false 8.258e+00            MaxIters
NonlinearSolve, LM with Cholesky          177379    55400.13μs false 7.244e+00            MaxIters
NonlinearSolve, Levenberg-Marquardt       177379   265349.86μs false 7.243e+00            MaxIters
NonlinearSolve, NR (BackTracking)         659966  2756542.92μs false 7.611e+00            MaxIters
NonlinearSolve, NR (HagerZhang)               58       34.02μs false 1.188e+09 InternalLineSearchF
NonlinearSolve, NR (MoreThuente)              53       36.38μs false 9.130e+12 InternalLineSearchF
NonlinearSolve, Newton Raphson                88       38.19μs  true 5.639e-10             Success
NonlinearSolve, Pseudo Transient             114       41.30μs  true 4.727e-08             Success
NonlinearSolve, TR (Bastin)                  504       74.96μs false 6.999e+00 ShrinkThresholdExce
NonlinearSolve, TR (Fan)                     109       59.58μs false 6.999e+00 ShrinkThresholdExce
NonlinearSolve, TR (Hei)                     109       57.28μs false 6.999e+00 ShrinkThresholdExce
NonlinearSolve, TR (NLsolve Update)          115       58.03μs false 6.999e+00 ShrinkThresholdExce
NonlinearSolve, TR (Nocedal Wright)          102       53.88μs false 6.999e+00 ShrinkThresholdExce
NonlinearSolve, TR (Yuan)                    511       75.05μs false 6.999e+00 ShrinkThresholdExce
NonlinearSolve, Trust Region                 102       55.86μs false 6.999e+00 ShrinkThresholdExce
Speedmapping, aa                              35        7.92μs  true 1.579e-10         first_order

Boggs function: 2 parameters, abstol = 1.0e-7.
Solver                                   f evals          time  conv     resid                 log
NLSolve, anderson                             15       17.08μs  true 1.251e-08  x_converged = true
NLSolve, newton                               25        3.37μs  true 6.661e-16  x_converged = true
NLSolve, trust_region                         41        6.73μs  true 4.950e-19  x_converged = true
NonlinearSolve, Default PolyAlg.              10       29.59μs  true 0.000e+00             Success
NonlinearSolve, LM (no Accln.)                22       41.00μs  true 6.630e-10             Success
NonlinearSolve, LM (no Accln.) Chol.          22       32.88μs  true 6.630e-10             Success
NonlinearSolve, LM (α_geodesic=0.5)           32       57.83μs  true 1.131e-08             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.     32       37.11μs  true 1.131e-08             Success
NonlinearSolve, LM with Cholesky              28       35.60μs  true 8.650e-08             Success
NonlinearSolve, Levenberg-Marquardt           28       53.01μs  true 8.650e-08             Success
NonlinearSolve, NR (BackTracking)             41       33.47μs  true 8.470e-08             Success
NonlinearSolve, NR (HagerZhang)              102       43.04μs  true 7.087e-10             Success
NonlinearSolve, NR (MoreThuente)             480      103.85μs  true 8.234e-08             Success
NonlinearSolve, Newton Raphson                10       29.34μs  true 0.000e+00             Success
NonlinearSolve, Pseudo Transient              18       30.72μs  true 3.187e-14             Success
NonlinearSolve, TR (Bastin)                   82       34.45μs  true 6.477e-12             Success
NonlinearSolve, TR (Fan)                      18       31.75μs  true 6.326e-11             Success
NonlinearSolve, TR (Hei)                      16       32.29μs  true 6.477e-12             Success
NonlinearSolve, TR (NLsolve Update)           16       31.64μs  true 6.477e-12             Success
NonlinearSolve, TR (Nocedal Wright)           18       31.99μs  true 1.336e-11             Success
NonlinearSolve, TR (Yuan)                     57       33.23μs  true 1.119e-08             Success
NonlinearSolve, Trust Region                  18       32.36μs  true 1.336e-11             Success
Speedmapping, aa                              50       11.11μs  true 9.268e-11         first_order

Chandrasekhar function: 10 parameters, abstol = 1.0e-7.
Solver                                   f evals          time  conv     resid                 log
NLSolve, anderson                           1031     5666.97μs false       NaN NLsolve.IsFiniteExc
NLSolve, newton                              126       20.44μs  true 4.441e-16  x_converged = true
NLSolve, trust_region                        126       22.46μs  true 4.441e-16  x_converged = true
NonlinearSolve, Default PolyAlg.              12       38.60μs  true 4.955e-14             Success
NonlinearSolve, LM (no Accln.)                18       95.74μs  true 9.729e-08             Success
NonlinearSolve, LM (no Accln.) Chol.          18       45.01μs  true 9.729e-08             Success
NonlinearSolve, LM (α_geodesic=0.5)           25      154.00μs  true 7.468e-08             Success
NonlinearSolve, LM (α_geodesic=0.5) Chol.     25       52.18μs  true 7.468e-08             Success
NonlinearSolve, LM with Cholesky              25       53.49μs  true 7.468e-08             Success
NonlinearSolve, Levenberg-Marquardt           25      155.34μs  true 7.468e-08             Success
NonlinearSolve, NR (BackTracking)             28       41.90μs  true 4.955e-14             Success
NonlinearSolve, NR (HagerZhang)               43       43.80μs  true 2.396e-08             Success
NonlinearSolve, NR (MoreThuente)              48       48.49μs  true 4.955e-14             Success
NonlinearSolve, Newton Raphson                12       37.64μs  true 4.955e-14             Success
NonlinearSolve, Pseudo Transient              12       38.23μs  true 3.441e-09             Success
NonlinearSolve, TR (Bastin)                  184       69.46μs  true 2.316e-08             Success
NonlinearSolve, TR (Fan)                      14       42.52μs  true 5.481e-09             Success
NonlinearSolve, TR (Hei)                      12       39.77μs  true 2.316e-08             Success
NonlinearSolve, TR (NLsolve Update)           12       39.13μs  true 4.955e-14             Success
NonlinearSolve, TR (Nocedal Wright)           18       47.24μs  true 1.558e-10             Success
NonlinearSolve, TR (Yuan)                    117       56.67μs  true 3.364e-10             Success
NonlinearSolve, Trust Region                  18       47.17μs  true 1.558e-10             Success
Speedmapping, aa                              14        8.33μs  true 1.278e-08         first_order
=#
