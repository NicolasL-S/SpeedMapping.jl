########################
# Fixed-point iterations

absolute_path_to_docs = "" # Update

using BenchmarkTools, JLD2, FileIO, Logging, FixedPointAcceleration, FixedPoint, SpeedMapping, 
	FixedPointTestProblems, LinearAlgebra

path_plots = absolute_path_to_docs*"assets/"
path_out = absolute_path_to_docs*"benchmarking_code/Output/"
include(absolute_path_to_docs*"benchmarking_code/Benchmarking_utils.jl")

Logging.disable_logging(Logging.Warn)

# This wrapper around map! automatically computes the number of map! calls and (hopefully) stops the 
# execution of a solver when maps[1] > maps_limit or time() > timesup by tricking the solver into
# thinking a fixed point has been reached.
@inline function _map!(map!, x_out, x_in, maps, maps_limit, timesup)
	if maps[] >= maps_limit || time() >= timesup
		x_out .= x_in
	else
		maps[] += 1
		map!(x_out, x_in)
	end
	return x_out
end

@inline _map(map!, x, maps, maps_limit, timesup) = _map!(map!, similar(x), x, maps, maps_limit, timesup)

# Creating a dictionary of solvers
fixed_point_solvers = Dict{AbstractString, Function}()

function Speedmapping_fp_wrapper(problem, abstol, timesup, maps_limit, options)
	x0, map!, obj = problem
	algo, ada_relax, composite, mono = options
	maps = Ref(0)
	f = mono ? obj : nothing
	res = speedmapping(x0; m! = (x_out, x_in) -> _map!(map!, x_out, x_in, maps, maps_limit, timesup), 
		f, algo, abstol, ada_relax, composite = composite, lags = 30, maps_limit, condition_max = 1e6)
	return res.minimizer, maps[], string(res.status)
end

fixed_point_solvers["SpeedMapping, acx"] = 
	(problem, abstol, start_time, maps_limit, time_limit) -> 
	Speedmapping_fp_wrapper(problem, abstol, start_time + time_limit, maps_limit, (:acx, :none, :none, false))

for (mono_t, mono_s) in (("", false), (", mono", true)), 
	(compo_t, compo_s) in (("", :none), (", compo (aa1)", :aa1), (", compo (acx2)", :acx2)), 
	(adap_t, adap_s) in (("",:none), (", adaptive", :minimum_distance))
	fixed_point_solvers["SpeedMapping, aa" * mono_t*compo_t*adap_t] = 
	(problem, abstol, start_time, maps_limit, time_limit) -> 
	Speedmapping_fp_wrapper(problem, abstol, start_time + time_limit, maps_limit, (:aa, adap_s, compo_s, mono_s))
end

for (mono_t, mono_s) in (("", false), (", mono", true)), 
	(compo_t, compo_s) in ((", compo (aa1)", :aa1), (", compo (acx2)", :acx2)), 
	(adap_t, adap_s) in (("",:none), (", adaptive", :minimum_distance))
	fixed_point_solvers["SpeedMapping, aa" * mono_t*compo_t*adap_t] = 
	(problem, abstol, start_time, maps_limit, time_limit) -> 
	Speedmapping_fp_wrapper(problem, abstol, start_time + time_limit, maps_limit, (:aa, adap_s, compo_s, mono_s))
end


function FixedPointAcceleration_wrapper(problem, abstol, timesup, maps_limit, Algorithm)
	x0, map! = problem
	maps = Ref(0)
	try
		res = FixedPointAcceleration.fixed_point(x -> _map(map!, x, maps, maps_limit, timesup), x0; 
			Algorithm = Algorithm, ConvergenceMetric = (x, y) -> norm(x .- y), 
			ConvergenceMetricThreshold = abstol, ConditionNumberThreshold = 1e6, MaxM = 30, 
			MaxIter = maps_limit)
		if res.TerminationCondition_ == :ReachedConvergenceThreshold
			return res.FixedPoint_, maps[], string(res.TerminationCondition_)
		else
			return NaN .* ones(length(x0)), maps[], string(res.TerminationCondition_)
		end
	catch e # To catch out-of-memory errors for the lid-driven cavity flow
		return NaN .* ones(length(x0)), maps[], sprint(showerror, typeof(e))
	end
end

for Algorithm in ("Anderson", "MPE", "RRE", "VEA", "SEA")
	fixed_point_solvers["FixedPointAcceleration, " * Algorithm] = 
		(problem, abstol, start_time, maps_limit, time_limit) -> 
		FixedPointAcceleration_wrapper(problem, abstol, start_time + time_limit, maps_limit, Symbol(Algorithm))
end

function FixedPoint_wrapper(problem, abstol, timesup, maps_limit)
	x0, map! = problem
	maps = Ref(0)
	res = FixedPoint.afps!((x_out, x_in) -> _map!(map!, x_out, x_in, maps, maps_limit, timesup), 
		copy(x0), grad_norm = norm, tol = abstol, iters = maps_limit) # Using a copy of x0 because afps! writes over it.
	res.x, maps[], ""
end

fixed_point_solvers["FixedPoint"] = (problem, abstol, start_time, maps_limit, time_limit) -> 
	FixedPoint_wrapper(problem, abstol, start_time + time_limit, maps_limit)

solver_names = sort([name for (name, wrapper) in fixed_point_solvers])

prob_sizes = Dict{String, Int}()
for (name, gen_problem) in testproblems
	prob_sizes[name] = length(gen_problem().x0)
end

order_length = sortperm([l for (name, l) in prob_sizes])
problems_names = [name for (name, l) in prob_sizes][order_length]
problems_names_len = problems_names .* " (" .* string.(sort([l for (name, l) in prob_sizes])) .* ")"

function compute_norm(problem, x)
	xout = similar(x)
	if sum(_isbad.(x)) == 0
		problem.map!(xout, x)
		xout .-= x
		last_res = norm(xout, 2)
	else
		last_res = NaN
	end
end

function gen_Feval_limit(problem, time_limit)
	xout = similar(problem.x0)
	bench = @benchmark $problem.map!($xout, $problem.x0)
	return min(Int(ceil(1.5time_limit / (median(bench.times)/1e9))),1_000_000)
end

# Since some problems take long time or are more challenging to solve, each ones will have specific 
# time unit (s, ms, μs), specific tolerance and specific time limits for algorithms.

time_units = [3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]
abstols = [1.0e-7, 1.0e-7, 1.0e-5, 1.0e-7, 1.0e-7, 1.0e-7, 1.0e-7, 1.0e-7, 1.0e-7, 1.0e-5, 1.0e-7, 
	1.0e-7, 1.0e-7, 1.0e-7, 1.0e-4]
time_limits = [10, 10, 100, 10, 10, 10, 10, 100, 100, 100, 200, 100, 100, 100, 1000]


# This should take several hours
res_maps_all = [Dict{String, Tuple{Float64, Float64}}() for i in eachindex(problems), j in 1:1]
res_maps_all = many_problems_many_solvers(testproblems, fixed_point_solvers, problems_names, 
	solver_names, compute_norm; tunits = time_units, F_name = "maps", gen_Feval_limit, 
	abstol = abstols, time_limit = time_limits, proper_benchmark = true, results = res_maps_all)

JLD2.@save path_out*"res_maps_all.jld2" res_maps_all
# res_maps_all = JLD2.load_object(path_out*"res_maps_all.jld2") # To load

title = "Performance of various Julia solvers for fixed-point mapping problems"
plot_res(res_maps_all, problems_names_len, solver_names, title, path_plots*"mapping_benchmarks.svg"; 
	size = (600, 600), legend_rowgap = -3, xticklabelrotation = pi/2.25)

#=
Hasselblad, Poisson mixtures: 3 parameters, abstol = 1.0e-7.
Solver                                               maps            time  conv   |resid|                             log                      obj
FixedPoint                                          21627       1372.47μs  true 9.998e-08                                         1989.94585988424
FixedPointAcceleration, Anderson                       23        102.00μs  true 1.240e-08     ReachedConvergenceThreshold       1989.9458598829701
FixedPointAcceleration, MPE                            38        105.92μs  true 6.107e-08     ReachedConvergenceThreshold       1989.9458598829892
FixedPointAcceleration, RRE                            42        130.55μs  true 7.180e-08     ReachedConvergenceThreshold       1989.9458598833305
FixedPointAcceleration, SEA                            37        101.95μs  true 6.430e-08     ReachedConvergenceThreshold       1989.9458598829667
FixedPointAcceleration, VEA                            43        102.66μs  true 1.741e-09     ReachedConvergenceThreshold       1989.9458598829644
SpeedMapping, aa                                       21          7.90μs  true 1.217e-08                     first_order       1989.9458598829644
SpeedMapping, aa, adaptive                             21          8.25μs  true 9.201e-09                     first_order        1989.945859882964
SpeedMapping, aa, compo (aa1)                          36          7.06μs  true 6.108e-08                     first_order       1989.9458598829679
SpeedMapping, aa, compo (aa1), adaptive                44          8.33μs  true 9.118e-08                     first_order         1989.94585988297
SpeedMapping, aa, compo (acx2)                         30          6.13μs  true 5.828e-08                     first_order        1989.945859882968
SpeedMapping, aa, compo (acx2), adaptive               32          6.74μs  true 4.808e-08                     first_order       1989.9458598829679
SpeedMapping, aa, mono                                 17         10.02μs  true 2.943e-08                     first_order       1989.9458598829656
SpeedMapping, aa, mono, adaptive                       17         10.58μs  true 2.917e-08                     first_order       1989.9458598829658
SpeedMapping, aa, mono, compo (aa1)                    36         10.80μs  true 7.442e-08                     first_order        1989.945859882967
SpeedMapping, aa, mono, compo (aa1), adaptive          44         13.12μs  true 6.297e-08                     first_order       1989.9458598829658
SpeedMapping, aa, mono, compo (acx2)                   37         10.29μs  true 4.650e-08                     first_order        1989.945859883055
SpeedMapping, aa, mono, compo (acx2), adaptive         37         10.87μs  true 5.957e-08                     first_order       1989.9458598830936
SpeedMapping, acx                                      52          4.02μs  true 3.214e-09                     first_order       1989.9458598829644

Exchange economy: 10 parameters, abstol = 1.0e-7.
Solver                                               maps            time  conv   |resid|                             log
FixedPoint                                            286         97.66μs  true 9.878e-08
FixedPointAcceleration, Anderson                        7         56.55μs  true 7.100e-11     ReachedConvergenceThreshold
FixedPointAcceleration, MPE                             7         26.34μs  true 2.156e-11     ReachedConvergenceThreshold
FixedPointAcceleration, RRE                             7         26.52μs  true 2.156e-11     ReachedConvergenceThreshold
FixedPointAcceleration, SEA                             8         33.68μs  true 1.228e-09     ReachedConvergenceThreshold
FixedPointAcceleration, VEA                             7         27.38μs  true 1.433e-09     ReachedConvergenceThreshold
SpeedMapping, aa                                        7          7.35μs  true 2.271e-09                     first_order
SpeedMapping, aa, adaptive                              7          7.47μs  true 2.294e-09                     first_order
SpeedMapping, aa, compo (aa1)                           7          5.82μs  true 2.017e-08                     first_order
SpeedMapping, aa, compo (aa1), adaptive                 7          5.50μs  true 2.017e-08                     first_order
SpeedMapping, aa, compo (acx2)                          7          5.87μs  true 6.982e-08                     first_order
SpeedMapping, aa, compo (acx2), adaptive                7          5.77μs  true 6.982e-08                     first_order
SpeedMapping, aa, mono                                  7          8.50μs  true 2.271e-09                     first_order
SpeedMapping, aa, mono, adaptive                        7          8.99μs  true 2.294e-09                     first_order
SpeedMapping, aa, mono, compo (aa1)                     7          5.62μs  true 2.017e-08                     first_order
SpeedMapping, aa, mono, compo (aa1), adaptive           7          7.14μs  true 2.017e-08                     first_order
SpeedMapping, aa, mono, compo (acx2)                    7          5.80μs  true 6.982e-08                     first_order
SpeedMapping, aa, mono, compo (acx2), adaptive          7          5.83μs  true 6.982e-08                     first_order
SpeedMapping, acx                                       7          2.81μs  true 6.193e-08                     first_order

Wang, PH interval censoring: 10 parameters, abstol = 1.0e-5.
Solver                                               maps            time  conv   |resid|                             log                      obj
FixedPoint                                         159056      38168.18ms  true 1.000e-05                                        946.0074640376806
FixedPointAcceleration, Anderson                      234         85.57ms  true 4.122e-06     ReachedConvergenceThreshold        946.0190255765672
FixedPointAcceleration, MPE                           122         45.54ms  true 6.321e-06     ReachedConvergenceThreshold        950.1698726583278
FixedPointAcceleration, RRE                         53111     100001.43ms false 1.212e-02                       Timed out        952.6516700475885
FixedPointAcceleration, SEA                            49         16.73ms  true 5.018e-07     ReachedConvergenceThreshold        960.3760393858493
FixedPointAcceleration, VEA                          2747       1278.54ms  true 9.694e-06     ReachedConvergenceThreshold        946.0074639471773
SpeedMapping, aa                                      213         76.61ms  true 8.270e-06                     first_order         950.169781946068
SpeedMapping, aa, adaptive                            302        113.90ms  true 2.393e-06                     first_order        950.1698731237138
SpeedMapping, aa, compo (aa1)                         682        213.93ms  true 3.635e-06                     first_order          949.10305978054
SpeedMapping, aa, compo (aa1), adaptive               224         76.16ms  true 2.095e-08                     first_order        950.1698730653361
SpeedMapping, aa, compo (acx2)                        157         56.75ms  true 3.713e-06                     first_order        950.1698945714584
SpeedMapping, aa, compo (acx2), adaptive              468        160.54ms  true 1.990e-06                     first_order        950.1698820807619
SpeedMapping, aa, mono                                127         57.68ms  true 8.746e-06                     first_order        946.0075245004447
SpeedMapping, aa, mono, adaptive                       99         43.60ms  true 3.299e-06                     first_order        946.0074638905028
SpeedMapping, aa, mono, compo (aa1)                   219         81.46ms  true 5.040e-06                     first_order        946.0074712977274
SpeedMapping, aa, mono, compo (aa1), adaptive         127         48.12ms  true 4.815e-07                     first_order        946.0074770329537
SpeedMapping, aa, mono, compo (acx2)                  829        307.87ms  true 1.794e-06                     first_order        946.0161909107874
SpeedMapping, aa, mono, compo (acx2), adaptive        721        269.37ms  true 8.146e-06                     first_order        946.0074574268376
SpeedMapping, acx                                     346        137.64ms  true 9.684e-06                     first_order        946.0074930611929

Mixture of 3 normals: 17 parameters, abstol = 1.0e-7.
Solver                                               maps            time  conv   |resid|                             log                      obj
FixedPoint                                           4220       2031.02ms  true 9.972e-08                                        4787.809009703242
FixedPointAcceleration, Anderson                       32         14.27ms  true 6.084e-09     ReachedConvergenceThreshold       4787.8090097032455
FixedPointAcceleration, MPE                            67         30.37ms  true 2.119e-08     ReachedConvergenceThreshold        4787.809009703251
FixedPointAcceleration, RRE                            61         28.70ms false 1.104e-07     ReachedConvergenceThreshold        4787.809009703241
FixedPointAcceleration, SEA                           151        243.88ms  true 7.732e-08     ReachedConvergenceThreshold        4862.643310378573
FixedPointAcceleration, VEA                           118         50.71ms  true 6.387e-08     ReachedConvergenceThreshold         4787.80900970324
SpeedMapping, aa                                       29         10.30ms  true 9.975e-09                     first_order        4787.809009703249
SpeedMapping, aa, adaptive                             33         11.46ms  true 7.225e-09                     first_order       4787.8090097032455
SpeedMapping, aa, compo (aa1)                          50         20.20ms  true 3.761e-08                     first_order        4787.809009703237
SpeedMapping, aa, compo (aa1), adaptive                49         17.98ms  true 9.719e-08                     first_order         4787.80900970324
SpeedMapping, aa, compo (acx2)                         50         26.17ms  true 5.482e-08                     first_order       4787.8090097032455
SpeedMapping, aa, compo (acx2), adaptive               53         22.45ms  true 7.177e-09                     first_order        4787.809009703235
SpeedMapping, aa, mono                                 28         29.70ms  true 3.780e-08                     first_order        4787.809009703247
SpeedMapping, aa, mono, adaptive                       27         27.43ms  true 1.553e-08                     first_order        4787.809009703242
SpeedMapping, aa, mono, compo (aa1)                    43         35.07ms  true 7.929e-08                     first_order       4787.8090097032355
SpeedMapping, aa, mono, compo (aa1), adaptive          41         32.36ms  true 2.612e-08                     first_order        4787.809009703243
SpeedMapping, aa, mono, compo (acx2)                   56         41.71ms  true 2.042e-08                     first_order        4787.809009703241
SpeedMapping, aa, mono, compo (acx2), adaptive         53         43.33ms  true 4.357e-08                     first_order        4787.809009703246
SpeedMapping, acx                                      55         24.74ms  true 3.493e-08                     first_order        4787.809009703232

Higham, correlation matrix mmb13: 42 parameters, abstol = 1.0e-7.
Solver                                               maps            time  conv   |resid|                             log
FixedPoint                                           3755         22.06ms  true 9.965e-08
FixedPointAcceleration, Anderson                       24          1.15ms  true 2.535e-09     ReachedConvergenceThreshold
FixedPointAcceleration, MPE                            43          0.54ms  true 2.917e-08     ReachedConvergenceThreshold
FixedPointAcceleration, RRE                            43          0.57ms  true 4.027e-08     ReachedConvergenceThreshold
FixedPointAcceleration, SEA                           342      10276.93ms false 4.827e-07                       Timed out
FixedPointAcceleration, VEA                           117          2.24ms  true 8.584e-08     ReachedConvergenceThreshold
SpeedMapping, aa                                       21          0.14ms  true 4.166e-08                     first_order
SpeedMapping, aa, adaptive                             25          0.17ms  true 1.129e-08                     first_order
SpeedMapping, aa, compo (aa1)                          32          0.18ms  true 4.543e-08                     first_order
SpeedMapping, aa, compo (aa1), adaptive                34          0.20ms  true 7.062e-08                     first_order
SpeedMapping, aa, compo (acx2)                         41          0.23ms  true 6.440e-09                     first_order
SpeedMapping, aa, compo (acx2), adaptive               38          0.21ms  true 4.723e-08                     first_order
SpeedMapping, aa, mono                                 21          0.14ms  true 4.166e-08                     first_order
SpeedMapping, aa, mono, adaptive                       25          0.17ms  true 1.129e-08                     first_order
SpeedMapping, aa, mono, compo (aa1)                    32          0.18ms  true 4.543e-08                     first_order
SpeedMapping, aa, mono, compo (aa1), adaptive          34          0.20ms  true 7.062e-08                     first_order
SpeedMapping, aa, mono, compo (acx2)                   41          0.22ms  true 6.440e-09                     first_order
SpeedMapping, aa, mono, compo (acx2), adaptive         38          0.22ms  true 4.723e-08                     first_order
SpeedMapping, acx                                      60          0.30ms  true 4.111e-08                     first_order

Power iter. for dom. eigenvalue: 100 parameters, abstol = 1.0e-7.
Solver                                               maps            time  conv   |resid|                             log
FixedPoint                                            303          0.28ms  true 9.470e-08
FixedPointAcceleration, Anderson                       14          0.31ms  true 1.496e-08     ReachedConvergenceThreshold
FixedPointAcceleration, MPE                            14          0.12ms  true 3.694e-08     ReachedConvergenceThreshold
FixedPointAcceleration, RRE                            17          0.17ms  true 4.944e-08     ReachedConvergenceThreshold
FixedPointAcceleration, SEA                            34          1.85ms  true 5.393e-08     ReachedConvergenceThreshold
FixedPointAcceleration, VEA                            17          0.16ms  true 4.891e-08     ReachedConvergenceThreshold
SpeedMapping, aa                                       14          0.03ms  true 2.693e-08                     first_order
SpeedMapping, aa, adaptive                             13          0.03ms  true 8.383e-08                     first_order
SpeedMapping, aa, compo (aa1)                          16          0.03ms  true 5.993e-08                     first_order
SpeedMapping, aa, compo (aa1), adaptive                16          0.03ms  true 5.135e-08                     first_order
SpeedMapping, aa, compo (acx2)                         16          0.03ms  true 5.385e-08                     first_order
SpeedMapping, aa, compo (acx2), adaptive               16          0.03ms  true 4.285e-08                     first_order
SpeedMapping, aa, mono                                 14          0.03ms  true 2.693e-08                     first_order
SpeedMapping, aa, mono, adaptive                       13          0.04ms  true 8.383e-08                     first_order
SpeedMapping, aa, mono, compo (aa1)                    16          0.03ms  true 5.993e-08                     first_order
SpeedMapping, aa, mono, compo (aa1), adaptive          16          0.03ms  true 5.135e-08                     first_order
SpeedMapping, aa, mono, compo (acx2)                   16          0.03ms  true 5.385e-08                     first_order
SpeedMapping, aa, mono, compo (acx2), adaptive         16          0.04ms  true 4.285e-08                     first_order
SpeedMapping, acx                                      18          0.02ms  true 6.470e-08                     first_order

Linear: 100 parameters, abstol = 1.0e-7.
Solver                                               maps            time  conv   |resid|                             log                      obj
FixedPoint                                           8008          1.94ms  true 9.992e-08                                      -130.98128232040014
FixedPointAcceleration, Anderson                       77          6.67ms  true 8.418e-08     ReachedConvergenceThreshold      -130.98128232040037
FixedPointAcceleration, MPE                           175          8.22ms  true 8.492e-08     ReachedConvergenceThreshold      -130.98128232040034
FixedPointAcceleration, RRE                           223         13.52ms  true 8.897e-08     ReachedConvergenceThreshold      -130.98128232040023
FixedPointAcceleration, SEA                             7          0.06ms  true 2.247e-14     ReachedConvergenceThreshold       -130.9812823204004
FixedPointAcceleration, VEA                           761        166.21ms  true 9.461e-08     ReachedConvergenceThreshold      -130.98128232040028
SpeedMapping, aa                                       77          0.22ms  true 9.975e-08                     first_order      -130.98128232040037
SpeedMapping, aa, adaptive                             72          0.23ms  true 7.713e-08                     first_order      -130.98128232040037
SpeedMapping, aa, compo (aa1)                          86          0.06ms  true 7.751e-08                     first_order       -130.9812823204004
SpeedMapping, aa, compo (aa1), adaptive                77          0.07ms  true 7.914e-08                     first_order      -130.98128232040037
SpeedMapping, aa, compo (acx2)                         86          0.06ms  true 5.317e-08                     first_order       -130.9812823204004
SpeedMapping, aa, compo (acx2), adaptive               82          0.07ms  true 3.207e-08                     first_order       -130.9812823204004
SpeedMapping, aa, mono                                 77          0.22ms  true 9.975e-08                     first_order      -130.98128232040037
SpeedMapping, aa, mono, adaptive                       72          0.22ms  true 7.713e-08                     first_order      -130.98128232040037
SpeedMapping, aa, mono, compo (aa1)                    86          0.06ms  true 7.751e-08                     first_order       -130.9812823204004
SpeedMapping, aa, mono, compo (aa1), adaptive          77          0.06ms  true 7.914e-08                     first_order      -130.98128232040037
SpeedMapping, aa, mono, compo (acx2)                   86          0.07ms  true 5.317e-08                     first_order       -130.9812823204004
SpeedMapping, aa, mono, compo (acx2), adaptive         82          0.07ms  true 3.207e-08                     first_order       -130.9812823204004
SpeedMapping, acx                                     108          0.02ms  true 7.039e-08                     first_order       -130.9812823204003

FixedPoint                                            798         100.07s false 2.824e-01                       Timed out
FixedPointAcceleration, Anderson                       20           2.20s  true 2.000e-08     ReachedConvergenceThreshold
FixedPointAcceleration, MPE                            91          11.21s  true 7.959e-08     ReachedConvergenceThreshold
FixedPointAcceleration, RRE                           797         100.09s false 4.772e-02                       Timed out
FixedPointAcceleration, SEA                            50           7.67s false       NaN InvalidInputOrOutputOfIteration
FixedPointAcceleration, VEA                           316          36.98s  true 4.343e-08     ReachedConvergenceThreshold
SpeedMapping, aa                                       18           2.64s  true 7.840e-09                     first_order
SpeedMapping, aa, adaptive                             20           2.21s  true 9.967e-09                     first_order
SpeedMapping, aa, compo (aa1)                          35           4.21s  true 6.680e-09                     first_order
SpeedMapping, aa, compo (aa1), adaptive                35           5.01s  true 3.742e-09                     first_order
SpeedMapping, aa, compo (acx2)                         38           6.60s  true 4.646e-08                     first_order
SpeedMapping, aa, compo (acx2), adaptive               38           4.14s  true 5.808e-08                     first_order
SpeedMapping, aa, mono                                 18           2.33s  true 7.840e-09                     first_order
SpeedMapping, aa, mono, adaptive                       20           2.29s  true 9.967e-09                     first_order
SpeedMapping, aa, mono, compo (aa1)                    35           3.79s  true 6.680e-09                     first_order
SpeedMapping, aa, mono, compo (aa1), adaptive          35           4.27s  true 3.742e-09                     first_order
SpeedMapping, aa, mono, compo (acx2)                   38           4.05s  true 4.646e-08                     first_order
SpeedMapping, aa, mono, compo (acx2), adaptive         38           4.24s  true 5.808e-08                     first_order
SpeedMapping, acx                                      66           7.77s  true 1.221e-08                     first_order

ALS for CANDECOMP: 450 parameters, abstol = 1.0e-7.
Solver                                               maps            time  conv   |resid|                             log
FixedPoint                                           9712           4.61s  true 9.999e-08
FixedPointAcceleration, Anderson                     3032           8.11s  true 1.290e-08     ReachedConvergenceThreshold
FixedPointAcceleration, MPE                            61           0.04s  true 9.024e-08     ReachedConvergenceThreshold
FixedPointAcceleration, RRE                           109           0.08s  true 2.601e-08     ReachedConvergenceThreshold
FixedPointAcceleration, SEA                          2104           8.43s  true 7.731e-08     ReachedConvergenceThreshold
FixedPointAcceleration, VEA                           264           0.24s  true 9.591e-08     ReachedConvergenceThreshold
SpeedMapping, aa                                      654           0.29s  true 9.550e-08                     first_order
SpeedMapping, aa, adaptive                            315           0.14s  true 1.378e-08                     first_order
SpeedMapping, aa, compo (aa1)                          88           0.03s  true 5.700e-08                     first_order
SpeedMapping, aa, compo (aa1), adaptive                78           0.03s  true 5.624e-08                     first_order
SpeedMapping, aa, compo (acx2)                        140           0.06s  true 5.972e-08                     first_order
SpeedMapping, aa, compo (acx2), adaptive               98           0.04s  true 9.203e-08                     first_order
SpeedMapping, aa, mono                                654           0.32s  true 9.550e-08                     first_order
SpeedMapping, aa, mono, adaptive                      315           0.16s  true 1.378e-08                     first_order
SpeedMapping, aa, mono, compo (aa1)                    88           0.04s  true 5.700e-08                     first_order
SpeedMapping, aa, mono, compo (aa1), adaptive          78           0.04s  true 5.624e-08                     first_order
SpeedMapping, aa, mono, compo (acx2)                  140           0.06s  true 5.972e-08                     first_order
SpeedMapping, aa, mono, compo (acx2), adaptive         98           0.04s  true 9.203e-08                     first_order
SpeedMapping, acx                                      64           0.02s  true 6.816e-08                     first_order

Lange, ancestry: 750 parameters, abstol = 1.0e-5.
Solver                                               maps            time  conv   |resid|                             log                      obj
FixedPoint                                          22973         100.00s false 5.347e-03                       Timed out       14748.932547804246
FixedPointAcceleration, Anderson                     5742         100.20s false 3.870e-02                       Timed out       19119.650806931684
FixedPointAcceleration, MPE                           989           9.07s  true 9.673e-06     ReachedConvergenceThreshold        14748.93082491167
FixedPointAcceleration, RRE                          3672         100.02s false 6.193e-02                       Timed out       19116.095793047876
FixedPointAcceleration, SEA                           259           1.32s false       NaN InvalidInputOrOutputOfIteration                      NaN
FixedPointAcceleration, VEA                          4372         100.08s false 1.136e-01                       Timed out       19115.585167364094
SpeedMapping, aa                                    22135         100.00s false 1.909e-02                       Timed out       19122.119001393497
SpeedMapping, aa, adaptive                          21361         100.00s false 3.735e-02                       Timed out        19116.20096151992
SpeedMapping, aa, compo (aa1)                       22076         100.00s false 2.822e-02                       Timed out        19121.25233896223
SpeedMapping, aa, compo (aa1), adaptive             23164         100.00s false 8.806e-03                       Timed out       19123.324406564952
SpeedMapping, aa, compo (acx2)                      21441         100.00s false 1.676e-02                       Timed out       19122.402879009176
SpeedMapping, aa, compo (acx2), adaptive            21840         100.00s false 1.465e-02                       Timed out        19122.67981966414
SpeedMapping, aa, mono                                929           4.69s  true 9.582e-06                     first_order       14748.930824886727
SpeedMapping, aa, mono, adaptive                      742           3.54s  true 9.887e-06                     first_order       14748.930824854917
SpeedMapping, aa, mono, compo (aa1)                   724           3.20s  true 9.473e-06                     first_order       14748.930824858986
SpeedMapping, aa, mono, compo (aa1), adaptive         687           3.02s  true 9.365e-06                     first_order        14748.93082487516
SpeedMapping, aa, mono, compo (acx2)                 1071           4.64s  true 9.574e-06                     first_order       14748.930824863759
SpeedMapping, aa, mono, compo (acx2), adaptive        781           3.36s  true 9.901e-06                     first_order       14748.930824862851
SpeedMapping, acx                                     934           3.92s  true 9.731e-06                     first_order       14748.930824916208

Bratu: 10000 parameters, abstol = 1.0e-7.
Solver                                               maps            time  conv   |resid|                             log
FixedPoint                                          13555         200.00s false 7.783e-03                       Timed out
FixedPointAcceleration, Anderson                     1425         130.72s  true 9.851e-08     ReachedConvergenceThreshold
FixedPointAcceleration, MPE                          1266         200.25s false 3.411e-03                       Timed out
FixedPointAcceleration, RRE                          1151         200.18s false 2.922e-03                       Timed out
FixedPointAcceleration, SEA                            32           0.51s false       NaN InvalidInputOrOutputOfIteration
FixedPointAcceleration, VEA                          1321         200.54s false 1.614e-02                       Timed out
SpeedMapping, aa                                      865          11.84s  true 9.951e-08                     first_order
SpeedMapping, aa, adaptive                            256           4.09s  true 9.980e-08                     first_order
SpeedMapping, aa, compo (aa1)                         470           7.14s  true 9.154e-08                     first_order
SpeedMapping, aa, compo (aa1), adaptive               259           3.97s  true 9.842e-08                     first_order
SpeedMapping, aa, compo (acx2)                        269           4.18s  true 9.572e-08                     first_order
SpeedMapping, aa, compo (acx2), adaptive              257           4.03s  true 9.285e-08                     first_order
SpeedMapping, aa, mono                                865          14.13s  true 9.951e-08                     first_order
SpeedMapping, aa, mono, adaptive                      256           4.23s  true 9.980e-08                     first_order
SpeedMapping, aa, mono, compo (aa1)                   470           7.37s  true 9.154e-08                     first_order
SpeedMapping, aa, mono, compo (aa1), adaptive         259           4.06s  true 9.842e-08                     first_order
SpeedMapping, aa, mono, compo (acx2)                  269           4.21s  true 9.572e-08                     first_order
SpeedMapping, aa, mono, compo (acx2), adaptive        257           4.06s  true 9.285e-08                     first_order
SpeedMapping, acx                                    2223          34.60s  true 9.915e-08                     first_order

Electric field, Gauss-Seidel: 10000 parameters, abstol = 1.0e-7.
Solver                                               maps            time  conv   |resid|                             log
FixedPoint                                         115278           3.91s  true 9.999e-08
FixedPointAcceleration, Anderson                      286           4.18s  true 9.342e-08     ReachedConvergenceThreshold
FixedPointAcceleration, MPE                          1103         100.05s false 4.835e-06                       Timed out
FixedPointAcceleration, RRE                           919         100.21s false 5.377e-05                       Timed out
FixedPointAcceleration, SEA                           153         114.03s false 7.822e-01                       Timed out
FixedPointAcceleration, VEA                          1108         100.19s false 3.296e-02                       Timed out
SpeedMapping, aa                                      286           0.10s  true 9.758e-08                     first_order
SpeedMapping, aa, adaptive                            281           0.11s  true 9.849e-08                     first_order
SpeedMapping, aa, compo (aa1)                         353           0.04s  true 9.151e-08                     first_order
SpeedMapping, aa, compo (aa1), adaptive               353           0.05s  true 9.767e-08                     first_order
SpeedMapping, aa, compo (acx2)                        389           0.05s  true 9.227e-08                     first_order
SpeedMapping, aa, compo (acx2), adaptive              370           0.05s  true 9.526e-08                     first_order
SpeedMapping, aa, mono                                286           0.10s  true 9.758e-08                     first_order
SpeedMapping, aa, mono, adaptive                      281           0.11s  true 9.849e-08                     first_order
SpeedMapping, aa, mono, compo (aa1)                   353           0.04s  true 9.151e-08                     first_order
SpeedMapping, aa, mono, compo (aa1), adaptive         353           0.05s  true 9.767e-08                     first_order
SpeedMapping, aa, mono, compo (acx2)                  389           0.05s  true 9.227e-08                     first_order
SpeedMapping, aa, mono, compo (acx2), adaptive        370           0.05s  true 9.526e-08                     first_order
SpeedMapping, acx                                     432           0.01s  true 7.963e-08                     first_order

Electric field, SOR: 10000 parameters, abstol = 1.0e-7.
Solver                                               maps            time  conv   |resid|                             log
FixedPoint                                         747654         530.90s false       NaN                       Timed out
FixedPointAcceleration, Anderson                      376           5.50s  true 9.939e-08     ReachedConvergenceThreshold
FixedPointAcceleration, MPE                           380           7.21s  true 8.010e-08     ReachedConvergenceThreshold
FixedPointAcceleration, RRE                           466          27.33s  true 9.552e-08     ReachedConvergenceThreshold
FixedPointAcceleration, SEA                           153         110.42s false 1.387e+00                       Timed out
FixedPointAcceleration, VEA                           660          28.14s  true 9.382e-08     ReachedConvergenceThreshold
SpeedMapping, aa                                      376           0.14s  true 9.660e-08                     first_order
SpeedMapping, aa, adaptive                            373           0.16s  true 9.904e-08                     first_order
SpeedMapping, aa, compo (aa1)                         356           0.05s  true 8.760e-08                     first_order
SpeedMapping, aa, compo (aa1), adaptive               403           0.07s  true 9.022e-08                     first_order
SpeedMapping, aa, compo (acx2)                        362           0.05s  true 9.957e-08                     first_order
SpeedMapping, aa, compo (acx2), adaptive              377           0.06s  true 9.705e-08                     first_order
SpeedMapping, aa, mono                                376           0.14s  true 9.660e-08                     first_order
SpeedMapping, aa, mono, adaptive                      373           0.16s  true 9.904e-08                     first_order
SpeedMapping, aa, mono, compo (aa1)                   356           0.05s  true 8.760e-08                     first_order
SpeedMapping, aa, mono, compo (aa1), adaptive         403           0.07s  true 9.022e-08                     first_order
SpeedMapping, aa, mono, compo (acx2)                  362           0.06s  true 9.957e-08                     first_order
SpeedMapping, aa, mono, compo (acx2), adaptive        377           0.07s  true 9.705e-08                     first_order
SpeedMapping, acx                                     420           0.02s  true 9.739e-08                     first_order

Electric field, Jacobi: 10000 parameters, abstol = 1.0e-7.
Solver                                               maps            time  conv   |resid|                             log
FixedPoint                                         196154           3.41s  true 1.000e-07
FixedPointAcceleration, Anderson                     1052          47.46s  true 9.969e-08     ReachedConvergenceThreshold
FixedPointAcceleration, MPE                          1148         100.06s false 3.336e-04                       Timed out
FixedPointAcceleration, RRE                           927         100.45s false 5.741e-03                       Timed out
FixedPointAcceleration, SEA                           153         114.12s false 1.265e+01                       Timed out
FixedPointAcceleration, VEA                          1134         100.18s false 1.596e-01                       Timed out
SpeedMapping, aa                                      768           0.29s  true 9.982e-08                     first_order
SpeedMapping, aa, adaptive                            353           0.15s  true 9.905e-08                     first_order
SpeedMapping, aa, compo (aa1)                         638           0.08s  true 9.501e-08                     first_order
SpeedMapping, aa, compo (aa1), adaptive               437           0.05s  true 9.300e-08                     first_order
SpeedMapping, aa, compo (acx2)                        482           0.05s  true 9.613e-08                     first_order
SpeedMapping, aa, compo (acx2), adaptive              443           0.05s  true 9.578e-08                     first_order
SpeedMapping, aa, mono                                768           0.24s  true 9.982e-08                     first_order
SpeedMapping, aa, mono, adaptive                      353           0.14s  true 9.905e-08                     first_order
SpeedMapping, aa, mono, compo (aa1)                   638           0.07s  true 9.501e-08                     first_order
SpeedMapping, aa, mono, compo (aa1), adaptive         437           0.04s  true 9.300e-08                     first_order
SpeedMapping, aa, mono, compo (acx2)                  482           0.05s  true 9.613e-08                     first_order
SpeedMapping, aa, mono, compo (acx2), adaptive        443           0.04s  true 9.578e-08                     first_order
SpeedMapping, acx                                     893           0.01s  true 9.339e-08                     first_order

Lid-driven cavity ﬂow: 52812 parameters, abstol = 0.0001.
Solver                                               maps            time  conv   |resid|                             log
FixedPoint                                         297706         763.78s  true 1.000e-04
FixedPointAcceleration, Anderson                     2053        1001.44s false 2.195e-03                       Timed out
FixedPointAcceleration, MPE                          1454        1000.59s false 4.449e-03                       Timed out
FixedPointAcceleration, RRE                             6           0.05s false       NaN                OutOfMemoryError
FixedPointAcceleration, SEA                           153        1084.13s false 3.637e-02                       Timed out
FixedPointAcceleration, VEA                          1465        1001.17s false 1.255e-02                       Timed out
SpeedMapping, aa                                     5789          56.78s  true 1.000e-04                     first_order
SpeedMapping, aa, adaptive                           4904          42.36s  true 9.999e-05                     first_order
SpeedMapping, aa, compo (aa1)                        7444          53.17s  true 9.988e-05                     first_order
SpeedMapping, aa, compo (aa1), adaptive              6190          34.13s  true 9.999e-05                     first_order
SpeedMapping, aa, compo (acx2)                       5032          26.75s  true 9.987e-05                     first_order
SpeedMapping, aa, compo (acx2), adaptive             5888          33.46s  true 1.000e-04                     first_order
SpeedMapping, aa, mono                               5789          46.79s  true 1.000e-04                     first_order
SpeedMapping, aa, mono, adaptive                     4904          50.03s  true 9.999e-05                     first_order
SpeedMapping, aa, mono, compo (aa1)                  7444          40.93s  true 9.988e-05                     first_order
SpeedMapping, aa, mono, compo (aa1), adaptive        6190          44.24s  true 9.999e-05                     first_order
SpeedMapping, aa, mono, compo (acx2)                 5032          27.92s  true 9.987e-05                     first_order
SpeedMapping, aa, mono, compo (acx2), adaptive       5888          33.12s  true 1.000e-04                     first_order
SpeedMapping, acx                                    4212          18.63s  true 9.983e-05                     first_order
=#
