########################
# Fixed-point iterations

using BenchmarkTools, JLD2, FileIO, Logging, FixedPointAcceleration, FixedPoint, SpeedMapping, 
	FixedPointTestProblems, LinearAlgebra

absolute_path = ""
path_plots = absolute_path*"assets/"
path_out = absolute_path*"benchmarking_code/Output/"

include("benchmark_code/Benchmarking_utils.jl")

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
	algo, adarel, composite, mono = options
	maps = Ref(0)
	f = mono ? obj : nothing
	res = speedmapping(x0; m! = (x_out, x_in) -> _map!(map!, x_out, x_in, maps, maps_limit, timesup), 
		f, algo, abstol, adarel, composite = composite, lags = 30, maps_limit, condition_max = 1e6)
	return res.minimizer, maps[], string(res.status)
end

for (name, options) in (
		("acx", (:acx, :none, :none, false)), 
		("aa", (:aa,:none, :none, false)),
	 	("aa: composite", (:aa, :none, :aa1, false)), 
		("aa: adaptive", (:aa, :minimum_distance, :none, false)), 
		("aa: adaptive, composite", (:aa, :minimum_distance, :aa1, false)),
		("aa: mono.", (:aa, :none, :none, true)),
		("aa: adaptive, mono.", (:aa, :minimum_distance, :none, true)))
	fixed_point_solvers["Speedmapping, " * name] = 
		(problem, abstol, start_time, maps_limit, time_limit) -> 
		Speedmapping_fp_wrapper(problem, abstol, start_time + time_limit, maps_limit, options)
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
res_maps_all = many_problems_many_solvers(testproblems, fixed_point_solvers, problems_names, 
	solver_names, compute_norm; tunits = time_units, F_name = "maps", gen_Feval_limit, 
	abstol = abstols, time_limit = time_limits, proper_benchmark = true)

JLD2.@save path_out*"res_maps_all.jld2" res_maps_all

title = "Performance of various Julia solvers for fixed-point mapping problems"
plot_res(res_maps_all, problems_names_len, solver_names, title, path_plots*"mapping_benchmarks.svg"; 
	size = (600, 400), legend_rowgap = -3)

#=
Hasselblad, Poisson mixtures: 3 parameters, abstol = 1.0e-7.
Solver                                      maps          time  conv   |resid|                 log                      obj
FixedPoint                                 21627     1299.92μs  true 9.998e-08                             1989.94585988424
FixedPointAcceleration, Anderson              23      102.07μs  true 1.240e-08 ReachedConvergenceT       1989.9458598829701
FixedPointAcceleration, MPE                   38      110.55μs  true 6.107e-08 ReachedConvergenceT       1989.9458598829892
FixedPointAcceleration, RRE                   42      126.58μs  true 7.180e-08 ReachedConvergenceT       1989.9458598833305
FixedPointAcceleration, SEA                   37      105.46μs  true 6.430e-08 ReachedConvergenceT       1989.9458598829667
FixedPointAcceleration, VEA                   43      128.76μs  true 1.741e-09 ReachedConvergenceT       1989.9458598829644
Speedmapping, aa                              21        7.25μs  true 1.217e-08         first_order       1989.9458598829644
Speedmapping, aa: adaptive                    21        8.14μs  true 9.201e-09         first_order        1989.945859882964
Speedmapping, aa: adaptive, composite         44        7.75μs  true 9.118e-08         first_order         1989.94585988297
Speedmapping, aa: adaptive, mono.             17        9.79μs  true 2.917e-08         first_order       1989.9458598829658
Speedmapping, aa: composite                   36        6.52μs  true 6.108e-08         first_order       1989.9458598829679
Speedmapping, aa: mono.                       17        9.89μs  true 2.943e-08         first_order       1989.9458598829656
Speedmapping, acx                             52        3.91μs  true 3.214e-09         first_order       1989.9458598829644

Exchange economy: 10 parameters, abstol = 1.0e-7.
Solver                                      maps          time  conv   |resid|                 log
FixedPoint                                   286       87.71μs  true 9.878e-08
FixedPointAcceleration, Anderson               7       59.42μs  true 7.100e-11 ReachedConvergenceT
FixedPointAcceleration, MPE                    7       28.41μs  true 2.156e-11 ReachedConvergenceT
FixedPointAcceleration, RRE                    7       33.21μs  true 2.156e-11 ReachedConvergenceT
FixedPointAcceleration, SEA                    8       36.23μs  true 1.228e-09 ReachedConvergenceT
FixedPointAcceleration, VEA                    7       35.75μs  true 1.433e-09 ReachedConvergenceT
Speedmapping, aa                               7        6.76μs  true 2.271e-09         first_order
Speedmapping, aa: adaptive                     7       10.09μs  true 2.294e-09         first_order
Speedmapping, aa: adaptive, composite          7        5.94μs  true 2.017e-08         first_order
Speedmapping, aa: adaptive, mono.              7        7.15μs  true 2.294e-09         first_order
Speedmapping, aa: composite                    7        6.01μs  true 2.017e-08         first_order
Speedmapping, aa: mono.                        7        6.91μs  true 2.271e-09         first_order
Speedmapping, acx                              7        2.34μs  true 6.193e-08         first_order

Wang, PH interval censoring: 10 parameters, abstol = 1.0e-5.
Solver                                      maps          time  conv   |resid|                 log                      obj
FixedPoint                                159056    34658.20ms  true 1.000e-05                            946.0074640376806
FixedPointAcceleration, Anderson             234       80.70ms  true 4.122e-06 ReachedConvergenceT        946.0190255765672
FixedPointAcceleration, MPE                  122       41.11ms  true 6.321e-06 ReachedConvergenceT        950.1698726583278
FixedPointAcceleration, RRE                51848   100005.87ms false 1.050e-02           Timed out        952.7601987817069
FixedPointAcceleration, SEA                   49       15.19ms  true 5.018e-07 ReachedConvergenceT        960.3760393858493
FixedPointAcceleration, VEA                 2747     1045.97ms  true 9.694e-06 ReachedConvergenceT        946.0074639471773
Speedmapping, aa                             213       65.14ms  true 8.270e-06         first_order         950.169781946068
Speedmapping, aa: adaptive                   302       99.75ms  true 2.393e-06         first_order        950.1698731237138
Speedmapping, aa: adaptive, composite        224       66.56ms  true 2.095e-08         first_order        950.1698730653361
Speedmapping, aa: adaptive, mono.             99       38.82ms  true 3.299e-06         first_order        946.0074638905028
Speedmapping, aa: composite                  682      196.79ms  true 3.635e-06         first_order          949.10305978054
Speedmapping, aa: mono.                      127       48.44ms  true 8.746e-06         first_order        946.0075245004447
Speedmapping, acx                            346      125.31ms  true 9.684e-06         first_order        946.0074930611929

Mixture of 3 normals: 17 parameters, abstol = 1.0e-7.
Solver                                      maps          time  conv   |resid|                 log                      obj
FixedPoint                                  4220     1732.64ms  true 9.972e-08                            4787.809009703242
FixedPointAcceleration, Anderson              32       11.44ms  true 6.084e-09 ReachedConvergenceT       4787.8090097032455
FixedPointAcceleration, MPE                   67       28.80ms  true 2.119e-08 ReachedConvergenceT        4787.809009703251
FixedPointAcceleration, RRE                   61       22.19ms false 1.104e-07 ReachedConvergenceT        4787.809009703241
FixedPointAcceleration, SEA                  151      224.42ms  true 7.732e-08 ReachedConvergenceT        4862.643310378573
FixedPointAcceleration, VEA                  118       49.83ms  true 6.387e-08 ReachedConvergenceT         4787.80900970324
Speedmapping, aa                              29        9.40ms  true 9.975e-09         first_order        4787.809009703249
Speedmapping, aa: adaptive                    33       10.70ms  true 7.225e-09         first_order       4787.8090097032455
Speedmapping, aa: adaptive, composite         49       16.12ms  true 9.719e-08         first_order         4787.80900970324
Speedmapping, aa: adaptive, mono.             27       20.21ms  true 1.553e-08         first_order        4787.809009703242
Speedmapping, aa: composite                   50       16.03ms  true 3.761e-08         first_order        4787.809009703237
Speedmapping, aa: mono.                       28       22.41ms  true 3.780e-08         first_order        4787.809009703247
Speedmapping, acx                             55       18.23ms  true 3.493e-08         first_order        4787.809009703232

Higham, correlation matrix mmb13: 42 parameters, abstol = 1.0e-7.
Solver                                      maps          time  conv   |resid|                 log
FixedPoint                                  3755       20.34ms  true 9.965e-08
FixedPointAcceleration, Anderson              24        1.03ms  true 2.535e-09 ReachedConvergenceT
FixedPointAcceleration, MPE                   43        0.48ms  true 2.917e-08 ReachedConvergenceT
FixedPointAcceleration, RRE                   43        0.50ms  true 4.027e-08 ReachedConvergenceT
FixedPointAcceleration, SEA                  349    10127.33ms false 3.661e-07           Timed out
FixedPointAcceleration, VEA                  117        1.90ms  true 8.584e-08 ReachedConvergenceT
Speedmapping, aa                              21        0.13ms  true 4.166e-08         first_order
Speedmapping, aa: adaptive                    25        0.17ms  true 1.129e-08         first_order
Speedmapping, aa: adaptive, composite         34        0.18ms  true 7.062e-08         first_order
Speedmapping, aa: adaptive, mono.             25        0.17ms  true 1.129e-08         first_order
Speedmapping, aa: composite                   32        0.18ms  true 4.543e-08         first_order
Speedmapping, aa: mono.                       21        0.14ms  true 4.166e-08         first_order
Speedmapping, acx                             60        0.30ms  true 4.111e-08         first_order

Power iter. for dom. eigenvalue: 100 parameters, abstol = 1.0e-7.
Solver                                      maps          time  conv   |resid|                 log
FixedPoint                                   303        0.27ms  true 9.470e-08
FixedPointAcceleration, Anderson              14        0.27ms  true 1.496e-08 ReachedConvergenceT
FixedPointAcceleration, MPE                   14        0.13ms  true 3.694e-08 ReachedConvergenceT
FixedPointAcceleration, RRE                   17        0.16ms  true 4.944e-08 ReachedConvergenceT
FixedPointAcceleration, SEA                   34        1.61ms  true 5.393e-08 ReachedConvergenceT
FixedPointAcceleration, VEA                   17        0.16ms  true 4.891e-08 ReachedConvergenceT
Speedmapping, aa                              14        0.03ms  true 2.693e-08         first_order
Speedmapping, aa: adaptive                    13        0.03ms  true 8.383e-08         first_order
Speedmapping, aa: adaptive, composite         16        0.03ms  true 5.135e-08         first_order
Speedmapping, aa: adaptive, mono.             13        0.04ms  true 8.383e-08         first_order
Speedmapping, aa: composite                   16        0.03ms  true 5.993e-08         first_order
Speedmapping, aa: mono.                       14        0.03ms  true 2.693e-08         first_order
Speedmapping, acx                             18        0.02ms  true 6.470e-08         first_order

Linear: 100 parameters, abstol = 1.0e-7.
Solver                                      maps          time  conv   |resid|                 log                      obj
FixedPoint                                  8008        2.42ms  true 9.992e-08                          -130.98128232040014
FixedPointAcceleration, Anderson              77        5.68ms  true 8.418e-08 ReachedConvergenceT       -130.9812823204004
FixedPointAcceleration, MPE                  175        6.53ms  true 8.492e-08 ReachedConvergenceT      -130.98128232040034
FixedPointAcceleration, RRE                  223       10.72ms  true 8.897e-08 ReachedConvergenceT      -130.98128232040023
FixedPointAcceleration, SEA                    7        0.06ms  true 1.700e-14 ReachedConvergenceT       -130.9812823204004
FixedPointAcceleration, VEA                  761      119.43ms  true 9.461e-08 ReachedConvergenceT      -130.98128232040028
Speedmapping, aa                              77        0.20ms  true 9.975e-08         first_order      -130.98128232040037
Speedmapping, aa: adaptive                    72        0.21ms  true 7.713e-08         first_order      -130.98128232040037
Speedmapping, aa: adaptive, composite         77        0.06ms  true 7.914e-08         first_order       -130.9812823204004
Speedmapping, aa: adaptive, mono.             72        1.19ms  true 7.713e-08         first_order      -130.98128232040037
Speedmapping, aa: composite                   86        0.06ms  true 7.751e-08         first_order       -130.9812823204004
Speedmapping, aa: mono.                       77        1.28ms  true 9.975e-08         first_order      -130.98128232040037
Speedmapping, acx                            108        0.02ms  true 7.037e-08         first_order       -130.9812823204003

Consumption smoothing: 107 parameters, abstol = 1.0e-7.
Solver                                      maps          time  conv   |resid|                 log
FixedPoint                                   940       100.01s false 1.338e-01           Timed out
FixedPointAcceleration, Anderson              20         2.13s  true 2.000e-08 ReachedConvergenceT
FixedPointAcceleration, MPE                   91        10.23s  true 7.959e-08 ReachedConvergenceT
FixedPointAcceleration, RRE                  966       100.05s false 3.386e-03           Timed out
FixedPointAcceleration, SEA                   50         6.69s false       NaN InvalidInputOrOutpu
FixedPointAcceleration, VEA                  316        31.51s  true 4.343e-08 ReachedConvergenceT
Speedmapping, aa                              18         1.77s  true 7.840e-09         first_order
Speedmapping, aa: adaptive                    20         1.95s  true 9.967e-09         first_order
Speedmapping, aa: adaptive, composite         35         3.84s  true 3.742e-09         first_order
Speedmapping, aa: adaptive, mono.             20         2.10s  true 9.967e-09         first_order
Speedmapping, aa: composite                   35         3.76s  true 6.680e-09         first_order
Speedmapping, aa: mono.                       18         1.81s  true 7.840e-09         first_order
Speedmapping, acx                             66         6.98s  true 1.221e-08         first_order

ALS for CANDECOMP: 450 parameters, abstol = 1.0e-7.
Solver                                      maps          time  conv   |resid|                 log
FixedPoint                                  9712         2.91s  true 9.999e-08
FixedPointAcceleration, Anderson            3032         8.58s  true 1.290e-08 ReachedConvergenceT
FixedPointAcceleration, MPE                   61         0.03s  true 9.024e-08 ReachedConvergenceT
FixedPointAcceleration, RRE                  109         0.07s  true 2.601e-08 ReachedConvergenceT
FixedPointAcceleration, SEA                 2104         9.63s  true 7.731e-08 ReachedConvergenceT
FixedPointAcceleration, VEA                  264         0.21s  true 9.591e-08 ReachedConvergenceT
Speedmapping, aa                             654         0.24s  true 9.550e-08         first_order
Speedmapping, aa: adaptive                   315         0.10s  true 1.378e-08         first_order
Speedmapping, aa: adaptive, composite         78         0.03s  true 5.624e-08         first_order
Speedmapping, aa: adaptive, mono.            315         0.11s  true 1.378e-08         first_order
Speedmapping, aa: composite                   88         0.03s  true 5.700e-08         first_order
Speedmapping, aa: mono.                      654         0.22s  true 9.550e-08         first_order
Speedmapping, acx                             64         0.02s  true 6.816e-08         first_order

Lange, ancestry: 750 parameters, abstol = 1.0e-5.
Solver                                      maps          time  conv   |resid|                 log                      obj
FixedPoint                                 22180       100.01s false 5.507e-03           Timed out        14643.44716334701
FixedPointAcceleration, Anderson            5729       100.03s false 9.779e-02           Timed out       18913.218515187917
FixedPointAcceleration, MPE                   55         0.27s false       NaN InvalidInputOrOutpu                      NaN
FixedPointAcceleration, RRE                 3751       100.03s false 4.289e-01           Timed out       18831.775092344986
FixedPointAcceleration, SEA                   79         0.39s false       NaN InvalidInputOrOutpu                      NaN
FixedPointAcceleration, VEA                 4356       100.18s false 6.461e-02           Timed out        18941.58131149373
Speedmapping, aa                           20277       100.00s false 3.057e-02           Timed out        18937.08042966592
Speedmapping, aa: adaptive                 22326       100.00s false 3.402e-02           Timed out        18935.61013479391
Speedmapping, aa: adaptive, composite      22659       100.00s false 1.151e-01           Timed out       18887.080670281284
Speedmapping, aa: adaptive, mono.            734         3.99s  true 9.865e-06         first_order        14643.44707939149
Speedmapping, aa: composite                20825       100.00s false 1.232e-01           Timed out       18887.185468358577
Speedmapping, aa: mono.                     1013         5.00s  true 9.727e-06         first_order       14643.447079393654
Speedmapping, acx                            570         2.61s  true 9.999e-06         first_order       14643.447079399115

Bratu: 10000 parameters, abstol = 1.0e-7.
Solver                                      maps          time  conv   |resid|                 log
FixedPoint                                 15169       200.01s false 7.405e-03           Timed out
FixedPointAcceleration, Anderson            1425       132.39s  true 9.851e-08 ReachedConvergenceT
FixedPointAcceleration, MPE                 1253       200.10s false 2.570e-03           Timed out
FixedPointAcceleration, RRE                 1147       200.11s false 2.928e-03           Timed out
FixedPointAcceleration, SEA                   32         0.49s false       NaN InvalidInputOrOutpu
FixedPointAcceleration, VEA                 1307       200.20s false 1.042e-02           Timed out
Speedmapping, aa                             865        11.55s  true 9.951e-08         first_order
Speedmapping, aa: adaptive                   256         3.40s  true 9.980e-08         first_order
Speedmapping, aa: adaptive, composite        259         3.25s  true 9.842e-08         first_order
Speedmapping, aa: adaptive, mono.            256         3.43s  true 9.980e-08         first_order
Speedmapping, aa: composite                  470         5.83s  true 9.154e-08         first_order
Speedmapping, aa: mono.                      865        13.07s  true 9.951e-08         first_order
Speedmapping, acx                           2223        32.71s  true 9.915e-08         first_order

Electric field, Gauss-Seidel: 10000 parameters, abstol = 1.0e-7.
Solver                                      maps          time  conv   |resid|                 log
FixedPoint                                115278         3.54s  true 9.999e-08
FixedPointAcceleration, Anderson             286         3.92s  true 9.342e-08 ReachedConvergenceT
FixedPointAcceleration, MPE                 1140       100.37s false 1.226e-06           Timed out
FixedPointAcceleration, RRE                  961       100.19s false 4.142e-05           Timed out
FixedPointAcceleration, SEA                  153       112.94s false 7.822e-01           Timed out
FixedPointAcceleration, VEA                 1140       100.30s false 9.395e-02           Timed out
Speedmapping, aa                             286         0.08s  true 9.758e-08         first_order
Speedmapping, aa: adaptive                   281         0.09s  true 9.849e-08         first_order
Speedmapping, aa: adaptive, composite        353         0.04s  true 9.767e-08         first_order
Speedmapping, aa: adaptive, mono.            281         0.09s  true 9.849e-08         first_order
Speedmapping, aa: composite                  353         0.04s  true 9.151e-08         first_order
Speedmapping, aa: mono.                      286         0.08s  true 9.758e-08         first_order
Speedmapping, acx                            432         0.01s  true 7.963e-08         first_order

Electric field, SOR: 10000 parameters, abstol = 1.0e-7.
Solver                                      maps          time  conv   |resid|                 log
FixedPoint                                749333       510.10s false       NaN           Timed out
FixedPointAcceleration, Anderson             376         5.40s  true 9.939e-08 ReachedConvergenceT
FixedPointAcceleration, MPE                  380         7.04s  true 8.010e-08 ReachedConvergenceT
FixedPointAcceleration, RRE                  466        24.76s  true 9.552e-08 ReachedConvergenceT
FixedPointAcceleration, SEA                  153       113.35s false 1.387e+00           Timed out
FixedPointAcceleration, VEA                  660        30.55s  true 9.382e-08 ReachedConvergenceT
Speedmapping, aa                             376         0.12s  true 9.660e-08         first_order
Speedmapping, aa: adaptive                   373         0.13s  true 9.904e-08         first_order
Speedmapping, aa: adaptive, composite        403         0.06s  true 9.022e-08         first_order
Speedmapping, aa: adaptive, mono.            373         0.13s  true 9.904e-08         first_order
Speedmapping, aa: composite                  356         0.05s  true 8.760e-08         first_order
Speedmapping, aa: mono.                      376         0.12s  true 9.660e-08         first_order
Speedmapping, acx                            420         0.02s  true 9.739e-08         first_order

Electric field, Jacobi: 10000 parameters, abstol = 1.0e-7.
Solver                                      maps          time  conv   |resid|                 log
FixedPoint                                196154         3.22s  true 1.000e-07
FixedPointAcceleration, Anderson            1052        51.11s  true 9.969e-08 ReachedConvergenceT
FixedPointAcceleration, MPE                 1112       100.07s false 4.183e-04           Timed out
FixedPointAcceleration, RRE                  935       100.17s false 5.466e-03           Timed out
FixedPointAcceleration, SEA                  153       109.99s false 1.265e+01           Timed out
FixedPointAcceleration, VEA                 1113       100.17s false 3.124e-01           Timed out
Speedmapping, aa                             768         0.21s  true 9.982e-08         first_order
Speedmapping, aa: adaptive                   353         0.11s  true 9.905e-08         first_order
Speedmapping, aa: adaptive, composite        437         0.04s  true 9.300e-08         first_order
Speedmapping, aa: adaptive, mono.            353         0.11s  true 9.905e-08         first_order
Speedmapping, aa: composite                  638         0.06s  true 9.501e-08         first_order
Speedmapping, aa: mono.                      768         0.21s  true 9.982e-08         first_order
Speedmapping, acx                            893         0.01s  true 9.339e-08         first_order

Lid-driven cavity ﬂow: 52812 parameters, abstol = 0.0001.
Solver                                      maps          time  conv   |resid|                 log
FixedPoint                                297706       816.75s  true 1.000e-04
FixedPointAcceleration, Anderson            2064      1001.24s false 2.164e-03           Timed out
FixedPointAcceleration, MPE                 1449      1001.66s false 4.167e-03           Timed out
FixedPointAcceleration, RRE                    6         0.06s false       NaN    OutOfMemoryError
FixedPointAcceleration, SEA                  153      1013.57s false 3.637e-02           Timed out
FixedPointAcceleration, VEA                 1459      1001.32s false 1.093e-02           Timed out
Speedmapping, aa                            5789        51.87s  true 1.000e-04         first_order
Speedmapping, aa: adaptive                  4904        41.92s  true 9.999e-05         first_order
Speedmapping, aa: adaptive, composite       6190        34.95s  true 9.999e-05         first_order
Speedmapping, aa: adaptive, mono.           4904        44.19s  true 9.999e-05         first_order
Speedmapping, aa: composite                 7444        43.59s  true 9.988e-05         first_order
Speedmapping, aa: mono.                     5789        48.20s  true 1.000e-04         first_order
Speedmapping, acx                           4212        19.92s  true 9.983e-05         first_order
=#
