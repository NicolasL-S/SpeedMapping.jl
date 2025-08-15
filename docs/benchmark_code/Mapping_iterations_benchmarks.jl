########################
# Fixed-point iterations

using BenchmarkTools, JLD2, FileIO, Logging, FixedPointAcceleration, FixedPoint, SpeedMapping, FixedPointTestProblems

path_out = ""
include("Benchmarking_utils.jl")

# Importing the problems
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
plot_res(res_maps_all, problems_names_len, solver_names, title, path_out*"maps_all.svg"; 
	size = (600, 400), legend_rowgap = -3)

#=
Hasselblad, Poisson mixtures: 3 parameters, abstol = 1.0e-7.
Solver                                    maps          time  conv   |resid|                 log                      obj
FixedPoint                               21627     1321.25μs  true 9.998e-08                             1989.94585988424
FixedPointAcceleration, Anderson            23       89.23μs  true 1.240e-08 ReachedConvergenceT       1989.9458598829701
FixedPointAcceleration, MPE                 38      101.54μs  true 6.107e-08 ReachedConvergenceT       1989.9458598829892
FixedPointAcceleration, RRE                 42      115.28μs  true 7.180e-08 ReachedConvergenceT       1989.9458598833305
FixedPointAcceleration, SEA                 37       87.34μs  true 6.430e-08 ReachedConvergenceT       1989.9458598829667
FixedPointAcceleration, VEA                 43       98.78μs  true 1.741e-09 ReachedConvergenceT       1989.9458598829644
Speedmapping, aa                            21        7.34μs  true 1.217e-08         first_order       1989.9458598829644
Speedmapping, aa: adaptive                  21        7.67μs  true 9.201e-09         first_order        1989.945859882964
Speedmapping, aa: adaptive, composite       44        8.02μs  true 9.118e-08         first_order         1989.94585988297
Speedmapping, aa: adaptive, mono.           17       10.44μs  true 2.917e-08         first_order       1989.9458598829658
Speedmapping, aa: composite                 36        6.71μs  true 6.108e-08         first_order       1989.9458598829679
Speedmapping, aa: mono.                     17        9.77μs  true 2.943e-08         first_order       1989.9458598829656
Speedmapping, acx                           52        4.15μs  true 3.214e-09         first_order       1989.9458598829644


Exchange economy: 10 parameters, abstol = 1.0e-7.
Solver                                    maps          time  conv   |resid|                 log
FixedPoint                                 286       80.53μs  true 9.878e-08
FixedPointAcceleration, Anderson             7       38.90μs  true 7.100e-11 ReachedConvergenceT
FixedPointAcceleration, MPE                  7       27.45μs  true 2.156e-11 ReachedConvergenceT
FixedPointAcceleration, RRE                  7       39.59μs  true 2.156e-11 ReachedConvergenceT
FixedPointAcceleration, SEA                  8       30.15μs  true 1.228e-09 ReachedConvergenceT
FixedPointAcceleration, VEA                  7       27.18μs  true 1.433e-09 ReachedConvergenceT
Speedmapping, aa                             7        6.78μs  true 2.271e-09         first_order
Speedmapping, aa: adaptive                   7        7.28μs  true 2.294e-09         first_order
Speedmapping, aa: adaptive, composite        7        5.68μs  true 2.017e-08         first_order
Speedmapping, aa: adaptive, mono.            7        7.34μs  true 2.294e-09         first_order
Speedmapping, aa: composite                  7        5.46μs  true 2.017e-08         first_order
Speedmapping, aa: mono.                      7        6.92μs  true 2.271e-09         first_order
Speedmapping, acx                            7        2.55μs  true 6.193e-08         first_order


Wang, PH interval censoring: 10 parameters, abstol = 1.0e-5.
Solver                                    maps          time  conv   |resid|                 log                      obj
FixedPoint                              159055    37356.68ms  true 1.000e-05                            946.0074704441713
FixedPointAcceleration, Anderson           415      172.79ms  true 7.102e-06 ReachedConvergenceT        946.0075253344736
FixedPointAcceleration, MPE                193       64.11ms  true 2.265e-08 ReachedConvergenceT        950.1698721250284
FixedPointAcceleration, RRE              50330   100004.47ms false 1.374e-02           Timed out        954.5264076927797
FixedPointAcceleration, SEA                 49       18.36ms  true 5.017e-07 ReachedConvergenceT        960.3760393857056
FixedPointAcceleration, VEA               1968      952.13ms  true 9.071e-06 ReachedConvergenceT        946.0074703622324
Speedmapping, aa                           295      103.53ms  true 9.054e-06         first_order        950.1698112628449
Speedmapping, aa: adaptive                 306      120.90ms  true 2.177e-06         first_order        950.1698067297523
Speedmapping, aa: adaptive, composite      669      213.65ms  true 4.420e-06         first_order        950.0857735450315
Speedmapping, aa: adaptive, mono.           86       37.81ms  true 5.642e-06         first_order        946.0074838607834
Speedmapping, aa: composite                443      154.35ms  true 8.998e-06         first_order        949.1267397988238
Speedmapping, aa: mono.                    149       59.26ms  true 3.691e-06         first_order         946.007477594379
Speedmapping, acx                          402      118.99ms  true 8.112e-06         first_order         946.007552007742


Mixture of 3 normals: 17 parameters, abstol = 1.0e-7.
Solver                                    maps          time  conv   |resid|                 log                      obj
FixedPoint                                4220     1724.18ms  true 9.972e-08                            4787.809009703242
FixedPointAcceleration, Anderson            32       11.87ms  true 6.084e-09 ReachedConvergenceT       4787.8090097032455
FixedPointAcceleration, MPE                 67       28.63ms  true 2.119e-08 ReachedConvergenceT        4787.809009703251
FixedPointAcceleration, RRE                 61       23.83ms false 1.104e-07 ReachedConvergenceT        4787.809009703241
FixedPointAcceleration, SEA                151      252.39ms  true 7.732e-08 ReachedConvergenceT        4862.643310378573
FixedPointAcceleration, VEA                118       48.38ms  true 6.387e-08 ReachedConvergenceT         4787.80900970324
Speedmapping, aa                            29       10.28ms  true 9.975e-09         first_order        4787.809009703249
Speedmapping, aa: adaptive                  33       12.07ms  true 7.225e-09         first_order       4787.8090097032455
Speedmapping, aa: adaptive, composite       49       18.98ms  true 9.719e-08         first_order        4787.809009703245
Speedmapping, aa: adaptive, mono.           27       22.57ms  true 1.553e-08         first_order        4787.809009703242
Speedmapping, aa: composite                 50       18.48ms  true 3.761e-08         first_order       4787.8090097032355
Speedmapping, aa: mono.                     28       23.15ms  true 3.780e-08         first_order        4787.809009703247
Speedmapping, acx                           55       20.32ms  true 3.476e-08         first_order        4787.809009703252


Higham, correlation matrix mmb13: 42 parameters, abstol = 1.0e-7.
Solver                                    maps          time  conv   |resid|                 log
FixedPoint                                3755       21.38ms  true 9.965e-08
FixedPointAcceleration, Anderson            24        1.09ms  true 2.535e-09 ReachedConvergenceT
FixedPointAcceleration, MPE                 43        0.51ms  true 2.917e-08 ReachedConvergenceT
FixedPointAcceleration, RRE                 43        0.53ms  true 4.027e-08 ReachedConvergenceT
FixedPointAcceleration, SEA                349    10797.02ms false 3.661e-07           Timed out
FixedPointAcceleration, VEA                117        1.99ms  true 8.584e-08 ReachedConvergenceT
Speedmapping, aa                            21        0.13ms  true 4.166e-08         first_order
Speedmapping, aa: adaptive                  25        0.17ms  true 1.129e-08         first_order
Speedmapping, aa: adaptive, composite       34        0.18ms  true 7.062e-08         first_order
Speedmapping, aa: adaptive, mono.           25        0.16ms  true 1.129e-08         first_order
Speedmapping, aa: composite                 32        0.17ms  true 4.543e-08         first_order
Speedmapping, aa: mono.                     21        0.13ms  true 4.166e-08         first_order
Speedmapping, acx                           60        0.31ms  true 4.111e-08         first_order


Power iter. for dom. eigenvalue: 100 parameters, abstol = 1.0e-7.
Solver                                    maps          time  conv   |resid|                 log
FixedPoint                                 303        0.28ms  true 9.470e-08
FixedPointAcceleration, Anderson            14        0.29ms  true 1.496e-08 ReachedConvergenceT
FixedPointAcceleration, MPE                 14        0.12ms  true 3.694e-08 ReachedConvergenceT
FixedPointAcceleration, RRE                 17        0.17ms  true 4.944e-08 ReachedConvergenceT
FixedPointAcceleration, SEA                 34        1.67ms  true 5.393e-08 ReachedConvergenceT
FixedPointAcceleration, VEA                 17        0.16ms  true 4.891e-08 ReachedConvergenceT
Speedmapping, aa                            14        0.04ms  true 2.693e-08         first_order
Speedmapping, aa: adaptive                  13        0.04ms  true 8.383e-08         first_order
Speedmapping, aa: adaptive, composite       16        0.03ms  true 5.135e-08         first_order
Speedmapping, aa: adaptive, mono.           13        0.04ms  true 8.383e-08         first_order
Speedmapping, aa: composite                 16        0.03ms  true 5.993e-08         first_order
Speedmapping, aa: mono.                     14        0.04ms  true 2.693e-08         first_order
Speedmapping, acx                           18        0.02ms  true 6.470e-08         first_order


Linear: 100 parameters, abstol = 1.0e-7.
Solver                                    maps          time  conv   |resid|                 log                      obj
FixedPoint                                8008        2.53ms  true 9.992e-08                          -130.98128232040014
FixedPointAcceleration, Anderson            77        4.70ms  true 8.418e-08 ReachedConvergenceT       -130.9812823204004
FixedPointAcceleration, MPE                175        8.01ms  true 8.492e-08 ReachedConvergenceT      -130.98128232040034
FixedPointAcceleration, RRE                223       14.45ms  true 8.897e-08 ReachedConvergenceT      -130.98128232040023
FixedPointAcceleration, SEA                  7        0.06ms  true 1.700e-14 ReachedConvergenceT       -130.9812823204004
FixedPointAcceleration, VEA                761      147.56ms  true 9.461e-08 ReachedConvergenceT      -130.98128232040028
Speedmapping, aa                            77        0.22ms  true 9.975e-08         first_order      -130.98128232040037
Speedmapping, aa: adaptive                  72        0.23ms  true 7.713e-08         first_order      -130.98128232040037
Speedmapping, aa: adaptive, composite       77        0.06ms  true 7.914e-08         first_order       -130.9812823204004
Speedmapping, aa: adaptive, mono.           72        0.90ms  true 7.713e-08         first_order      -130.98128232040037
Speedmapping, aa: composite                 86        0.06ms  true 7.751e-08         first_order       -130.9812823204004
Speedmapping, aa: mono.                     77        0.96ms  true 9.975e-08         first_order      -130.98128232040037
Speedmapping, acx                          108        0.03ms  true 7.036e-08         first_order       -130.9812823204003


Consumption smoothing: 107 parameters, abstol = 1.0e-7.
Solver                                    maps          time  conv   |resid|                 log
FixedPoint                                2361       100.03s false 7.617e-05           Timed out
FixedPointAcceleration, Anderson            20         0.86s  true 2.001e-08 ReachedConvergenceT
FixedPointAcceleration, MPE                223        10.18s  true 4.670e-08 ReachedConvergenceT
FixedPointAcceleration, RRE               2291       100.01s false 4.772e-02           Timed out
FixedPointAcceleration, SEA               2215       100.04s false 1.173e+01           Timed out
FixedPointAcceleration, VEA               2415       100.03s false 3.338e-04           Timed out
Speedmapping, aa                            18         0.79s  true 7.841e-09         first_order
Speedmapping, aa: adaptive                  20         0.86s  true 9.965e-09         first_order
Speedmapping, aa: adaptive, composite       35         1.48s  true 3.738e-09         first_order
Speedmapping, aa: adaptive, mono.           20         0.84s  true 9.965e-09         first_order
Speedmapping, aa: composite                 35         1.52s  true 6.680e-09         first_order
Speedmapping, aa: mono.                     18         0.78s  true 7.841e-09         first_order
Speedmapping, acx                           79         3.45s  true 9.650e-09         first_order


ALS for CANDECOMP: 450 parameters, abstol = 1.0e-7.
Solver                                    maps          time  conv   |resid|                 log
FixedPoint                                9712         3.29s  true 9.999e-08
FixedPointAcceleration, Anderson          3032         8.08s  true 1.290e-08 ReachedConvergenceT
FixedPointAcceleration, MPE                 61         0.03s  true 9.024e-08 ReachedConvergenceT
FixedPointAcceleration, RRE                109         0.07s  true 2.601e-08 ReachedConvergenceT
FixedPointAcceleration, SEA               2104         8.19s  true 7.731e-08 ReachedConvergenceT
FixedPointAcceleration, VEA                264         0.22s  true 9.591e-08 ReachedConvergenceT
Speedmapping, aa                           654         0.23s  true 9.550e-08         first_order
Speedmapping, aa: adaptive                 315         0.10s  true 1.378e-08         first_order
Speedmapping, aa: adaptive, composite       78         0.02s  true 5.624e-08         first_order
Speedmapping, aa: adaptive, mono.          315         0.10s  true 1.378e-08         first_order
Speedmapping, aa: composite                 88         0.03s  true 5.700e-08         first_order
Speedmapping, aa: mono.                    654         0.21s  true 9.550e-08         first_order
Speedmapping, acx                           64         0.02s  true 6.816e-08         first_order


Lange, ancestry: 750 parameters, abstol = 1.0e-5.
Solver                                    maps          time  conv   |resid|                 log                      obj
FixedPoint                               21341       100.01s false 9.766e-04           Timed out       14658.897740624303
FixedPointAcceleration, Anderson          6504       100.04s false 7.235e-02           Timed out       18902.552447357768
FixedPointAcceleration, MPE                472         2.76s  true 9.167e-06 ReachedConvergenceT        14658.89773568876
FixedPointAcceleration, RRE               4337       100.02s false 4.962e-01           Timed out       18796.436362910554
FixedPointAcceleration, SEA                  7         0.03s false       NaN InvalidInputOrOutpu                      NaN
FixedPointAcceleration, VEA               4997       100.02s false 9.577e-01           Timed out       18849.438769553024
Speedmapping, aa                         21552       100.00s false 6.337e-02           Timed out       18894.318895993056
Speedmapping, aa: adaptive               22022       100.00s false 5.959e-02           Timed out       18890.711683316727
Speedmapping, aa: adaptive, composite    22157       100.01s false 5.572e-02           Timed out       18894.239103620785
Speedmapping, aa: adaptive, mono.          370         1.76s  true 9.154e-06         first_order       14658.897735700908
Speedmapping, aa: composite              22174       100.00s false 6.311e-02           Timed out       18893.826135544437
Speedmapping, aa: mono.                    404         1.95s  true 9.176e-06         first_order       14658.897735703162
Speedmapping, acx                          372         1.71s  true 9.615e-06         first_order       14658.897735703118


Bratu: 10000 parameters, abstol = 1.0e-7.
Solver                                    maps          time  conv   |resid|                 log
FixedPoint                               13972       200.01s false 7.683e-03           Timed out
FixedPointAcceleration, Anderson          1425       124.98s  true 9.851e-08 ReachedConvergenceT
FixedPointAcceleration, MPE               1356       200.11s false 3.387e-03           Timed out
FixedPointAcceleration, RRE               1214       200.27s false 2.725e-03           Timed out
FixedPointAcceleration, SEA                 32         0.55s false       NaN InvalidInputOrOutpu
FixedPointAcceleration, VEA               1369       200.15s false 1.590e-02           Timed out
Speedmapping, aa                           865        13.18s  true 9.951e-08         first_order
Speedmapping, aa: adaptive                 256         4.16s  true 9.980e-08         first_order
Speedmapping, aa: adaptive, composite      259         3.97s  true 9.842e-08         first_order
Speedmapping, aa: adaptive, mono.          256         4.13s  true 9.980e-08         first_order
Speedmapping, aa: composite                470         7.26s  true 9.154e-08         first_order
Speedmapping, aa: mono.                    865        14.00s  true 9.951e-08         first_order
Speedmapping, acx                          842        12.71s  true 8.972e-08         first_order


Electric field, Gauss-Seidel: 10000 parameters, abstol = 1.0e-7.
Solver                                    maps          time  conv   |resid|                 log
FixedPoint                              115278         4.49s  true 9.999e-08
FixedPointAcceleration, Anderson           286         2.65s  true 9.342e-08 ReachedConvergenceT
FixedPointAcceleration, MPE               1131       100.13s false 1.374e-06           Timed out
FixedPointAcceleration, RRE                985       100.27s false 3.434e-05           Timed out
FixedPointAcceleration, SEA                146       100.80s false 8.264e-01           Timed out
FixedPointAcceleration, VEA               1149       100.16s false 2.807e-02           Timed out
Speedmapping, aa                           286         0.09s  true 9.758e-08         first_order
Speedmapping, aa: adaptive                 281         0.10s  true 9.849e-08         first_order
Speedmapping, aa: adaptive, composite      353         0.05s  true 9.767e-08         first_order
Speedmapping, aa: adaptive, mono.          281         0.11s  true 9.849e-08         first_order
Speedmapping, aa: composite                353         0.04s  true 9.151e-08         first_order
Speedmapping, aa: mono.                    286         0.10s  true 9.758e-08         first_order
Speedmapping, acx                          434         0.01s  true 9.408e-08         first_order


Electric field, SOR: 10000 parameters, abstol = 1.0e-7.
Solver                                    maps          time  conv   |resid|                 log
FixedPoint                              747069       561.02s false       NaN           Timed out
FixedPointAcceleration, Anderson           376         3.43s  true 9.939e-08 ReachedConvergenceT
FixedPointAcceleration, MPE                380         5.18s  true 8.010e-08 ReachedConvergenceT
FixedPointAcceleration, RRE                466        22.08s  true 9.552e-08 ReachedConvergenceT
FixedPointAcceleration, SEA                153       101.33s false 1.387e+00           Timed out
FixedPointAcceleration, VEA                660        26.66s  true 9.382e-08 ReachedConvergenceT
Speedmapping, aa                           376         0.15s  true 9.660e-08         first_order
Speedmapping, aa: adaptive                 373         0.16s  true 9.904e-08         first_order
Speedmapping, aa: adaptive, composite      403         0.07s  true 9.022e-08         first_order
Speedmapping, aa: adaptive, mono.          373         0.16s  true 9.904e-08         first_order
Speedmapping, aa: composite                356         0.06s  true 8.760e-08         first_order
Speedmapping, aa: mono.                    376         0.14s  true 9.660e-08         first_order
Speedmapping, acx                          420         0.03s  true 9.739e-08         first_order

res = speedmapping(x0; m! = (xout, xin) -> power_iteration!(xout, xin, A), algo = :aa)

Electric field, Jacobi: 10000 parameters, abstol = 1.0e-7.
Solver                                    maps          time  conv   |resid|                 log
FixedPoint                              196154         3.83s  true 1.000e-07
FixedPointAcceleration, Anderson          1052        41.48s  true 9.969e-08 ReachedConvergenceT
FixedPointAcceleration, MPE               1158       100.19s false 3.440e-04           Timed out
FixedPointAcceleration, RRE                975       100.19s false 4.469e-03           Timed out
FixedPointAcceleration, SEA                153       103.20s false 1.265e+01           Timed out
FixedPointAcceleration, VEA               1169       100.11s false 1.638e-01           Timed out
Speedmapping, aa                           768         0.24s  true 9.982e-08         first_order
Speedmapping, aa: adaptive                 353         0.12s  true 9.905e-08         first_order
Speedmapping, aa: adaptive, composite      437         0.05s  true 9.300e-08         first_order
Speedmapping, aa: adaptive, mono.          353         0.13s  true 9.905e-08         first_order
Speedmapping, aa: composite                638         0.07s  true 9.501e-08         first_order
Speedmapping, aa: mono.                    768         0.26s  true 9.982e-08         first_order
Speedmapping, acx                          565         0.01s  true 8.688e-08         first_order


Lid-driven cavity ﬂow: 52812 parameters, abstol = 0.0001.
Solver                                    maps          time  conv   |resid|                 log
FixedPoint                              297706       888.91s  true 1.000e-04
FixedPointAcceleration, Anderson          2264      1000.82s false 1.741e-03           Timed out
FixedPointAcceleration, MPE               1601      1001.44s false 3.048e-03           Timed out
FixedPointAcceleration, RRE                  6         0.08s false       NaN    OutOfMemoryError
FixedPointAcceleration, SEA                160      1216.94s false 3.519e-02           Timed out
FixedPointAcceleration, VEA               1583      1000.70s false 8.315e-03           Timed out
Speedmapping, aa                          5789        53.69s  true 1.000e-04         first_order
Speedmapping, aa: adaptive                4904        43.90s  true 9.999e-05         first_order
Speedmapping, aa: adaptive, composite     5881        37.63s  true 1.000e-04         first_order
Speedmapping, aa: adaptive, mono.         4904        44.95s  true 9.999e-05         first_order
Speedmapping, aa: composite               7387        46.57s  true 9.998e-05         first_order
Speedmapping, aa: mono.                   5789        51.86s  true 1.000e-04         first_order
Speedmapping, acx                         4303        15.59s  true 9.930e-05         first_order
=#
