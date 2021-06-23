using SpeedMapping, Test, LinearAlgebra

function f(x) # Rosenbrock objective
	f_out = 0.0
	for i ∈ 1:Int(length(x)/2)
		f_out += 100(x[2i - 1]^2 - x[2i])^2 + (x[2i - 1] - 1)^2
	end
	return f_out
end

function g!(∇,x) # Rosenbrock gradient
	for i ∈ 1:Int(length(x)/2)
		∇[2i] = -200 * (x[2i - 1]^2 - x[2i])
		∇[2i - 1] =  400 * (x[2i - 1]^2 - x[2i]) * x[2i - 1] + 2(x[2i - 1] - 1)
	end
	return nothing
end

# Power method
C = [1 2 3; 4 5 6; 7 8 9]
A = C + C'

function m!(x_out, x_in) # map for the power method
	mul!(x_out, A, x_in)
	x_out ./= norm(x_out, Inf)
end

# Testing
function exception(expr, error)
	goodexception = false
	try
		eval(expr)
	catch e
		goodexception = isa(e, error)
	end
	return goodexception
end

@testset "SpeedMapping.jl" begin
	# Testing the complete algorithm with f, g!, ForwardDiff, m!, lower, upper
	@test speedmapping([-2.0,5.0]; g!).minimizer ≈ [1,1]
	@test speedmapping(zeros(2); f, g!).minimizer ≈ [1,1]
	@test speedmapping(zeros(2); f, g!, check_obj = true).minimizer ≈ [1,1]
	@test speedmapping(zeros(4); f).minimizer ≈ [1,1,1,1]
	@test speedmapping([5.0,5.0]; f, g!, lower = [1.5,-Inf]).minimizer ≈ [1.5, 2.25]
	@test speedmapping([0.0,0.0]; f, g!, upper = [Inf,0.25]).minimizer ≈ [0.5048795424100077, 0.25]
	@test speedmapping(ones(3); m!).minimizer' * A[:,3] ≈ 32.916472867168714
	@test speedmapping(ones(3); m!, stabilize = true).minimizer' * A[:,3] ≈ 32.916472867168714
	@test speedmapping(ones(3); m!, Lp = Inf).minimizer' * A[:,3] ≈ 32.916472867168714

	# Exceptions
	# Starting point outside boundary
	@test exception(:(speedmapping([0.0, 0.0]; f, g!, lower = [1,1])), DomainError) 
	# Can't provide both g! and m!
	@test exception(:(speedmapping([0.0, 0.0]; g!, m!)), ArgumentError) 
	# eltype(x_in) is Int
	@test exception(:(speedmapping([0, 0]; g!)), ArgumentError) 
	# check_obj without providing f
	@test exception(:(speedmapping([0.0, 0.0]; g!, check_obj = true)), ArgumentError) 
	# The gradient is zero
	@test exception(:(speedmapping([1.0, 1.0]; g!)), DomainError) 
end
