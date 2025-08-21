using Test
using LinearAlgebra: Hermitian, Diagonal, norm, mul!
using SpeedMapping

function f(x) # Easy Rosenbrock objective
    f_out = 0.0
    for i in 1:Int(length(x) / 2)
        f_out += (x[2i-1]^2 - x[2i])^2 + (x[2i-1] - 1)^2
    end
    return f_out
end

function g!(grad, x) # Easy Rosenbrock gradient
    for i in 1:Int(length(x) / 2)
        grad[2i] = -2 * (x[2i-1]^2 - x[2i])
        grad[2i-1] = 4 * (x[2i-1]^2 - x[2i]) * x[2i-1] + 2(x[2i-1] - 1)
    end
    return nothing
end

# With tuple
gt(x) = (4 * (x[1]^2 - x[2]) * x[1] + 2(x[1] - 1),-2 * (x[1]^2 - x[2]))

function g_matrix!(grad, x) # Easy Rosenbrock gradient for N × 2 input
    for i in axes(x, 1)
        grad[i, 2] = -2 * (x[i, 1]^2 - x[i, 2])
        grad[i, 1] = 4 * (x[i, 1]^2 - x[i, 2]) * x[i, 1] + 2(x[i, 1] - 1)
    end
    return nothing
end

# Power method
C = [1 2 3; 4 5 6; 7 8 9]
A = C + C'
B = Hermitian(ones(10) * ones(10)' .* im + Diagonal(1:10))

function m!(x_out, x_in, M) # map for the power method
    mul!(x_out, M, x_in)
    x_out ./= norm(x_out, Inf)
end

function m_horizontal!(x_out, x_in) # map for the power method
    x_out .= (A * x_in')'
    x_out ./= norm(x_out, Inf)
end

# Diagonal linear problem
n_lin = 5
tr = 2(1:n_lin)/(n_lin+1)
map_diag! = (x_out, x_in) -> @. x_out = x_in - tr .* x_in + 1.
r_diag! = (res, x) -> @. res = x - tr .* x + 1. .- x
x0_lin = zeros(n_lin)

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

@testset "ACX" begin

    # Testing the complete algorithm with f, g!, ForwardDiff, m!, lower, upper
	@test speedmapping([-2.0, 1.0]; g!, algo = :acx).minimizer ≈ [1, 1]
    @test speedmapping([-2.0, 1.0]; g!, store_trace=true, algo = :acx).acx_trace[1].x[1] == -2
    @test speedmapping(zeros(2); f, g!, algo = :acx).minimizer ≈ [1, 1]
    @test speedmapping(zeros(4); f, algo = :acx).minimizer ≈ [1, 1, 1, 1]
    @test speedmapping([5.0, 5.0]; f, g!, lower=[1.5, -Inf], algo = :acx).minimizer ≈ [1.5, 2.25]
    @test speedmapping([0.0, 0.0]; f, g!, upper=[Inf, 0.25], algo = :acx).minimizer ≈ [0.689398350053853, 0.25]
    @test speedmapping(ones(3); (m!)=(x_out, x_in) -> m!(x_out, x_in, A), algo = :acx).minimizer' * A[:, 3] ≈ 32.916472867168096
end

@testset "other number types" begin

    #support for bigfloats
    @test speedmapping(zeros(BigFloat, 2); g!).minimizer isa Vector{BigFloat}

    #support for float32 and matrices
    @test speedmapping(Float32.([-2.0 5.0]); (g!)=g_matrix!, abstol=1e-4).minimizer ≈ [1 1]
    @test (speedmapping(Float32.(ones(3)'); (m!)=m_horizontal!).minimizer*A[:, 3])[1] ≈ 32.916473f0

    #support for Complex type
    @test speedmapping(ones(10) .+ 0im; (m!)=(x_out, x_in) -> m!(x_out, x_in, B)).minimizer' * B[:, 10] ≈ 4.083655765461623 + 12.373570801838728im

    # support for tuples
    @test collect(speedmapping((0.,0.); g = gt).minimizer) ≈ [1, 1]
end

# Test for aa
@testset "AA" begin
	# Test with m!
	aa_res_lin = speedmapping(x0_lin; m! = map_diag!, algo = :aa, store_trace=true)
	@test aa_res_lin.minimizer'aa_res_lin.minimizer ≈ 13.1725
	@test aa_res_lin.aa_trace[1].x[1] == 0

	# Test with m! and other specs
    min_aa_res_lin2 = speedmapping(x0_lin; m! = map_diag!, algo = :aa, store_trace=true, composite = :aa1, ada_relax = :none, pnorm = Inf).minimizer
	@test min_aa_res_lin2'min_aa_res_lin2 ≈ 13.1725

    min_aa_res_lin3 = speedmapping(x0_lin; m! = map_diag!, algo = :aa, store_trace=true, composite = :acx2, lags = 3, abstol = 1e-10, pnorm = 1).minimizer
    @test min_aa_res_lin3'min_aa_res_lin3 ≈ 13.1725

	# Test with r!
	aa_res_lin_r = speedmapping(zeros(n_lin); r! = r_diag!, algo = :aa)
	@test aa_res_lin_r.minimizer'aa_res_lin_r.minimizer ≈ 13.1725
end

@testset "Exceptions" begin

    # Can't provide both g! and m!
    @test exception(:(speedmapping([0.0, 0.0]; g!, m!)), ArgumentError)

    # eltype(x_in) is Int
    @test exception(:(speedmapping([0, 0]; g!)), ArgumentError)

    # g! with :aa
    @test exception(:(speedmapping([0., 0.]; g!, algo = :aa)), ArgumentError) 

    # r! with :acx
    @test exception(:(speedmapping(zeros(n_lin); r! = r_diag!, algo = :acx)), ArgumentError) 

    # tuple with g!
    @test exception(:(speedmapping((0., 0.); g!, algo = :aa)), ArgumentError) 

    # array with g
    @test exception(:(speedmapping([0., 0.]; g = gt, algo = :aa)), ArgumentError) 
end