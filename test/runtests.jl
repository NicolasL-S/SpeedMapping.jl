using SpeedMapping, Test, LinearAlgebra, ForwardDiff

function f(x) # Rosenbrock objective
    f_out = 0.0
    for i ∈ 1:Int(length(x) / 2)
        f_out += 100(x[2i-1]^2 - x[2i])^2 + (x[2i-1] - 1)^2
    end
    return f_out
end

function g!(∇, x) # Rosenbrock gradient
    for i ∈ 1:Int(length(x) / 2)
        ∇[2i] = -200 * (x[2i-1]^2 - x[2i])
        ∇[2i-1] = 400 * (x[2i-1]^2 - x[2i]) * x[2i-1] + 2(x[2i-1] - 1)
    end
    return nothing
end

function g_matrix!(∇, x) # Rosenbrock gradient for N × 2 input
    for i ∈ 1:size(x, 1)
        ∇[i, 2] = -200 * (x[i, 1]^2 - x[i, 2])
        ∇[i, 1] = 400 * (x[i, 1]^2 - x[i, 2]) * x[i, 1] + 2(x[i, 1] - 1)
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
    @test speedmapping([-2.0, 5.0]; g!).minimizer ≈ [1, 1]
    @test speedmapping([-2.0, 5.0]; g!, store_info=true).info.x[1][1] == -2
    @test speedmapping(zeros(2); f, g!).minimizer ≈ [1, 1]
    @test speedmapping(zeros(2); f, g!, check_obj=true).minimizer ≈ [1, 1]
    @test speedmapping(zeros(4); f).minimizer ≈ [1, 1, 1, 1]
    @test speedmapping([5.0, 5.0]; f, g!, lower=[1.5, -Inf]).minimizer ≈ [1.5, 2.25]
    @test speedmapping([0.0, 0.0]; f, g!, upper=[Inf, 0.25]).minimizer ≈ [0.5048795423935251, 0.25]
    @test speedmapping(ones(3); (m!)=(x_out, x_in) -> m!(x_out, x_in, A)).minimizer' * A[:, 3] ≈ 32.916472867168096
    @test speedmapping(ones(3); (m!)=(x_out, x_in) -> m!(x_out, x_in, A), stabilize=true).minimizer' * A[:, 3] ≈ 32.91647286145264

    # Exceptions

    # Starting point outside boundary
    @test exception(:(speedmapping([0.0, 0.0]; f, g!, lower=[1, 1])), DomainError)

    # Can't provide both g! and m!
    @test exception(:(speedmapping([0.0, 0.0]; g!, m!)), ArgumentError)

    # eltype(x_in) is Int
    @test exception(:(speedmapping([0, 0]; g!)), ArgumentError)

    # check_obj without providing f
    @test exception(:(speedmapping([0.0, 0.0]; g!, check_obj=true)), ArgumentError)
end

@testset "other number types" begin

    #support for bigfloats
    @test speedmapping(zeros(BigFloat, 2); g!).minimizer isa Vector{BigFloat}

    #support for ForwardDiff.Dual
    @test ForwardDiff.jacobian(x -> speedmapping(x; g!).minimizer, [0.0, 0.0]) isa Matrix{Float64}

    #support for float32 and matrices
    @test speedmapping(Float32.([-2.0 5.0]); (g!)=g_matrix!, tol=1e-4).minimizer ≈ [1 1]
    @test (speedmapping(Float32.(ones(3)'); (m!)=m_horizontal!).minimizer*A[:, 3])[1] ≈ 32.916473f0

    #support for Complex type
    @test speedmapping(ones(10) .+ 0im; (m!)=(x_out, x_in) -> m!(x_out, x_in, B)).minimizer' * B[:, 10] ≈ 4.083655765461623 + 12.373570801838728im
end