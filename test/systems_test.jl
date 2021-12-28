using naumer_ICML_2022, Test

@testset "LinearSystem" begin

    dynamic_mat = [2.0 0.0; 0.0 1.0]
    system = LinearSystem(dynamic_mat)
    x = [1.0, 1.0]
    τ = 1.0

    @testset "differential" begin
        @test differential(system, x) ≈ [2.0, 1.0] 
    end

    @testset "flow" begin
        @test flow(x, τ, system) ≈ [exp(2.0), exp(1.0)]
    end

    @testset "flowJacobian" begin
        @test flowJacobian(x, τ, system) ≈ [exp(2.0) 0; 0 exp(1.0)]
    end
end


@testset "VanDerPolSystem" begin

    μ = 1
    system = VanDerPolSystem(μ)
    x = [0.0, 1.0]
    τ = 1

    @testset "differential" begin
        @test isapprox(differential(system, x), [1.0, 1.0], atol=1e-5)
    end

    @testset "flow" begin

    end

    @testset "flowJacobian" begin

    end
end

@testset "HopfSystem" begin

    λ = 1
    b = -1
    system = HopfSystem(λ, b)
    x = [0.0, 1.0]
    τ = 2π

    @testset "differential" begin
        @test isapprox(differential(system, x), [-1.0, 0.0], atol=1e-5)
    end

    @testset "flow" begin
        @test isapprox(flow(x, τ, system), [0.0, 1.0], atol=1e-5)
    end

    @testset "flowJacobian" begin
        jacobian = flowJacobian(x, τ, system)
        @test isapprox(jacobian, [1.0 0; 0 0.0], atol=1e-4)
    end
end
