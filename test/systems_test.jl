using naumer_Dimensionality_2022, Test

@testset "LinearSystem" begin

    dynamic_mat = [2.0 0.0; 0.0 1.0]
    system = LinearSystem(dynamic_mat)
    x = [1.0, 1.0]
    τ = 1.0

    @testset "dimension" begin
        @test dimension(system) == 2
    end

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

    @testset "dimension" begin
        @test dimension(system) == 2
    end

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

    @testset "dimension" begin
        @test dimension(system) == 2
    end

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


@testset "LorenzSystem" begin

    σ = 10.0
    ρ = 28.0
    β = 8.0/3.0

    system = LorenzSystem(σ,ρ,β)
    x = [0.0, 0.0, 1.0]
    τ = 1

    @testset "dimension" begin
        @test dimension(system) == 3
    end

    @testset "differential" begin
        @test isapprox(differential(system, x), [0.0, 0.0, -8.0/3.0], atol=1e-5)
    end

    @testset "flow" begin

    end

    @testset "flowJacobian" begin

    end
end

