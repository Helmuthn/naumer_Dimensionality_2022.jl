using naumer_Dimensionality_2022, Test

@testset "StaticSystem" begin

    dim = 2
    system = StaticSystem(dim)
    x = [1.0, 1.0]
    τ = 1.0

    @testset "dimension" begin
        @test dimension(system) == dim
    end

    @testset "differential" begin
        @test differential(system, x) ≈ [0.0, 0.0] 
    end

    @testset "flow" begin
        @test flow(x, τ, system) ≈ x
    end

    @testset "flowJacobian" begin
        @test flowJacobian(x, τ, system) ≈ [1.0 0.0; 0.0 1.0]
    end
end

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
        # Very simple test, assuming correct functionality on May 04, 2022 to check for later issues
        @test isapprox(flow(x, τ, system), [1.250701247938, 0.9793760435906557], atol=1e-5)
    end

    @testset "flowJacobian" begin
        # Very simple test, assuming correct functionality on May 04, 2022 to check for later issues
        @test isapprox(flowJacobian(x, τ, system), [0.0435376 0.935839; -1.72123 -0.0820911], atol=1e-5)
        
    end
end

@testset "VanDerPolSystem_expanded" begin

    μ = 1
    dim = 1
    system = VanDerPolSystem_expanded(μ, dim)
    x = [0.0, 1.0, 1.0]
    τ = 1

    @testset "dimension" begin
        @test dimension(system) == dim + 2
    end

    @testset "differential" begin
        
        @test isapprox(differential(system, x), [1.0, 1.0, -x[3]], atol=1e-5)
    end

    @testset "flow" begin
        # Very simple test, assuming correct functionality on May 04, 2022 to check for later issues
        @test isapprox(flow(x, τ, system), [1.250701247938, 0.9793760435906557, exp(-1.0)*x[3]], atol=1e-5)
    end

    @testset "flowJacobian" begin
        # Very simple test, assuming correct functionality on May 04, 2022 to check for later issues
        @test isapprox(flowJacobian(x, τ, system), [0.0435376 0.935839 0; -1.72123 -0.0820911 0; 0 0 exp(-1.0)], atol=1e-5)
        
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
        # Very simple test, assuming correct functionality on May 04, 2022 to check for later issues
        @test isapprox(flow(x, τ, system), [0.0, 0.0, 0.06948364385801184], atol=1e-5)
    end

    @testset "flowJacobian" begin
        # Very simple test, assuming correct functionality on May 04, 2022 to check for later issues
        @test isapprox(flowJacobian(x, τ, system), [45246.6 36030.0 0; 98664.2 78567.0 0; 0 0 0.0694835], atol=1e-5, rtol=1e-4)
    end
end

