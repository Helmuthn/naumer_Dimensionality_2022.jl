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
