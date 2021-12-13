using naumer_ICML_2022, Test
using LinearAlgebra: eigvals


@testset "randomPSD" begin
    random_matrix = randomPSD(3)

    # Check size is correct
    @test size(random_matrix) == (3,3)

    # Check for PSD
    @test minimum(eigvals(random_matrix)) >= 0
end

@testset "nearestNeighbor" begin

end

@testset "updateFisherInformation" begin

end

@testset "optimalAction_NearestNeighbor" begin

end

@testset "valueUpdate_NearestNeighbor" begin

end
