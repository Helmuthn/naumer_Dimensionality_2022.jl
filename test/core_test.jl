using naumer_ICML_2022, Test
using LinearAlgebra: eigvals


@testset "randomPSD" begin
    random_matrix = randomPSD(3)

    # Check size is correct
    @test size(random_matrix) == (3,3)

    # Check for PSD
    @test minimum(eigvals(random_matrix)) >= 0

    # Check eigenvalue mean for appropriate scaling
    total = 0
    for i in 1:10000
        random_matrix = randomPSD(1,0.5)
        total += eigvals(random_matrix)[1]
    end
    total /= 10000
    @test isapprox(total, 2, atol=1e-1)
end

@testset "nearestNeighbor" begin
    # 1D form
    target = 1.1
    samples = [1.0,2.0]
    values = [-1,1]
    @test nearestNeighbor(target,samples,values) == -1
    @test nearestNeighbor(target+1,samples,values) == 1

    # 2D form
    target = [1.0,2.0]
    samples = [ 1.1 2.0;
                2.0 1.1 ]
    values = [-1, 1]
    @test nearestNeighbor(target,samples,values) == -1

    # Matrix form
    target = [1.0 0.0;
              0.0 1.0 ]
    samples = zeros(2,2,2)
    samples[:,:,1] = [ 1.2 0.1;
                      -0.1 0.9 ]
    values = [-1, 1]

    @test nearestNeighbor(target,samples,values) == -1
    
end

@testset "updateCRLB" begin

    # Identity Jacobian, Identity FIM
    crlb         = [1.0 0.0
                    0.0 1.0]

    action       = [1.0; 0]

    jacobian     = [1.0 0.0
                    0.0 1.0]
    σ² = 1.0

    new_info = updateCRLB(crlb, action, jacobian, σ²)
    @test new_info ≈ [0.5 0.0; 0.0 1.0]

    # Scale Second Eigenvalue, same setup
    jacobian     = [1.0 0.0
                    0.0 0.5]
    new_info = updateCRLB(crlb, action, jacobian, σ²)
    @test new_info ≈ [0.5 0.0; 0.0 0.25]
end

@testset "optimalAction_NearestNeighbor" begin

    # Test without value function impact
    crlb         = [1.0 0.0;
                    0.0 5.0]

    actionSpace  = [[1.0, 0.0], [0.0, 1.0]]

    jacobian     = [1.0 0.0
                    0.0 1.0]
    σ² = 1.0

    samples = zeros(2,2,2)
    samples[:,:,1] = [0.5 0.0;
                      0.0 5.0 ]

    samples[:,:,2] = [1.0 0.0;
                      0.0 0.8 ]
    values =  [5.5,1.8]

    action = optimalAction_NearestNeighbor(crlb, actionSpace, samples, values, jacobian, σ²)
    @test action == 2

end

@testset "valueUpdate_NearestNeighbor" begin

end
