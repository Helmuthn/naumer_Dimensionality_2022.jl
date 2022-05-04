using naumer_Dimensionality_2022, Test
using LinearAlgebra: eigvals


@testset "randomPSD" begin
    random_matrix = randomPSD(3, 1)

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

@testset "samplePSD" begin
    dataset = samplePSD(2,3,1)

    # Because this just call randomPSD, just check dimensionality
    @test size(dataset) == (3,3,2)
end

@testset "sampleStateSpace" begin
    # TODO
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

@testset "buildNearestNeighbor" begin
    # 1D test
    samples = -1:0.1:1
    values = buildNearestNeighbor(sin, samples)
    @test values == sin.(samples) 

    # 2D test
    samples = [1 2;
               2 1]
    f(x) = x[2] - x[1]
    values = buildNearestNeighbor(f,samples)
    @test values == [1,-1]
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
    crlb         = [1.0 0.0;
                    0.0 5.0]

    actionSpace  = [[1.0, 0.0], [0.0, 1.0]]

    jacobian     = [1.0 0.0
                    0.0 1.0]
    σ² = 1.0
    γ = 0.5

    samples = zeros(2,2,2)
    samples[:,:,1] = [0.5 0.0;
                      0.0 5.0 ]

    samples[:,:,2] = [1.0 0.0;
                      0.0 0.8 ]

    values =  [5.5,1.8]

    update = valueUpdate_NearestNeighbor(   crlb,
                                            γ,
                                            jacobian,
                                            actionSpace,
                                            samples, values,
                                            σ²)

    truth = 6.0 + 0.9
    @test update ≈ truth
end

@testset "valueIterate_NearestNeighbor" begin
    actionSpace  = [[1.0, 0.0], [0.0, 1.0]]

    jacobian     = [1.0 0.0
                    0.0 1.0]
    σ² = 1.0
    γ = 0.5

    samples = zeros(2,2,1)
    samples[:,:,1] = [1.0 0.0;
                      0.0 1.0 ]


    values =  [2.0]

    new_values = valueIterate_NearestNeighbor(γ, jacobian, actionSpace, samples, values, σ²)

    @test new_values[1] ≈ 3.0
end

@testset "valueIterate_NearestNeighbor_precompute!" begin
    # Basic test to ensure there are no crashes
    actionSpace = [[1.0,0], [0,1.0]] 
    system = LinearSystem([0.5 0; 0 0.5])
    γ = 0.8
    τ = 0.25
    λ = 1
    psdSampleCount = 5
    trajectorySampleCount = 3
    timestepSampleCount = 2
    σ² = 1
    max_iterations = 2

    ValueFunctionApproximation_NearestNeighbor_precompute(system, 
                                                          τ,
                                                          γ,
                                                          actionSpace,
                                                          λ,
                                                          psdSampleCount,
                                                          trajectorySampleCount,
                                                          timestepSampleCount,
                                                          σ²,
                                                          max_iterations)
end
