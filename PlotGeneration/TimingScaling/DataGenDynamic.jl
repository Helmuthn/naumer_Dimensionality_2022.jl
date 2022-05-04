using naumer_Dimensionality_2022
using LinearAlgebra: I, tr, norm, svd
using CSV
using BenchmarkTools

sample_lst = 1:40

const μ = 1.0
const σ² = 1.0
const τ = 0.05
const stepcount = 1000


const actionSpace = [[sin(θ), cos(θ)] for θ in 0:.1:π]
const γ = 0.99


timing = zeros(length(sample_lst))

function FunctionToTime(samples, μ, σ², τ, stepcount, actionSpace, γ)
    # Initialize system
    system = VanDerPolSystem(μ)
    crlb = I(dimension(system))
    optimal_sampling_trace = zeros(stepcount)
    state = randn(2)

    
values, psdSamples, stateSamples = ValueFunctionApproximation_LocalAverage_precompute(  system,
                                                                                        τ,
                                                                                        γ,
                                                                                        actionSpace,
                                                                                        1,
                                                                                        samples,
                                                                                        samples,
                                                                                        1,
                                                                                        σ²,
                                                                                        100,
                                                                                        1)



    NN_Policy_tuple = (system, σ², τ, values, psdSamples, stateSamples, actionSpace, 1)
    for i in 2:stepcount


    # Choose Action and Update System
    action, index, state, crlb = LocalAverage_OptimalPolicy(    state,
                                                                crlb,
                                                                NN_Policy_tuple...)



        # Record cost
        optimal_sampling_trace[i] = tr(crlb)
    end
    return tr(optimal_sampling_trace[end])
end

FunctionToTime(sample_lst[1],μ, σ², τ, stepcount, actionSpace, γ)

for i in 1:length(sample_lst)
    @info sample_lst[i]
    global timing
    timing[i] = @belapsed FunctionToTime($(sample_lst[i]), μ, σ², τ, stepcount, actionSpace, γ)
    @info timing[i]
end


CSV.write("DimensionTimingDynamic.csv", ( timing = timing, samples = sample_lst.^2))
