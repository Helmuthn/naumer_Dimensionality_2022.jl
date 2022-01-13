using CairoMakie
using naumer_ICML_2022
using LinearAlgebra: tr, svd, dot, det, Diagonal, pinv, eigen
using CSV

################################################
########### Plot Parameters ####################
################################################

max_steps = 1000
state = 2*randn(2)
state_cp = copy(state)
crlb = [4.0 0; 0 4.0]


b = -1
a = 1
τ = 0.01
γ = 0.99
actionSpace = [[sin(θ), cos(θ)] for θ in 0:.1:π]

λ = .5
psdSampleCount = [100]
trajectorySampleCount = 20
timestepSampleCount = 1
σ² = 1
max_iterations = 1000
d_max = 2

#################################################
################# Data Generation ###############
#################################################

system = StaticSystem(2)

optimal_sampling_trace = zeros(max_steps)
random_sampling_trace  = zeros(max_steps)

optimal_sampling_trace[1] = 8 
random_sampling_trace[1]  = 8

@info("Approximating Value Function")

values, psdSamples, stateSamples = ValueFunctionApproximation_LocalAverage_precompute( system,
                                                                            τ,
                                                                            γ,
                                                                            actionSpace,
                                                                            λ,
                                                                            psdSampleCount[1],
                                                                            trajectorySampleCount,
                                                                            timestepSampleCount,
                                                                            σ²,
                                                                            max_iterations,
                                                                            d_max)

@info("Solving Optimal Sampling Problem")

NN_Policy_tuple = (system, σ², τ, values, psdSamples, stateSamples, actionSpace, d_max)

for i in 2:max_steps
    global state
    global crlb
    global NN_Policy_tuple
    action, index, state, crlb = LocalAverage_OptimalPolicy(    state,
                                                                crlb,
                                                                NN_Policy_tuple...)
    ~, S, ~ = svd(crlb)
    optimal_sampling_trace[i] = sum(S)
end

crlb = [4.0 0; 0 4.0]
state = state_cp
                                             
for i in 2:max_steps
    global state
    global crlb
    action = actionSpace[rand(1:length(actionSpace))]

    jacobian = flowJacobian(state, τ, system)
    crlb = updateCRLB(crlb, action, jacobian, σ²)

    state = flow(state, τ, system)
    ~, S, ~ = svd(crlb)
    random_sampling_trace[i] = sum(S)
end


#################################################
################ Save Data CSV ##################
#################################################

CSV.write("StaticSampling.csv", ( random_sampling_trace = random_sampling_trace, 
                                  optimal_sampling_trace = optimal_sampling_trace,
                                  samples = Array(1:max_steps)))

