using CairoMakie
using naumer_ICML_2022
using LinearAlgebra: tr, svd, dot
using CSV

################################################
########### Plot Parameters ####################
################################################

max_steps = 1000
state = 2*randn(3)
state_cp = copy(state)
crlb = [4.0 0 0; 0 4.0 0;0 0 4.0]

τ = 0.01
γ = 0.9
actionSpace = [randn(3) for i in 1:20]
for i in 1:20
    actionSpace[i] /= sqrt(sum(abs2,actionSpace[i]))
end

λ = 0.05
psdSampleCount = [50]
trajectorySampleCount = 100
timestepSampleCount = 1
σ² = 1
max_iterations = 10000
d_max = 2

#################################################
################# Data Generation ###############
#################################################

system = LorenzSystem(10.0, 28.0, 8.0/3.0)


optimal_sampling_trace  = zeros(max_steps)
random_sampling_trace   = zeros(max_steps)

optimal_sampling_trace[1]    = 8
random_sampling_trace[1]     = 8

values, psdSamples, stateSamples = ValueFunctionApproximation_LocalAverage_precompute(system,
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



NN_Policy_tuple = (system, σ², τ, values, psdSamples, stateSamples, actionSpace, d_max)

for i in 2:max_steps
    global state
    global crlb
    global NN_Policy_tuple
    action, index, state, crlb = LocalAverage_OptimalPolicy(  state,
                                                                 crlb,
                                                                 NN_Policy_tuple...)
    optimal_sampling_trace[i] = tr(crlb)
end

crlb = [4.0 0 0; 0 4.0 0; 0.0 0.0 4.0]
state = state_cp
                                             
for i in 2:max_steps
    global state
    global crlb
    action = actionSpace[rand(1:length(actionSpace))]

    jacobian = flowJacobian(state, τ, system)
    crlb = updateCRLB(crlb, action, jacobian, σ²)

    state = flow(state, τ, system)
    random_sampling_trace[i] = tr(crlb)
end


#################################################
################ Save Data CSV ##################
#################################################

CSV.write("LorenzSampling.csv",(random_sampling_trace = random_sampling_trace, 
                                optimal_sampling_trace = optimal_sampling_trace,
                                samples = Array(1:max_steps)))

