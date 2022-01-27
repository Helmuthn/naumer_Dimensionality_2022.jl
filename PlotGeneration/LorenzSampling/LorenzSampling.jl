using CairoMakie
using naumer_ICML_2022
using LinearAlgebra: tr, svd, dot, I
using CSV
using Random: MersenneTwister

################################################
########### Plot Parameters ####################
################################################
mrng = MersenneTwister(1234)

max_steps = 1000
state     = 2*randn(mrng,3)#5*ones(3) 
state_cp  = copy(state)
crlb      = I(3) 
τ = 0.01
γ = 0.9
actionSpace = [randn(mrng,3) for i in 1:50]
for i in 1:50
    actionSpace[i] /= sqrt(sum(abs2,actionSpace[i]))
end

λ = 1
psdSampleCount = [50]
trajectorySampleCount = 100
timestepSampleCount = 1
σ² = 1
max_iterations = 1000
d_max = 2

T = 1

#################################################
################# Data Generation ###############
#################################################

system = LorenzSystem(10.0, 28.0, 8.0/3.0)


optimal_sampling_trace      = zeros(max_steps)
random_sampling_trace       = zeros(max_steps)
approx_sampling_trace       = zeros(max_steps)
optimal_sampling_trace_ekf  = zeros(max_steps)
approx_sampling_trace_ekf   = zeros(max_steps)

optimal_sampling_trace[1]   = tr(crlb)
random_sampling_trace[1]    = tr(crlb)
random_sampling_trace[1]    = tr(crlb)

@info("Approximating Value Function")

values, psdSamples, stateSamples = ValueFunctionApproximation_LocalAverage_precompute(  mrng,
                                                                                        system,
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



@info("Solving Dynamic Programming Optimal Sampling")
NN_Policy_tuple = (system, σ², τ, values, psdSamples, stateSamples, actionSpace, d_max)

for i in 2:max_steps
    global state
    global crlb
    global NN_Policy_tuple
    action, index, state, crlb = LocalAverage_OptimalPolicy(    state,
                                                                crlb,
                                                                NN_Policy_tuple...)

    optimal_sampling_trace[i] = tr(crlb)
end



@info("Solving Optimal Sampling Dynamic Programming Problem - EKF")

NN_Policy_tuple = (system, σ², τ, values, psdSamples, stateSamples, actionSpace, d_max)
beliefState = zeros(3) 
beliefCRLB  = I(3)
crlb        = I(3)
state       = copy(state_cp)

for i in 2:max_steps
    global state
    global crlb
    global beliefState
    global beliefCRLB

    # Choose Action
    action, ~, ~, ~ = LocalAverage_OptimalPolicy(   beliefState,
                                                    beliefCRLB,
                                                    NN_Policy_tuple...)

    # Update True System
    jacobian = flowJacobian(state, τ, system)
    crlb = updateCRLB(crlb, action, jacobian, σ²)
    state = flow(state, τ, system)

    # Update Belief System
    observation = dot(action, state) + sqrt(σ²) * randn(mrng)
    jacobian = flowJacobian(beliefState, τ, system)
    beliefCRLB = updateCRLB(beliefCRLB, action, jacobian, σ²)
    beliefState = flow(beliefState, τ, system)
    beliefState = stateupdate_EKF(beliefState, beliefCRLB, observation, action, σ²)

    # Record cost
    optimal_sampling_trace_ekf[i] = tr(crlb)
end

 
@info("Solving Optimal Sampling 1D Approximation Problem")

crlb = I(3) 
state = copy(state_cp)

for i in 2:max_steps
    global state
    global crlb
    global T
    global σ²
    global system

    # Decide action
    ~, action = optimalaction_1DApprox(actionSpace, system, state, crlb, σ², T)
    
    # Update System
    jacobian = flowJacobian(state, τ, system)
    crlb = updateCRLB(crlb, action, jacobian, σ²)
    state = flow(state, τ, system)

    # Record Cost
    approx_sampling_trace[i] = tr(crlb)
end


@info("Solving Optimal Sampling 1D Approximation Problem - EKF")

crlb = I(3)
state = copy(state_cp)
beliefState = zeros(3)
beliefCRLB = I(3)

for i in 2:max_steps
    global state
    global crlb
    global beliefState
    global beliefCRLB

    # Decide action
    ~, action = optimalaction_1DApprox(actionSpace, system, beliefState, beliefCRLB, σ², T)
    
    # Update True System
    jacobian = flowJacobian(state, τ, system)
    crlb = updateCRLB(crlb, action, jacobian, σ²)
    state = flow(state, τ, system)

    # Update Belief System
    observation = dot(action, state) + sqrt(σ²) * randn(mrng)
    jacobian = flowJacobian(beliefState, τ, system)
    beliefCRLB = updateCRLB(beliefCRLB, action, jacobian, σ²)
    beliefState = flow(beliefState, τ, system)
    beliefState = stateupdate_EKF(beliefState, beliefCRLB, observation, action, σ²)

    # Record Cost
    approx_sampling_trace_ekf[i] = tr(crlb)
end

@info("Solving Random Sampling")

crlb = I(3)
state = copy(state_cp)
                                             
for i in 2:max_steps
    global state
    global crlb
    action = actionSpace[rand(mrng, 1:length(actionSpace))]

    jacobian = flowJacobian(state, τ, system)
    crlb = updateCRLB(crlb, action, jacobian, σ²)

    state = flow(state, τ, system)
    random_sampling_trace[i] = tr(crlb)
end


#################################################
################ Save Data CSV ##################
#################################################

CSV.write("LorenzSampling.csv",(random_sampling_trace       = random_sampling_trace, 
                                optimal_sampling_trace      = optimal_sampling_trace,
                                optimal_sampling_trace_ekf  = optimal_sampling_trace_ekf,
                                approx_sampling_trace       = approx_sampling_trace,
                                approx_sampling_trace_ekf   = approx_sampling_trace_ekf,
                                samples = Array(1:max_steps)))

