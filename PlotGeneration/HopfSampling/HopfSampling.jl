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

λ = 1
psdSampleCount = [30]
trajectorySampleCount = 30
timestepSampleCount = 4
σ² = 1
max_iterations = 100
d_max = 2

T = 30

#################################################
################# Data Generation ###############
#################################################

system = HopfSystem(a, b)

optimal_sampling_trace     = zeros(max_steps)
random_sampling_trace      = zeros(max_steps)
approx_sampling_trace      = zeros(max_steps)
approx_sampling_trace_ekf  = zeros(max_steps)
optimal_sampling_trace_ekf = zeros(max_steps)

optimal_sampling_trace[1] = 8 
random_sampling_trace[1]  = 8
approx_sampling_trace[1] = 8
approx_sampling_trace_ekf[1] = 8
optimal_sampling_trace_ekf[1] = 8


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

@info("Solving Optimal Sampling Dynamic Programming Problem")

NN_Policy_tuple = (system, σ², τ, values, psdSamples, stateSamples, actionSpace, d_max)

for i in 2:max_steps
    global state
    global crlb
    global NN_Policy_tuple

    # Choose Action and Update System
    action, index, state, crlb = LocalAverage_OptimalPolicy(    state,
                                                                crlb,
                                                                NN_Policy_tuple...)



    # Record cost
    optimal_sampling_trace[i] = tr(crlb)
end


@info("Solving Optimal Sampling Dynamic Programming Problem - EKF")

NN_Policy_tuple = (system, σ², τ, values, psdSamples, stateSamples, actionSpace, d_max)
beliefState = copy(state_cp) + 2*randn(2)
beliefCRLB = [4.0 0; 0 4.0]
crlb = [4.0 0; 0 4.0]

for i in 2:max_steps
    global state
    global crlb
    global beliefState
    global beliefCRLB

    # Choose Action
    action, ~, ~, ~ = LocalAverage_OptimalPolicy(   state,
                                                    crlb,
                                                    NN_Policy_tuple...)

    # Update True System
    jacobian = flowJacobian(state, τ, system)
    crlb = updateCRLB(crlb, action, jacobian, σ²)
    state = flow(state, τ, system)

    # Update Belief System
    observation = dot(action, state) + sqrt(σ²) * randn()
    jacobian = flowJacobian(beliefState, τ, system)
    beliefCRLB = updateCRLB(beliefCRLB, action, jacobian, σ²)
    beliefState = flow(beliefState, τ, system)
    beliefState = stateupdate_EKF(beliefState, beliefCRLB, observation, action, σ²)

    # Record cost
    optimal_sampling_trace_ekf[i] = tr(crlb)
end

@info("Solving Optimal Sampling 1D Approximation Problem")

crlb = [4.0 0; 0 4.0]
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

crlb = [4.0 0; 0 4.0]
state = copy(state_cp)
beliefState = state + 2*randn(2)
beliefCRLB = [4.0 0; 0 4.0]

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
    observation = dot(action, state) + sqrt(σ²) * randn()
    jacobian = flowJacobian(beliefState, τ, system)
    beliefCRLB = updateCRLB(beliefCRLB, action, jacobian, σ²)
    beliefState = flow(beliefState, τ, system)
    beliefState = stateupdate_EKF(beliefState, beliefCRLB, observation, action, σ²)

    # Record Cost
    approx_sampling_trace_ekf[i] = tr(crlb)
end

@info("Generating Random Sampling Trajectory")

crlb = [4.0 0; 0 4.0]
state = copy(state_cp)
                                             
for i in 2:max_steps
    global state
    global crlb

    # Choose Action
    action = actionSpace[rand(1:length(actionSpace))]

    # Update System
    jacobian = flowJacobian(state, τ, system)
    crlb = updateCRLB(crlb, action, jacobian, σ²)
    state = flow(state, τ, system)

    # Record Cost
    random_sampling_trace[i] = tr(crlb)
end


#################################################
################ Save Data CSV ##################
#################################################

CSV.write("HopfSampling.csv", ( random_sampling_trace = random_sampling_trace, 
                                optimal_sampling_trace = optimal_sampling_trace,
                                optimal_sampling_trace_ekf = optimal_sampling_trace_ekf,
                                approx_sampling_trace = approx_sampling_trace,
                                approx_sampling_trace_ekf = approx_sampling_trace_ekf,
                                samples = Array(1:max_steps)))

