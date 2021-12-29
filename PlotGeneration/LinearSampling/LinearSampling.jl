using CairoMakie
using naumer_ICML_2022
using LinearAlgebra: tr, svd, dot


#################################################
################# Data Generation ###############
#################################################

dynamics = [-10.0 0.0; 0.0 -0.1]
system = LinearSystem(dynamics)
τ = 0.01
γ = 0.95
actionSpace = [[sin(θ), cos(θ)] for θ in 0:.05:π]

λ = 0.01
psdSampleCount = 10000
trajectorySampleCount = 1
timestepSampleCount = 1
σ² = 1
max_iterations = 10000

values, psdSamples, stateSamples = ValueFunctionApproximation_NearestNeighbor_precompute(system,
                                                                                        τ,
                                                                                        γ,
                                                                                        actionSpace,
                                                                                        λ,
                                                                                        psdSampleCount,
                                                                                        trajectorySampleCount,
                                                                                        timestepSampleCount,
                                                                                        σ²,
                                                                                        max_iterations)

NN_Policy_tuple = (system, σ², τ, values, psdSamples, stateSamples, actionSpace)

max_steps = 1000
state = 2*randn(2)
state_cp = copy(state)
crlb = [4.0 0; 0 4.0]

optimal_sampling_trace = zeros(max_steps)
random_sampling_trace = zeros(max_steps)

geometric_amplification = [1.0/(1.0 - γ * exp(τ*dynamics[1,1])), 1.0/(1.0 - γ * exp(τ*dynamics[2,2]))]

optimal_sampling_trace[1] = dot([4,4],geometric_amplification)
random_sampling_trace[1] = dot([4,4],geometric_amplification)

for i in 2:max_steps
    global state
    global crlb
    global NN_Policy_tuple
    action, index, state, crlb = NearestNeighbor_OptimalPolicy( state,
                                                                 crlb,
                                                                 NN_Policy_tuple...)
    ~, S, ~ = svd(crlb)
    optimal_sampling_trace[i] = dot(S, geometric_amplification)
end


crlb = [4.0 0; 0 4.0]
state = state_cp
                                              
for i in 2:max_steps
    global state
    global crlb
    action = actionSpace[rand(1:length(actionSpace))]

    jacobian = flowJacobian(state, τ, system)
    crlb = inv(inv(crlb) + (action * action' / σ²))
    crlb = jacobian * crlb * jacobian'

    state = flow(state, τ, system)
    ~, S, ~ = svd(crlb)
    random_sampling_trace[i] = dot(S, geometric_amplification)
end



#################################################
################# Plot Generation ###############
#################################################

noto_sans = "./resources/NotoSans-Regular.ttf"

tickfontsize    = 28
labelfontsize   = 32

f = Figure(resolution=(600,600))

ax = Axis(  f[1,1],
            xticklabelsize=tickfontsize, 
            yticklabelsize=tickfontsize, 
            yticklabelpad=2,
            xlabel="Sample", ylabel = "log10(tr(CRLB))",
            xlabelsize=labelfontsize,
            ylabelsize=labelfontsize,
            yscale = log10,
            yminorticksvisible = true,
            yminorgridvisible = true,
            yminorticks = IntervalsBetween(10),
            xminorticksvisible = true,
            xminorgridvisible = true,
            xminorticks = IntervalsBetween(10))

lines!(ax, 1:max_steps, optimal_sampling_trace, color=:black)
lines!(ax, 1:max_steps, random_sampling_trace, color=:blue)

ylims!(ax,(1e-3,1))
xlims!(ax,(0,1000))

save("LinearSampling.pdf",f)

run(`mailme hnaumer2@illinois.edu "Finished Running"`)
