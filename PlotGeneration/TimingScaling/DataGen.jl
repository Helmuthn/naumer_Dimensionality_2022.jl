using naumer_ICML_2022
using Random: MersenneTwister
using LinearAlgebra: I, tr, norm, svd
using CSV
using BenchmarkTools

dim_lst = 1:50
const μ = 1.0
const σ² = 1.0
const τ = 0.05
const stepcount = 1000
const T = 10

const stepsize = 1e4


timing = zeros(length(dim_lst))

function FunctionToTime(dim,μ,σ²,τ,stepcount,T,stepsize)
    # Initialize system
    system = VanDerPolSystem_expanded(μ,dim)
    crlb = I(dimension(system))

    state = ones(dimension(system))

    # Run 1D approximate optimal sampling procedure
    for j in 1:stepcount

        _jacobian = flowJacobian(state, T, system)
        ~, ~, V = svd(_jacobian)
        limitvector = Array(V[:,1])

        # Decide action
        action = limitvector + .01 * randn(length(limitvector))
        action ./= norm(action) 
        for step in 1:100
            action .= actiongradientstep_1DApprox(action, limitvector, crlb, σ², stepsize*dimension(system) / step^(2.0/3.0))
        end
        
        # Update System
        jacobian = flowJacobian(state, τ, system)
        crlb = updateCRLB(crlb, action, jacobian, σ²)
        state = flow(state, τ, system)
    end
    return tr(crlb)
end

FunctionToTime(dim_lst[1],μ,σ²,τ,stepcount,T,stepsize)

for i in 1:length(dim_lst)
    global timing
    timing[i] = @belapsed FunctionToTime($(dim_lst[i]),μ,σ²,τ,stepcount,T,stepsize) samples=10
end


CSV.write("DimensionTiming.csv", ( timing = timing, junk = rand(length(timing))))
