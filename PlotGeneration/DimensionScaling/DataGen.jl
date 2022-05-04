using naumer_Dimensionality_2022
using Random: MersenneTwister
using LinearAlgebra: I, tr, norm, svd
using CSV

dim_lst = 1:50
μ = 1.0
σ² = 1.0
τ = 0.05
stepcount = 1000
T = 10

stepsize = 1e4


crlb_trace = zeros(length(dim_lst))

Threads.@threads for i in 1:length(dim_lst)
    global crlb_trace

    # Initialize system
    system = VanDerPolSystem_expanded(μ,dim_lst[i])
    crlb = I(dimension(system))

    actionseed = MersenneTwister(1234)

    state = ones(dimension(system))

    # Run 1D approximate optimal sampling procedure
    for j in 1:stepcount

        _jacobian = flowJacobian(state, T, system)
        ~, ~, V = svd(_jacobian)
        limitvector = Array(V[:,1])

        # Decide action
        action = limitvector + .01 * randn(length(limitvector))
        action ./= norm(action) 
        for step in 1:1000
            action .= actiongradientstep_1DApprox(action, limitvector, crlb, σ², stepsize*dimension(system) / step^(2.0/3.0))
        end
        
        # Update System
        jacobian = flowJacobian(state, τ, system)
        crlb = updateCRLB(crlb, action, jacobian, σ²)
        state = flow(state, τ, system)
    end

    # Now record final crlb trace
    crlb_trace[i] = tr(crlb)
end
    

# Now random smapling for comparison
random_crlb_trace = zeros(length(crlb_trace))
for i in 1:length(dim_lst)
    @info(i)
    global random_crlb_trace

    # Initialize system
    system = VanDerPolSystem_expanded(μ,dim_lst[i])
    SystemMat = -1*I(dim_lst[i] + 1)
    SystemMat[1,1] = 0
    system = LinearSystem(SystemMat)
    crlb = I(dimension(system)) 

    state = 2*ones(dimension(system))

    # Run 1D approximate optimal sampling procedure
    for j in 1:stepcount
        # Decide action
        action = randn(dimension(system))
        action ./= norm(action)
        
        # Update System
        jacobian = flowJacobian(state, τ, system)
        crlb = updateCRLB(crlb, action, jacobian, σ²)
        state = flow(state, τ, system)
    end

    # Now record final crlb trace
    random_crlb_trace[i] = tr(crlb)
end


println(random_crlb_trace ./ crlb_trace)

CSV.write("DimensionScaling.csv", ( crlb_trace = crlb_trace,
                                    random_crlb_trace = random_crlb_trace))
