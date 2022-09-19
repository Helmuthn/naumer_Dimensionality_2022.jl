using CairoMakie
using DiffEqSensitivity
using Zygote
using ForwardDiff
using LinearAlgebra
using DifferentialEquations

"""
  vanderpol!(du,u,μ,t)

Van der Pol oscillator dynamics for DifferentialEquations.jl
inline
"""
function vanderpol!(du,u,μ,t)
    x,y = u
    du[1] = μ * (x - x^3/3 - y)
    du[2] = x/μ
end

"""
  vanderpol(u,μ,t)

Van der Pol oscillator dynamics for DifferentialEquations.jl 
not inline
"""
function vanderpol(u,μ,t)
    du = zeros(2)
    vanderpol!(du,u,μ,t)
    return du
end


"""
  phasemap(u, positions)

Applies to Van der Pol oscillator.

Assigns a point `u` in the statespace to a position on the 
limit cycle of the oscillator. Given a list of `positions`
spaced equally in time along the limit cycle, this function
simulates the system from initial condition `u` then determines
the closest element in `positions`

Returns: The index of the closest element.
"""
function phasemap(u, positions)
    μ = 1.0
    tspan = (0.0,40.0)

    prob = ODEProblem(vanderpol!,u,tspan,μ)

    sol = solve(prob)
    final = sol.u[end]
    ~, out = findmin(sum(abs2,positions .- final,dims=1))
    return out[2]
end


"""
  simystem(u)

Compute the value of the Van der Pol oscillator initialized at
`u` 20 units of time in the future.
"""
function simsystem(u)
    μ = 1.0
    tspan = (0.0,20.0)

    prob = ODEProblem(vanderpol!,u,tspan,μ)

    sol = solve(prob,reltol=1e-6,abstol=1e-6,sensealg=QuadratureAdjoint())
    return sol.u[end]
end

"""
  informativeVec(u)

Find the non-zero eigenvalue of dφ₂₀ in the Van Der Pol
oscillator around state space point `u`. 
"""
function informativeVec(u)
    dx = ForwardDiff.jacobian(simsystem,u)
    ~, Σ, Vt = svd(dx)
    # Force counterclockwise
    test_vec = [u[2], -u[1]]
    if dot(test_vec,Vt[1,:]) > 0
        return Vt[1,:]
    else
        return -Vt[1,:]
    end
end


