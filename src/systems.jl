# Dynamical system definitions used in the manuscript
using DifferentialEquations: Tsit5, solve, ODEProblem, remake
using ForwardDiff

############################
####### Basic System #######
############################
export differential, flow, flowJacobian, dimension

abstract type AbstractSystem{T} end;

"""
    differential(system::AbstractSystem, x::Vector)

Computes the time derivative of a given continuous-time dynamical
system evaluated at a given point.

### Arguments
 - `system` - Continuous-Time Dynamical System
 - `x`      - Point to Evaluate differential

### Returns
The derivative of the trajectory of the system with respect to time
evaluated at a fixed point `x`.
"""
function differential end;

"""
    flow(system::AbstractSystem, τ, x::Vector)

Advances the state of a dynamical `system` from state `x` by time `τ`.

### Arguments
 - `system` - Continuous-Time Dynamical System
 - `τ`      - Advancement time
 - `x`      - Initial state

### Returns
The state advanced by `τ` units of time
"""
function flow end;

"""
    flowJacobian(system::AbstractSystem, τ, x::Vector)

Computes the Jacobian of the flow of a `system` for `τ` units of time 
around an initial condition `x`.

### Arguments
 - `system` - Continuous-Time Dynamical System
 - `τ`      - Advancement time
 - `x`      - Initial State

### Returns
Jacobian matrix of the flow representing the derivative with respect to the
initial state `x`.
"""
function flowJacobian end;

"""
    dimension(system::AbstractSystem)

Returns the dimensionality of the system.
"""
function dimension end;


#############################
###### Linear Systems #######
#############################
export LinearSystem

"""
    LinearSystem{T} <: AbstractSystem{T}

Defines a linear system dx/dt = Ax

### Fields
 - `dynamics::Matrix{T}` - Matrix defining dynamics
"""
struct LinearSystem{T} <: AbstractSystem{T}
    dynamics::Matrix{T}
end

function differential(system::LinearSystem, x::Vector)
    return system.dynamics * x
end

function flow(x::Vector, τ, system::LinearSystem)
    return exp(τ * system.dynamics) * x
end

function flowJacobian(x::Vector, τ, system::LinearSystem)
    return exp(τ * system.dynamics)
end

function dimension(system::LinearSystem)
    return size(system.dynamics)[1]
end


#############################
######## Van Der Pol ########
#############################
export VanDerPolSystem

struct VanDerPolSystem{T} <: AbstractSystem{T}
    μ::T
end

function dimension(system::VanDerPolSystem)
    return 2
end

function differential(system::VanDerPolSystem, x::Vector)
    out = zeros(2)
    out[1] = x[2]
    out[2] = system.μ * (1 - x[1]^2) * x[2] - x[1]
    return out
end

function vanderpolDynamics!(du, u, p, t)
    x, y = u
    μ = p
    du[1] = y
    du[2] = μ * (1 - x^2) * y - x
end

function flow(x::Vector, τ, system::VanDerPolSystem)
    problem = ODEProblem(vanderpolDynamics!, x, (0.0, τ), system.μ)
    sol = solve(problem, Tsit5(), reltol=1e-8, save_everystep=false)
    return sol[end]
end

function flowJacobian(x::Vector, τ, system::VanDerPolSystem)
    problem = ODEProblem(hopfDynamics!, x, (0.0, τ), system.μ)

    function solvesystem(init)
        prob = remake(problem, u0=init)
        sol = solve(prob, Tsit5(), reltol=1e-8, save_everystep=false)
        return sol[end]
    end

    return ForwardDiff.jacobian(solvesystem, x)
end

#############################
######## Hopf System ########
#############################
export HopfSystem

struct HopfSystem{T} <: AbstractSystem{T}
    λ::T
    b::T
end

function dimension(system::HopfSystem)
    return 2
end

function differential(system::HopfSystem, x::Vector)
    out = zeros(2)
    nonlinear = system.λ + system.b * (x[1]^2 + x[2]^2)
    out[1] = x[1] * nonlinear - x[2]
    out[2] = x[2] * nonlinear + x[1]
    return out
end

function hopfDynamics!(du, u, p, t)
    x, y = u
    λ, b = p 
    nonlinear = λ + b * (x^2 + y^2)
    du[1] = x * nonlinear - y
    du[2] = y * nonlinear + x
end

function flow(x::Vector, τ, system::HopfSystem)
    problem = ODEProblem(hopfDynamics!, x, (0.0, τ), (system.λ, system.b))
    sol = solve(problem, Tsit5(), reltol=1e-8, save_everystep=false)
    return sol[end]
end

function flowJacobian(x::Vector, τ, system::HopfSystem)
    problem = ODEProblem(hopfDynamics!, x, (0.0, τ), (system.λ, system.b))

    function solvesystem(init)
        prob = remake(problem, u0=init)
        sol = solve(prob, Tsit5(), reltol=1e-8, save_everystep=false)
        return sol[end]
    end

    return ForwardDiff.jacobian(solvesystem, x)
end

#############################
###### Lorenz System ########
#############################
export LorenzSystem

struct LorenzSystem{T} <: AbstractSystem{T}
    σ::T
    ρ::T
    β::T
end

function dimension(system::LorenzSystem)
    return 3
end

function differential(system::LorenzSystem, x::Vector)
    out = zeros(3)
    out[1] = system.σ * (x[2] - x[1])
    out[2] = x[1] * (system.ρ - x[3]) - x[2]
    out[3] = x[1]*x[2] - system.β * x[3]
    return out
end

function lorenzDynamics!(du, u, p, t)
    x, y, z = u
    σ, ρ, β = p
    du[1] = σ * (y - x)
    du[2] = x * (ρ - z) - y
    du[3] = x * y - β * z
end

function flow(x::Vector, τ, system::LorenzSystem)
    problem = ODEProblem(lorenzDynamics!, x, (0.0, τ), (system.σ, system.ρ, system.β))
    sol = solve(problem, Tsit5(), reltol=1e-8, save_everystep=false)
    return sol[end]
end

function flowJacobian(x::Vector, τ, system::LorenzSystem)
    problem = ODEProblem(lorenzDynamics!, x, (0.0, τ), (system.σ, system.ρ, system.β))

    function solvesystem(init)
        prob = remake(problem, u0=init)
        sol = solve(prob, Tsit5(), reltol=1e-8, save_everystep=false)
        return sol[end]
    end

    return ForwardDiff.jacobian(solvesystem, x)
end

