# Dynamical system definitions used in the manuscript
using LinearAlgebra: I
using DifferentialEquations: Tsit5, solve, ODEProblem, remake
using ForwardDiff
using SparseArrays: spzeros

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
    flow(x::Vector, τ, system::AbstractSystem)

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
    flowJacobian(x::Vector, τ, system::AbstractSystem)

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
###### Static Systems #######
#############################
export StaticSystem

"""
    StaticSystem <: AbstractSystem

Defines a system dx/dt = 0
"""
struct StaticSystem <: AbstractSystem{Float64}
    dim::Int64
end

function differential(system::StaticSystem, x::Vector)
    return zeros(length(x))
end

function flow(x::Vector, τ, system::StaticSystem)
    return copy(x)
end

function flowJacobian(x::Vector, τ, system::StaticSystem)
    return I(system.dim)
end

function dimension(system::StaticSystem)
    return system.dim
end

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
    dynamics::AbstractMatrix{T}
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

"""
    VanDerPolSystem{T} <: AbstractSystem{T}
"""
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
    problem = ODEProblem(vanderpolDynamics!, x, (0.0, τ), system.μ)

    function solvesystem(init)
        prob = remake(problem, u0=init)
        sol = solve(prob, Tsit5(), reltol=1e-8, save_everystep=false)
        return sol[end]
    end

    return ForwardDiff.jacobian(solvesystem, x)
end


#############################
######## Van Der Pol ########
#############################
export VanDerPolSystem_expanded

"""
    VanDerPolSystem_expanded

Represents dynamics in a product space where two axis
represent a Van Der Pol System, while the rest represent
stable linear systems.

Used to demonstrate scaling of dimensionality collapse benefit
by allowing the introduction of arbitrary numbers of
collapsing dimensions.

### Fields
 - `μ`               - Van Der Pol Parameter
 - `linearDimension` - Extra Collapsing Dimension Count
"""
struct VanDerPolSystem_expanded{T} <: AbstractSystem{T}
    μ::T
    linearDimension::Int
end

function dimension(system::VanDerPolSystem_expanded)
    return 2 + system.linearDimension
end

function differential(system::VanDerPolSystem_expanded, x::Vector)
    out = zeros(dimension(system))
    out[1] = x[2]
    out[2] = system.μ * (1 - x[1]^2) * x[2] - x[1]
    out[3:end] .= -x[3:end]
    return out
end

function flow(x::Vector, τ, system::VanDerPolSystem_expanded)
    problem = ODEProblem(vanderpolDynamics!, x[1:2], (0.0, τ), system.μ)
    sol = solve(problem, Tsit5(), reltol=1e-8, save_everystep=false)
    return [sol[end]; exp(-τ) .* x[3:end]]
end

function flowJacobian(x::Vector, τ, system::VanDerPolSystem_expanded)
    problem = ODEProblem(vanderpolDynamics!, x[1:2], (0.0, τ), system.μ)

    function solvesystem(init)
        prob = remake(problem, u0=init)
        sol = solve(prob, Tsit5(), reltol=1e-8, save_everystep=false)
        return sol[end]
    end

    out = zeros(dimension(system),dimension(system))
    out[1:2,1:2] = ForwardDiff.jacobian(solvesystem, x[1:2])
    for i in 3:dimension(system)
        out[i,i] = exp(-τ)
    end

    return out
end

#############################
######## Hopf System ########
#############################
export HopfSystem

"""
    HopfSystem{T} <: AbstractSystem{T}
"""
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

"""
    LorenzSystem{T} <: AbstractSystem{T}
"""
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
    sol = solve(problem, Tsit5(), reltol=1e-4, save_everystep=false)
    return sol[end]
end

function flowJacobian(x::Vector, τ, system::LorenzSystem)
    problem = ODEProblem(lorenzDynamics!, x, (0.0, τ), (system.σ, system.ρ, system.β))

    function solvesystem(init)
        prob = remake(problem, u0=init)
        sol = solve(prob, Tsit5(), reltol=1e-4, save_everystep=false)
        return sol[end]
    end

    return ForwardDiff.jacobian(solvesystem, x)
end

