# Dynamical system definitions used in the manuscript

############################
####### Basic System #######
############################
export differential, flow, flowJacobian

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


#############################
######## Van Der Pol ########
#############################
export VanDerPolSystem

struct VanDerPolSystem{T} <: AbstractSystem{T}
    μ::T
end

function differential(system::VanDerPolSystem, x::Vector)
    out = zeros(2)
    out[1] = x[2]
    out[2] = system.μ * (1 - x[1]^2) * x[2] - x[1]
end

#function flow(x::vector, τ, system::VanDerPolSystem)
#
#end


