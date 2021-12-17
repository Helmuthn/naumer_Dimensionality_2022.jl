# Dynamical system definitions used in the manuscript

############################
####### Basic System #######
############################

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
function differential(system::AbstractSystem, x::Vector)
    error("Function not implemented for $(typeof(system))")
end;

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
function flow(system::AbstractSystem, τ, x::Vector)
    error("Function not implemented for $(typeof(system))")
end;

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
function flowJacobian(system::AbstractSystem, τ, x::Vector)
    error("Function not implemented for $(typeof(system))")
end;

export differential, flow, flowJacobian

#############################
###### Linear Systems #######
#############################
export LinearSystem

# Define a struct containing the relevant details
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


