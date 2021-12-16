# Dynamical system definitions used in the manuscript

############################
####### Basic System #######
############################

abstract type AbstractSystem{T} end;

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


