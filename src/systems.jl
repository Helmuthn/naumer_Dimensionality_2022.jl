# Dynamical system definitions used in the manuscript

############################
####### Basic System #######
############################

abstract type AbstractSystem{T} end;

#############################
###### Linear Systems #######
#############################

# Define a struct containing the relevant details
struct LinearSystem{T} <: AbstractSystem{T}
    dynamics::Matrix{T}
end

function differential(system::LinearSystem, x::Vector{T})
    return system.dynamics * x
end

function flow(x::Vector{T}, τ, system::LinearSystem)
    return exp(τ * system.dynamics) * x
end

function jacobian(x::Vector{T}, τ, system::Linearsystem)
    return exp(τ * system.dynamics)
end


