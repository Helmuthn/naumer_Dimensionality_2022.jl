# The core functionality in the manuscript.
#
# The functions in this file could be applied to real data.

using LinearAlgebra: qr, Diagonal, tr
using Random: MersenneTwister, randexp, GLOBAL_RNG
using EllipsisNotation


#################################################
################ Random Sampling ################
#################################################

export randomPSD, samplePSD

"""
    randomPSD([rng = GLOBAL_RNG,] n, λ=1)

Generate a random `n×n` positive definite matrix with i.i.d.
exponentially distributed eigenvalues.

### Arguments
 - `n`: Dimensionality of the matrix
 - `λ`: Exponential distribution parameter

### Returns
A random `n×n` matrix
"""
function randomPSD(rng, n, λ)
    mat = randn(rng,n,n)
    ortho, ~ = qr(mat)
	return ortho' * Diagonal(randexp(rng,n)/λ) * ortho
end

randomPSD(n, λ) = randomPSD(GLOBAL_RNG, n, λ)


"""
    samplePSD([rng = GLOBAL_RNG,] K, n, λ=1)

Generate `k` random `n×n` positive definite matrices with
i.i.d. exponentially distributed eigenvalues.

### Arguments
 - `K` - Number of samples
 - `n` - Dimensionality of the matrix
 - `λ` - Exponential distribution parameter

### Returns
An `n×n×K` array representing `K` random matrices
"""
function samplePSD(rng, K, n, λ)
    out = zeros(n,n,K)
    for i in 1:K
        out[:,:,i] = randomPSD(rng, n, λ)
    end
    return out
end

samplePSD(K, n, λ) = samplePSD(GLOBAL_RNG, K, n, λ)


"""
    sampleStateSpace(system::AbstractSystem, trajectorySampleCount, timestepCount, burnin, τ)

Generates `stateSampleCount` random samples of the statespace.
For now, initializes according to a high-variance Gaussian distribution,
then steps the system forward in time to shape the density according to the system.

A small i.i.d. Gaussian random vector is added after each step to avoid degeneracy in value iteration.

### Arguments
 - `system`                 - Chosen dynamical system
 - `trajectorySampleCount`  - Number of trajectory samples
 - `timestepCount`          - Number of timesteps per trajectory
 - 'burnin`                 - Initial time advancement
 - `τ`                      - timestep size

### Returns
A 2D array representing `trajectorySampleCount * timestepCount` samples, where columns represent the state.
"""
function sampleStateSpace(system::AbstractSystem, trajectorySampleCount, timestepCount, burnin, τ)
    out = zeros(dimension(system), trajectorySampleCount * timestepCount)
    state = zeros(dimension(system))
    for i in 1:trajectorySampleCount
        if burnin == 0
            state .= 5*randn(dimension(system))
        else
            state = flow(5*randn(dimension(system)), burnin, system) 
        end
        for j in 1:timestepCount
            index = j + (i-1) * timestepCount
            out[:,i] .= state
            state .= flow(state, τ, system) + .25 * randn(dimension(system))
        end
    end
    return out
end


#################################################
################ Interpolation ##################
#################################################

export nearestNeighbor, buildNearestNeighbor

"""
    min_dist(v, D)

Finds the minimum distance vector in `D` from `v`.
Allows multidimensional array representation, 
assumes last axis of `D` indexes the vectors.

### Arguments
 - `v` - Vector being compared
 - `D` - Set of vectors

### Returns
    (out, out_ind)

Where `out` is the closest vector, and `out_ind` is the position.

### Notes
This function is not exported
"""
function min_dist(v,D)
	alloc = zeros(size(v))
	alloc .= v.- @view D[..,1]
	out = sum(abs2,alloc)
	out_ind = 1
    @inbounds for i in 2:size(D)[end]
		alloc .= v .- @view D[..,i]
		d = sum(abs2,alloc)
		if d < out
			out = d
			out_ind = i
		end
	end
	return out, out_ind
end


"""
    nearestNeighbor(target, samples, values)

Approximates the value at `target` with the value at
the closest known sample.

### Arguments
 - `target`  - Point being evaluated, domain of function
 - `samples` - Known sample locations, domain of function
 - `values`  - Values at known locations, range of function

### Returns
Approximate value at `target`
"""
function nearestNeighbor(target, samples, values)
	~, i = min_dist(target, samples)
	return values[i]
end


"""
    buildNearestNeighbor(f, samples)

Evaluates a function `f` at a set of chosen locations
to build the nearest neighbor approximation.
Glorified broadcast function.

The last axis of `samples` is assumed to index the sample locations.

### Arguments
 - `f`      - The function to be approximated
 - `samples`- Set of representation points

### Returns
An array of values at the given sample locations.
"""
function buildNearestNeighbor(f, samples)
    values = zeros(size(samples)[end])
    Threads.@threads for i in 1:size(samples)[end]
        values[i] = f(samples[..,i])
    end
    return values
end


#################################################
################# POMDP Tools ###################
#################################################

export updateCRLB, optimalAction_NearestNeighbor, valueUpdate_NearestNeighbor, valueIterate_NearestNeighbor, valueIterate_NearestNeighbor_precompute!

"""
    updateFisherInformation(    crlb, 
                                action, 
                                jacobian,
                                σ²)

Updates the Fisher information based on the Jacobian of the
flow and the current measurement under Gaussian noise.

### Arguments
 - `crlb`- Previous Fisher information matrix
 - `action`     - Measurement vector
 - `jacobian`   - Jacobian of flow for current timestep
 - `σ²`         - Measurement variance

### Returns
Updated Fisher Information
"""
function updateCRLB(crlb, action, jacobian, σ²)
	return jacobian * inv(inv(crlb) + action * action'/σ²) * jacobian'
end


"""
    optimalAction_NearestNeighbor(  crlb, actionSpace, 
                                    samples, values, 
                                    jacobian, σ²)

Computes the optimal action for a given state based on the current 
nearest neighbor approximation of the value function.
Assumes linear measurement under additive Gaussian noise.

### Arguments
 - `crlb`       - Current Cramér-Rao bound
 - `actionSpace`- Action Space (finite)
 - `samples`    - Sample locations in value function
 - `values`     - Sample evaluations of value function
 - `jacobian`   - Jacobian of flow for current timestep
 - `σ²`         - Measurement variance

### Returns
The action that minimizes the Cramér-Rao bound
"""
function optimalAction_NearestNeighbor( crlb, actionSpace, 
                                        samples, values, jacobian, σ²)

	vals = zeros(length(actionSpace))

	for i in 1:length(actionSpace)

		new_crlb  = updateCRLB(crlb, actionSpace[i], jacobian, σ²)
        vals[i]   = nearestNeighbor(new_crlb, samples, values)

	end
	return findmin(vals)[2]
end


""" 
    valueUpdate_NearestNeighbor(   crlb, 
                                   γ, 
                                   jacobian, 
                                   actionSpace, 
                                   samples, values)

Iterates the discounted Bellman equation for minimizing the
CRLB at a given current Fisher information under the
nearest neighbor value approximation for a given point.

Assumes linear measurement under additive Gaussian noise.

### Arguments
 - `crlb`       - Current Cramér-Rao bound
 - `γ`          - Discount factor
 - `jacobian`   - Jacobian of flow for current timestep
 - `actionSpace`- Action Space (finite)
 - `samples`    - Sample locations in value function
 - `values`     - Sample evaluations of value function
 - `σ²`         - Noise variance

### Returns
The updated value function approximation evaluated at `crlb`
"""
function valueUpdate_NearestNeighbor(   crlb, 
                                        γ, 
                                        jacobian, 
                                        actionSpace, 
                                        samples, values,
                                        σ²)

	update_vals = zeros(length(actionSpace))
	for i in 1:length(actionSpace)
		new_crlb = updateCRLB(crlb, actionSpace[i], jacobian, σ²)
		update_vals[i] = nearestNeighbor(new_crlb, samples, values)
	end
	return γ * minimum(update_vals) + tr(crlb)
end



"""
    valueIterate_NearestNeighbor(γ, jacobian, actionSpace, samples, values)

Iterates the discounted Bellman equation for minimizing the
CRLB at a given current Fisher information under the
nearest neighbor value approximation.

Assumes linear measurement under additive Gaussian noise.

### Arguments
 - `γ`          - Discount factor
 - `jacobian`   - Jacobian of flow for current timestep
 - `actionSpace`- Action Space (finite)
 - `samples`    - Sample locations in value function
 - `values`     - Sample evaluations of value function
 - `σ²`         - Measurement variance

### Returns
The updated value function approximation values for all points in `samples`.

### Notes
This function is multithreaded, remember to give Julia multiple threads when launching with
`julia -t NTHREADS`, where `NTHREADS` is the desired number of threads.
"""
function valueIterate_NearestNeighbor(γ, jacobian, actionSpace, samples, values, σ²)
	new_out = zeros(length(values))
	Threads.@threads for i in 1:length(values)
		new_out[i] = valueUpdate_NearestNeighbor(  view(samples,:,:,i), 
                                                           γ, 
                                                           jacobian, 
                                                           actionSpace, 
                                                           samples, values,
                                                           σ²)

	end
	return new_out
end

"""
    valueIterate_NearestNeighbor_precompute!(out, values, transitionMap, psdSamples, γ, stateSampleCount)

Completes one step of value iteration using precomputed transition maps. 

### Arguments
 - `out`              - Output Array (new value function approximation)
 - `values`           - Current value function approximation
 - `transitionMap`    - Transitions of state/crlb based on the action
 - `psdSamples`       - CRLB samples
 - `γ`                - Discount Factor
 - `stateSampleCount` - Number of state-space samples (for indexing psdSamples)

### Returns
Nothing 

### Notes
This function is multithreaded
"""
function valueIterate_NearestNeighbor_precompute!(  out,
                                                    values,
                                                    transitionMap::Array{Int,2},
                                                    psdSamples,
                                                    γ,
                                                    psdSampleCount)

    Threads.@threads for i in 1:length(values)
        crlbIndex = ((i-1) % psdSampleCount) + 1
        crlb = @view psdSamples[:,:,crlbIndex]
        updatedStates = @view transitionMap[:,i]

        futureValue = minimum(values[updatedStates])
        out[i] = tr(crlb) + γ * futureValue
    end
end

#################################################
################# POMDP Tools ###################
#################################################
export ValueFunctionApproximation_NearestNeighbor_precompute

"""
    ValueFunctionApproximation_NearestNeighbor_precompute(  system::AbstractSystem, 
                                                            τ,
                                                            γ, 
                                                            actionSpace, 
                                                            λ = 1,
                                                            psdSampleCount = 1000,
                                                            trajectorySampleCount = 100
                                                            timestepSampleCount = 10,
                                                            σ² = 1,
                                                            max_iterations=50)

Approximates the value function in the discounted cost optimal sampling problem.
Requires a model of the system dynamics, timestep, discount-factor, and a discrete action space.

Precomputes all state transitions for all actions. Uses significant memory in exchange for
less expensive computation.

Computes `max_iterations` steps of value iteration.

### Arguments
 - `system` - Representation of the dynamical system
 - `τ`                      - Timestep size
 - `γ`                      - Discount Factor
 - `actionSpace`            - Vector of possible actions
 - `λ`                      - CRLB space sampling parameter
 - `psdSampleCount`         - Number of covariance-matrix samples
 - `trajectorySampleCount`  - Number of state-space trajectory samples
 - `timestepSampleCount`    - Number of state-space timestep samples
 - `σ²`                     - Measurement variance
 - `max_iterations`         - Maximum number of iterations

### Returns
    (values, psdSamples, stateSamples)

 - `values`         - Output of value function at sample points 
 - `psdSamples`     - CRLB sample locations
 - `stateSamples`   - State sample locations

### Note

`values` is a 1D array indexed as crlbIndex + psdSampleCount * (stateIndex - 1)
"""
function ValueFunctionApproximation_NearestNeighbor_precompute( system::AbstractSystem, 
                                                                τ,
                                                                γ, 
                                                                actionSpace, 
                                                                λ = 1,
                                                                psdSampleCount = 1000,
                                                                trajectorySampleCount = 100,
                                                                timestepSampleCount = 10,
                                                                σ² = 1,
                                                                max_iterations=50)

    @info("Discretizing the Space")
    # Discretize the space
    stateSampleCount = trajectorySampleCount * timestepSampleCount
    psdSamples   = samplePSD(psdSampleCount, dimension(system), λ) 
    stateSamples = sampleStateSpace(system, trajectorySampleCount, timestepSampleCount, 0, τ)
    values = rand(psdSampleCount*stateSampleCount)
    updateValues = copy(values)

    @info("Precomputing transition maps")
    # Precompute the nearest-neighbor maps based on actions
    systemMap = zeros(Int,length(actionSpace), psdSampleCount*stateSampleCount)
    Threads.@threads for i in 1:stateSampleCount
        # Compute flow and jacobian
        nextState = flow(stateSamples[:,i], τ, system)
        jacobian = flowJacobian(stateSamples[:,i], τ, system)

        # Find nearest state
        ~, nearestState = min_dist(nextState, stateSamples)

        for j in 1:psdSampleCount
            crlbInv = inv(psdSamples[:,:,j])
            for k in 1:length(actionSpace)
                # Compute Updated CRLB
                nextCRLB = jacobian * inv(crlbInv + actionSpace[k] * (actionSpace[k]')) * jacobian'
                ~, nearestJacobian = min_dist(nextCRLB, psdSamples)
                
                systemMap[k, (i-1)*psdSampleCount + j] = (nearestState-1)*psdSampleCount + nearestJacobian
            end
        end
    end

    @info("Applying Value Iteration")
    # Apply Value Iteration 
    for i in 1:max_iterations
        valueIterate_NearestNeighbor_precompute!(updateValues, values, systemMap, psdSamples, γ, psdSampleCount)
        values .= updateValues
    end

    return values, psdSamples, stateSamples 
end
