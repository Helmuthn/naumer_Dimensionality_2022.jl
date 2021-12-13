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
function randomPSD(rng, n, λ=1)
    mat = randn(rng,n,n)
    ortho, ~ = qr(mat)
	return ortho' * Diagonal(randexp(rng,n)/λ) * ortho
end

randomPSD(n, λ=1) = randomPSD(GLOBAL_RNG, n, λ)


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
function samplePSD(rng, K, n, λ=1)
    out = zeros(n,n,K)
    for i in 1:K
        out[:,:,i] = randomPSD(rng, n, λ)
    end
    return out
end

samplePSD(K, n, λ=1) = samplePSD(GLOBAL_RNG, K, n, λ)


#################################################
################ Interpolation ##################
#################################################

export nearestNeighbor

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



#################################################
################# POMDP Tools ###################
#################################################

export updateCRLB, optimalAction_NearestNeighbor, valueUpdate_NearestNeighbor, valueIterate_NearestNeighbor

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
		new_out[i] = valueUpdate_NearestNeighbor(   view(samples,:,:,i), 
                                                    γ, 
                                                    jacobian, 
                                                    actionSpace, 
                                                    samples, values,
                                                    σ²)

	end
	return new_out
end
