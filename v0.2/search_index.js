var documenterSearchIndex = {"docs":
[{"location":"#naumer_ICML_2022","page":"Home","title":"naumer_ICML_2022","text":"","category":"section"},{"location":"#Overview","page":"Home","title":"Overview","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The repository naumer_ICML_2022 contains code related to my anticipated ICML submission in 2022.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The code is split into two main sections.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The folder src represents a library which represents the main building blocks of the work and could be applied to other problems.\nThe folder PlotGeneration contains the scripts to reproduce the plots in the manuscript.","category":"page"},{"location":"","page":"Home","title":"Home","text":"This library is not registered through the Julia package registry, but the package manager can still install it through: ","category":"page"},{"location":"","page":"Home","title":"Home","text":"] add https://github.com/helmuthn/naumer_Dimensionality_2022.jl","category":"page"},{"location":"#Random-Sampling","page":"Home","title":"Random Sampling","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"In our approximation of the value function, we need to generate random positive definite matrices. In this section, we include helper functions to generate samples from the appropriate distributions. ","category":"page"},{"location":"","page":"Home","title":"Home","text":"randomPSD","category":"page"},{"location":"#naumer_Dimensionality_2022.randomPSD","page":"Home","title":"naumer_Dimensionality_2022.randomPSD","text":"randomPSD([rng = GLOBAL_RNG,] n, λ)\n\nGenerate a random n×n positive definite matrix with i.i.d. exponentially distributed eigenvalues.\n\nArguments\n\nn: Dimensionality of the matrix\nλ: Exponential distribution parameter\n\nReturns\n\nA random n×n matrix\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"samplePSD","category":"page"},{"location":"#naumer_Dimensionality_2022.samplePSD","page":"Home","title":"naumer_Dimensionality_2022.samplePSD","text":"samplePSD([rng = GLOBAL_RNG,] K, n, λ)\n\nGenerate k random n×n positive definite matrices with i.i.d. exponentially distributed eigenvalues.\n\nArguments\n\nK - Number of samples\nn - Dimensionality of the matrix\nλ - Exponential distribution parameter\n\nReturns\n\nAn n×n×K array representing K random matrices\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"sampleStateSpace","category":"page"},{"location":"#naumer_Dimensionality_2022.sampleStateSpace","page":"Home","title":"naumer_Dimensionality_2022.sampleStateSpace","text":"sampleStateSpace(system::AbstractSystem, trajectorySampleCount, timestepCount, burnin, τ)\n\nGenerates stateSampleCount random samples of the statespace. For now, initializes according to a high-variance Gaussian distribution, then steps the system forward in time to shape the density according to the system.\n\nA small i.i.d. Gaussian random vector is added after each step to avoid degeneracy in value iteration.\n\nArguments\n\nsystem                 - Chosen dynamical system\ntrajectorySampleCount  - Number of trajectory samples\ntimestepCount          - Number of timesteps per trajectory\n'burnin`                 - Initial time advancement\nτ                      - timestep size\n\nReturns\n\nA 2D array representing trajectorySampleCount * timestepCount samples, where columns represent the state.\n\n\n\n\n\n","category":"function"},{"location":"#Interpolation","page":"Home","title":"Interpolation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This section includes functions involved in our local average interpolation.","category":"page"},{"location":"","page":"Home","title":"Home","text":"min_dist","category":"page"},{"location":"#naumer_Dimensionality_2022.min_dist","page":"Home","title":"naumer_Dimensionality_2022.min_dist","text":"min_dist(v, D)\n\nFinds the minimum distance vector in D from v. Allows multidimensional array representation,  assumes last axis of D indexes the vectors.\n\nArguments\n\nv - Vector being compared\nD - Set of vectors\n\nReturns\n\n(out, out_ind)\n\nWhere out is the closest vector, and out_ind is the position.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"nearestNeighbor","category":"page"},{"location":"#naumer_Dimensionality_2022.nearestNeighbor","page":"Home","title":"naumer_Dimensionality_2022.nearestNeighbor","text":"nearestNeighbor(target, samples, values)\n\nApproximates the value at target with the value at the closest known sample.\n\nArguments\n\ntarget  - Point being evaluated, domain of function\nsamples - Known sample locations, domain of function\nvalues  - Values at known locations, range of function\n\nReturns\n\nApproximate value at target\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"buildNearestNeighbor","category":"page"},{"location":"#naumer_Dimensionality_2022.buildNearestNeighbor","page":"Home","title":"naumer_Dimensionality_2022.buildNearestNeighbor","text":"buildNearestNeighbor(f, samples)\n\nEvaluates a function f at a set of chosen locations to build the nearest neighbor approximation. Glorified broadcast function.\n\nThe last axis of samples is assumed to index the sample locations.\n\nArguments\n\nf      - The function to be approximated\nsamples- Set of representation points\n\nReturns\n\nAn array of values at the given sample locations.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"localAverage","category":"page"},{"location":"#naumer_Dimensionality_2022.localAverage","page":"Home","title":"naumer_Dimensionality_2022.localAverage","text":"localAverage(targetPSD, targetState, psdSamples, stateSamples, values, d_max)\n\nComputes the local average interpolation, defaulting to a  nearest neighbor interpolation if there are no points within d_max of the desired target location.\n\nThe last axis of samples is assumed to index the sample locations.\n\nArguments\n\ntargetPSD    - PSD point being evaluated, domain of function \ntargetState  - State point being evaluated, domain of function\npsdSamples   - Known PSD sample locations, domain of function\nstateSamples - Known state sample locations, domain of function\nvalues       - Values at known locations, range of function\nd_max        - Maximum distance to average\n\nReturns\n\nA local average approximation at a given location\n\nNotes\n\nvalues is indexed as ...\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"localAverageWeights","category":"page"},{"location":"#naumer_Dimensionality_2022.localAverageWeights","page":"Home","title":"naumer_Dimensionality_2022.localAverageWeights","text":"localAverageWeights(targetPSD, targetState, psdSamples, stateSamples, values, d_max)\n\nComputes the local average interpolation, defaulting to a  nearest neighbor interpolation if there are no points within d_max of the desired target location.\n\nThe last axis of samples is assumed to index the sample locations.\n\nArguments\n\ntargetPSD    - PSD point being evaluated, domain of function \ntargetState  - State point being evaluated, domain of function\npsdSamples   - Known PSD sample locations, domain of function\nstateSamples - Known state sample locations, domain of function\nd_max        - Maximum distance to average\n\nReturns\n\n(indices, weights)\n\nNotes\n\nvalues is indexed as ...\n\n\n\n\n\n","category":"function"},{"location":"#POMDP-Tools","page":"Home","title":"POMDP Tools","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"updateCRLB","category":"page"},{"location":"#naumer_Dimensionality_2022.updateCRLB","page":"Home","title":"naumer_Dimensionality_2022.updateCRLB","text":"updateCRLB(   crlb, \n              action, \n              jacobian,\n              σ²)\n\nUpdates the Fisher information based on the Jacobian of the flow and the current measurement under Gaussian noise.\n\nArguments\n\ncrlb       - Previous Fisher information matrix\naction     - Measurement vector\njacobian   - Jacobian of flow for current timestep\nσ²         - Measurement variance\n\nReturns\n\nUpdated CRLB\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"optimalAction_NearestNeighbor","category":"page"},{"location":"#naumer_Dimensionality_2022.optimalAction_NearestNeighbor","page":"Home","title":"naumer_Dimensionality_2022.optimalAction_NearestNeighbor","text":"optimalAction_NearestNeighbor(  crlb, actionSpace, \n                                samples, values, \n                                jacobian, σ²)\n\nComputes the optimal action for a given state based on the current  nearest neighbor approximation of the value function. Assumes linear measurement under additive Gaussian noise.\n\nArguments\n\ncrlb       - Current Cramér-Rao bound\nactionSpace- Action Space (finite)\nsamples    - Sample locations in value function\nvalues     - Sample evaluations of value function\njacobian   - Jacobian of flow for current timestep\nσ²         - Measurement variance\n\nReturns\n\nThe action that minimizes the Cramér-Rao bound\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"valueUpdate_NearestNeighbor","category":"page"},{"location":"#naumer_Dimensionality_2022.valueUpdate_NearestNeighbor","page":"Home","title":"naumer_Dimensionality_2022.valueUpdate_NearestNeighbor","text":"valueUpdate_NearestNeighbor(   crlb, \n                               γ, \n                               jacobian, \n                               actionSpace, \n                               samples, values)\n\nIterates the discounted Bellman equation for minimizing the CRLB at a given current Fisher information under the nearest neighbor value approximation for a given point.\n\nAssumes linear measurement under additive Gaussian noise.\n\nArguments\n\ncrlb       - Current Cramér-Rao bound\nγ          - Discount factor\njacobian   - Jacobian of flow for current timestep\nactionSpace- Action Space (finite)\nsamples    - Sample locations in value function\nvalues     - Sample evaluations of value function\nσ²         - Noise variance\n\nReturns\n\nThe updated value function approximation evaluated at crlb\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"valueIterate_NearestNeighbor","category":"page"},{"location":"#naumer_Dimensionality_2022.valueIterate_NearestNeighbor","page":"Home","title":"naumer_Dimensionality_2022.valueIterate_NearestNeighbor","text":"valueIterate_NearestNeighbor(γ, jacobian, actionSpace, samples, values)\n\nIterates the discounted Bellman equation for minimizing the CRLB at a given current Fisher information under the nearest neighbor value approximation.\n\nAssumes linear measurement under additive Gaussian noise.\n\nArguments\n\nγ          - Discount factor\njacobian   - Jacobian of flow for current timestep\nactionSpace- Action Space (finite)\nsamples    - Sample locations in value function\nvalues     - Sample evaluations of value function\nσ²         - Measurement variance\n\nReturns\n\nThe updated value function approximation values for all points in samples.\n\nNotes\n\nThis function is multithreaded, remember to give Julia multiple threads when launching with julia -t NTHREADS, where NTHREADS is the desired number of threads.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"valueIterate_NearestNeighbor_precompute!","category":"page"},{"location":"#naumer_Dimensionality_2022.valueIterate_NearestNeighbor_precompute!","page":"Home","title":"naumer_Dimensionality_2022.valueIterate_NearestNeighbor_precompute!","text":"valueIterate_NearestNeighbor_precompute!(   out, \n                                            values, \n                                            transitionMap::Array{Int,2}, \n                                            psdSamples, \n                                            γ, \n                                            stateSampleCount)\n\nCompletes one step of value iteration using precomputed transition maps. \n\nArguments\n\nout              - Output Array (new value function approximation)\nvalues           - Current value function approximation\ntransitionMap    - Transitions of state/crlb based on the action\npsdSamples       - CRLB samples\nγ                - Discount Factor\nstateSampleCount - Number of state-space samples (for indexing psdSamples)\n\nReturns\n\nNothing \n\nNotes\n\nThis function is multithreaded\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"optimalAction_LocalAverage","category":"page"},{"location":"#naumer_Dimensionality_2022.optimalAction_LocalAverage","page":"Home","title":"naumer_Dimensionality_2022.optimalAction_LocalAverage","text":"optimalAction_LocalAverage(     crlb, newState, actionSpace, \n                                psdSamples, stateSamples, values, \n                                jacobian, σ², d_max)\n\nComputes the optimal action for a given state based on the current  nearest neighbor approximation of the value function. Assumes linear measurement under additive Gaussian noise.\n\nArguments\n\ncrlb         - Current Cramér-Rao bound\nnewState     - Updated system state\nactionSpace  - Action Space (finite)\npsdSamples   - Sample PSD locations in value function\nstateSamples - Sample state locations in value function\nvalues       - Sample evaluations of value function\njacobian     - Jacobian of flow for current timestep\nσ²           - Measurement variance\nd_max        - Maximum distance for local average\n\nReturns\n\nThe action that minimizes the Cramér-Rao bound\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"valueUpdate_LocalAverage","category":"page"},{"location":"#naumer_Dimensionality_2022.valueUpdate_LocalAverage","page":"Home","title":"naumer_Dimensionality_2022.valueUpdate_LocalAverage","text":"valueUpdate_LocalAverage(   crlb, \n                            newState,\n                            γ, \n                            jacobian, \n                            actionSpace, \n                            psdSamples, stateSamples,\n                            values,\n                            σ²,\n                            d_max)\n\nIterates the discounted Bellman equation for minimizing the CRLB at a given current Fisher information under the nearest neighbor value approximation for a given point.\n\nAssumes linear measurement under additive Gaussian noise.\n\nArguments\n\ncrlb         - Current Cramér-Rao bound\nnewState     - Updated state value\nγ            - Discount factor\njacobian     - Jacobian of flow for current timestep\nactionSpace  - Action Space (finite)\npsdSamples   - Sample PSD locations in value function\nstateSamples - Sample state locations in value function\nvalues       - Sample evaluations of value function\nσ²           - Noise variance\nd_max        - Maximum distance for local average\n\nReturns\n\nThe updated value function approximation evaluated at crlb\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"valueIterate_LocalAverage","category":"page"},{"location":"#Dynamic-Programming-Solvers","page":"Home","title":"Dynamic Programming Solvers","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Finally, we have the full value iteration solvers for dynamic programming.","category":"page"},{"location":"","page":"Home","title":"Home","text":"ValueFunctionApproximation_NearestNeighbor_precompute","category":"page"},{"location":"#naumer_Dimensionality_2022.ValueFunctionApproximation_NearestNeighbor_precompute","page":"Home","title":"naumer_Dimensionality_2022.ValueFunctionApproximation_NearestNeighbor_precompute","text":"ValueFunctionApproximation_NearestNeighbor_precompute(  system::AbstractSystem, \n                                                        τ,\n                                                        γ, \n                                                        actionSpace, \n                                                        λ = 1,\n                                                        psdSampleCount = 1000,\n                                                        trajectorySampleCount = 100\n                                                        timestepSampleCount = 10,\n                                                        σ² = 1,\n                                                        max_iterations=50)\n\nApproximates the value function in the discounted cost optimal sampling problem. Requires a model of the system dynamics, timestep, discount-factor, and a discrete action space.\n\nPrecomputes all state transitions for all actions. Uses significant memory in exchange for less expensive computation.\n\nComputes max_iterations steps of value iteration.\n\nArguments\n\nsystem - Representation of the dynamical system\nτ                      - Timestep size\nγ                      - Discount Factor\nactionSpace            - Vector of possible actions\nλ                      - CRLB space sampling parameter\npsdSampleCount         - Number of covariance-matrix samples\ntrajectorySampleCount  - Number of state-space trajectory samples\ntimestepSampleCount    - Number of state-space timestep samples\nσ²                     - Measurement variance\nmax_iterations         - Maximum number of iterations\n\nReturns\n\n(values, psdSamples, stateSamples)\n\nvalues         - Output of value function at sample points \npsdSamples     - CRLB sample locations\nstateSamples   - State sample locations\n\nNote\n\nvalues is a 1D array indexed as crlbIndex + psdSampleCount * (stateIndex - 1)\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"NearestNeighbor_OptimalPolicy","category":"page"},{"location":"#naumer_Dimensionality_2022.NearestNeighbor_OptimalPolicy","page":"Home","title":"naumer_Dimensionality_2022.NearestNeighbor_OptimalPolicy","text":"NearestNeighbor_OptimalPolicy(  state, \n                                crlb, \n                                system::AbstractSystem, \n                                σ²,\n                                τ, \n                                values, \n                                psdSamples, \n                                stateSamples, \n                                actionSpace)\n\nGiven a nearest neighbor approximation method for the value function, as well as the current state and crlb, returns the optimal action.\n\nArguments\n\nstate        - Current state of the system\ncrlb         - Current CRLB\nsystem       - Represntation of the system\nσ²           - Measurement noise power\nτ            - Timestep size\nvalues       - Value function evaluated at samples\npsdSamples   - Positive Semidefinite matrix samples\n`stateSamples  - Samples in the state-space\nactionSpace  - Set of possible actions\n\nReturns\n\n(action, index, new_state, new_crlb)\n\naction - The optimal action\nindex  - The index of the action\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"ValueFunctionApproximation_LocalAverage","category":"page"},{"location":"#naumer_Dimensionality_2022.ValueFunctionApproximation_LocalAverage","page":"Home","title":"naumer_Dimensionality_2022.ValueFunctionApproximation_LocalAverage","text":"ValueFunctionApproximation_LocalAverage(    system::AbstractSystem, \n                                            τ,\n                                            γ, \n                                            actionSpace, \n                                            λ = 1,\n                                            psdSampleCount = 1000,\n                                            trajectorySampleCount = 100\n                                            timestepSampleCount = 10,\n                                            σ² = 1,\n                                            max_iterations=50,\n                                            d_max = 0.05)\n\nApproximates the value function using a LocalAverage interpolation in the discounted cost optimal sampling problem. Requires a model of the system dynamics, timestep, discount-factor, and a discrete action space.\n\nComputes max_iterations steps of value iteration.\n\nArguments\n\nsystem                 - Representation of the dynamical system\nτ                      - Timestep size\nγ                      - Discount Factor\nactionSpace            - Vector of possible actions\nλ                      - CRLB space sampling parameter\npsdSampleCount         - Number of covariance-matrix samples\ntrajectorySampleCount  - Number of state-space trajectory samples\ntimestepSampleCount    - Number of state-space timestep samples\nσ²                     - Measurement variance\nmax_iterations         - Maximum number of iterations\nd_max                  - \n\nReturns\n\n(values, psdSamples, stateSamples)\n\nvalues         - Output of value function at sample points \npsdSamples     - CRLB sample locations\nstateSamples   - State sample locations\n\nNote\n\nvalues is a 1D array indexed as crlbIndex + psdSampleCount * (stateIndex - 1)\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"LocalAverage_OptimalPolicy","category":"page"},{"location":"#naumer_Dimensionality_2022.LocalAverage_OptimalPolicy","page":"Home","title":"naumer_Dimensionality_2022.LocalAverage_OptimalPolicy","text":"LocalAverage_OptimalPolicy( state, \n                            crlb, \n                            system::AbstractSystem, \n                            σ²,\n                            τ, \n                            values, \n                            psdSamples, \n                            stateSamples, \n                            actionSpace,\n                            d_max)\n\nGiven a nearest neighbor approximation method for the value function, as well as the current state and crlb, returns the optimal action.\n\nArguments\n\nstate        - Current state of the system\ncrlb         - Current CRLB\nsystem       - Represntation of the system\nσ²           - Measurement noise power\nτ            - Timestep size\nvalues       - Value function evaluated at samples\npsdSamples   - Positive Semidefinite matrix samples\n`stateSamples  - Samples in the state-space\nactionSpace  - Set of possible actions\nd_max        - Maximum distance for local average\n\nReturns\n\n(action, index, new_state, new_crlb)\n\naction - The optimal action\nindex  - The index of the action\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"ValueFunctionApproximation_LocalAverage_precompute","category":"page"},{"location":"#naumer_Dimensionality_2022.ValueFunctionApproximation_LocalAverage_precompute","page":"Home","title":"naumer_Dimensionality_2022.ValueFunctionApproximation_LocalAverage_precompute","text":"ValueFunctionApproximation_LocalAverage_precompute(    system::AbstractSystem, \n                                                       τ,\n                                                       γ, \n                                                       actionSpace, \n                                                       λ = 1,\n                                                       psdSampleCount = 1000,\n                                                       trajectorySampleCount = 100\n                                                       timestepSampleCount = 10,\n                                                       σ² = 1,\n                                                       max_iterations=50,\n                                                       d_max = 0.05)\n\nApproximates the value function using a LocalAverage interpolation in the discounted cost optimal sampling problem. Requires a model of the system dynamics, timestep, discount-factor, and a discrete action space.\n\nComputes max_iterations steps of value iteration.\n\nPrecomputes the local averaging weights to accelerate the value iteration steps.\n\nArguments\n\nsystem                 - Representation of the dynamical system\nτ                      - Timestep size\nγ                      - Discount Factor\nactionSpace            - Vector of possible actions\nλ                      - CRLB space sampling parameter\npsdSampleCount         - Number of covariance-matrix samples\ntrajectorySampleCount  - Number of state-space trajectory samples\ntimestepSampleCount    - Number of state-space timestep samples\nσ²                     - Measurement variance\nmax_iterations         - Maximum number of iterations\nd_max                  - \n\nReturns\n\n(values, psdSamples, stateSamples)\n\nvalues         - Output of value function at sample points \npsdSamples     - CRLB sample locations\nstateSamples   - State sample locations\n\nNote\n\nvalues is a 1D array indexed as crlbIndex + psdSampleCount * (stateIndex - 1)\n\n\n\n\n\n","category":"function"},{"location":"#Extended-Kalman-Filter","page":"Home","title":"Extended Kalman Filter","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"We include a function for the extended Kalman filter under linear functional measurements.","category":"page"},{"location":"","page":"Home","title":"Home","text":"stateupdate_EKF","category":"page"},{"location":"#naumer_Dimensionality_2022.stateupdate_EKF","page":"Home","title":"naumer_Dimensionality_2022.stateupdate_EKF","text":"stateupdate_EKF(prediction, covariance, observation, action, σ²)\n\nUpdates the state based on the measurement according to the Extended Kalman Filter (EKF).\n\nArguments\n\nprediction  - Predicted State\ncovariance  - Covariance approximation\nobservation - Observed value\naction      - Measurement functional\nσ²          - Noise variance\n\nReturns\n\nThe updated state vector\n\n\n\n\n\n","category":"function"},{"location":"#D-Approximation","page":"Home","title":"1D Approximation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"optimalaction_1DApprox","category":"page"},{"location":"#naumer_Dimensionality_2022.optimalaction_1DApprox","page":"Home","title":"naumer_Dimensionality_2022.optimalaction_1DApprox","text":"optimalaction_1DApprox(actionspace, limitvector::Vector, crlb, σ²)\n\nReturns the optimal action out of a finite actionspace.\n\nArguments\n\nactionspace - List of measurement functionals\nlimitvector - Non-zero right singular vector of limiting flow Jacobian\ncrlb        - Current CRLB for state estimation\nσ²          - Measurement noise variance\n\nReturns\n\n`index, action`\n\nindex  - The index of the optimal action\naction - The optimal action\n\n\n\n\n\noptimalaction_1DApprox(actionspace, system::AbstractSystem, state, crlb, σ², T)\n\nReturns the optimal action out of a finite actionspace.\n\nArguments\n\nactionspace - List of measurement functionals\nsystem      - Dynamical System\nstate       - Current system state\ncrlb        - Current CRLB for state estimation\nσ²          - Measurement noise variance\nT           - Time horizon to approximate limit\n\nReturns\n\n`index, action`\n\nindex  - The index of the optimal action\naction - The optimal action\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"measurementvalue_1DApprox","category":"page"},{"location":"#naumer_Dimensionality_2022.measurementvalue_1DApprox","page":"Home","title":"naumer_Dimensionality_2022.measurementvalue_1DApprox","text":"measurementvalue_1DApprox(action, limitvector::Vector, crlb, σ²)\n\nGives a value for the measurement action based on the 1D approximation of the limiting behavior.\n\nArguments\n\naction      - Measurement functional\nlimitvector - Non-zero right singular vector of limiting flow Jacobian\ncrlb        - Current CRLB for state estimation\nσ²          - Measurement noise variance\n\nReturns\n\nA value of the action which should be maximized to minimize the CRLB for prediction.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"actiongradientstep_1DApprox","category":"page"},{"location":"#naumer_Dimensionality_2022.actiongradientstep_1DApprox","page":"Home","title":"naumer_Dimensionality_2022.actiongradientstep_1DApprox","text":"actiongradientstep_1DApprox(action, limitvector, crlb, σ², τ)\n\nGradient step along action space.\n\n\n\n\n\n","category":"function"},{"location":"#System-API","page":"Home","title":"System API","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"To add the new abstract system, create a new subtype of AbstractSystem.","category":"page"},{"location":"","page":"Home","title":"Home","text":"differential","category":"page"},{"location":"#naumer_Dimensionality_2022.differential","page":"Home","title":"naumer_Dimensionality_2022.differential","text":"differential(system::AbstractSystem, x::Vector)\n\nComputes the time derivative of a given continuous-time dynamical system evaluated at a given point.\n\nArguments\n\nsystem - Continuous-Time Dynamical System\nx      - Point to Evaluate differential\n\nReturns\n\nThe derivative of the trajectory of the system with respect to time evaluated at a fixed point x.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"flow","category":"page"},{"location":"#naumer_Dimensionality_2022.flow","page":"Home","title":"naumer_Dimensionality_2022.flow","text":"flow(x::Vector, τ, system::AbstractSystem)\n\nAdvances the state of a dynamical system from state x by time τ.\n\nArguments\n\nsystem - Continuous-Time Dynamical System\nτ      - Advancement time\nx      - Initial state\n\nReturns\n\nThe state advanced by τ units of time\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"flowJacobian","category":"page"},{"location":"#naumer_Dimensionality_2022.flowJacobian","page":"Home","title":"naumer_Dimensionality_2022.flowJacobian","text":"flowJacobian(x::Vector, τ, system::AbstractSystem)\n\nComputes the Jacobian of the flow of a system for τ units of time  around an initial condition x.\n\nArguments\n\nsystem - Continuous-Time Dynamical System\nτ      - Advancement time\nx      - Initial State\n\nReturns\n\nJacobian matrix of the flow representing the derivative with respect to the initial state x.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"dimension","category":"page"},{"location":"#naumer_Dimensionality_2022.dimension","page":"Home","title":"naumer_Dimensionality_2022.dimension","text":"dimension(system::AbstractSystem)\n\nReturns the dimensionality of the system.\n\n\n\n\n\n","category":"function"},{"location":"#Example-Systems","page":"Home","title":"Example Systems","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"We include a number of example systems in this work.","category":"page"},{"location":"","page":"Home","title":"Home","text":"StaticSystem","category":"page"},{"location":"#naumer_Dimensionality_2022.StaticSystem","page":"Home","title":"naumer_Dimensionality_2022.StaticSystem","text":"StaticSystem <: AbstractSystem\n\nDefines a system dx/dt = 0\n\n\n\n\n\n","category":"type"},{"location":"","page":"Home","title":"Home","text":"LinearSystem","category":"page"},{"location":"#naumer_Dimensionality_2022.LinearSystem","page":"Home","title":"naumer_Dimensionality_2022.LinearSystem","text":"LinearSystem{T} <: AbstractSystem{T}\n\nDefines a linear system dx/dt = Ax\n\nFields\n\ndynamics::Matrix{T} - Matrix defining dynamics\n\n\n\n\n\n","category":"type"},{"location":"","page":"Home","title":"Home","text":"VanDerPolSystem","category":"page"},{"location":"#naumer_Dimensionality_2022.VanDerPolSystem","page":"Home","title":"naumer_Dimensionality_2022.VanDerPolSystem","text":"VanDerPolSystem{T} <: AbstractSystem{T}\n\n\n\n\n\n","category":"type"},{"location":"","page":"Home","title":"Home","text":"VanDerPolSystem_expanded","category":"page"},{"location":"#naumer_Dimensionality_2022.VanDerPolSystem_expanded","page":"Home","title":"naumer_Dimensionality_2022.VanDerPolSystem_expanded","text":"VanDerPolSystem_expanded\n\nRepresents dynamics in a product space where two axis represent a Van Der Pol System, while the rest represent stable linear systems.\n\nUsed to demonstrate scaling of dimensionality collapse benefit by allowing the introduction of arbitrary numbers of collapsing dimensions.\n\nFields\n\nμ               - Van Der Pol Parameter\nlinearDimension - Extra Collapsing Dimension Count\n\n\n\n\n\n","category":"type"},{"location":"","page":"Home","title":"Home","text":"HopfSystem","category":"page"},{"location":"#naumer_Dimensionality_2022.HopfSystem","page":"Home","title":"naumer_Dimensionality_2022.HopfSystem","text":"HopfSystem{T} <: AbstractSystem{T}\n\n\n\n\n\n","category":"type"},{"location":"","page":"Home","title":"Home","text":"LorenzSystem","category":"page"},{"location":"#naumer_Dimensionality_2022.LorenzSystem","page":"Home","title":"naumer_Dimensionality_2022.LorenzSystem","text":"LorenzSystem{T} <: AbstractSystem{T}\n\n\n\n\n\n","category":"type"}]
}