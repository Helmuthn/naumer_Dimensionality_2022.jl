var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = naumer_ICML_2022","category":"page"},{"location":"#naumer_ICML_2022","page":"Home","title":"naumer_ICML_2022","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for naumer_ICML_2022.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [naumer_ICML_2022]","category":"page"},{"location":"#naumer_ICML_2022.nearestNeighbor-Tuple{Any, Any, Any}","page":"Home","title":"naumer_ICML_2022.nearestNeighbor","text":"nearestNeighbor(target, samples, values)\n\nApproximates the value at target with the value at the closest known sample.\n\nArguments\n\ntarget  - Point being evaluated, domain of function\nsamples - Known sample locations, domain of function\nvalues  - Values at known locations, range of function\n\nReturns\n\nApproximate value at target\n\n\n\n\n\n","category":"method"},{"location":"#naumer_ICML_2022.optimalAction_NearestNeighbor-NTuple{5, Any}","page":"Home","title":"naumer_ICML_2022.optimalAction_NearestNeighbor","text":"optimalAction_NearestNeighbor(  information, actionSpace, \n                                samples, values, \n                                jacobian)\n\nComputes the optimal action for a given state based on the current  nearest neighbor approximation of the value function.\n\nArguments\n\ninformation- Current Fisher information\nactionSpace- Action Space (finite)\nsamples    - Sample locations in value function\nvalues     - Sample evaluations of value function\njacobian   - Jacobian of flow for current timestep\n\nReturns\n\nThe action that minimizes the Cramér-Rao bound\n\n\n\n\n\n","category":"method"},{"location":"#naumer_ICML_2022.randomPSD","page":"Home","title":"naumer_ICML_2022.randomPSD","text":"randomPSD([rng,] n, λ=1)\n\nGenerate a random n×n positive definite matrix with i.i.d. exponentially distributed eigenvalues.\n\nArguments\n\nn: Dimensionality of the matrix\nλ: Exponential distribution parameter\n\nReturns\n\nA random n×n matrix\n\n\n\n\n\n","category":"function"},{"location":"#naumer_ICML_2022.updateFisherInformation-NTuple{4, Any}","page":"Home","title":"naumer_ICML_2022.updateFisherInformation","text":"updateFisherInformation(    information, \n                            action, \n                            jacobian,\n                            σ²)\n\nUpdates the Fisher information based on the Jacobian of the flow and the current measurement under Gaussian noise.\n\nArguments\n\ninformation- Previous Fisher information matrix\naction     - Measurement vector\njacobian   - Jacobian of flow for current timestep\nσ²         - Measurement variance\n\nReturns\n\nUpdated Fisher Information\n\n\n\n\n\n","category":"method"},{"location":"#naumer_ICML_2022.valueIterate-NTuple{5, Any}","page":"Home","title":"naumer_ICML_2022.valueIterate","text":"ValueIterate_NearestNeighbor(γ, jacobian, actionSpace, samples, values)\n\nIterates the discounted Bellman equation for minimizing the CRLB at a given current Fisher information under the nearest neighbor value approximation.\n\nArguments\n\nγ          - Discount factor\njacobian   - Jacobian of flow for current timestep\nactionSpace- Action Space (finite)\nsamples    - Sample locations in value function\nvalues     - Sample evaluations of value function\n\nReturns\n\nThe updated value function approximation values for all points in samples.\n\nNotes\n\nThis function is multithreaded, remember to give Julia multiple threads when launching with julia -t NTHREADS, where NTHREADS is the desired number of threads.\n\n\n\n\n\n","category":"method"},{"location":"#naumer_ICML_2022.valueUpdate_NearestNeighbor-NTuple{6, Any}","page":"Home","title":"naumer_ICML_2022.valueUpdate_NearestNeighbor","text":"valueUpdate_NearestNeighbor(   information, \n                               γ, \n                               jacobian, \n                               actionSpace, \n                               samples, values)\n\nIterates the discounted Bellman equation for minimizing the CRLB at a given current Fisher information under the nearest neighbor value approximation for a given point.\n\nArguments\n\ninformation- Current Fisher information\nγ          - Discount factor\njacobian   - Jacobian of flow for current timestep\nactionSpace- Action Space (finite)\nsamples    - Sample locations in value function\nvalues     - Sample evaluations of value function\n\nReturns\n\nThe updated value function approximation evaluated at information\n\n\n\n\n\n","category":"method"}]
}
