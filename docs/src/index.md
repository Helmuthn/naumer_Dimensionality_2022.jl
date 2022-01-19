# naumer\_ICML\_2022

## Overview
The repository [naumer\_ICML\_2022](https://github.com/helmuthn/naumer_ICML_2022.jl) contains code related to my anticipated ICML submission in 2022.

The code is split into two main sections.

 * The folder `src` represents a library which represents the main building blocks of the work and could be applied to other problems.

 * The folder `PlotGeneration` contains the scripts to reproduce the plots in the manuscript.

This library is not registered through the Julia package registry, but the package manager can still install it through: 

    ] add https://github.com/helmuthn/naumer_ICML_2022.jl


## Random Sampling

In our approximation of the value function, we need to generate random positive definite matrices.
In this section, we include helper functions to generate samples from the appropriate distributions. 

```@docs
randomPSD
```

```@docs
samplePSD
```

```@docs
sampleStateSpace
```

## Interpolation

This section includes functions involved in our local average interpolation.

```@docs
min_dist
```

```@docs
nearestNeighbor
```

```@docs
buildNearestNeighbor
```

```@docs
localAverage
```

```@docs
localAverageWeights
```

## POMDP Tools


```@docs
updateCRLB
```

```@docs
optimalAction_NearestNeighbor
```

```@docs
valueUpdate_NearestNeighbor
```

```@docs
valueIterate_NearestNeighbor
```

```@docs
valueIterate_NearestNeighbor_precompute!
```

```@docs
optimalAction_LocalAverage
```

```@docs
valueUpdate_LocalAverage
```

```@docs
valueIterate_LocalAverage
```

## Dynamic Programming Solvers

Finally, we have the full value iteration solvers for dynamic programming.

```@docs
ValueFunctionApproximation_NearestNeighbor_precompute
```

```@docs
NearestNeighbor_OptimalPolicy
```

```@docs
ValueFunctionApproximation_LocalAverage
```

```@docs
LocalAverage_OptimalPolicy
```

```@docs
ValueFunctionApproximation_LocalAverage_precompute
```

## Extended Kalman Filter

We include a function for the extended Kalman filter under linear functional measurements.

```@docs
stateupdate_EKF
```

## 1D Approximation

```@docs
optimalaction_1DApprox
```

```@docs
measurementvalue_1DApprox
```

```@docs
actiongradientstep_1DApprox
```


## System API
To add the new abstract system, create a new subtype of `AbstractSystem`.

```@docs
differential
```

```@docs
flow
```

```@docs
flowJacobian
```

```@docs
dimension
```

## Example Systems
We include a number of example systems in this work.

```@docs
StaticSystem
```

```@docs
LinearSystem
```

```@docs
VanDerPolSystem
```

```@docs
VanDerPolSystem_expanded
```

```@docs
HopfSystem
```

```@docs
LorenzSystem
```
