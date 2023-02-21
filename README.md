# Dimensionality Collapse Code

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://helmuthn.github.io/naumer_Dimensionality_2022.jl/dev)
[![Build Status](https://github.com/helmuthn/naumer_Dimensionality_2022.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/helmuthn/naumer_Dimensionality_2022.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/helmuthn/naumer_Dimensionality_2022.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/helmuthn/naumer_Dimensionality_2022.jl)

Repository of implementations for the manuscript
*Dimensionality Collapse: Optimal Measurement Selection for Low-Error Infinite-Horizon Forecasting*

Includes implementations of a dynamic programming approach to optimal experimental design for infinite-horizon forecasting based on the Bellman equation, as well as the proposed approach based on a local 1D approximation.

## Important Note
The plotting code depends on an old version of [Makie.jl](https://github.com/MakieOrg/Makie.jl).

Due to breaking changes in the library, use version 0.15.3 of Makie.jl to regenerate the figures.
