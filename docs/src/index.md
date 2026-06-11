# VariogramAnalysis.jl

`VariogramAnalysis.jl` is a pure Julia implementation of the Variogram Analysis
of Response Surfaces (VARS) method for global sensitivity analysis, based on
the original research by M. Razavi and H. V. Gupta and inspired by the Python
implementation available at
[vars-tool/vars-tool](https://github.com/vars-tool/vars-tool).

The package provides:

* Input parameter sampling using quasi-Monte Carlo methods (Sobol and Latin Hypercube).
* Two strategies for generating VARS "star" samples (`:relative` and `:shifted_grid`).
* Total-order sensitivity indices (ST) via VARS and G-VARS (correlated inputs).
* Bootstrap confidence intervals for the sensitivity indices.
* D-VARS sensitivity indices from given data, available as package extensions.

## Installation

```julia
pkg> add VariogramAnalysis
```

## Usage

See the
[README](https://github.com/msturroc/VariogramAnalysis.jl#usage-example-sobol-g-function)
for a complete worked example on the Sobol-G function.

## D-VARS extensions

The D-VARS methods rely on heavier dependencies and are loaded on demand:

```julia
using VariogramAnalysis
using DataFrames, Surrogates      # activates dvars_sensitivities
using DataFrames, BlackBoxOptim   # activates dvars_sensitivities_robust
```
