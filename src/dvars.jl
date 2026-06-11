# src/dvars.jl
#
# D-VARS public entry points. The implementations live in package extensions
# so that the surrogate/optimization stacks are only loaded when the user
# needs them:
#   - `dvars_sensitivities`        -> ext/VariogramAnalysisSurrogatesExt.jl
#                                     (load DataFrames and Surrogates)
#   - `dvars_sensitivities_robust` -> ext/VariogramAnalysisBlackBoxOptimExt.jl
#                                     (load DataFrames and BlackBoxOptim)

using QuadGK: quadgk

"""
    dvars_sensitivities(df::DataFrame, outvarname::Symbol; Hj::Float64=1.0, verbose::Bool=false)

Calculates D-VARS global sensitivity indices using a Kriging surrogate model
from Surrogates.jl for hyperparameter optimization. This version is useful for
comparison against other surrogate-based methods.

Requires `using DataFrames, Surrogates` to activate the implementation.
"""
function dvars_sensitivities(args...; kwargs...)
    error("dvars_sensitivities requires DataFrames.jl and Surrogates.jl. " *
          "Run `using DataFrames, Surrogates` to activate it.")
end

"""
    dvars_sensitivities_robust(df::DataFrame, outvarname::Symbol; Hj::Float64=1.0, verbose::Bool=false)

Calculates D-VARS global sensitivity indices using maximum-likelihood
estimation of a squared-exponential kernel via differential evolution.
This is the recommended method for accuracy and stability.

Requires `using DataFrames, BlackBoxOptim` to activate the implementation.
"""
function dvars_sensitivities_robust(args...; kwargs...)
    error("dvars_sensitivities_robust requires DataFrames.jl and BlackBoxOptim.jl. " *
          "Run `using DataFrames, BlackBoxOptim` to activate it.")
end

"""
    _dvars_integrated_variogram(variance, theta, p, Hj)

Integrated variogram of the exponential kernel,
∫₀^Hj variance * (1 - exp(-theta * h^p)) dh, used to turn fitted kernel
hyperparameters into D-VARS sensitivity indices.
"""
function _dvars_integrated_variogram(variance::Float64, theta::Float64, p::Float64, Hj::Float64)
    integral, _ = quadgk(h -> variance * (1.0 - exp(-theta * h^p)), 0.0, Hj; rtol=1e-6, atol=1e-6)
    return integral
end
