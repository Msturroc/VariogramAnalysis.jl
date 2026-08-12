module VariogramAnalysisBlackBoxOptimExt

using VariogramAnalysis
using DataFrames
using BlackBoxOptim
using LinearAlgebra
using Statistics

import VariogramAnalysis: dvars_sensitivities_robust, _dvars_integrated_variogram

function calc_R_sqexp(theta::Vector{Float64}, X::Matrix{Float64})
    m, d = size(X)
    R = ones(Float64, m, m)
    for u in 1:m, w in 1:(u-1)
        dist_sq = sum(theta[j] * (X[u, j] - X[w, j])^2 for j in 1:d)
        val = exp(-dist_sq)
        R[u, w] = val
        R[w, u] = val
    end
    return R
end

function calc_L_cholesky(theta::Vector{Float64}, X::Matrix{Float64}, Y::Vector{Float64})
    if any(theta .<= 0) return 1e12 end
    R = calc_R_sqexp(theta, X)
    R += I * 1e-8
    try
        C = cholesky(R)
        m = length(Y)
        M = ones(m, 1)
        C_inv_M = C.L \ M
        mu = (C_inv_M' * (C.L \ Y))[1] / (C_inv_M' * C_inv_M)[1]
        Y_minus_mu = Y .- mu
        alpha = C.U \ (C.L \ Y_minus_mu)
        logdetR = 2 * sum(log.(diag(C.L)))
        L = m * log(Y_minus_mu' * alpha) + logdetR
        return isfinite(L) ? L : 1e12
    catch e
        # Numerical failures mean "bad hyperparameters"; anything else is a real bug.
        e isa Union{PosDefException, SingularException, DomainError} || rethrow()
        return 1e12
    end
end

function dvars_sensitivities_robust(df::AbstractDataFrame, outvarname::Symbol; Hj::Float64=1.0, verbose::Bool=false)
    # --- Data Preparation ---
    df_norm = copy(df)
    for col in names(df_norm)
        min_val, max_val = minimum(df_norm[!, col]), maximum(df_norm[!, col])
        if max_val - min_val > 1e-9
            df_norm[!, col] = (df_norm[!, col] .- min_val) ./ (max_val - min_val)
        end
    end
    invar_names = [name for name in names(df) if name != String(outvarname)]
    X_matrix = Matrix(df_norm[!, invar_names])
    y_vector = df_norm[!, outvarname]
    ninvars = length(invar_names)
    variance = var(y_vector)

    # --- Optimization ---
    verbose && println("Optimizing hyperparameters (θ) using Cholesky + Sq. Exp. Kernel + DE...")
    objective(theta) = calc_L_cholesky(theta, X_matrix, y_vector)
    search_range = [(1e-6, 100.0) for _ in 1:ninvars]

    opt_results = bboptimize(objective,
                             SearchRange=search_range,
                             NumDimensions=ninvars,
                             Method=:adaptive_de_rand_1_bin,
                             MaxFuncEvals=1500,
                             TraceMode=:silent)

    theta_opt = best_candidate(opt_results)

    if verbose
        println("Optimized Theta values: ", round.(theta_opt, digits=4))
        println("Final Likelihood Value: ", best_fitness(opt_results))
    end

    # --- Sensitivity Calculation ---
    sensitivities = [_dvars_integrated_variogram(variance, theta_opt[j], 2.0, Hj) for j in 1:ninvars]

    total_sensitivity = sum(sensitivities)
    ratios = total_sensitivity > 0 ? sensitivities ./ total_sensitivity : zeros(ninvars)

    return sensitivities, ratios, theta_opt, variance
end

end # module
