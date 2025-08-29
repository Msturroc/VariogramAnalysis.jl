using Surrogates
using DataFrames
using Statistics
using LinearAlgebra
using Integrals: IntegralProblem, solve, QuadGKJL
using BlackBoxOptim

#==============================================================================#
#           Implementation 1: Based on Surrogates.jl
#==============================================================================#

"""
    dvars_sensitivities(df::DataFrame, outvarname::Symbol; Hj::Float64=1.0)

Calculates D-VARS global sensitivity indices using a Kriging surrogate model
from Surrogates.jl for hyperparameter optimization. This version is useful for
comparison against other surrogate-based methods.
"""
function dvars_sensitivities(df::DataFrame, outvarname::Symbol; Hj::Float64=1.0)
    # --- Data Preparation ---
    df_norm = copy(df)
    for col in names(df_norm)
        min_val, max_val = minimum(df_norm[!, col]), maximum(df_norm[!, col])
        if max_val - min_val > 1e-9
            df_norm[!, col] = (df_norm[!, col] .- min_val) ./ (max_val - min_val)
        end
    end
    
    invar_names = [name for name in names(df) if name != String(outvarname)]
    X = [collect(row) for row in eachrow(Matrix(df_norm[!, invar_names]))]
    y = df_norm[!, outvarname]
    
    ninvars = length(invar_names)
    variance = var(y)

    # --- Surrogate Model Training ---
    println("Building and optimizing Anisotropic Kriging surrogate...")
    p_values = fill(2.0, ninvars)
    lower_bounds_theta = fill(1e-6, ninvars)
    upper_bounds_theta = fill(100.0, ninvars)
    
    surrogate = Kriging(X, y, lower_bounds_theta, upper_bounds_theta, p=p_values)
    phi_opt = surrogate.theta

    # --- Sensitivity Calculation ---
    sensitivities = zeros(ninvars)
    for j in 1:ninvars
        theta_j = surrogate.theta[j]
        p_j = surrogate.p[j]
        integrand(h, p_ignored) = variance * (1.0 - exp(-theta_j * h[1]^p_j))
        prob = IntegralProblem(integrand, [0.0], [Hj])
        sol = solve(prob, QuadGKJL(), reltol=1e-6, abstol=1e-6)
        sensitivities[j] = sol.u
    end
    
    total_sensitivity = sum(sensitivities)
    ratios = total_sensitivity > 0 ? sensitivities ./ total_sensitivity : zeros(ninvars)
    
    return sensitivities, ratios, phi_opt, variance
end


#==============================================================================#
#           Implementation 2: The Robust, State-of-the-Art Version
#==============================================================================#

# --- Helper Functions for the Robust Implementation ---

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
        return 1e12
    end
end

"""
    dvars_sensitivities_robust(df::DataFrame, outvarname::Symbol; Hj::Float64 = 1.0)

Calculates D-VARS global sensitivity indices using a state-of-the-art approach.
This is the recommended method for accuracy and stability.
"""
function dvars_sensitivities_robust(df::DataFrame, outvarname::Symbol; Hj::Float64 = 1.0)
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
    println("Optimizing hyperparameters (θ) using Cholesky + Sq. Exp. Kernel + DE...")
    objective(theta) = calc_L_cholesky(theta, X_matrix, y_vector)
    search_range = [(1e-6, 100.0) for _ in 1:ninvars]
    
    @time opt_results = bboptimize(objective,
                                   SearchRange=search_range,
                                   NumDimensions=ninvars,
                                   Method=:adaptive_de_rand_1_bin,
                                   MaxFuncEvals=1500,
                                   TraceMode=:silent)
    
    theta_opt = best_candidate(opt_results)
    
    println("\n--- JULIA DE OPTIMIZER DEBUG ---")
    println("Optimized Theta values: ", round.(theta_opt, digits=4))
    println("Final Likelihood Value: ", best_fitness(opt_results))
    println("-------------------------------------\n")

    # --- Sensitivity Calculation ---
    sensitivities = zeros(ninvars)
    for j in 1:ninvars
        integrand(h, p) = variance * (1.0 - exp(-theta_opt[j] * h[1]^2))
        prob = IntegralProblem(integrand, [0.0], [Hj])
        sol = solve(prob, QuadGKJL(), reltol=1e-6, abstol=1e-6)
        sensitivities[j] = sol.u
    end
    
    total_sensitivity = sum(sensitivities)
    ratios = total_sensitivity > 0 ? sensitivities ./ total_sensitivity : zeros(ninvars)
    
    return sensitivities, ratios, theta_opt, variance
end

using SurrogatesPolyChaos
using Combinatorics

"""
    pce_sensitivities(d::Int, lb::Vector{Float64}, ub::Vector{Float64}, a_params::Vector{Float64})

Calculates Sobol' indices using Polynomial Chaos Expansion.
"""
function _compute_pce_st_indices(pce)
    coeffs = pce.coeff
    multiidx = pce.orthopolys.ind
    d = size(multiidx, 2)
    
    varY_pce = sum(coeffs[2:end].^2)
    if varY_pce < 1e-12
        return zeros(d)
    end
    
    ST = zeros(d)
    for k in 2:length(coeffs)
        coeff_sq = coeffs[k]^2
        vars_idx = findall(multiidx[k, :] .> 0)
        if !isempty(vars_idx)
            for i in vars_idx
                ST[i] += coeff_sq
            end
        end
    end
    return ST ./ varY_pce
end

"""
    pce_sensitivities(X_pce, Y_pce, lb, ub)

Calculates Sobol' indices using Polynomial Chaos Expansion from a given dataset.
"""
function pce_sensitivities(X_pce::AbstractMatrix, Y_pce::AbstractVector, lb::Vector{Float64}, ub::Vector{Float64})
    println("\n--- Running Julia PCE Analysis ---")
    d = size(X_pce, 1)
    println("PCE using $(size(X_pce, 2)) samples...")

    poly_degree = 2
    
    xpoints = [collect(X_pce[:, i]) for i in 1:size(X_pce, 2)]
    
    orthos = SurrogatesPolyChaos.MultiOrthoPoly([SurrogatesPolyChaos.GaussOrthoPoly(poly_degree) for _ in 1:d], poly_degree)
    pce = SurrogatesPolyChaos.PolynomialChaosSurrogate(xpoints, Y_pce, lb, ub, orthopolys=orthos)
    
    ST_pce = _compute_pce_st_indices(pce)
    
    # Normalize the final ratios to sum to 1 for comparison with other methods
    return ST_pce ./ sum(ST_pce)
end