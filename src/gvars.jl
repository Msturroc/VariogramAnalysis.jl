# src/gvars.jl

using Distributions
using Roots
using HCubature
using ProgressMeter
using LinearAlgebra

const NORMAL_0_1 = Normal(0, 1)

"""
    rx_to_rn(dist_pair::Tuple, params1::NamedTuple, params2::NamedTuple, rx_pair::Real)

Transforms a correlation coefficient `rx` from the original parameter space to `rn`
in the standard normal space using the Nataf transformation integral.
"""
function rx_to_rn(dist_pair::Tuple{String, String}, params1::NamedTuple, params2::NamedTuple, rx_pair::Real)
    dist1, μ1, σ1 = _get_distribution_and_stats(params1)
    dist2, μ2, σ2 = _get_distribution_and_stats(params2)

    if isapprox(σ1, 0) || isapprox(σ2, 0)
        return 0.0
    end

    # The integrand is defined over two INDEPENDENT standard normal variables, z = [z₁, z₂].
    integrand = function(z)
        z₁, z₂ = z[1], z[2]

        # The PDF is for two independent standard normals.
        pdf_val = pdf(NORMAL_0_1, z₁) * pdf(NORMAL_0_1, z₂)

        # Correlation is introduced by transforming the arguments to the CDFs.
        # This creates two correlated standard uniform variables.
        u₁ = cdf(NORMAL_0_1, z₁ * sqrt(1 - rx_pair^2) + rx_pair * z₂)
        u₂ = cdf(NORMAL_0_1, z₂)

        # We then find the quantiles from the original distributions.
        val1 = quantile(dist1, u₁)
        val2 = quantile(dist2, u₂)
        
        return val1 * val2 * pdf_val
    end

    # Perform the double integration over the standard normal domain.
    integral_val, _ = hcubature(integrand, [-8.0, -8.0], [8.0, 8.0], rtol=1e-6)

    # Normalise the result to get the correlation coefficient `rn`.
    rn = (integral_val - μ1 * μ2) / (σ1 * σ2)

    return rn
end

"""
    rn_to_rx(dist_pair::Tuple, params1::NamedTuple, params2::NamedTuple, rn_pair::Real)

The inverse of `rx_to_rn`. Finds the `rx` in the original space that corresponds
to a given correlation `rn` in the standard normal space by solving a root-finding problem.
"""
function rn_to_rx(dist_pair::Tuple{String, String}, params1::NamedTuple, params2::NamedTuple, rn_pair::Real)
    # Handle trivial and edge cases immediately for efficiency and stability.
    if isapprox(abs(rn_pair), 1.0, atol=1e-9)
        return sign(rn_pair) * 1.0
    elseif isapprox(rn_pair, 0.0, atol=1e-9)
        return 0.0
    end

    # Define the function whose root we want to find: f(r) = rn_pair - rx_to_rn(r)
    f(r) = rn_pair - rx_to_rn(dist_pair, params1, params2, r)

    # --- Attempt 1: Use a tight, intelligent bracket for the root search ---
    safe_min = nextfloat(-1.0)
    safe_max = prevfloat(1.0)
    bracket = rn_pair > 0 ? (0.0, safe_max) : (safe_min, 0.0)

    try
        rx = find_zero(f, bracket, Bisection())
        return clamp(rx, -1.0, 1.0) # Success! Return the result immediately.
    catch e
        @warn "Root finding with tight bracket failed for rn_pair = $rn_pair. Trying full interval. Error: $e"
    end

    # --- Attempt 2: Fallback to the full interval if the first attempt fails ---
    try
        rx = find_zero(f, (safe_min, safe_max), Bisection())
        return clamp(rx, -1.0, 1.0) # Success on the second try! Return the result.
    catch final_e
        @error "Root finding failed completely for rn_pair = $rn_pair. Returning rn_pair as a fallback. Final Error: $final_e"
        return rn_pair # All attempts failed. Return the input as a fallback.
    end
end

"""
    map_to_fictive_corr(parameters::Dict, corr_mat::AbstractMatrix)

Computes the "fictive" correlation matrix required for G-VARS. This matrix represents
the correlations in the standard normal space that will produce the desired correlations
in the original parameter space after transformation.
"""
function map_to_fictive_corr(parameters::OrderedDict, corr_mat::AbstractMatrix{<:Real})
    d = size(corr_mat, 1)
    fictive_corr = Matrix{Float64}(I, d, d) # Start with an identity matrix

    param_keys = collect(keys(parameters))
    
    @showprogress "Building fictive matrix..." for i in 1:(d-1)
        for j in (i+1):d
            params_i = parameters[param_keys[i]]
            params_j = parameters[param_keys[j]]
            dist_pair = (params_i.dist, params_j.dist)
            
            # For each off-diagonal element, find the corresponding fictive correlation
            fictive_corr[i, j] = rn_to_rx(dist_pair, params_i, params_j, corr_mat[i, j])
            
            # The matrix is symmetric
            fictive_corr[j, i] = fictive_corr[i, j]
        end
    end
    
    return fictive_corr
end


"""
    normal_to_original_dist(norm_vectors::AbstractMatrix, parameters::OrderedDict)

Transforms samples from a correlated standard normal distribution back into their
original, specified marginal distributions. This is the final step of the inverse
Nataf transformation.
"""
function normal_to_original_dist(norm_vectors::AbstractMatrix{<:Real}, parameters::OrderedDict)
    d = size(norm_vectors, 2)
    if d != length(parameters)
        error("Number of columns in norm_vectors must match the number of parameters.")
    end

    x = similar(norm_vectors) # Pre-allocate output matrix
    param_keys = collect(keys(parameters))

    for i in 1:d
        params_i = parameters[param_keys[i]]
        dist_i, _, _ = _get_distribution_and_stats(params_i)
        
        # 1. Apply the standard normal CDF to get uniformly distributed samples in [0, 1]
        u = cdf.(NORMAL_0_1, @view norm_vectors[:, i])
        
        # 2. Apply the inverse CDF (quantile function) of the target distribution
        x[:, i] = quantile.(dist_i, u)
    end
    
    return x
end