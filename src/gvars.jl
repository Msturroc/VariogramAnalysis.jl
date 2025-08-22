# src/gvars.jl

using Distributions
using Roots
using HCubature
using ProgressMeter
using LinearAlgebra

const NORMAL_0_1 = Normal(0, 1)

"""
    rx_to_rn(...)
"""
function rx_to_rn(dist_pair::Tuple{String, String}, params1::NamedTuple, params2::NamedTuple, rx_pair::Real)
    dist1, μ1, σ1 = _get_distribution_and_stats(params1)
    dist2, μ2, σ2 = _get_distribution_and_stats(params2)

    if isapprox(σ1, 0) || isapprox(σ2, 0) return 0.0 end

    integrand = function(z)
        z₁, z₂ = z[1], z[2]
        pdf_val = pdf(NORMAL_0_1, z₁) * pdf(NORMAL_0_1, z₂)
        
        u₁_raw = cdf(NORMAL_0_1, z₁ * sqrt(1 - rx_pair^2) + rx_pair * z₂)
        u₂_raw = cdf(NORMAL_0_1, z₂)

        epsilon_clamp = 1e-15
        u₁ = clamp(u₁_raw, epsilon_clamp, 1.0 - epsilon_clamp)
        u₂ = clamp(u₂_raw, epsilon_clamp, 1.0 - epsilon_clamp)
        
        val1 = quantile(dist1, u₁)
        val2 = quantile(dist2, u₂)
        
        result = val1 * val2 * pdf_val
        return isnan(result) ? 0.0 : result
    end

    # --- CRITICAL FIX: Added maxevals to prevent hangs ---
    # This forces the integrator to stop after a reasonable number of evaluations,
    # preventing the infinite loops on difficult integrands.
    integral_val, _ = hcubature(integrand, [-8.0, -8.0], [8.0, 8.0], rtol=1e-6, maxevals=100000)
    
    rn = (integral_val - μ1 * μ2) / (σ1 * σ2)
    
    return rn
end

"""
    rn_to_rx(...)
"""
function rn_to_rx(dist_pair::Tuple{String, String}, params1::NamedTuple, params2::NamedTuple, rn_pair::Real)
    if isapprox(abs(rn_pair), 1.0, atol=1e-9) return sign(rn_pair) * 1.0 end
    if isapprox(rn_pair, 0.0, atol=1e-9) return 0.0 end

    f(r) = rn_pair - rx_to_rn(dist_pair, params1, params2, r)

    try
        rx = find_zero(f, (-1.0, 1.0), Bisection())
        return clamp(rx, -1.0, 1.0)
    catch e
        @error "Root finding failed for rn_pair = $rn_pair. Returning rn_pair as fallback. Error: $e"
        return rn_pair
    end
end

"""
    map_to_fictive_corr(...)
"""
function map_to_fictive_corr(parameters::OrderedDict, corr_mat::AbstractMatrix{<:Real})
    d = size(corr_mat, 1)
    fictive_corr = Matrix{Float64}(I, d, d)
    param_keys = collect(keys(parameters))
    
    @showprogress "Building fictive matrix..." for i in 1:(d-1)
        for j in (i+1):d
            params_i = parameters[param_keys[i]]
            params_j = parameters[param_keys[j]]
            dist_pair = (params_i.dist, params_j.dist)
            fictive_corr[i, j] = rn_to_rx(dist_pair, params_i, params_j, corr_mat[i, j])
            fictive_corr[j, i] = fictive_corr[i, j]
        end
    end
    
    return fictive_corr
end

"""
    normal_to_original_dist(...)
"""
function normal_to_original_dist(norm_vectors::AbstractMatrix{<:Real}, parameters::OrderedDict)
    d = size(norm_vectors, 2)
    if d != length(parameters)
        error("Number of columns in norm_vectors must match the number of parameters.")
    end
    x = similar(norm_vectors)
    param_keys = collect(keys(parameters))
    for i in 1:d
        params_i = parameters[param_keys[i]]
        dist_i, _, _ = _get_distribution_and_stats(params_i)
        u = cdf.(NORMAL_0_1, @view norm_vectors[:, i])
        x[:, i] = quantile.(dist_i, u)
    end
    return x
end