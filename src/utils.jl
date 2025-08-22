# src/utils.jl

using Distributions

"""
    _get_distribution_and_stats(params::NamedTuple)

A helper function to parse parameter information and return a corresponding
distribution object from Distributions.jl, along with its mean and standard deviation.

# Arguments
- `params::NamedTuple`: A named tuple containing the distribution parameters,
  e.g., `(p1=0.0, p2=1.0, p3=nothing, dist="unif")`.

# Returns
- A tuple `(dist, μ, σ)` where:
  - `dist`: The corresponding `Distribution` object.
  - `μ`: The mean of the distribution.
  - `σ`: The standard deviation of the distribution.
"""
function _get_distribution_and_stats(params::NamedTuple)
    dist_type = params.dist
    p = (params.p1, params.p2, params.p3)

    if dist_type == "unif"
        dist = Uniform(p[1], p[2])
    elseif dist_type == "norm"
        dist = Normal(p[1], p[2])
    elseif dist_type == "triangle"
        dist = TriangularDist(p[1], p[2], p[3])
    elseif dist_type == "lognorm"
        # Note: Distributions.jl LogNormal is parameterised by log-mean and log-std.
        # The python code seems to take mean and std, then convert. We'll do the same.
        μ, σ_val = p[1], p[2]
        cv_sq = (σ_val / μ)^2
        log_μ = log(μ / sqrt(1 + cv_sq))
        log_σ = sqrt(log(1 + cv_sq))
        dist = LogNormal(log_μ, log_σ)
    elseif dist_type == "expo"
        # Python uses lambda (rate), Distributions.jl uses theta (scale = 1/lambda)
        dist = Exponential(1 / p[1])
    elseif dist_type == "gev"
        # location, scale, shape
        dist = GeneralizedExtremeValue(p[1], p[2], p[3])
    else
        error("Unsupported distribution type: $dist_type")
    end

    return dist, mean(dist), std(dist)
end