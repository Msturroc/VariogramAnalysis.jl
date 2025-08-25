# src/api.jl

using LinearAlgebra, OrderedCollections, Random, Distributions, Combinatorics

"""
    VARS.sample(params::OrderedDict, num_stars::Int, delta_h::Float64; ...)

Generate VARS/GVARS design of experiments. This function now acts as a dispatcher.
"""
function sample(params::OrderedDict, num_stars::Int, delta_h::Float64;
                sampler_type="lhs", corr_mat=nothing,
                num_dir_samples=10, # Default from python
                use_fictive_corr=false, seed=1234)

    if corr_mat === nothing
        # --- VARS Path ---
        method = :VARS
        X_norm, info = VARS.generate_vars_samples(params, num_stars, delta_h; seed=seed, sampler_type=sampler_type)
        X = VARS.uniform_to_original_dist(X_norm, params)
        d = length(params)
        return (method=method, X=X, X_norm=X_norm, info=info, N=num_stars, d=d, delta_h=delta_h)
    else
        # --- G-VARS Path ---
        method = :GVARS
        X, info = VARS.generate_gvars_samples(params, num_stars, corr_mat, num_dir_samples, delta_h;
                                              seed=seed, use_fictive_corr=use_fictive_corr, sampler_type=sampler_type)
        # For G-VARS, X_norm must be derived from X by applying the CDF of each parameter
        X_norm = VARS.scale_to_unity(X, params)
        d = length(params)
        return (method=method, X=X, X_norm=X_norm, info=info, N=num_stars, d=d, delta_h=delta_h)
    end
end

# --- The "Tell" Function (Analysis) ---
function analyse(method::Symbol, X::Matrix{Float64}, X_norm::Matrix{Float64}, info::Vector, parameters::OrderedDict, N::Int, d::Int, delta_h::Float64, Y::Vector{Float64})
    if method == :VARS
        return vars_analyse(Y, info, N, d, delta_h)
    elseif method == :GVARS
        # Pass the original samples X and the parameters dict for the new analysis logic
        return gvars_analyse(Y, X, info, N, d, delta_h, parameters)
    else
        error("Unknown analysis method: $(method)")
    end
end

# --- Helper Function ---
"""
    scale_to_unity(X::Matrix, parameters::OrderedDict)

Scales a sample matrix `X` to the unit hypercube [0, 1]ᵈ using the
probability integral transform (i.e., applying the CDF of each parameter's
distribution). This is the correct way to "uniformize" the space.
"""
function scale_to_unity(X::Matrix, parameters::OrderedDict)
    d = size(X, 1)
    X_norm = similar(X)
    param_defs = collect(values(parameters))

    for i in 1:d
        # Get the distribution object for the current parameter
        dist, _, _ = VARS._get_distribution_and_stats(param_defs[i])
        
        # Apply the CDF of that distribution to its samples
        X_norm[i, :] = cdf.(dist, @view X[i, :])
    end
    return X_norm
end

"""
    uniform_to_original_dist(X_norm::Matrix, parameters::OrderedDict)

Transforms samples from the unit hypercube to their specified distributions.
"""
function uniform_to_original_dist(X_norm::Matrix, parameters::OrderedDict)
    d, n_samples = size(X_norm)
    X = similar(X_norm)
    param_defs = collect(values(parameters))

    for i in 1:d
        dist, _, _ = VARS._get_distribution_and_stats(param_defs[i])
        # Add a small epsilon to prevent exact 0 or 1
        u_clamped = clamp.(@view(X_norm[i, :]), 1e-15, 1.0 - 1e-15)
        X[i, :] = quantile.(dist, u_clamped)
    end
    return X
end