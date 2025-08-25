# src/api.jl

using LinearAlgebra, OrderedCollections

# --- Data Structure to Pass Between Sample and Analyse ---
struct VARSProblem
    method::Symbol
    X::Matrix{Float64}
    info::Vector
    parameters::OrderedDict
    N::Int
    d::Int
    delta_h::Float64
end

# --- The "Ask" Function (Sample Generator with Dispatcher) ---
function sample(parameters::OrderedDict, N::Int, delta_h::Float64;
                corr_mat::Union{AbstractMatrix, Nothing}=nothing,
                num_dir_samples::Int=50,
                seed::Union{Nothing, Int}=nothing,
                use_fictive_corr::Bool=true,
                sampler_type::String="sobol")

    d = length(parameters)
    is_uniform = all(p.dist == "unif" for p in values(parameters))
    is_independent = isnothing(corr_mat) || corr_mat == Matrix{Float64}(I, d, d)

    if is_uniform && is_independent
        println("Info: Detected independent, uniform case. Using standard VARS sampling.")
        method = :VARS
        samples = generate_vars_samples(parameters, N, delta_h, seed=seed, sampler_type=sampler_type)
        X, info = samples.X, samples.info
    else
        println("Info: Detected correlated or non-uniform case. Using G-VARS sampling.")
        if !use_fictive_corr
            println("Warning: Skipping Nataf transformation (`use_fictive_corr=false`). Results will be less accurate.")
        end
        method = :GVARS
        final_corr_mat = isnothing(corr_mat) ? Matrix{Float64}(I, d, d) : corr_mat
        samples = generate_gvars_samples(parameters, N, final_corr_mat, num_dir_samples, 
                                         seed=seed, 
                                         use_fictive_corr=use_fictive_corr, sampler_type=sampler_type)
        X, info = samples.X, samples.info
    end

    return VARSProblem(method, X, info, parameters, N, d, delta_h)
end

# --- The "Tell" Function (Analysis) ---
function analyse(problem::VARSProblem, Y::Vector{Float64})
    if problem.method == :VARS
        return vars_analyse(Y, problem.info, problem.N, problem.d, problem.delta_h)
    elseif problem.method == :GVARS
        # G-VARS analysis requires inputs to be correctly scaled to the unit hypercube
        X_norm = scale_to_unity(problem.X, problem.parameters)
        return gvars_analyse(Y, X_norm, problem.info, problem.N, problem.d, problem.delta_h)
    else
        error("Unknown analysis method: $(problem.method)")
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