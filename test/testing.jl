
using Pkg
Pkg.activate("")
include("../src/VARS.jl")
using OrderedCollections
using Distributions
using Statistics
using LinearAlgebra
using Printf
using PyCall
using Combinatorics

println("--- G-VARS Analysis: True Julia vs. Python Library Verification ---")

# --- Debug the Python Environment (Excellent idea!) ---
println("\n=== Debugging Python Environment ===")
try
    # This block confirms that PyCall can execute multi-line Python
    # and that it can define and then look up a function.
    py"""
    import sys
    import numpy
    import varstool
    print("Python version:", sys.version)
    print("Numpy path:", numpy.__file__)
    print("Varstool path:", varstool.__file__)

    def test_func():
        return "Hello from Python"
    """
    test_result = py"test_func"()
    println("Test function result: ", test_result)
    println("Python environment seems OK.")
catch e
    @error "Python environment debug failed. This indicates a core PyCall issue."
    rethrow(e)
end
println("==================================\n")


# --- Python Wrapper Function (Using the Correct "Define, then Call" Pattern) ---
function run_python_gvars_analyse(X_norm::Matrix, Y::Vector, info::Vector, N::Int, d::Int, delta_h::Float64, parameters::OrderedDict)
    # 1. Prepare data for Python
    param_keys = collect(keys(parameters))
    rays_data = Dict()
    for i in 1:length(Y)
        p_info = info[i]
        key = (p_info.star_id, p_info.dim_id)
        if !haskey(rays_data, key)
            rays_data[key] = []
        end
        x_val = p_info.dim_id > 0 ? X_norm[p_info.dim_id, i] : -1.0
        push!(rays_data[key], Dict("y" => Y[i], "x_norm" => x_val))
    end

    # 2. DEFINE the function in Python's main namespace.
    # We do not assign the result of this block to a Julia variable.
    py"""
    import numpy as np
    from itertools import combinations

    def analyse_in_python(rays_data, N, d, delta_h, param_keys):
        centre_ys = [rays_data.get((s, 0), [{}])[0].get('y') for s in range(1, N + 1)]
        centre_ys = [y for y in centre_ys if y is not None]
        VY = np.var(centre_ys, ddof=1)
        if VY < 1e-12:
            return np.zeros(d)

        ST = np.zeros(d)
        for dim_idx, param_name in enumerate(param_keys):
            dim = dim_idx + 1
            gamma_sum = 0.0
            ecov_sum = 0.0
            stars_with_data = 0
            for star in range(1, N + 1):
                centre_point_data = rays_data.get((star, 0), [])
                conditional_points = rays_data.get((star, dim), [])
                if not centre_point_data or not conditional_points:
                    continue
                full_ray_data = conditional_points + [{'y': centre_point_data[0]['y'], 'x_norm': -1}]
                star_bin_pairs = []
                for p1_dict, p2_dict in combinations(full_ray_data, 2):
                    if p1_dict['x_norm'] != -1 and p2_dict['x_norm'] != -1:
                        actual_h = abs(p1_dict['x_norm'] - p2_dict['x_norm'])
                        if 0 < actual_h <= delta_h:
                            star_bin_pairs.append((p1_dict['y'], p2_dict['y']))
                if not star_bin_pairs:
                    continue
                p1_vals = np.array([p[0] for p in star_bin_pairs])
                p2_vals = np.array([p[1] for p in star_bin_pairs])
                gamma_i = 0.5 * np.mean(np.square(p1_vals - p2_vals))
                mu_star = np.mean([p['y'] for p in full_ray_data])
                ecov_i = np.mean((p1_vals - mu_star) * (p2_vals - mu_star))
                gamma_sum += gamma_i
                ecov_sum += ecov_i
                stars_with_data += 1
            if stars_with_data > 0:
                avg_gamma = gamma_sum / stars_with_data
                avg_ecov = ecov_sum / stars_with_data
                ST[dim_idx] = (avg_gamma + avg_ecov) / VY
            else:
                ST[dim_idx] = np.nan
        return ST
    """

    # 3. LOOK UP the function by name and CALL it.
    return py"analyse_in_python"(rays_data, N, d, delta_h, param_keys)
end


# --- Model and G-VARS Setup (same as before) ---
function sobol_g_function_batch(X::AbstractMatrix, a::Vector)
    d, n_points = size(X)
    results = ones(n_points)
    for i in 1:d
        results .*= (abs.(4 .* X[i,:] .- 2) .+ a[i]) ./ (1 .+ a[i])
    end
    return results
end

a = [0, 0.5, 3, 9, 99, 99, 99, 99]
d = length(a)
parameters = OrderedDict("x$i" => (p1=0.0, p2=1.0, p3=nothing, dist="unif") for i in 1:d)
corr_mat = Matrix{Float64}(I, d, d)
N = 200
num_dir_samples = 50
delta_h = 0.1

# --- Generate Inputs (ONCE) ---
println("Step 1: Generating G-VARS samples...")
samples = VARS.generate_gvars_samples(parameters, N, corr_mat, num_dir_samples, seed=123, use_fictive_corr=false)
X = samples.X
Y = sobol_g_function_batch(X, a)
X_norm = X

# --- Run and Compare ---
println("Step 2: Running analysis in Julia and Python...")
julia_results = VARS.gvars_analyse(Y, X_norm, samples.info, N, d, delta_h)
python_results = run_python_gvars_analyse(X_norm, Y, samples.info, N, d, delta_h, parameters)

# --- Display Results ---
println("\n--- Verification of G-VARS Analysis Implementation ---")
println("Parameter | Julia ST  | Python ST | Difference")
println("------------------------------------------------------")
for i in 1:d
    diff = abs(julia_results.ST[i] - python_results[i])
    @printf("x%-8d | %-9.4f | %-9.4f | %.6f\n", i, julia_results.ST[i], python_results[i], diff)
end