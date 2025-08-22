using Pkg
Pkg.activate("") # Activates the current environment

# Make sure your VARS.jl source file is accessible and includes the new API
include("../src/VARS.jl")

using OrderedCollections
using Statistics
using LinearAlgebra
using Printf
using PyCall

println("--- VARS Analysis (Independent/Uniform): Julia vs. Python vs. Analytical ---")

# --- Python Environment Setup and Helper Function ---
try
    py"""
    import sys
    import numpy as np
    import pandas as pd
    import traceback
    import varstool 
    from varstool.sensitivity_analysis import vars_funcs
    from itertools import combinations

    print("Python version:", sys.version)
    print("Varstool path:", varstool.__file__)

    # --- 1. DEFINE THE CORRECTED LIBRARY FUNCTION ---
    # This is a corrected version of varstool.sensitivity_analysis.vars_funcs.section_df
    def section_df_fixed(df, delta_h):
        # THE CRITICAL FIX: Reset the index to be a simple 0-based positional index.
        df_reset = df.reset_index(drop=True)
        
        # The original library's helper function to create pairs
        def pairs_h(iterable):
            # This now correctly operates on the new 0-based index [0, 1, ..., n-1]
            interval = range(min(iterable), max(iterable) - min(iterable))
            pairs = {key + 1: [j for j in combinations(iterable, 2) if abs(
                j[0] - j[1]) == key + 1] for key in interval}
            return pairs

        # Now, the index labels match the positional indices of the numpy array
        pairs = pairs_h(df_reset.index)
        df_values = df_reset.to_numpy()
        
        # This dict comprehension can be empty if a group has < 2 points.
        pair_dict = {h * delta_h:
                     pd.DataFrame.from_dict({str(idx_tup): [
                                            df_values[idx_tup[0]], df_values[idx_tup[1]]] for idx_tup in idx}, 'index')
                     for h, idx in pairs.items()}

        # If no pairs were found, return an empty DataFrame to prevent the concat error.
        if not pair_dict:
            return pd.DataFrame()

        return pd.concat(pair_dict)

    # --- 2. MONKEY-PATCH THE LIBRARY ---
    # We dynamically replace the buggy function in the loaded module with our fixed version.
    vars_funcs.section_df = section_df_fixed
    print("Monkey-patch applied successfully to varstool.sensitivity_analysis.vars_funcs.section_df")


    # --- 3. MAIN HELPER FUNCTION ---
    def run_vars_offline(Y, info_dicts, parameters_dict, N, delta_h):
        try:
            # 4. Construct the full model_df as the library expects
            all_Y = np.array(Y)
            all_info = pd.DataFrame(info_dicts)
            param_keys = list(parameters_dict.keys())
            
            all_info['param_name'] = [param_keys[d_id - 1] if d_id > 0 else 'centre_point' for d_id in all_info['dim_id']]
            all_info['points'] = list(range(len(all_info)))

            multi_index = pd.MultiIndex.from_frame(
                all_info[['star_id', 'param_name', 'points']],
                names=['centre', 'param', 'points']
            )
            model_df = pd.DataFrame(all_Y, index=multi_index, columns=['output'])

            # 5. Instantiate the VARS class
            vars_instance = varstool.VARS(
                num_stars=N,
                parameters=parameters_dict,
                delta_h=delta_h,
                report_verbose=False
            )

            # 6. Run the original offline analysis. It will now use our patched function.
            vars_instance.run_offline(model_df)

            # 7. Return the results
            return vars_instance.output['ST']

        except Exception as e:
            print("--- PYTHON EXCEPTION ---", file=sys.stderr)
            print(f"An error occurred in the Python helper function: {e}", file=sys.stderr)
            print("Traceback:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            print("--- END PYTHON EXCEPTION ---", file=sys.stderr)
            raise
    """
    println("Python helper function defined successfully.")
catch e
    @error "Failed to set up Python environment or define helper function. Check PyCall and varstool installation."
    rethrow(e)
end

# --- Julia Wrapper to Call the Python VARS Helper ---
function run_python_library_vars(Y::Vector, info::Vector, parameters::OrderedDict, N::Int, delta_h::Float64)
    info_dicts = [Dict("star_id" => i.star_id, "dim_id" => i.dim_id, "h" => i.h) for i in info]
    
    py_params = Dict{String, Any}()
    for (key, nt) in parameters
        py_params[key] = (nt.p1, nt.p2, nt.p3, nt.dist)
    end
    
    python_st_series = py"run_vars_offline"(Y, info_dicts, py_params, N, delta_h)
    
    return convert(Vector, python_st_series.values)
end

# --- Analytical Solution for Sobol G-function ---
function sobol_g_analytical_st(a::Vector)
    d = length(a)
    Vi = @. 1 / (3 * (1 + a)^2)
    total_variance = prod(Vi .+ 1) - 1
    st_indices = zeros(d)
    for i in 1:d
        st_indices[i] = (Vi[i] / total_variance) * (total_variance + 1) / (Vi[i] + 1)
    end
    return st_indices
end

# --- Model Setup ---
function sobol_g_function_batch(X::AbstractMatrix, a::Vector)
    d, n_points = size(X)
    results = ones(n_points)
    for i in 1:d
        results .*= (abs.(4 .* X[i,:] .- 2) .+ a[i]) ./ (1 .+ a[i])
    end
    return results
end

# --- Problem Definition ---
a = [0, 0.5, 3, 9, 99, 99, 99, 99]
d = length(a)
parameters = OrderedDict("x$i" => (p1=0.0, p2=1.0, p3=nothing, dist="unif") for i in 1:d)
corr_mat_identity = Matrix{Float64}(I, d, d)
N = 256 
delta_h = 0.1

# --- Step 1: ASK for samples using the dispatcher ---
println("\nStep 1: Generating VARS samples using the dispatcher...")
problem = VARS.sample(parameters, N, delta_h, corr_mat=corr_mat_identity, seed=123)

# --- Step 2: Run the model ---
println("Step 2: Running the model function...")
Y = sobol_g_function_batch(problem.X, a)

# --- Step 3: TELL the results to the dispatcher for analysis ---
println("Step 3: Running analyses...")

println("  - Running Julia VARS implementation via dispatcher...")
julia_results = VARS.analyse(problem, Y)

println("  - Running original Python 'varstool.VARS' library...")
python_results = run_python_library_vars(Y, problem.info, problem.parameters, N, delta_h)

println("  - Calculating analytical solution...")
analytical_results = sobol_g_analytical_st(a)

# --- Step 4: Display Final Comparison Table ---
println("\n--- VARS Analysis Results ---")
@printf("%-12s | %-10s | %-10s | %-10s | %-20s\n", "Parameter", "Julia ST", "Python ST", "Analytic", "Julia vs. Analytic")
println(repeat("-", 78))
for i in 1:d
    diff = abs(julia_results.ST[i] - analytical_results[i])
    @printf("x%-11d | %-10.4f | %-10.4f | %-10.4f | %-20.6f\n", i, julia_results.ST[i], python_results[i], analytical_results[i], diff)
end

using Pkg
Pkg.activate("") # Activates the current environment

# Make sure your VARS.jl source file is accessible and includes the new API
include("../src/VARS.jl")

using OrderedCollections
using Statistics
using LinearAlgebra
using Printf

println("--- VARS Analysis (Independent/Uniform): Julia vs. Analytical ---")

# --- Analytical Solution for Sobol G-function ---
function sobol_g_analytical_st(a::Vector)
    d = length(a)
    Vi = @. 1 / (3 * (1 + a)^2)
    total_variance = prod(Vi .+ 1) - 1
    st_indices = zeros(d)
    for i in 1:d
        st_indices[i] = (Vi[i] / total_variance) * (total_variance + 1) / (Vi[i] + 1)
    end
    return st_indices
end

# --- Model Setup ---
function sobol_g_function_batch(X::AbstractMatrix, a::Vector)
    d, n_points = size(X)
    results = ones(n_points)
    for i in 1:d
        results .*= (abs.(4 .* X[i,:] .- 2) .+ a[i]) ./ (1 .+ a[i])
    end
    return results
end

# --- Problem Definition ---
a = [0, 0.5, 3, 9, 99, 99, 99, 99]
d = length(a)
parameters = OrderedDict("x$i" => (p1=0.0, p2=1.0, p3=nothing, dist="unif") for i in 1:d)
corr_mat_identity = Matrix{Float64}(I, d, d)
N = 256 
delta_h = 0.1

# --- Step 1: ASK for samples using the dispatcher ---
println("\nStep 1: Generating VARS samples using the dispatcher...")
# Because params are uniform and corr_mat is identity, this will call `generate_vars_samples`
problem = VARS.sample(parameters, N, delta_h, corr_mat=corr_mat_identity, seed=123)

# --- Step 2: Run the model ---
println("Step 2: Running the model function...")
Y = sobol_g_function_batch(problem.X, a)

# --- Step 3: TELL the results to the dispatcher for analysis ---
println("Step 3: Running analyses...")

println("  - Running Julia VARS implementation via dispatcher...")
# This will call `vars_analyse` because problem.method is :VARS
julia_results = VARS.analyse(problem, Y)

println("  - Calculating analytical solution...")
analytical_results = sobol_g_analytical_st(a)

# --- Step 4: Display Final Comparison Table ---
println("\n--- VARS Analysis Results ---")
@printf("%-12s | %-10s | %-10s | %-20s\n", "Parameter", "Julia ST", "Analytic", "Difference")
println(repeat("-", 58))
for i in 1:d
    diff = abs(julia_results.ST[i] - analytical_results[i])
    @printf("x%-11d | %-10.4f | %-10.4f | %-20.6f\n", i, julia_results.ST[i], analytical_results[i], diff)
end

println("\n\nNext step: We will create a new test for G-VARS using a correlated function.")