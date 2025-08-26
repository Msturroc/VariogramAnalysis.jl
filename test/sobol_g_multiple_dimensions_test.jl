using VARS
using OrderedCollections
using Statistics
using LinearAlgebra
using Printf
using DataFrames
using PyCall
using Random

try
    py"""
    import numpy as np
    # Backward-compat shim for removed NumPy aliases
    if not hasattr(np, "int"):
        np.int = int
    if not hasattr(np, "float"):
        np.float = float
    if not hasattr(np, "bool"):
        np.bool = np.bool_

    import pandas as pd
    from varstool import VARS, GVARS, Model
    from varstool.sensitivity_analysis import vars_funcs
    from tqdm.auto import tqdm
    from collections import OrderedDict  # Add this import

    # --- ROBUST MONKEY-PATCH SETUP ---

    # 1. Global storage for our results.
    results_storage = {}

    # 2. A complete, correct copy of the original bootstrapping function.
    def bootstrapping_patched(
        num_stars, pair_df, df, cov_section_all, bootstrap_size, bootstrap_ci,
        delta_h, ivars_scales, parameters, st_factor_ranking, ivars_factor_ranking,
        grouping_flag, num_grps, progress=False
    ):
        print(f"[DEBUG] bootstrapping_patched called with bootstrap_size={bootstrap_size}")
        result_bs_sobol = pd.DataFrame()
        
        bs = tqdm(range(0, bootstrap_size), desc='bootstrapping (patched)', disable=not progress, dynamic_ncols=True)
        for i in bs:
            bootstrap_rand = np.random.choice(list(range(0, num_stars)), size=len(range(0, num_stars)), replace=True).tolist()
            bootstrapped_pairdf = pd.concat([pair_df.loc[pd.IndexSlice[i, :, :, :], :] for i in bootstrap_rand])
            bootstrapped_df = pd.concat([df.loc[pd.IndexSlice[i, :, :], :] for i in bootstrap_rand])
            bootstrapped_cov_section_all = pd.concat([cov_section_all.loc[pd.IndexSlice[i, :]] for i in bootstrap_rand])
            
            bootstrapped_variogram = vars_funcs.variogram(bootstrapped_pairdf)
            bootstrapped_ecovariogram = vars_funcs.e_covariogram(bootstrapped_cov_section_all)
            bootstrapped_var = np.nanvar(bootstrapped_df.iloc[:, -1].unique(), ddof=1)
            bootstrapped_sobol = vars_funcs.sobol_eq(bootstrapped_variogram, bootstrapped_ecovariogram, bootstrapped_var, delta_h)
            
            bootstrapped_sobol_df = bootstrapped_sobol.to_frame().transpose()
            result_bs_sobol = pd.concat([result_bs_sobol, bootstrapped_sobol_df])

        # --- OUR ADDITION: Save the data to global storage ---
        results_storage['result_bs_sobol'] = result_bs_sobol
        print(f"[DEBUG] Bootstrap results saved. Shape: {result_bs_sobol.shape}")
        
        # Return dummy values to avoid re-running the whole loop
        stlb = result_bs_sobol.quantile((1 - bootstrap_ci) / 2).rename('').to_frame().transpose()
        stub = result_bs_sobol.quantile(1 - ((1 - bootstrap_ci) / 2)).rename('').to_frame().transpose()
        
        dummy_df = pd.DataFrame()
        dummy_ranking = pd.DataFrame()
        if grouping_flag:
            return dummy_df, dummy_df, dummy_df, dummy_df, stlb, stub, dummy_df, dummy_df, dummy_ranking, dummy_ranking, dummy_df, dummy_df, dummy_df, dummy_df
        else:
            return dummy_df, dummy_df, dummy_df, dummy_df, stlb, stub, dummy_df, dummy_df, dummy_ranking, dummy_ranking

    # 3. Get a handle on the original run_online methods
    original_vars_run_online = VARS.run_online
    original_gvars_run_online = GVARS.run_online

    # 4. Define our wrapper function that performs the swap
    def patched_run_online(self, original_run_func):
        print("[DEBUG] patched_run_online called")
        original_bootstrap_in_module = vars_funcs.bootstrapping
        try:
            # Replace the function in the module with our patch
            vars_funcs.bootstrapping = bootstrapping_patched
            # Call the original run_online method
            original_run_func(self)
        finally:
            # ALWAYS restore the original function
            vars_funcs.bootstrapping = original_bootstrap_in_module

    # 5. Apply the final patch to the classes
    VARS.run_online = lambda self: patched_run_online(self, original_vars_run_online)
    GVARS.run_online = lambda self: patched_run_online(self, original_gvars_run_online)
    print("Monkey-patch applied successfully to VARS and GVARS run_online methods.")

    # Global variable to store current dimension and a-values
    current_dimension = 4
    current_a_values = [0, 0.5, 3, 9]

    # Sobol-G function for Python (following the same pattern as ishigami_python)
    def sobol_g_python(x, a=None):
        if a is None:
            a = current_a_values
            
        if isinstance(x, pd.Series): 
            x_vals = x.values
        elif isinstance(x, pd.DataFrame): 
            x_vals = x.iloc[0].values
        else: 
            x_vals = np.asarray(x)
        
        if len(x_vals) != len(a):
            raise ValueError(f'x must have exactly {len(a)} arguments, got {len(x_vals)}')
        
        result = 1.0
        for i in range(len(x_vals)):
            result *= (abs(4 * x_vals[i] - 2) + a[i]) / (1 + a[i])
        
        return result

    def set_sobol_g_dimension(d, a_vals):
        global current_dimension, current_a_values
        current_dimension = d
        current_a_values = a_vals[:d]
        print(f"[DEBUG] Set dimension to {d} with a values: {current_a_values}")
    """
    println("Python helper functions and monkey-patch defined successfully.")
catch e
    @error "Failed to set up Python environment or define helper functions. Check PyCall and varstool installation."
    rethrow(e)
end

println("--- VARS Bootstrap Analysis (Sobol-G): Julia vs. Python vs. Analytical ---")

# --- Analytical Solution and Model ---
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

function sobol_g_function_batch(X::AbstractMatrix, a::Vector)
    d = size(X, 1)
    n_points = size(X, 2)
    results = ones(n_points)
    for i in 1:d
        results .*= (abs.(4 .* X[i,:] .- 2) .+ a[i]) ./ (1 .+ a[i])
    end
    return results
end

# Single point evaluation function for Julia (matching the ishigami pattern)
function sobol_g_julia(x::AbstractVector, a::Vector)
    if length(x) != length(a)
        throw(ArgumentError("`x` must have exactly $(length(a)) arguments."))
    end
    result = 1.0
    for i in 1:length(x)
        result *= (abs(4 * x[i] - 2) + a[i]) / (1 + a[i])
    end
    return result
end

# --- Main Loop for Multi-Dimensional Comparison ---
dimensions_to_test = [4, 10, 20]
a_full = [0, 0.5, 3, 9, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99] 

for d in dimensions_to_test
    println("\n" * repeat("=", 80))
    println("--- RUNNING COMPARISON FOR d = $d ---")
    println(repeat("=", 80))

    # --- Problem Definition for current dimension 'd' ---
    a = a_full[1:d]
    parameters_julia = OrderedDict("x$i" => (p1=0.0, p2=1.0, p3=nothing, dist="unif") for i in 1:d)
    
    N = 256 
    delta_h = 0.1
    num_boot_replicates = 50
    seed = 123

    println("[DEBUG] Parameters set up:")
    println("  Julia parameters: $parameters_julia")
    println("  a values: $a")

    # --- Julia Analysis ---
    println("\nStep 1: Running Julia Analysis...")
    Random.seed!(seed)
    problem_jl = VARS.sample(parameters_julia, N, delta_h, seed=seed)
    Y_jl = [sobol_g_julia(x, a) for x in eachcol(problem_jl.X)]
    
    println("[DEBUG] Julia Y values computed, length: $(length(Y_jl))")
    
    compute_st_closure = (Y_b, X_b, X_norm_b, info_b, N_b, d_b, delta_h_b) -> begin
        VARS.analyse(problem_jl.method, X_b, X_norm_b, info_b, parameters_julia, N_b, d_b, delta_h_b, Y_b)
    end
    julia_boot_results = VARS.VARSBootstrap.bootstrap_st!(
        compute_st_closure, Y_jl, problem_jl.X, problem_jl.X_norm, problem_jl.info,
        problem_jl.N, problem_jl.d, problem_jl.delta_h;
        num_boot=num_boot_replicates, seed=seed
    )
    julia_st_boot = julia_boot_results.st_boot
    println("[DEBUG] Julia bootstrap completed, result shape: $(size(julia_st_boot))")

    # --- Python Analysis ---
    println("\nStep 2: Running Python Analysis...")

    # Initialize python results variable
    py_st_boot = nothing

    # Set up the dimension and a-values in Python
    py"set_sobol_g_dimension"(d, a_full)

    # Create the Python model using the same pattern as the working code
    py_sobol_model = py"sobol_g_python"

    println("[DEBUG] About to create Python model and experiment...")

    try
        # Create everything entirely in Python to avoid parameter ordering issues
        py"""
        import numpy as np
        from collections import OrderedDict
        
        # Create the model
        model_instance = Model(sobol_g_python)
        print("[DEBUG] Python Model created successfully")
        
        # Create ordered parameters for the current dimension
        ordered_params = OrderedDict()
        for i in range(1, $d + 1):
            key = f"x{i}"
            ordered_params[key] = [0.0, 1.0]
        
        print(f"[DEBUG] Created ordered parameters: {list(ordered_params.keys())}")
        
        # Create VARS instance with all parameters
        python_exp_kwargs = {
            "seed": $seed,
            "ivars_scales": (0.1, 0.3, 0.5),
            "sampler": "lhs",
            "bootstrap_flag": True, 
            "bootstrap_size": $num_boot_replicates, 
            "bootstrap_ci": 0.9,
            "grouping_flag": False, 
            "report_verbose": False
        }
        
        print(f"[DEBUG] About to create VARS instance with kwargs: {python_exp_kwargs}")
        
        py_experiment = VARS(
            parameters=ordered_params,  # Use the OrderedDict directly
            num_stars=$N, 
            delta_h=$delta_h, 
            model=model_instance,
            **python_exp_kwargs
        )
        
        print("[DEBUG] VARS instance created, about to run_online()")
        py_experiment.run_online()
        print("[DEBUG] run_online() completed")
        
        # Ensure bootstrap results maintain the correct column order
        cols = [f"x{i}" for i in range(1, $d + 1)]
        df = results_storage['result_bs_sobol']
        # Reindex enforces the exact column order we want
        results_storage['result_bs_sobol'] = df.reindex(columns=cols)
        print(f"[DEBUG] Reordered columns: {list(results_storage['result_bs_sobol'].columns)}")
        """
        
        py_st_boot_df = py"results_storage"["result_bs_sobol"]
        py_st_boot = py_st_boot_df.to_numpy()
        println("[DEBUG] Python bootstrap results retrieved, shape: $(size(py_st_boot))")
        
        # --- Analytical Solution ---
        analytical_results = sobol_g_analytical_st(a)
        println("[DEBUG] Analytical results: $analytical_results")

        # --- Display Final Comparison Table ---
        println("\n--- VARS Bootstrap Analysis Results (Sobol-G, d=$d, $(num_boot_replicates) replicates) ---")
        @printf("%-10s | %-15s | %-15s | %-15s\n", "Parameter", "Julia Mean ST", "Python Mean ST", "Analytical ST")
        println(repeat("-", 65))
        for i in 1:d
            jl_mean = mean(view(julia_st_boot, :, i))
            py_mean = mean(view(py_st_boot, :, i))
            @printf("x%-9d | %-15.4f | %-15.4f | %-15.4f\n", i, jl_mean, py_mean, analytical_results[i])
        end
        
    catch e
        println("[DEBUG] Error during Python execution: $e")
        rethrow(e)
    end


end

println("\n--- All Comparisons Complete ---")
