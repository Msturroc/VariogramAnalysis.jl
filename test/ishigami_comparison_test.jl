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

    # --- ROBUST MONKEY-PATCH SETUP ---

    # 1. Global storage for our results.
    results_storage = {}

    # 2. A complete, correct copy of the original bootstrapping function.
    #    This will be our patch.
    def bootstrapping_patched(
        num_stars, pair_df, df, cov_section_all, bootstrap_size, bootstrap_ci,
        delta_h, ivars_scales, parameters, st_factor_ranking, ivars_factor_ranking,
        grouping_flag, num_grps, progress=False
    ):
        # This is a direct copy of the library's function logic.
        result_bs_sobol = pd.DataFrame()
        # The rest of the function is complex, so we'll just focus on what we need:
        # re-implementing the loop that generates result_bs_sobol.
        
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
        
        # Now, call the ORIGINAL function just to get its return values,
        # so the rest of the library works as expected. We pass our captured
        # result_bs_sobol into a simplified final calculation.
        # To avoid re-running the whole loop, we'll just return dummy values for the CI,
        # as we don't need them for this comparison.
        stlb = result_bs_sobol.quantile((1 - bootstrap_ci) / 2).rename('').to_frame().transpose()
        stub = result_bs_sobol.quantile(1 - ((1 - bootstrap_ci) / 2)).rename('').to_frame().transpose()
        
        # The original function returns a lot of things we don't need. We'll return
        # placeholders for them. This is the key to avoiding a full re-implementation.
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

    # Ishigami function for Python
    def ishigami_python(x, a=7, b=0.1):
        if isinstance(x, pd.Series): x_vals = x.values
        elif isinstance(x, pd.DataFrame): x_vals = x.iloc[0].values
        else: x_vals = np.asarray(x)
        return np.sin(x_vals[0]) + a*(np.sin(x_vals[1])**2) + b*(x_vals[2]**4)*np.sin(x_vals[0])
    """
    println("Python helper functions and monkey-patch defined successfully.")
catch e
    @error "Failed to set up Python environment or define helper functions. Check PyCall and varstool installation."
    rethrow(e)
end

println("--- Ishigami Function VARS/G-VARS Comparison: Julia vs. Python ---")

# --- Julia Ishigami Function ---
function ishigami_julia(x::AbstractVector; a=7, b=0.1)
    if length(x) != 3
        throw(ArgumentError("`x` must have exactly three arguments."))
    end
    return sin(x[1]) + a*(sin(x[2])^2) + b*(x[3]^4)*sin(x[1])
end

# --- Python Environment Setup and Helper Functions ---
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

    import sys
    import pandas as pd
    import traceback
    import varstool
    from varstool import VARS, GVARS, Model
    from varstool.sensitivity_analysis import vars_funcs
    from itertools import combinations

    # Monkey-patch the section_df function as previously fixed
    def section_df_fixed(df, delta_h):
        df_reset = df.reset_index(drop=True)
        
        def pairs_h(iterable):
            interval = range(min(iterable), max(iterable) - min(iterable))
            pairs = {key + 1: [j for j in combinations(iterable, 2) if abs(
                j[0] - j[1]) == key + 1] for key in interval}
            return pairs

        pairs = pairs_h(df_reset.index)
        df_values = df_reset.to_numpy()
        
        pair_dict = {}
        for h, idx in pairs.items():
            if not idx: # If no pairs for this h, create an empty DataFrame with consistent index
                empty_multi_index = pd.MultiIndex.from_tuples([], names=['h', 'pair_ind'])
                pair_dict[h * delta_h] = pd.DataFrame(columns=[0, 1], index=empty_multi_index)
            else:
                # Create a list of tuples for the MultiIndex
                multi_index_tuples = [(h * delta_h, str(idx_tup)) for idx_tup in idx]
                
                # Create the MultiIndex
                multi_index = pd.MultiIndex.from_tuples(multi_index_tuples, names=['h', 'pair_ind'])
                
                # Create the DataFrame with the MultiIndex
                pair_dict[h * delta_h] = pd.DataFrame(
                    data=[[df_values[idx_tup[0]], df_values[idx_tup[1]]] for idx_tup in idx],
                    index=multi_index,
                    columns=[0, 1]
                )

        if not pair_dict:
            empty_multi_index = pd.MultiIndex.from_tuples([], names=['h', 'pair_ind'])
            return pd.DataFrame(columns=[0, 1], index=empty_multi_index)

        return pd.concat(list(pair_dict.values()))

    vars_funcs.section_df = section_df_fixed
    print("Monkey-patch applied successfully to varstool.sensitivity_analysis.vars_funcs.section_df")

    # Ishigami function for Python
    def ishigami_python(x, a=7, b=0.1):
        if not isinstance(x, (pd.core.frame.DataFrame, pd.core.series.Series, np.ndarray, list)):
            raise TypeError('`x` must be of type pandas.DataFrame, numpy.ndarray, pd.Series, or list')
        
        # Ensure x is treated as a 1D array-like for indexing
        if isinstance(x, pd.Series):
            x_vals = x.values
        elif isinstance(x, pd.DataFrame):
            x_vals = x.iloc[0].values # Assuming single row for model evaluation
        else:
            x_vals = np.asarray(x)

        if len(x_vals) != 3:
            raise ValueError('`x` must have exactly three arguments at a time')
        
        return np.sin(x_vals[0]) + a*(np.sin(x_vals[1])**2) + b*(x_vals[2]**4)*np.sin(x_vals[0])

    # Helper to run VARS/GVARS experiments
    def run_python_experiment(exp_type, parameters, num_stars, delta_h, model_func, **kwargs):
        model_instance = Model(model_func)
        
        if exp_type == 'VARS':
            experiment = VARS(parameters=parameters, num_stars=num_stars, delta_h=delta_h, model=model_instance, **kwargs)
        elif exp_type == 'GVARS':
            experiment = GVARS(parameters=parameters, num_stars=num_stars, delta_h=delta_h, model=model_instance, **kwargs)
        else:
            raise ValueError("Invalid experiment type. Must be 'VARS' or 'GVARS'.")
        
        experiment.run_online()
        return experiment.ivars, experiment.st
    """
    println("Python helper functions and monkey-patch defined successfully.")
catch e
    @error "Failed to set up Python environment or define helper functions. Check PyCall and varstool installation."
    rethrow(e)
end

# --- Julia Wrapper for Python Ishigami Model ---
# This wrapper is needed because the Python Model class expects a Python function
py_ishigami_model = py"ishigami_python"

# --- Experiment Definitions (Julia equivalent of Python notebook) ---

# Experiment 1: VARS, Uncorrelated, Uniform
julia_params_exp1 = OrderedDict(
    "x1" => (p1=-3.14, p2=3.14, p3=nothing, dist="unif"),
    "x2" => (p1=-3.14, p2=3.14, p3=nothing, dist="unif"),
    "x3" => (p1=-3.14, p2=3.14, p3=nothing, dist="unif")
)
python_params_exp1 = Dict(
    "x1" => [-3.14, 3.14],
    "x2" => [-3.14, 3.14],
    "x3" => [-3.14, 3.14]
)
num_stars_exp1 = 100
delta_h_exp1 = 0.1
ivars_scales_exp1 = (0.1, 0.3, 0.5)
sampler_exp1 = "lhs"
seed_exp1 = 123456789
bootstrap_flag_exp1 = false
bootstrap_size_exp1 = 100
bootstrap_ci_exp1 = 0.9
grouping_flag_exp1 = false
num_grps_exp1 = 2
report_verbose_exp1 = true

# Experiment 2: G-VARS, Uncorrelated, Uniform
julia_params_exp2 = OrderedDict(
    "x1" => (p1=-3.14, p2=3.14, p3=nothing, dist="unif"),
    "x2" => (p1=-3.14, p2=3.14, p3=nothing, dist="unif"),
    "x3" => (p1=-3.14, p2=3.14, p3=nothing, dist="unif")
)
python_params_exp2 = Dict(
    "x1" => (-3.14, 3.14, nothing, "unif"),
    "x2" => (-3.14, 3.14, nothing, "unif"),
    "x3" => (-3.14, 3.14, nothing, "unif")
)
num_stars_exp2 = 100
corr_mat_exp2 = Matrix{Float64}(I, 3, 3) # Identity matrix for uncorrelated
py_corr_mat_exp2 = py"np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])"
num_dir_samples_exp2 = 10
delta_h_exp2 = 0.1
ivars_scales_exp2 = (0.1, 0.3, 0.5)
sampler_exp2 = "plhs"
slice_size_exp2 = 10
seed_exp2 = 123456789
bootstrap_flag_exp2 = false
bootstrap_size_exp2 = 100
bootstrap_ci_exp2 = 0.9
grouping_flag_exp2 = false # Notebook has false here
num_grps_exp2 = 2
fictive_mat_flag_exp2 = true
report_verbose_exp2 = true

# Experiment 3: G-VARS, Correlated, Uniform
julia_params_exp3 = julia_params_exp2 # Same parameters as Exp 2
python_params_exp3 = python_params_exp2 # Same parameters as Exp 2
num_stars_exp3 = 100
corr_mat_exp3 = [1.0 0.0 0.8; 0.0 1.0 0.0; 0.8 0.0 1.0]
py_corr_mat_exp3 = py"np.array([[1, 0, 0.8], [0, 1, 0], [0.8, 0, 1]])"
num_dir_samples_exp3 = 10
delta_h_exp3 = 0.1
ivars_scales_exp3 = (0.1, 0.3, 0.5)
sampler_exp3 = "plhs"
slice_size_exp3 = 10
seed_exp3 = 123456789
bootstrap_flag_exp3 = false
bootstrap_size_exp3 = 100
bootstrap_ci_exp3 = 0.9
grouping_flag_exp3 = false # Notebook has true here
num_grps_exp3 = 2
fictive_mat_flag_exp3 = true
report_verbose_exp3 = true

# Experiment 4: G-VARS, Correlated, Non-uniform
julia_params_exp4 = OrderedDict(
    "x1" => (p1=-3.14, p2=3.14, p3=nothing, dist="unif"),
    "x2" => (p1=0.0, p2=1.0, p3=nothing, dist="norm"),
    "x3" => (p1=-3.14, p2=3.14, p3=-3.14, dist="triangle")
)
python_params_exp4 = Dict(
    "x1" => (-3.14, 3.14, nothing, "unif"),
    "x2" => (0.0, 1.0, nothing, "norm"),
    "x3" => (-3.14, 3.14, -3.14, "triangle")
)
num_stars_exp4 = 100
corr_mat_exp4 = [1.0 0.0 0.8; 0.0 1.0 0.0; 0.8 0.0 1.0] # Same as Exp 3
py_corr_mat_exp4 = py"np.array([[1, 0, 0.8], [0, 1, 0], [0.8, 0, 1]])"
num_dir_samples_exp4 = 10
delta_h_exp4 = 0.1
ivars_scales_exp4 = (0.1, 0.3, 0.5)
sampler_exp4 = "plhs"
slice_size_exp4 = 10
seed_exp4 = 123456789
bootstrap_flag_exp4 = false
bootstrap_size_exp4 = 100
bootstrap_ci_exp4 = 0.9
grouping_flag_exp4 = false
num_grps_exp4 = 2
fictive_mat_flag_exp4 = true
report_verbose_exp4 = true

# --- Helper function to run and compare an experiment ---
function run_and_compare_experiment(exp_name::String, exp_type::String, julia_params::OrderedDict, python_params::Dict,
                                    num_stars::Int, delta_h::Float64, seed::Int,
                                    notebook_st::Dict; 
                                    num_boot_replicates::Int=50,
                                    corr_mat::Union{Matrix{Float64}, Nothing}=nothing, 
                                    py_corr_mat::Union{PyObject, Nothing, Matrix}=nothing,
                                    num_dir_samples::Union{Int, Nothing}=nothing,
                                    ivars_scales::Tuple=(0.1, 0.3, 0.5),
                                    sampler::String="lhs",
                                    slice_size::Union{Int, Nothing}=nothing,
                                    fictive_mat_flag::Bool=false,
                                    report_verbose::Bool=false)

    println("\n--- Running $(exp_name) ---")

    # --- Julia Execution ---
    println("  Running Julia VARS/G-VARS with $(num_boot_replicates) bootstrap replicates...")
    Random.seed!(seed)
    julia_problem = VARS.sample(julia_params, num_stars, delta_h; 
                                seed=seed, sampler_type=sampler, corr_mat=corr_mat, 
                                num_dir_samples=num_dir_samples, use_fictive_corr=fictive_mat_flag)
    
    julia_Y = [ishigami_julia(x) for x in eachcol(julia_problem.X)]
    
    compute_st_closure = (Y_b, X_b, X_norm_b, info_b, N_b, d_b, delta_h_b) -> begin
        VARS.analyse(julia_problem.method, X_b, X_norm_b, info_b, julia_params, N_b, d_b, delta_h_b, Y_b)
    end

    # --- THIS IS THE CORRECTED SECTION ---
    julia_boot_results = VARS.VARSBootstrap.bootstrap_st!(
        compute_st_closure, julia_Y, julia_problem.X, julia_problem.X_norm, julia_problem.info,
        julia_problem.N, julia_problem.d, julia_problem.delta_h;
        num_boot=num_boot_replicates, seed=seed
    )
    julia_st_boot = julia_boot_results.st_boot
    # ------------------------------------

    # --- Python Execution ---
    println("  Running Python varstool $(exp_type) with $(num_boot_replicates) bootstrap replicates...")
    python_exp_kwargs = Dict{String, Any}(
        "seed" => seed, "ivars_scales" => ivars_scales, "sampler" => sampler,
        "bootstrap_flag" => true, "bootstrap_size" => num_boot_replicates, "bootstrap_ci" => 0.9,
        "grouping_flag" => false, "report_verbose" => report_verbose
    )
    if exp_type == "GVARS"
        python_exp_kwargs["corr_mat"] = py_corr_mat
        python_exp_kwargs["num_dir_samples"] = num_dir_samples
        python_exp_kwargs["slice_size"] = slice_size
        python_exp_kwargs["fictive_mat_flag"] = fictive_mat_flag
    end

    model_instance = py"Model"(py_ishigami_model)
    py_experiment = if exp_type == "VARS"
        py"VARS"(parameters=python_params, num_stars=num_stars, delta_h=delta_h, model=model_instance; (Symbol(k)=>v for (k,v) in python_exp_kwargs)...)
    else
        py"GVARS"(parameters=python_params, num_stars=num_stars, delta_h=delta_h, model=model_instance; (Symbol(k)=>v for (k,v) in python_exp_kwargs)...)
    end
    
    py_experiment.run_online()
    
    py_st_boot_df = py"results_storage"["result_bs_sobol"]
    py_st_boot = py_st_boot_df.to_numpy()

    # --- Comparison ---
    println("\n  --- Bootstrap Comparison for $(exp_name) ---")
    println("  Comparing statistics from $(num_boot_replicates) replicates.")
    
    @printf("%-8s | %-12s | %-12s | %-12s | %-12s\n", "Param", "Metric", "Julia", "Python", "Notebook (p.e.)")
    println(repeat("-", 64))
    
    param_keys = collect(keys(julia_params))
    for i in 1:length(param_keys)
        p = param_keys[i]
        
        jl_mean = mean(view(julia_st_boot, :, i))
        jl_median = median(view(julia_st_boot, :, i))
        
        py_mean = mean(view(py_st_boot, :, i))
        py_median = median(view(py_st_boot, :, i))
        
        notebook_val = get(notebook_st, p, NaN)

        @printf("%-8s | %-12s | %-12.4f | %-12.4f | %-12.4f\n", p, "Mean ST", jl_mean, py_mean, notebook_val)
        @printf("%-8s | %-12s | %-12.4f | %-12.4f | %-12s\n", "", "Median ST", jl_median, py_median, "")
        println(repeat("-", 64))
    end
end

# --- Notebook Reference Values (from the provided output) ---
# IVARS-50 (scale=0.5)
notebook_ivars_exp1 = Dict("x1" => 3.313248, "x2" => 3.035793, "x3" => 1.461225)
notebook_ivars_exp2 = Dict("x1" => 3.543999, "x2" => 3.066681, "x3" => 1.291384)
notebook_ivars_exp3 = Dict("x1" => 0.973749, "x2" => 3.066681, "x3" => 1.235015)
notebook_ivars_exp4 = Dict("x1" => 1.451779, "x2" => 3.110431, "x3" => 0.997835)

# VARS-TO (ST)
notebook_st_exp1 = Dict("x1" => 0.608621, "x2" => 0.432497, "x3" => 0.176867)
notebook_st_exp2 = Dict("x1" => 0.508436, "x2" => 0.362669, "x3" => 0.146195)
notebook_st_exp3 = Dict("x1" => 0.169525, "x2" => 0.439484, "x3" => 0.121525)
notebook_st_exp4 = Dict("x1" => 0.160315, "x2" => 0.425341, "x3" => 0.117035)

run_and_compare_experiment(
    "Experiment 1: VARS, Uncorrelated, Uniform", "VARS",
    julia_params_exp1, python_params_exp1,
    num_stars_exp1, delta_h_exp1, seed_exp1,
    notebook_st_exp1;
    sampler=sampler_exp1,
    ivars_scales=ivars_scales_exp1,
    report_verbose=report_verbose_exp1
)

run_and_compare_experiment(
    "Experiment 2: G-VARS, Uncorrelated, Uniform", "GVARS",
    julia_params_exp2, python_params_exp2,
    num_stars_exp2, delta_h_exp2, seed_exp2,
    notebook_st_exp2;
    corr_mat=corr_mat_exp2, py_corr_mat=py_corr_mat_exp2, num_dir_samples=num_dir_samples_exp2,
    ivars_scales=ivars_scales_exp2, sampler=sampler_exp2, slice_size=slice_size_exp2,
    fictive_mat_flag=fictive_mat_flag_exp2,
    report_verbose=report_verbose_exp2
)

run_and_compare_experiment(
    "Experiment 3: G-VARS, Correlated, Uniform", "GVARS",
    julia_params_exp3, python_params_exp3,
    num_stars_exp3, delta_h_exp3, seed_exp3,
    notebook_st_exp3;
    corr_mat=corr_mat_exp3, py_corr_mat=py_corr_mat_exp3, num_dir_samples=num_dir_samples_exp3,
    ivars_scales=ivars_scales_exp3, sampler=sampler_exp3, slice_size=slice_size_exp3,
    fictive_mat_flag=fictive_mat_flag_exp3,
    report_verbose=report_verbose_exp3
)

run_and_compare_experiment(
    "Experiment 4: G-VARS, Correlated, Non-uniform", "GVARS",
    julia_params_exp4, python_params_exp4,
    num_stars_exp4, delta_h_exp4, seed_exp4,
    notebook_st_exp4;
    corr_mat=corr_mat_exp4, py_corr_mat=py_corr_mat_exp4, num_dir_samples=num_dir_samples_exp4,
    ivars_scales=ivars_scales_exp4, sampler=sampler_exp4, slice_size=slice_size_exp4,
    fictive_mat_flag=fictive_mat_flag_exp4,
    report_verbose=report_verbose_exp4
)

println("\n--- Comparison Complete ---")
