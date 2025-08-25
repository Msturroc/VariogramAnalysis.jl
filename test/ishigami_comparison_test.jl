using Pkg
Pkg.activate("") # Activates the current environment

include("../src/VARS.jl")

using OrderedCollections
using Statistics
using LinearAlgebra
using Printf
using DataFrames
using PyCall

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
    import sys
    import numpy as np
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
bootstrap_flag_exp1 = true
bootstrap_size_exp1 = 100
bootstrap_ci_exp1 = 0.9
grouping_flag_exp1 = true
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
bootstrap_flag_exp2 = true
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
bootstrap_flag_exp3 = true
bootstrap_size_exp3 = 100
bootstrap_ci_exp3 = 0.9
grouping_flag_exp3 = true # Notebook has true here
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
bootstrap_flag_exp4 = true
bootstrap_size_exp4 = 100
bootstrap_ci_exp4 = 0.9
grouping_flag_exp4 = true
num_grps_exp4 = 2
fictive_mat_flag_exp4 = true
report_verbose_exp4 = true

# --- Helper function to run and compare an experiment ---
function run_and_compare_experiment(exp_name::String, exp_type::String, julia_params::OrderedDict, python_params::Dict,
                                    num_stars::Int, delta_h::Float64, seed::Int,
                                    notebook_ivars::Dict, notebook_st::Dict; 
                                    corr_mat::Union{Matrix{Float64}, Nothing}=nothing, 
                                    py_corr_mat::Union{PyObject, Nothing, Matrix}=nothing,
                                    num_dir_samples::Union{Int, Nothing}=nothing,
                                    ivars_scales::Tuple=(0.1, 0.3, 0.5),
                                    sampler::String="lhs",
                                    slice_size::Union{Int, Nothing}=nothing,
                                    bootstrap_flag::Bool=false,
                                    bootstrap_size::Int=0,
                                    bootstrap_ci::Float64=0.0,
                                    grouping_flag::Bool=false,
                                    num_grps::Int=0,
                                    fictive_mat_flag::Bool=false,
                                    report_verbose::Bool=false)

    println("\n--- Running $(exp_name) ---")

    # --- Julia Execution ---
    println("  Running Julia VARS/G-VARS...")
    julia_problem_kwargs = Dict{Symbol, Any}(
        :seed => seed,
    )
    # Map Python sampler names to Julia sampler types
    julia_sampler_type = if sampler == "plhs"
        "lhs" # QuasiMonteCarlo.jl doesn't have 'plhs', use 'lhs'
    else
        sampler
    end
    julia_problem_kwargs[:sampler_type] = julia_sampler_type

    if exp_type == "GVARS"
        julia_problem_kwargs[:corr_mat] = corr_mat
        julia_problem_kwargs[:num_dir_samples] = num_dir_samples
        julia_problem_kwargs[:use_fictive_corr] = fictive_mat_flag # Assuming use_fictive_corr is derived from fictive_mat_flag
    end

    julia_problem = VARS.sample(julia_params, num_stars, delta_h; julia_problem_kwargs...)
    julia_Y = [ishigami_julia(x) for x in eachcol(julia_problem.X)]
    julia_results = VARS.analyse(julia_problem, julia_Y)
    julia_st = julia_results.ST

    # --- Python Execution ---
    println("  Running Python varstool $(exp_type)...")
    python_exp_kwargs_symbols = Dict{Symbol, Any}()
    python_exp_kwargs_strings = Dict{String, Any}(
        "seed" => seed,
        "ivars_scales" => ivars_scales,
        "sampler" => sampler,
        "bootstrap_flag" => bootstrap_flag,
        "bootstrap_size" => bootstrap_size,
        "bootstrap_ci" => bootstrap_ci,
        "grouping_flag" => grouping_flag,
        "num_grps" => num_grps,
        "report_verbose" => report_verbose
    )
    if exp_type == "GVARS"
        python_exp_kwargs_strings["corr_mat"] = py_corr_mat
        python_exp_kwargs_strings["num_dir_samples"] = num_dir_samples
        python_exp_kwargs_strings["slice_size"] = slice_size
        python_exp_kwargs_strings["fictive_mat_flag"] = fictive_mat_flag
    end

    for (k, v) in python_exp_kwargs_strings
        python_exp_kwargs_symbols[Symbol(k)] = v
    end

    py_ivars, py_st = try
        py"run_python_experiment"(exp_type, python_params, num_stars, delta_h, py_ishigami_model; python_exp_kwargs_symbols...)
    catch e
        if e isa PyCall.PyError
            @error "A Python error occurred in run_python_experiment. Details:"
            println(e)
        end
        rethrow(e)
    end
    
    # Extract Python IVARS for the largest scale
    # Get the specific row as a Python Series
    py_ivars_row_series = pycall(py_ivars.loc.__getitem__, PyObject, ivars_scales[end])
    
    # Convert the Python Series to a Python dictionary
    py_ivars_row_dict = py_ivars_row_series.to_dict()
    
    # Convert the Python dictionary to a Julia Dict
    py_ivars_dict = convert(Dict{String, Any}, py_ivars_row_dict)

    # Extract Python ST
    # The `py_st` object is a pandas Series, so we can iterate over its items directly.
    py_st_dict = Dict(String(k) => v for (k, v) in py_st.items())

    # --- Comparison ---
    println("\n  --- Comparison for $(exp_name) ---")

    

    # ST Comparison
    println("\n  ST (Sobol Total-Order Effect):")
    @printf("% -10s | % -10s | % -10s | % -10s\n", "Param", "Julia", "Python", "Notebook")
    println(repeat("-", 48))
    for (i, p) in enumerate(keys(julia_params))
        julia_val = julia_st[i]
        python_val = get(py_st_dict, p, NaN)
        notebook_val = get(notebook_st, p, NaN)
        @printf("% -10s | % -10.4f | % -10.4f | % -10.4f\n", p, julia_val, python_val, notebook_val)
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

# --- Run all experiments ---
run_and_compare_experiment(
    "Experiment 1: VARS, Uncorrelated, Uniform", "VARS",
    julia_params_exp1, python_params_exp1,
    num_stars_exp1, delta_h_exp1, seed_exp1,
    notebook_ivars_exp1, notebook_st_exp1;
    ivars_scales=ivars_scales_exp1, sampler=sampler_exp1,
    bootstrap_flag=bootstrap_flag_exp1, bootstrap_size=bootstrap_size_exp1, bootstrap_ci=bootstrap_ci_exp1,
    grouping_flag=grouping_flag_exp1, num_grps=num_grps_exp1, report_verbose=report_verbose_exp1
)

run_and_compare_experiment(
    "Experiment 2: G-VARS, Uncorrelated, Uniform", "GVARS",
    julia_params_exp2, python_params_exp2,
    num_stars_exp2, delta_h_exp2, seed_exp2,
    notebook_ivars_exp2, notebook_st_exp2;
    corr_mat=corr_mat_exp2, py_corr_mat=py_corr_mat_exp2, num_dir_samples=num_dir_samples_exp2,
    ivars_scales=ivars_scales_exp2, sampler=sampler_exp2, slice_size=slice_size_exp2,
    bootstrap_flag=bootstrap_flag_exp2, bootstrap_size=bootstrap_size_exp2, bootstrap_ci=bootstrap_ci_exp2,
    grouping_flag=grouping_flag_exp2, num_grps=num_grps_exp2, fictive_mat_flag=fictive_mat_flag_exp2,
    report_verbose=report_verbose_exp2
)

run_and_compare_experiment(
    "Experiment 3: G-VARS, Correlated, Uniform", "GVARS",
    julia_params_exp3, python_params_exp3,
    num_stars_exp3, delta_h_exp3, seed_exp3,
    notebook_ivars_exp3, notebook_st_exp3;
    corr_mat=corr_mat_exp3, py_corr_mat=py_corr_mat_exp3, num_dir_samples=num_dir_samples_exp3,
    ivars_scales=ivars_scales_exp3, sampler=sampler_exp3, slice_size=slice_size_exp3,
    bootstrap_flag=bootstrap_flag_exp3, bootstrap_size=bootstrap_size_exp3, bootstrap_ci=bootstrap_ci_exp3,
    grouping_flag=grouping_flag_exp3, num_grps=num_grps_exp3, fictive_mat_flag=fictive_mat_flag_exp3,
    report_verbose=report_verbose_exp3
)

run_and_compare_experiment(
    "Experiment 4: G-VARS, Correlated, Non-uniform", "GVARS",
    julia_params_exp4, python_params_exp4,
    num_stars_exp4, delta_h_exp4, seed_exp4,
    notebook_ivars_exp4, notebook_st_exp4;
    corr_mat=corr_mat_exp4, py_corr_mat=py_corr_mat_exp4, num_dir_samples=num_dir_samples_exp4,
    ivars_scales=ivars_scales_exp4, sampler=sampler_exp4, slice_size=slice_size_exp4,
    bootstrap_flag=bootstrap_flag_exp4, bootstrap_size=bootstrap_size_exp4, bootstrap_ci=bootstrap_ci_exp4,
    grouping_flag=grouping_flag_exp4, num_grps=num_grps_exp4, fictive_mat_flag=fictive_mat_flag_exp4,
    report_verbose=report_verbose_exp4
)

println("\n--- Comparison Complete ---")
