using VARS
using OrderedCollections
using Statistics
using LinearAlgebra
using Printf
using DataFrames
using PyCall
using Random
using Test # Make sure Test is imported

with_patched_varstool() do
    try
        py"""
        import numpy as np
        import pandas as pd
        from varstool import VARS, GVARS, Model
        from varstool.sensitivity_analysis import vars_funcs
        from itertools import combinations

        def section_df_fixed(df, delta_h):
            df_reset = df.reset_index(drop=True)
            def pairs_h(iterable):
                interval = range(min(iterable), max(iterable) - min(iterable))
                return {key + 1: [j for j in combinations(iterable, 2) if abs(j[0] - j[1]) == key + 1] for key in interval}
            pairs = pairs_h(df_reset.index)
            df_values = df_reset.to_numpy()
            pair_dict = {}
            for h, idx in pairs.items():
                if not idx:
                    empty_multi_index = pd.MultiIndex.from_tuples([], names=['h', 'pair_ind'])
                    pair_dict[h * delta_h] = pd.DataFrame(columns=[0, 1], index=empty_multi_index)
                else:
                    multi_index_tuples = [(h * delta_h, str(idx_tup)) for idx_tup in idx]
                    multi_index = pd.MultiIndex.from_tuples(multi_index_tuples, names=['h', 'pair_ind'])
                    pair_dict[h * delta_h] = pd.DataFrame(data=[[df_values[idx_tup[0]], df_values[idx_tup[1]]] for idx_tup in idx], index=multi_index, columns=[0, 1])
            if not pair_dict:
                return pd.DataFrame(columns=[0, 1], index=pd.MultiIndex.from_tuples([], names=['h', 'pair_ind']))
            return pd.concat(list(pair_dict.values()))
        vars_funcs.section_df = section_df_fixed

        def ishigami_python(x, a=7, b=0.1):
            if isinstance(x, pd.Series): x_vals = x.values
            elif isinstance(x, pd.DataFrame): x_vals = x.iloc[0].values
            else: x_vals = np.asarray(x)
            return np.sin(x_vals[0]) + a*(np.sin(x_vals[1])**2) + b*(x_vals[2]**4)*np.sin(x_vals[0])
        """
        println("Ishigami test: Python helper functions defined successfully.")
    catch e
        @error "Ishigami test: Failed to set up Python environment or define helper functions."
        rethrow(e)
    end

    println("\n--- Ishigami Function VARS/G-VARS Comparison: Julia vs. Python ---")

    function ishigami_julia(x::AbstractVector; a=7, b=0.1)
        if length(x) != 3 throw(ArgumentError("`x` must have exactly three arguments.")) end
        return sin(x[1]) + a*(sin(x[2])^2) + b*(x[3]^4)*sin(x[1])
    end

    py_ishigami_model = py"ishigami_python"
    include("ishigami_test_parameters.jl")

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

        println("  Running Julia VARS/G-VARS with $(num_boot_replicates) bootstrap replicates...")
        Random.seed!(seed)
        julia_problem = VARS.sample(julia_params, num_stars, delta_h; seed=seed, sampler_type=sampler, corr_mat=corr_mat, num_dir_samples=num_dir_samples, use_fictive_corr=fictive_mat_flag)
        julia_Y = [ishigami_julia(x) for x in eachcol(julia_problem.X)]
        compute_st_closure = (Y_b, X_b, X_norm_b, info_b, N_b, d_b, delta_h_b) -> VARS.analyse(julia_problem.method, X_b, X_norm_b, info_b, julia_params, N_b, d_b, delta_h_b, Y_b)
        julia_boot_results = VARS.VARSBootstrap.bootstrap_st!(compute_st_closure, julia_Y, julia_problem.X, julia_problem.X_norm, julia_problem.info, julia_problem.N, julia_problem.d, julia_problem.delta_h; num_boot=num_boot_replicates, seed=seed)
        julia_st_boot = julia_boot_results.st_boot

        println("  Running Python varstool $(exp_type) with $(num_boot_replicates) bootstrap replicates...")
        python_exp_kwargs = Dict{String, Any}("seed" => seed, "ivars_scales" => ivars_scales, "sampler" => sampler, "bootstrap_flag" => true, "bootstrap_size" => num_boot_replicates, "bootstrap_ci" => 0.9, "grouping_flag" => false, "report_verbose" => report_verbose)
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

        println("\n  --- Bootstrap Comparison for $(exp_name) ---")
        @printf("%-8s | %-12s | %-12s | %-12s | %-12s\n", "Param", "Metric", "Julia", "Python", "Notebook (p.e.)")
        println(repeat("-", 64))
        
        param_keys = collect(keys(julia_params))
        for i in 1:length(param_keys)
            p = param_keys[i]
            jl_mean = mean(view(julia_st_boot, :, i))
            py_mean = mean(view(py_st_boot, :, i))
            notebook_val = get(notebook_st, p, NaN)
            @printf("%-8s | %-12s | %-12.4f | %-12.4f | %-12.4f\n", p, "Mean ST", jl_mean, py_mean, notebook_val)
            
            # --- THIS IS THE NEW TEST ASSERTION ---
            # Test that Julia and Python results are approximately equal with a tolerance of 0.1
            @test jl_mean ≈ py_mean atol=0.1
        end
    end

    run_and_compare_experiment("Experiment 1: VARS, Uncorrelated, Uniform", "VARS", julia_params_exp1, python_params_exp1, num_stars_exp1, delta_h_exp1, seed_exp1, notebook_st_exp1; sampler=sampler_exp1, ivars_scales=ivars_scales_exp1, report_verbose=report_verbose_exp1)
    run_and_compare_experiment("Experiment 2: G-VARS, Uncorrelated, Uniform", "GVARS", julia_params_exp2, python_params_exp2, num_stars_exp2, delta_h_exp2, seed_exp2, notebook_st_exp2; corr_mat=corr_mat_exp2, py_corr_mat=py_corr_mat_exp2, num_dir_samples=num_dir_samples_exp2, ivars_scales=ivars_scales_exp2, sampler=sampler_exp2, slice_size=slice_size_exp2, fictive_mat_flag=fictive_mat_flag_exp2, report_verbose=report_verbose_exp2)
    run_and_compare_experiment("Experiment 3: G-VARS, Correlated, Uniform", "GVARS", julia_params_exp3, python_params_exp3, num_stars_exp3, delta_h_exp3, seed_exp3, notebook_st_exp3; corr_mat=corr_mat_exp3, py_corr_mat=py_corr_mat_exp3, num_dir_samples=num_dir_samples_exp3, ivars_scales=ivars_scales_exp3, sampler=sampler_exp3, slice_size=slice_size_exp3, fictive_mat_flag=fictive_mat_flag_exp3, report_verbose=report_verbose_exp3)
    run_and_compare_experiment("Experiment 4: G-VARS, Correlated, Non-uniform", "GVARS", julia_params_exp4, python_params_exp4, num_stars_exp4, delta_h_exp4, seed_exp4, notebook_st_exp4; corr_mat=corr_mat_exp4, py_corr_mat=py_corr_mat_exp4, num_dir_samples=num_dir_samples_exp4, ivars_scales=ivars_scales_exp4, sampler=sampler_exp4, slice_size=slice_size_exp4, fictive_mat_flag=fictive_mat_flag_exp4, report_verbose=report_verbose_exp4)

    println("\n--- Ishigami Comparison Complete ---")
end