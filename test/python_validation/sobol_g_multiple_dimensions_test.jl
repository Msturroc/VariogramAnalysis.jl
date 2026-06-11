using VariogramAnalysis
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
        from collections import OrderedDict

        current_dimension = 4
        current_a_values = [0, 0.5, 3, 9]

        def sobol_g_python(x, a=None):
            if a is None: a = current_a_values
            if isinstance(x, pd.Series): x_vals = x.values
            elif isinstance(x, pd.DataFrame): x_vals = x.iloc[0].values
            else: x_vals = np.asarray(x)
            if len(x_vals) != len(a): raise ValueError(f'x must have exactly {len(a)} arguments, got {len(x_vals)}')
            result = 1.0
            for i in range(len(x_vals)):
                result *= (abs(4 * x_vals[i] - 2) + a[i]) / (1 + a[i])
            return result

        def set_sobol_g_dimension(d, a_vals):
            global current_dimension, current_a_values
            current_dimension = d
            current_a_values = a_vals[:d]
        """
        println("Sobol-G test: Python helper functions defined successfully.")
    catch e
        @error "Sobol-G test: Failed to define Python helper functions."
        rethrow(e)
    end

    println("\n--- VARS Bootstrap Analysis (Sobol-G): Julia vs. Python vs. Analytical ---")

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

    function sobol_g_julia(x::AbstractVector, a::Vector)
        if length(x) != length(a) throw(ArgumentError("`x` must have exactly $(length(a)) arguments.")) end
        result = 1.0
        for i in 1:length(x)
            result *= (abs(4 * x[i] - 2) + a[i]) / (1 + a[i])
        end
        return result
    end

    dimensions_to_test = [4, 10, 20]
    a_full = [0, 0.5, 3, 9, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99] 

    for d in dimensions_to_test
        println("\n" * repeat("=", 80))
        println("--- RUNNING COMPARISON FOR d = $d ---")
        println(repeat("=", 80))

        a = a_full[1:d]
        parameters_julia = OrderedDict("x$i" => (p1=0.0, p2=1.0, p3=nothing, dist="unif") for i in 1:d)
        N = 256 
        delta_h = 0.1
        num_boot_replicates = 50
        seed = 123

        println("\nStep 1: Running Julia Analysis...")
        Random.seed!(seed)
        problem_jl = VariogramAnalysis.sample(parameters_julia, N, delta_h, seed=seed)
        Y_jl = [sobol_g_julia(x, a) for x in eachcol(problem_jl.X)]
        
        compute_st_closure = (Y_b, X_b, X_norm_b, info_b, N_b, d_b, delta_h_b) -> VariogramAnalysis.analyse(problem_jl.method, X_b, X_norm_b, info_b, parameters_julia, N_b, d_b, delta_h_b, Y_b)
        julia_boot_results = VariogramAnalysis.VARSBootstrap.bootstrap_st!(compute_st_closure, Y_jl, problem_jl.X, problem_jl.X_norm, problem_jl.info, problem_jl.N, problem_jl.d, problem_jl.delta_h; num_boot=num_boot_replicates, seed=seed)
        julia_st_boot = julia_boot_results.st_boot

        println("\nStep 2: Running Python Analysis...")
        py"set_sobol_g_dimension"(d, a_full)
        
        py"""
        from collections import OrderedDict
        from varstool import VARS, Model
        model_instance = Model(sobol_g_python)
        ordered_params = OrderedDict()
        for i in range(1, $d + 1): ordered_params[f"x{i}"] = [0.0, 1.0]
        py_experiment = VARS(parameters=ordered_params, num_stars=$N, delta_h=$delta_h, model=model_instance, seed=$seed, sampler="lhs", bootstrap_flag=True, bootstrap_size=$num_boot_replicates, bootstrap_ci=0.9, report_verbose=False)
        py_experiment.run_online()
        cols = [f"x{i}" for i in range(1, $d + 1)]
        df = results_storage['result_bs_sobol']
        results_storage['result_bs_sobol'] = df.reindex(columns=cols)
        """
        
        py_st_boot_df = py"results_storage"["result_bs_sobol"]
        py_st_boot = py_st_boot_df.to_numpy()
        analytical_results = sobol_g_analytical_st(a)

        println("\n--- VARS Bootstrap Analysis Results (Sobol-G, d=$d, $(num_boot_replicates) replicates) ---")
        @printf("%-10s | %-15s | %-15s | %-15s\n", "Parameter", "Julia Mean ST", "Python Mean ST", "Analytical ST")
        println(repeat("-", 65))
        for i in 1:d
            jl_mean = mean(view(julia_st_boot, :, i))
            py_mean = mean(view(py_st_boot, :, i))
            @printf("x%-9d | %-15.4f | %-15.4f | %-15.4f\n", i, jl_mean, py_mean, analytical_results[i])
            
            # --- THIS IS THE NEW TEST ASSERTION ---
            # Test that Julia and Python results are approximately equal with a tolerance of 0.1
            @test jl_mean ≈ py_mean atol=0.1
        end
    end
    println("\n--- All Sobol-G Comparisons Complete ---")
end