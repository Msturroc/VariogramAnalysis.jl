using VariogramAnalysis
using DataFrames
using QuasiMonteCarlo
using Printf
using Statistics
using PyCall

# --- Model and Analytical Solution Definition (unchanged) ---
function sobol_g_batch(X, a)
    d = size(X, 1)
    n = size(X, 2)
    result = ones(n)
    for i in 1:d
        result .*= (abs.(4 .* X[i,:] .- 2) .+ a[i]) ./ (1 .+ a[i])
    end
    return result
end

function sobol_g_analytical_st(a::Vector)
    d = length(a)
    Vi = @. 1 / (3 * (1 + a)^2)
    total_variance = prod(Vi .+ 1) - 1
    st_indices = zeros(d)
    for i in 1:d
        st_indices[i] = (Vi[i] / total_variance) * (total_variance + 1) / (Vi[i] + 1)
    end
    return st_indices ./ sum(st_indices)
end

# --- Main Analysis Function ---
function run_dvars_benchmark()
    dimensions_to_test = [4, 8]
    a_full = [0, 0.5, 3, 9, 99, 99, 99, 99]

    for d in dimensions_to_test
        println("\n" * "="^80)
        println("--- Running D-VARS Benchmark for d = $d ---")
        println("="^80)

        # --- THE FAIRNESS REFINEMENT: Use one sample size for all methods ---
        num_samples = d == 4 ? 1000 : 2000
        
        # --- Step 1: Generate ONE Shared Dataset ---
        println("Generating one shared dataset of $num_samples samples...")
        a = a_full[1:d]
        lb = zeros(d); ub = ones(d)
        X_sample = QuasiMonteCarlo.sample(num_samples, lb, ub, SobolSample())
        y_sample = sobol_g_batch(X_sample, a)
        df_data = DataFrame(X_sample', :auto)
        df_data[!, :y] = y_sample
        println("Data generation complete.")

        # --- Step 2: Run All Analyses on the SAME data ---
        ratios_jl_kriging = VariogramAnalysis.dvars_sensitivities(df_data, :y)[2]
        ratios_jl_robust = VariogramAnalysis.dvars_sensitivities_robust(df_data, :y)[2]
        
        # Pass the generated X and y matrices directly to the PCE function
        ratios_pce = VariogramAnalysis.pce_sensitivities(X_sample, y_sample, lb, ub)
        
        # Python Analysis (also uses the same data via df_data)
        ratios_py = fill(NaN, d)
        try
            py_dvars = pyimport("varstool.sensitivity_analysis.dvars_funcs")
            pd = pyimport("pandas")
            data_dict = Dict(String(col) => df_data[!, col] for col in names(df_data))
            df_pandas = pd.DataFrame(data_dict)
            _, ratios_py, _, _ = py_dvars.calc_sensitivities(df_pandas, "y")
        catch e
            println("ERROR during Python execution: $e")
        end

        # --- Step 3: Compare All Results ---
        analytical_st_ratios = sobol_g_analytical_st(a)

        println("\n--- FINAL COMPARISON for d = $d (using $num_samples samples) ---")
        @printf("%-10s | %-15s | %-18s | %-15s | %-15s | %-15s\n", "Parameter", "Julia (PCE)(%)", "Julia (Robust DE)(%)", "Julia (Kriging)(%)", "Python ST (%)", "Analytical (%)")
        println(repeat("-", 115))
        for i in 1:d
            py_ratio_val = isnan(ratios_py[i]) ? "Error" : @sprintf("%.2f", ratios_py[i] * 100)
            @printf("x%-9d | %-15.2f | %-18.2f | %-15.2f | %-15s | %-15.2f\n", i, ratios_pce[i] * 100, ratios_jl_robust[i] * 100, ratios_jl_kriging[i] * 100, py_ratio_val, analytical_st_ratios[i] * 100)
        end
    end
end

# --- Run the benchmark ---
run_dvars_benchmark()