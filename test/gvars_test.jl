# test/gvars_simple_test.jl

using Pkg
Pkg.activate("") # Activate the project environment from the test folder

# Make sure your VARS.jl source file is accessible
include("../src/VARS.jl")

using OrderedCollections
using Statistics
using LinearAlgebra
using Printf

println("--- G-VARS Simplified Test (Correlated, All-Normal): Julia vs. Analytical ---")

# --- 1. Define the Simplified Correlated Test Problem ---

# A simple linear model: Y = 2*X₁ + 5*X₂
function linear_model_batch(X::AbstractMatrix)
    a = [2.0, 5.0]
    return (a' * X)'
end

# Analytical solution for First-Order indices (S₁) of a correlated linear model
function linear_model_analytical_s1(parameters::OrderedDict, corr_mat::Matrix, a::Vector)
    d = length(parameters)
    param_defs = collect(values(parameters))
    
    variances = map(1:d) do i
        dist, _, _ = VARS._get_distribution_and_stats(param_defs[i])
        var(dist)
    end
    
    var_y = 0.0
    for i in 1:d
        var_y += a[i]^2 * variances[i]
        for j in (i+1):d
            cov_ij = corr_mat[i, j] * sqrt(variances[i] * variances[j])
            var_y += 2 * a[i] * a[j] * cov_ij
        end
    end
    
    s1_indices = (a.^2 .* variances) ./ var_y
    return s1_indices
end

# --- 2. Problem Definition ---

a = [2.0, 5.0]
d = length(a)

# All parameters are Normal distributions to simplify the Nataf transformation
parameters = OrderedDict(
    "x1" => (p1=0.0, p2=1.0, p3=nothing, dist="norm"), # Normal(0, 1)
    "x2" => (p1=2.0, p2=0.5, p3=nothing, dist="norm")  # Normal(2, 0.5)
)

# A simple correlation matrix
corr_mat = [1.0  0.7;
            0.7  1.0]

# G-VARS parameters
N = 256
num_dir_samples = 100
delta_h = 0.1

# --- 3. Run the Analysis ---

println("\nStep 1: Generating G-VARS samples (with Nataf transformation)...")
# We are now running the full G-VARS with the fictive correlation
problem = VARS.sample(parameters, N, delta_h, 
                      corr_mat=corr_mat, 
                      num_dir_samples=num_dir_samples, 
                      seed=789,
                      use_fictive_corr=true)

println("Step 2: Running the model function...")
Y = linear_model_batch(problem.X)

println("Step 3: Running analyses...")
println("  - Running Julia G-VARS implementation via dispatcher...")
julia_results = VARS.analyse(problem, vec(Y))

println("  - Calculating analytical solution...")
analytical_results = linear_model_analytical_s1(parameters, corr_mat, a)

# --- 4. Display Final Comparison Table ---
println("\n--- G-VARS Analysis Results ---")
@printf("%-12s | %-12s | %-12s | %-20s\n", "Parameter", "Julia ST", "Analytic S1", "Difference")
println(repeat("-", 62))
for i in 1:d
    param_name = collect(keys(parameters))[i]
    diff = abs(julia_results.ST[i] - analytical_results[i])
    @printf("%-12s | %-12.4f | %-12.4f | %-20.6f\n", param_name, julia_results.ST[i], analytical_results[i], diff)
end