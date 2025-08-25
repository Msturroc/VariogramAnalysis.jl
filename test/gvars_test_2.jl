# test/gvars_ishigami_test.jl

using Pkg
Pkg.activate("")

include("../src/VARS.jl")

using OrderedCollections
using Statistics
using LinearAlgebra
using Printf
using GlobalSensitivity, QuasiMonteCarlo, Distributions

println("--- G-VARS Definitive Test (Correlated, Non-Uniform Ishigami Function) ---")

# --- 1. Define the Ishigami Function & Problem Space ---

const ISHIGAMI_A = 7.0
const ISHIGAMI_B = 0.1

# Model function for gsa (takes N x d matrix)
function ishigami_model_for_gsa(X)
    return sin.(X[:,1]) .+ ISHIGAMI_A .* sin.(X[:,2]).^2 .+ ISHIGAMI_B .* (X[:,3].^4) .* sin.(X[:,1])
end

# Model function for VARS.jl (takes d x N matrix)
function ishigami_model_for_vars(X)
    return sin.(X[1,:]) .+ ISHIGAMI_A .* sin.(X[2,:]).^2 .+ ISHIGAMI_B .* (X[3,:].^4) .* sin.(X[1,:])
end

parameters = OrderedDict(
    "x1" => (p1=-π, p2=π, p3=nothing, dist="unif"),
    "x2" => (p1=0.0, p2=2.0, p3=nothing, dist="norm"),
    "x3" => (p1=-π, p2=π, p3=nothing, dist="unif")
)
d = length(parameters)

corr_mat = [1.0  0.4  0.0;
            0.4  1.0  0.7;
            0.0  0.7  1.0]

# --- 2. Calculate the "Ground Truth" ST via Brute-Force Monte Carlo ---

function calculate_ground_truth_st()
    println("\nStep 1: Calculating Ground Truth ST with massive Monte Carlo sample...")
    
    # --- MEMORY-ADJUSTED LINE ---
    # N_truth is now 2^18, so 2*N_truth is 2^19 = 524,288.
    # This should be manageable on a 64GB system.
    N_truth = 262144
    
    fictive_corr = VARS.map_to_fictive_corr(parameters, corr_mat)
    fictive_corr_pd = VARS._ensure_pos_def(fictive_corr)
    
    uniform_samples_AB = QuasiMonteCarlo.sample(2 * N_truth, d, SobolSample(R=OwenScramble(base=2)))
    
    uniform_A = uniform_samples_AB[:, 1:N_truth]
    uniform_B = uniform_samples_AB[:, (N_truth+1):end]

    y_A = quantile.(Normal(0, 1), uniform_A)
    y_B = quantile.(Normal(0, 1), uniform_B)
    
    C = cholesky(fictive_corr_pd).L
    z_A = C * y_A
    z_B = C * y_B
    
    X_A = VARS.normal_to_original_dist(z_A', parameters)
    X_B = VARS.normal_to_original_dist(z_B', parameters)

    sobol_result = gsa(ishigami_model_for_gsa, Sobol(), X_A, X_B)
    
    println("...Ground Truth calculation complete.")
    return sobol_result.ST
end

ground_truth_st = calculate_ground_truth_st()


# --- 3. Run Your G-VARS Implementation with a Realistic Budget ---

N = 512
num_dir_samples = 500
delta_h = 0.1

println("\nStep 2: Generating G-VARS samples with a realistic budget...")
problem = VARS.sample(parameters, N, delta_h, 
                      corr_mat=corr_mat, 
                      num_dir_samples=num_dir_samples, 
                      seed=123,
                      use_fictive_corr=true)

println("Step 3: Running the model function on G-VARS samples...")
Y = ishigami_model_for_vars(problem.X)

println("Step 4: Running Julia G-VARS implementation...")
julia_results = VARS.analyse(problem, vec(Y))


# --- 4. Display Final Comparison Table ---
println("\n--- G-VARS Final Validation ---")
@printf("%-12s | %-15s | %-15s | %-20s\n", "Parameter", "Julia G-VARS ST", "Ground Truth ST", "Difference")
println(repeat("-", 70))
for i in 1:d
    param_name = collect(keys(parameters))[i]
    diff = abs(julia_results.ST[i] - ground_truth_st[i])
    @printf("%-12s | %-15.4f | %-15.4f | %-20.6f\n", param_name, julia_results.ST[i], ground_truth_st[i], diff)
end