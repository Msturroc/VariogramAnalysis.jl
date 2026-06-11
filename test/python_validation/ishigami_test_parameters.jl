using LinearAlgebra
using OrderedCollections
using PyCall

# --- Experiment 1: VARS, Uncorrelated, Uniform ---
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
report_verbose_exp1 = false

# --- Experiment 2: G-VARS, Uncorrelated, Uniform ---
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
fictive_mat_flag_exp2 = true
report_verbose_exp2 = false

# --- Experiment 3: G-VARS, Correlated, Uniform ---
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
fictive_mat_flag_exp3 = true
report_verbose_exp3 = false

# --- Experiment 4: G-VARS, Correlated, Non-uniform ---
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
fictive_mat_flag_exp4 = true
report_verbose_exp4 = false

# --- Notebook Reference Values (from the provided output) ---
# VARS-TO (ST) values for comparison
notebook_st_exp1 = Dict("x1" => 0.608621, "x2" => 0.432497, "x3" => 0.176867)
notebook_st_exp2 = Dict("x1" => 0.508436, "x2" => 0.362669, "x3" => 0.146195)
notebook_st_exp3 = Dict("x1" => 0.169525, "x2" => 0.439484, "x3" => 0.121525)
notebook_st_exp4 = Dict("x1" => 0.160315, "x2" => 0.425341, "x3" => 0.117035)