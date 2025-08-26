# VARS.jl

`VARS.jl` is a pure Julia implementation of the Variogram Analysis of Response Surfaces (VARS) method for global sensitivity analysis, as described by M. Razavi and H. V. Gupta. VARS is a robust and efficient method for sensitivity analysis of complex models.

This package provides tools to:
*   Generate the required input parameter samples using Latin Hypercube Sampling.
*   Calculate total-order sensitivity indices (ST).
*   Perform bootstrap analysis to estimate confidence intervals for the sensitivity indices.

## Installation

From the Julia REPL, type `]` to enter the Pkg REPL mode and run:
```
pkg> add VARS
```

## Usage Example: Sobol-G Function

Let's walk through an example using the Sobol-G function to understand the main workflow of `VARS.jl`.

First, bring the necessary packages into scope.
```julia
using VARS
using OrderedCollections
using Random
using Statistics
```

### 1. Define Your Model

You need a Julia function that represents your model. The function should take a vector of input parameters and return a single output value.

```julia
# Sobol-G function
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
```

### 2. Define Input Parameters

Next, define the input parameters for your model using an `OrderedDict`. For each parameter, specify its distribution and range. Here, we define a 4-dimensional problem with uniform distributions.

```julia
d = 4 # Number of dimensions
parameters_julia = OrderedDict("x$i" => (p1=0.0, p2=1.0, p3=nothing, dist="unif") for i in 1:d)
```

### 3. Sample the Input Space

Use the `VARS.sample` function to generate the input samples required for the VARS method.

```julia
N = 256             # Number of star centers
delta_h = 0.1       # Step size for radial sampling
seed = 123          # for reproducibility

Random.seed!(seed)
problem = VARS.sample(parameters_julia, N, delta_h, seed=seed)
```

### 4. Run Your Model

Evaluate your model for each of the generated input samples.

```julia
a = [0, 0.5, 3, 9] # 'a' coefficients for the Sobol-G function
Y = [sobol_g_julia(x, a) for x in eachcol(problem.X)];
```

### 5. Perform Sensitivity Analysis

Now, you can perform the sensitivity analysis. For robust results, it's highly recommended to use bootstrapping to get confidence intervals on the sensitivity indices.

```julia
num_boot_replicates = 50 # Number of bootstrap replicates

# Define a closure for the analysis function
compute_st_closure = (Y_b, X_b, X_norm_b, info_b, N_b, d_b, delta_h_b) -> begin
    VARS.analyse(problem.method, X_b, X_norm_b, info_b, parameters_julia, N_b, d_b, delta_h_b, Y_b)
end

# Run the bootstrap analysis
julia_boot_results = VARS.VARSBootstrap.bootstrap_st!(
    compute_st_closure, Y, problem.X, problem.X_norm, problem.info,
    problem.N, problem.d, problem.delta_h;
    num_boot=num_boot_replicates, seed=seed
)
```

### 6. View the Results

You can now analyze the results, for example, by calculating the mean of the bootstrapped sensitivity indices.

```julia
mean_st = mean(julia_boot_results.st_boot, dims=1)

println("Mean Total-Order Sensitivity Indices (ST):")
for i in 1:d
    println("  x$i: $(mean_st[i])")
end
```