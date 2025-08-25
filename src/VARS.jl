# src/VARS.jl
module VARS

# Dependencies
using QuasiMonteCarlo, Statistics, Combinatorics, Distributions, Roots, HCubature, ProgressMeter, LinearAlgebra, OrderedCollections, Random 

# Include source files
include("utils.jl")
include("sampling.jl")
include("analysis.jl")
include("gvars.jl")
include("api.jl") # <-- ADD THIS LINE
include("bootstrap.jl")

# Export user functions
export generate_vars_samples, vars_analyse
export map_to_fictive_corr, normal_to_original_dist
export rx_to_rn, rn_to_rx
export generate_gvars_samples

end