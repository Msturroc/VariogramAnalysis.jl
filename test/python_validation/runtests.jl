# Validation of VariogramAnalysis.jl against the original Python varstool
# implementation. Not part of the default `Pkg.test` run -- see README.md in
# this directory for how to set up and run it.
using Test

@testset "Python validation" begin
    include("test_helpers.jl")

    @testset "Ishigami Comparison" begin
        include("ishigami_comparison_test.jl")
    end

    @testset "Sobol-G Comparison" begin
        include("sobol_g_multiple_dimensions_test.jl")
    end
end
