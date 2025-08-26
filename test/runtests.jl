using Test

@testset "VARS.jl" begin
    # First, include the helper function so it's defined for the tests that follow.
    println("Including test helpers...")
    include("test_helpers.jl")

    # Now, run the actual tests. Each will use the helper to manage the Python state.
    println("\nRunning Ishigami comparison tests...")
    @testset "Ishigami Comparison" begin
        include("ishigami_comparison_test.jl")
    end

    println("\nRunning Sobol-G comparison tests...")
    @testset "Sobol-G Comparison" begin
        include("sobol_g_multiple_dimensions_test.jl")
    end
end