using VARS
using Test

@testset "VARS.jl" begin
    # Include your test files
    include("ishigami_comparison_test.jl")
    include("sobol_g_multiple_dimensions_test.jl")
    # You can add more @test assertions here
end
