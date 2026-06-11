using Test

# Pure-Julia test suite. Validation against the Python varstool implementation
# lives in test/python_validation/ and is run separately (see its README.md).
@testset "VariogramAnalysis.jl" begin
    @testset "Sampling" begin
        include("sampling_test.jl")
    end

    @testset "VARS Accuracy" begin
        include("vars_accuracy_test.jl")
    end

    @testset "G-VARS" begin
        include("gvars_test.jl")
    end

    @testset "Bootstrap" begin
        include("bootstrap_test.jl")
    end

    @testset "D-VARS" begin
        include("dvars_test.jl")
    end
end
