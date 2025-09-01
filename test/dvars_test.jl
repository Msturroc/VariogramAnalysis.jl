using Test
using VariogramAnalysis
using DataFrames
using QuasiMonteCarlo

@testset "D-VARS Analysis" begin
    # a. Create a sample dataset (e.g., from the Ishigami function)
    function ishigami(x)
        a = 7.0
        b = 0.1
        sin(x[1]) + a * sin(x[2])^2 + b * x[3]^4 * sin(x[1])
    end

    num_samples = 200
    d = 3
    lb = [-pi, -pi, -pi]
    ub = [pi, pi, pi] .* 1.0

    X_sample = QuasiMonteCarlo.sample(num_samples, lb, ub, SobolSample())
    y_sample = [ishigami(x) for x in eachcol(X_sample)]

    # b. Convert to a DataFrame
    df_ishigami = DataFrame(
        x1 = X_sample[1, :],
        x2 = X_sample[2, :],
        x3 = X_sample[3, :],
        y = y_sample
    )

    # c. Run the D-VARS analysis
    sens, ratios, phis, v = dvars_sensitivities(df_ishigami, :y, Hj=1.0)

    # d. Print and check the results
    println("\n--- D-VARS Results ---")
    println("Output Variance: ", round(v, digits=4))
    println("Optimized Hyperparameters (phi): ", round.(phis, digits=4))
    println("Sensitivity Indices (Gamma): ", round.(sens, digits=4))
    println("Sensitivity Ratios (%): ", round.(ratios .* 100, digits=2))

    @test length(sens) == d
    @test length(ratios) == d
    @test length(phis) == d
    @test isapprox(sum(ratios), 1.0, atol=1e-9)
    @test all(ratios .>= 0)

end
