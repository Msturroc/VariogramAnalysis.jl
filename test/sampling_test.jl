using VariogramAnalysis
using OrderedCollections
using Random
using Test

params3 = OrderedDict("x$i" => (p1=0.0, p2=1.0, p3=nothing, dist="unif") for i in 1:3)

@testset "N = 0 edge case" begin
    X_norm, info = generate_vars_samples(params3, 0, 0.1)
    @test size(X_norm) == (3, 0)
    @test isempty(info)
end

@testset "structure of the star design" begin
    N, delta_h = 8, 0.1
    X_norm, info = generate_vars_samples(params3, N, delta_h; seed=1)
    @test size(X_norm, 1) == 3
    @test size(X_norm, 2) == length(info)
    @test all(0.0 .<= X_norm .<= 1.0)
    # One centre per star
    @test count(p -> p.dim_id == 0, info) == N
    # Every ray point differs from its centre in exactly one dimension
    centres = Dict(p.star_id => i for (i, p) in enumerate(info) if p.dim_id == 0)
    for (i, p) in enumerate(info)
        p.dim_id == 0 && continue
        diff_dims = findall(X_norm[:, i] .!= X_norm[:, centres[p.star_id]])
        @test diff_dims == [p.dim_id]
    end
end

@testset "reproducibility with seed" begin
    for sampler in ("lhs", "sobol_shift"), logic in (:relative, :shifted_grid)
        X1, info1 = generate_vars_samples(params3, 8, 0.1; seed=42, sampler_type=sampler, ray_logic=logic)
        X2, info2 = generate_vars_samples(params3, 8, 0.1; seed=42, sampler_type=sampler, ray_logic=logic)
        @test X1 == X2
        @test info1 == info2
    end
end

@testset "seeded sampling does not touch the global RNG" begin
    Random.seed!(1234)
    expected = rand()
    Random.seed!(1234)
    generate_vars_samples(params3, 4, 0.1; seed=7, sampler_type="sobol_shift")
    @test rand() == expected
end

@testset "invalid arguments" begin
    @test_throws ErrorException generate_vars_samples(params3, 4, 0.1; sampler_type="bogus")
    @test_throws ErrorException generate_vars_samples(params3, 4, 0.1; ray_logic=:bogus)
end

@testset "distribution transforms round-trip" begin
    params_mixed = OrderedDict(
        "u" => (p1=-2.0, p2=3.0, p3=nothing, dist="unif"),
        "n" => (p1=1.0, p2=0.5, p3=nothing, dist="norm"),
        "t" => (p1=0.0, p2=4.0, p3=1.0, dist="triangle"),
    )
    problem = VariogramAnalysis.sample(params_mixed, 8, 0.1; seed=3)
    back = VariogramAnalysis.scale_to_unity(problem.X, params_mixed)
    @test isapprox(back, problem.X_norm; atol=1e-10)
end
