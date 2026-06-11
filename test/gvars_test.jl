using VariogramAnalysis
using OrderedCollections
using LinearAlgebra
using Statistics
using Test

@testset "G-VARS sampling and analysis" begin
    d = 3
    parameters = OrderedDict("x$i" => (p1=0.0, p2=1.0, p3=nothing, dist="unif") for i in 1:d)
    corr_mat = Matrix{Float64}(I, d, d)
    corr_mat[1, 2] = corr_mat[2, 1] = 0.6

    N, num_dir_samples, delta_h, seed = 32, 10, 0.1, 11

    problem = VariogramAnalysis.sample(parameters, N, delta_h;
                                       corr_mat=corr_mat, num_dir_samples=num_dir_samples,
                                       use_fictive_corr=false, seed=seed)

    @test problem.method == :GVARS
    @test size(problem.X) == (d, N * (1 + d * num_dir_samples))
    @test all(0.0 .<= problem.X .<= 1.0)

    # The induced rank correlation between x1 and x2 should be clearly positive
    centres = problem.X[:, 1:N]
    @test cor(centres[1, :], centres[2, :]) > 0.3

    # Analysis runs and produces finite indices on an additive model
    Y = [x[1] + 2 * x[2] + 0.1 * x[3] for x in eachcol(problem.X)]
    res = VariogramAnalysis.analyse(problem.method, problem.X, problem.X_norm, problem.info,
                                    parameters, problem.N, problem.d, problem.delta_h, Y)
    @test length(res.ST) == d
    @test all(isfinite, res.ST)
    @test res.ST[2] > res.ST[3]
end

@testset "fictive correlation mapping" begin
    p_unif = (p1=0.0, p2=1.0, p3=nothing, dist="unif")
    # For identical uniform marginals the fictive correlation stays close to
    # the target correlation and preserves the sign
    rn = rx_to_rn(("unif", "unif"), p_unif, p_unif, 0.5)
    @test 0.3 < rn < 0.7
    rx = rn_to_rx(("unif", "unif"), p_unif, p_unif, rn)
    @test rx ≈ 0.5 atol = 1e-3
    @test rn_to_rx(("unif", "unif"), p_unif, p_unif, 0.0) == 0.0
    @test rn_to_rx(("unif", "unif"), p_unif, p_unif, 1.0) == 1.0
end
