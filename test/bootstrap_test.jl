using VariogramAnalysis
using OrderedCollections
using Statistics
using Test

@testset "bootstrap_st!" begin
    d = 3
    parameters = OrderedDict("x$i" => (p1=0.0, p2=1.0, p3=nothing, dist="unif") for i in 1:d)
    problem = VariogramAnalysis.sample(parameters, 32, 0.1; seed=5,
                                       sampler_type="sobol_shift", ray_logic=:shifted_grid)
    Y = [x[1] + 0.5 * x[2]^2 for x in eachcol(problem.X)]

    compute_st = (Y_b, X_b, X_norm_b, info_b, N_b, d_b, delta_h_b) ->
        VariogramAnalysis.analyse(problem.method, X_b, X_norm_b, info_b,
                                  parameters, N_b, d_b, delta_h_b, Y_b)

    num_boot = 25
    res = VariogramAnalysis.VARSBootstrap.bootstrap_st!(
        compute_st, Y, problem.X, problem.X_norm, problem.info,
        problem.N, problem.d, problem.delta_h;
        num_boot=num_boot, seed=99)

    @test size(res.st_boot) == (num_boot, d)
    @test length(res.st_point) == d
    @test length(res.st_ci) == d
    @test all(ci -> ci[1] <= ci[2], res.st_ci)
    # The point estimate of an unused parameter should be near zero
    @test abs(res.st_point[3]) < 0.05

    # Same seed gives identical replicates
    res2 = VariogramAnalysis.VARSBootstrap.bootstrap_st!(
        compute_st, Y, problem.X, problem.X_norm, problem.info,
        problem.N, problem.d, problem.delta_h;
        num_boot=num_boot, seed=99)
    @test res.st_boot == res2.st_boot
end

@testset "rank_from_bootstrap and group_factors" begin
    st_boot = [0.9 0.5 0.1; 0.8 0.6 0.05; 0.85 0.55 0.12]
    names = ["a", "b", "c"]

    ranks = VariogramAnalysis.VARSBootstrap.rank_from_bootstrap(st_boot, names)
    @test ranks.rank_mode == [1, 2, 3]
    @test all(ranks.rank_agreement .== 1.0)

    grouping = VariogramAnalysis.VARSBootstrap.group_factors(st_boot, names; num_groups=2)
    @test length(grouping.groups) == 3
    @test grouping.groups[1] != grouping.groups[3]
end
