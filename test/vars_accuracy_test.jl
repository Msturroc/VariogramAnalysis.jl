using VariogramAnalysis
using OrderedCollections
using Statistics
using Test

# VARS total-order indices are checked against the analytical Sobol' total-order
# indices of well-known benchmark functions. VARS is an estimator of ST, so the
# comparisons use loose tolerances; the tight per-pair agreement with the
# original Python implementation is covered by test/python_validation/.

function run_vars(model, parameters; N=256, delta_h=0.1, seed=123,
                  sampler_type="sobol_shift", ray_logic=:shifted_grid)
    problem = VariogramAnalysis.sample(parameters, N, delta_h;
                                       seed=seed, sampler_type=sampler_type, ray_logic=ray_logic)
    Y = [model(x) for x in eachcol(problem.X)]
    return VariogramAnalysis.analyse(problem.method, problem.X, problem.X_norm, problem.info,
                                     parameters, problem.N, problem.d, problem.delta_h, Y).ST
end

@testset "Ishigami" begin
    ishigami(x; a=7, b=0.1) = sin(x[1]) + a * sin(x[2])^2 + b * x[3]^4 * sin(x[1])

    # Analytical total-order indices for a=7, b=0.1 on [-pi, pi]^3
    a, b = 7.0, 0.1
    V = a^2 / 8 + b * pi^4 / 5 + b^2 * pi^8 / 18 + 1 / 2
    st_analytical = [
        (1 / 2 * (1 + b * pi^4 / 5)^2 + 8 * b^2 * pi^8 / 225) / V,
        (a^2 / 8) / V,
        (8 * b^2 * pi^8 / 225) / V,
    ]

    parameters = OrderedDict("x$i" => (p1=-pi, p2=pi, p3=nothing, dist="unif") for i in 1:3)
    st = run_vars(ishigami, parameters)

    @test length(st) == 3
    @test all(isfinite, st)
    for i in 1:3
        @test st[i] ≈ st_analytical[i] atol = 0.1
    end
    # Ranking must be correct: x1 > x2 > x3
    @test sortperm(st; rev=true) == [1, 2, 3]
end

@testset "Sobol-G" begin
    function sobol_g(x, a)
        result = 1.0
        for i in eachindex(x)
            result *= (abs(4 * x[i] - 2) + a[i]) / (1 + a[i])
        end
        return result
    end

    function sobol_g_analytical_st(a)
        Vi = @. 1 / (3 * (1 + a)^2)
        total_variance = prod(Vi .+ 1) - 1
        return [(Vi[i] / total_variance) * (total_variance + 1) / (Vi[i] + 1) for i in eachindex(a)]
    end

    a = [0.0, 0.5, 3.0, 9.0]
    d = length(a)
    parameters = OrderedDict("x$i" => (p1=0.0, p2=1.0, p3=nothing, dist="unif") for i in 1:d)
    st = run_vars(x -> sobol_g(x, a), parameters)
    st_analytical = sobol_g_analytical_st(a)

    @test all(isfinite, st)
    for i in 1:d
        @test st[i] ≈ st_analytical[i] atol = 0.15
    end
    @test sortperm(st; rev=true) == [1, 2, 3, 4]
end

@testset "constant model returns zero indices" begin
    parameters = OrderedDict("x$i" => (p1=0.0, p2=1.0, p3=nothing, dist="unif") for i in 1:2)
    st = run_vars(x -> 3.14, parameters; N=8)
    @test st == zeros(2)
end
