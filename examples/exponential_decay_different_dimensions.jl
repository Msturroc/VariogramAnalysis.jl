using Distributed

addprocs(19)

# Load packages and define functions on all processes
@everywhere begin
    using QuasiMonteCarlo
    using SurrogatesPolyChaos
    using LinearAlgebra
    using Statistics
    using GlobalSensitivity
    using Plots
    using Distributions
    using Combinatorics
    using OrderedCollections
    using VariogramAnalysis

    # ---- Exponential Decay Model and Analytical Solutions ----
    function exp_decay_batch(X, lambda)
        return vec(exp.(-lambda' * X))
    end

    function calculate_exp_moments(lambda_i)
        if abs(lambda_i) < 1e-10
            A_i = 1.0
            B_i = 1.0
        else
            A_i = (1.0 - exp(-lambda_i)) / lambda_i
            B_i = (1.0 - exp(-2.0 * lambda_i)) / (2.0 * lambda_i)
        end
        return A_i, B_i
    end

    function exp_decay_analytical_total_indices(lambda)
        d = length(lambda)
        A = zeros(d)
        B = zeros(d)
        for i in 1:d
            A[i], B[i] = calculate_exp_moments(lambda[i])
        end

        prod_B = prod(B)
        prod_A_sq = prod(A)^2
        VarY = prod_B - prod_A_sq

        if VarY < 1e-12
            return zeros(d)
        end

        ST = zeros(d)
        for i in 1:d
            prod_B_noti = prod_B / B[i]
            prod_A_noti_sq = prod_A_sq / A[i]^2
            var_g_noti = prod_B_noti - prod_A_noti_sq
            var_E_Y_Xnoti = A[i]^2 * var_g_noti
            ST[i] = 1.0 - var_E_Y_Xnoti / VarY
        end
        return ST
    end

    # ---- VARS Index Estimator ----
    # This is the direct Razavi-Gupta point estimator for VARS, which takes the
    # output from VariogramAnalysis.sample
    function compute_vars_indices(Y::AbstractVector{<:Real}, info::Vector, d::Int)
        center_idx = findall(p -> p.dim_id == 0, info)
        if length(center_idx) < 2 return fill(NaN, d) end
        
        VY = var(view(Y, center_idx))
        if !(VY > 1e-12) return zeros(d) end
        
        stars = unique(p.star_id for p in info)
        Ti = zeros(d)
        
        for dim in 1:d
            variogram_sum, covariogram_sum, stars_with_data = 0.0, 0.0, 0
            for star in stars
                traj_idx = findall(k -> info[k].star_id == star && info[k].dim_id == dim, eachindex(info))
                if length(traj_idx) < 2 continue end
                
                vals = view(Y, traj_idx)
                pairs = collect(Combinatorics.combinations(vals, 2))
                if isempty(pairs) continue end
                
                p1 = [p[1] for p in pairs]
                p2 = [p[2] for p in pairs]
                variogram_i = 0.5 * mean((p1 .- p2).^2)
                covariogram_i = cov(p1, p2)
                
                if isfinite(variogram_i) && isfinite(covariogram_i)
                    variogram_sum += variogram_i
                    covariogram_sum += covariogram_i
                    stars_with_data += 1
                end
            end
            Ti[dim] = stars_with_data > 0 ? (variogram_sum / stars_with_data + covariogram_sum / stars_with_data) / VY : NaN
        end
        return Ti
    end

    # ---- PCE Helper ----
    function compute_sobol_indices_st_only(pce)
        coeffs = pce.coeff; multiidx = pce.orthopolys.ind; d = size(multiidx, 2)
        varY_pce = sum(coeffs[2:end].^2)
        if varY_pce < 1e-12 return zeros(d) end
        ST = zeros(d)
        for k in 2:length(coeffs)
            coeff_sq = coeffs[k]^2
            vars_idx = findall(multiidx[k, :] .> 0)
            if !isempty(vars_idx)
                for i in vars_idx
                    ST[i] += coeff_sq
                end
            end
        end
        return ST ./ varY_pce
    end

    # ---- Core Distributed Function ----
    function run_single_method_repeat_distributed(method_name, d, total_cost, lambda, analytical_ST; va_params=nothing)
        try
            batch_function = X -> exp_decay_batch(X, lambda)
            lb, ub = zeros(d), ones(d)
            
            N_pce = floor(Int, total_cost / 2)
            N_sobol = floor(Int, total_cost / (d + 2))
            N_efast = floor(Int, total_cost / d)

            if method_name == "Sobol" && N_sobol > 1
                sampler = SobolSample(QuasiMonteCarlo.Shift())
                A, B = QuasiMonteCarlo.generate_design_matrices(N_sobol, lb, ub, sampler, 2)
                res_sobol = gsa(batch_function, Sobol(), A, B, batch=true)
                return norm(res_sobol.ST - analytical_ST) / norm(analytical_ST)
            elseif method_name == "eFAST" && N_efast > 1
                param_ranges = [[lb[i], ub[i]] for i in 1:d]
                res_efast = gsa(batch_function, eFAST(), param_ranges, samples=N_efast, batch=true)
                return norm(vec(res_efast.ST) - analytical_ST) / norm(analytical_ST)
            elseif method_name == "VARS"
                if va_params.r < 10 return NaN end # Check if valid params were found
                parameters = OrderedDict("x$i" => (p1=0.0, p2=1.0, p3=nothing, dist="unif") for i in 1:d)
                problem = VariogramAnalysis.sample(parameters, va_params.r, va_params.h; seed=rand(Int), sampler_type="sobol_shift", ray_logic=:shifted_grid)
                Y_vars = batch_function(problem.X)
                ST_vars = compute_vars_indices(vec(Y_vars), problem.info, d)
                return norm(ST_vars - analytical_ST) / norm(analytical_ST)
            elseif method_name in ["PCE_deg2", "PCE_deg3"]
                pce_deg = method_name == "PCE_deg2" ? 2 : 3
                if N_pce > d * pce_deg + 1
                    sampler = SobolSample(QuasiMonteCarlo.Shift())
                    A, B = QuasiMonteCarlo.generate_design_matrices(N_pce, lb, ub, sampler, 2)
                    X_combined = hcat(A, B)
                    Y_combined = batch_function(X_combined)
                    xpoints = [Vector(X_combined[:, i]) for i in 1:size(X_combined, 2)]
                    orthos = SurrogatesPolyChaos.MultiOrthoPoly([SurrogatesPolyChaos.GaussOrthoPoly(pce_deg) for _ in 1:d], pce_deg)
                    pce = SurrogatesPolyChaos.PolynomialChaosSurrogate(xpoints, Y_combined, lb, ub; orthopolys=orthos)
                    ST_pce = compute_sobol_indices_st_only(pce)
                    return norm(ST_pce - analytical_ST) / norm(analytical_ST)
                end
            end
        catch e
            println("Worker $(myid()): $(method_name) for d=$d, cost=$total_cost failed with error: $e")
            return NaN
        end
        return NaN
    end
end

# ---- MAIN PROCESS ONLY FUNCTIONS for VARS Parameter Finding ----
# These functions are only needed on the main process to find the parameters.
function estimate_va_cost(r::Int, d::Int, h::Float64)
    center = fill(0.5, d) # A representative center point
    points_for_one_star = 1 # The center itself
    for j in 1:d
        c_dim = center[j]
        traj_values = filter(x -> x != c_dim, unique(vcat(c_dim % h : h : 1.0, c_dim % h : -h : 0.0)))
        points_for_one_star += length(traj_values)
    end
    return r * points_for_one_star
end

function find_va_params(target_cost::Int, d::Int)
    h_options = [0.05, 0.1, 0.15, 0.2]
    r_options = [10, 20, 50, 100, 200, 500, 1000]
    best_params = (r=0, h=0.0, cost=0, diff=Inf)
    
    for h_val in h_options, r_val in r_options
        estimated_cost = estimate_va_cost(r_val, d, h_val)
        if estimated_cost > target_cost * 1.5 continue end # Don't overshoot too much
        diff = abs(estimated_cost - target_cost)
        # Score prioritizes lower h for better variogram estimation
        score = diff + (h_val - 0.05) * 1000 
        if score < best_params.diff
            best_params = (r=r_val, h=h_val, cost=estimated_cost, diff=score)
        end
    end
    return best_params
end


# ---- Main Study Function with Distributed Computing ----
function run_sensitivity_study_distributed(d_values, cost_values; selected_methods, n_repeats)
    results = Dict()
    println("Using distributed computing with $(nworkers()) worker processes")

    for d in d_values
        results[d] = Dict()
        println("\n" * "="^60 * "\nAnalyzing dimension d=$d for Exponential Decay Model\n" * "="^60)
        
        lambda = ones(d) * 1e-6
        if d >= 1; lambda[1] = 1e-5; end
        if d >= 2; lambda[2] = 1e-4; end
        if d >= 3; lambda[3] = 1e-3; end
        if d >= 4; lambda[4] = 1e-2; end
        if d >= 5; lambda[5] = 1e-4; end
        
        analytical_ST = exp_decay_analytical_total_indices(lambda)
        results[d]["analytical"] = (ST=analytical_ST,)

        for total_cost in cost_values
            println("\n  Target Total Cost = $total_cost")
            results[d][total_cost] = Dict()

            # Find the best VARS parameters for this specific cost and dimension
            va_params_for_cost = find_va_params(total_cost, d)
            println("    VARS params: r=$(va_params_for_cost.r), h=$(va_params_for_cost.h) (Est. Cost: $(va_params_for_cost.cost))")

            for method_name in selected_methods
                futures = [@spawnat :any run_single_method_repeat_distributed(method_name, d, total_cost, lambda, analytical_ST; va_params=va_params_for_cost) for _ in 1:n_repeats]
                method_results = [fetch(f) for f in futures]
                
                valid_errors = filter(!isnan, method_results)
                if !isempty(valid_errors)
                    mean_err = mean(valid_errors)
                    std_err = length(valid_errors) > 1 ? std(valid_errors) : 0.0
                    results[d][total_cost][Symbol(method_name)] = (error=mean_err, std_error=std_err, n_successful=length(valid_errors))
                    println("    $(rpad(method_name, 10)): Mean Error = $(round(mean_err, digits=4)) ± $(round(std_err, digits=4)) (n=$(length(valid_errors))/$n_repeats)")
                else
                    results[d][total_cost][Symbol(method_name)] = (error=NaN, std_error=NaN, n_successful=0)
                    println("    $(rpad(method_name, 10)): All repeats failed")
                end
            end
        end
    end
    return results
end

# ---- Enhanced Plotting Function (Unchanged from your original) ----
function plot_st_error_comparison_with_errorbars(results; selected_methods)
    dimensions = sort(collect(keys(results)))
    plots_array = []
    method_styles = Dict(
        "PCE_deg2" => (color=:blue, marker=:circle, label="PCE (d=2)"),
        "PCE_deg3" => (color=:purple, marker=:cross, label="PCE (d=3)"),
        "Sobol" => (color=:red, marker=:square, label="Sobol"),
        "eFAST" => (color=:green, marker=:diamond, label="eFAST"),
        "VARS" => (color=:orange, marker=:utriangle, label="VARS")
    )
    
    for (d_idx, d) in enumerate(dimensions)
        p = plot(xscale=:log10, yscale=:log10,
                 xlabel="Total Sample Size (Cost)", 
                 ylabel=d_idx == 1 ? "Relative Error (ST)" : "",
                 title="d = $d",
                 legend=d_idx == length(dimensions) ? :outertopright : false,
                 size=(400, 300), margin=3Plots.mm)
        
        cost_values = sort(collect(filter(k -> isa(k, Number), keys(results[d]))))
        
        for method_name in selected_methods
            plot_data = []
            for cost in cost_values
                if haskey(results[d][cost], Symbol(method_name))
                    res = results[d][cost][Symbol(method_name)]
                    if !isnan(res.error) && res.error > 0
                        push!(plot_data, (cost, res.error, res.std_error))
                    end
                end
            end
            
            if !isempty(plot_data)
                sort!(plot_data)
                x_vals = [p[1] for p in plot_data]; y_vals = [p[2] for p in plot_data]; y_errs = [p[3] for p in plot_data]
                style = method_styles[method_name]
                plot!(p, x_vals, y_vals, label=style.label, color=style.color, marker=style.marker, linewidth=2, markersize=5, yerror=y_errs, capsize=3)
            end
        end
        push!(plots_array, p)
    end
    
    return plot(plots_array..., layout=(1, length(dimensions)), size=(1200, 400), margin=5Plots.mm, plot_title="Exponential Decay Model: Total-Order Error Comparison")
end

# ---- EXECUTION ----
println("Starting distributed sensitivity study for Exponential Decay model...")

d_values = [3, 10, 20, 40]
cost_values = [5000, 10000, 25000, 50000]
selected_methods_all = ["PCE_deg2", "Sobol", "eFAST", "VARS"]
n_repeats = 20

results_exp_decay = run_sensitivity_study_distributed(
    d_values, cost_values; 
    selected_methods=selected_methods_all, 
    n_repeats=n_repeats
)

try
    plot_exp_decay = plot_st_error_comparison_with_errorbars(results_exp_decay; selected_methods=selected_methods_all)
    display(plot_exp_decay)
    println("\nDistributed comparison plot for Exponential Decay model generated (n=$n_repeats repeats).")
catch e
    println("\nError during plotting: $e")
end

# Clean up worker processes
rmprocs(workers())
println("Worker processes removed. Analysis complete.")