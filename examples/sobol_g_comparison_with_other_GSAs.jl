using Distributed

# Add worker processes (adjust based on your CPU cores)
addprocs(19)

# Load packages on all processes
@everywhere using QuasiMonteCarlo
@everywhere using SurrogatesPolyChaos
@everywhere using LinearAlgebra
@everywhere using Statistics
@everywhere using GlobalSensitivity
@everywhere using Plots
@everywhere using Distributions
@everywhere using Combinatorics
@everywhere using OrderedCollections
@everywhere using VariogramAnalysis

# ---- DEFINE MAIN PROCESS FUNCTIONS ----
# These functions are only needed on the main process to find the parameters.
# They have been moved OUTSIDE the @everywhere block.

function estimate_va_cost(r::Int, d::Int, h::Float64)
    # Create a single, representative center point.
    # A point with all coordinates at 0.5 is a good average case.
    center = fill(0.5, d)
    
    points_for_one_star = 1 # The center itself
    
    for j in 1:d
        c_dim = center[j]
        # Use your superior "shifted_grid" logic to count the points for one ray
        traj_values = filter(x -> x != c_dim, unique(vcat(c_dim % h : h : 1.0, c_dim % h : -h : 0.0)))
        points_for_one_star += length(traj_values)
    end
    
    # Extrapolate the cost from one average star to `r` stars.
    return r * points_for_one_star
end

function find_va_params(target_cost::Int, d::Int)
    h_options = [0.05, 0.1, 0.15, 0.2]
    best_params = (r=0, h=0.0, cost=0, diff=Inf)
    for h_val in h_options
        for r_val in [10, 20, 50, 100, 200, 500] 
            if r_val < 10 continue end
            estimated_cost = estimate_va_cost(r_val, d, h_val)
            if estimated_cost > target_cost * 1.2 continue end
            diff = abs(estimated_cost - target_cost)
            score = diff + (h_val - 0.05) * 1000
            if score < best_params.diff
                best_params = (r=r_val, h=h_val, cost=estimated_cost, diff=score)
            end
        end
    end
    return best_params
end
@everywhere begin
    # Sobol-G and Analytical Functions
    function sobol_g_batch(X, a)
        n = size(X, 2); result = ones(n)
        for i in 1:length(a)
            result .*= (abs.(4 .* X[i,:] .- 2) .+ a[i]) ./ (1 .+ a[i])
        end
        return result
    end

    function sobol_g_analytical_total_indices(a)
        d = length(a); V_i = [1 / (3 * (1 + a_i)^2) for a_i in a]
        V = prod(1 .+ V_i) - 1; if V < 1e-12 return zeros(d) end
        ST = zeros(d)
        for i in 1:d
            ST[i] = V_i[i] * prod(1 .+ V_i[setdiff(1:d, i)]) / V
        end
        return ST
    end

    # --- THIS IS THE FIX: The PCE function is now back inside @everywhere ---
    function compute_sobol_indices_st_only(pce)
        coeffs = pce.coeff; multiidx = pce.orthopolys.ind; d = size(multiidx, 2)
        varY_pce = sum(coeffs[2:end].^2); if varY_pce < 1e-12 return zeros(d) end
        ST = zeros(d)
        for k in 2:length(coeffs)
            coeff_sq = coeffs[k]^2; vars_idx = findall(multiidx[k, :] .> 0)
            if !isempty(vars_idx)
                for i in vars_idx; ST[i] += coeff_sq; end
            end
        end
        return ST ./ varY_pce
    end

    # The direct Razavi-Gupta point estimator for VARS
    function compute_st_rg(Y::AbstractVector{<:Real}, info::Vector, d::Int)
        center_idx = findall(p -> p.dim_id == 0, info)
        if length(center_idx) < 2 return fill(NaN, d) end
        VY = var(view(Y, center_idx)); if !(VY > 1e-12) return zeros(d) end
        stars = unique(p.star_id for p in info); Ti = zeros(d)
        for dim in 1:d
            variogram_sum = 0.0; covariogram_sum = 0.0; stars_with_data = 0
            for star in stars
                traj_idx = findall(k -> info[k].star_id == star && info[k].dim_id == dim, eachindex(info))
                if length(traj_idx) < 2 continue end
                vals = view(Y, traj_idx); pairs = collect(Combinatorics.combinations(vals, 2))
                if isempty(pairs) continue end
                p1 = [p[1] for p in pairs]; p2 = [p[2] for p in pairs]
                variogram_i = 0.5 * mean((p1 .- p2).^2); covariogram_i = cov(p1, p2)
                if isfinite(variogram_i) && isfinite(covariogram_i)
                    variogram_sum += variogram_i; covariogram_sum += covariogram_i; stars_with_data += 1
                end
            end
            Ti[dim] = stars_with_data > 0 ? (variogram_sum / stars_with_data + covariogram_sum / stars_with_data) / VY : NaN
        end
        return Ti
    end

    # The updated worker function using the direct estimator
    function run_single_method_repeat_distributed(method_name, d, total_cost, a, analytical_ST; va_params=nothing)
        try
            batch_function = X -> sobol_g_batch(X, a)
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
                if va_params.r < 10 return NaN end
                parameters = OrderedDict("x$i" => (p1=0.0, p2=1.0, p3=nothing, dist="unif") for i in 1:d)
                problem = VariogramAnalysis.sample(parameters, va_params.r, va_params.h; seed=rand(Int), sampler_type="sobol_shift", ray_logic=:shifted_grid)
                Y_vars = sobol_g_batch(problem.X, a)
                ST_vars = compute_st_rg(vec(Y_vars), problem.info, d)
                return norm(ST_vars - analytical_ST) / norm(analytical_ST)
            elseif method_name == "PCE_deg2"
                # A more robust check for required samples for PCE
                num_coeffs = binomial(d + 2, 2)
                if N_pce > 2 * num_coeffs
                    sampler = SobolSample(QuasiMonteCarlo.Shift())
                    A, B = QuasiMonteCarlo.generate_design_matrices(N_pce, lb, ub, sampler, 2)
                    X_combined = hcat(A, B); Y_combined = batch_function(X_combined)
                    xpoints = [Vector(X_combined[:, i]) for i in 1:size(X_combined, 2)]
                    orthos = SurrogatesPolyChaos.MultiOrthoPoly([SurrogatesPolyChaos.GaussOrthoPoly(2) for _ in 1:d], 2)
                    pce = SurrogatesPolyChaos.PolynomialChaosSurrogate(xpoints, Y_combined, lb, ub; orthopolys=orthos)
                    ST_pce = compute_sobol_indices_st_only(pce)
                    return norm(ST_pce - analytical_ST) / norm(analytical_ST)
                end
            end
        catch e
            println("Worker $(myid()): $(method_name) failed with error: $e")
            return NaN
        end
        return NaN
    end
end

# ---- Main Study Function (Unchanged from your version) ----
function run_sensitivity_study_distributed(d_values, cost_values; selected_methods=["PCE_deg2", "Sobol", "eFAST", "VARS"], n_repeats=5)
    results = Dict()
    println("Using distributed computing with $(nworkers()) worker processes")
    for d in d_values
        results[d] = Dict()
        println("\n" * "="^60 * "\nAnalyzing dimension d=$d (Distributed)\n" * "="^60)
        a = zeros(d); a[1] = 0; if d >= 2; a[2] = 1; end; if d >= 3; a[3] = 4.5; end
        if d >= 4; a[4] = 9; end; if d >= 5; a[5:d] .= 99; end
        analytical_ST = sobol_g_analytical_total_indices(a)
        results[d]["analytical"] = (ST=analytical_ST,)
        for total_cost in cost_values
            println("\n  Target Total Cost = $total_cost")
            results[d][total_cost] = Dict()
            println("    Finding optimal VARS parameters for cost ≈ $total_cost...")
            va_params_for_cost = find_va_params(total_cost, d)
            println("    Using VARS with r=$(va_params_for_cost.r), h=$(va_params_for_cost.h) (cost=$(va_params_for_cost.cost))")
            for method_name in selected_methods
                method_sym = Symbol(method_name)
                println("    Running $method_name with $n_repeats repeats...")
                futures = [@spawnat :any run_single_method_repeat_distributed(method_name, d, total_cost, a, analytical_ST; va_params=va_params_for_cost) for _ in 1:n_repeats]
                method_results = [fetch(f) for f in futures]
                valid_errors = filter(x -> !isnan(x), method_results)
                if !isempty(valid_errors)
                    mean_err = mean(valid_errors)
                    std_err = length(valid_errors) > 1 ? std(valid_errors) : 0.0
                    results[d][total_cost][method_sym] = (error=mean_err, std_error=std_err, n_successful=length(valid_errors))
                    println("    $(method_name): Mean Error = $(round(mean_err, digits=4)) ± $(round(std_err, digits=4)) (n=$(length(valid_errors))/$n_repeats)")
                else
                    results[d][total_cost][method_sym] = (error=NaN, std_error=NaN, n_successful=0)
                    println("    $(method_name): All repeats failed")
                end
            end
        end
    end
    return results
end

function plot_st_error_comparison_with_errorbars(results; selected_methods=["PCE_deg2", "Sobol", "eFAST", "VARS"])
    dimensions = sort(collect(keys(results)))
    
    plots_array = []
    method_colors = Dict("PCE_deg2" => :blue, "PCE_deg3" => :lightblue, "Sobol" => :red, "eFAST" => :green, "VARS" => :orange)
    method_markers = Dict("PCE_deg2" => :circle, "PCE_deg3" => :circle, "Sobol" => :square, "eFAST" => :diamond, "VARS" => :utriangle)
    method_labels = Dict("PCE_deg2" => "PCE (d=2)", "PCE_deg3" => "PCE (d=3)", "Sobol" => "Sobol", "eFAST" => "eFAST", "VARS" => "VARS")
    
    # Create data plots (all without legends)
    for (d_idx, d) in enumerate(dimensions)
        p = plot(xscale=:log10, yscale=:log10,
                 xlabel="Total Sample Size (Cost)", 
                 ylabel=d_idx == 1 ? "Relative Error (ST)" : "",
                 title="d = $d",
                 legend=false,  # Remove all legends from data plots
                 size=(400, 300), margin=1Plots.mm)
        
        cost_values = sort(collect(filter(k -> isa(k, Number), keys(results[d]))))
        
        for method_name in selected_methods
            method_sym = Symbol(method_name)
            plot_data = []
            
            for cost in cost_values
                if haskey(results[d][cost], method_sym)
                    res = results[d][cost][method_sym]
                    if !isnan(res.error) && res.error > 0
                        push!(plot_data, (cost, res.error, res.std_error))
                    end
                end
            end
            
            if !isempty(plot_data)
                sort!(plot_data)
                x_vals = [p[1] for p in plot_data]
                y_vals = [p[2] for p in plot_data]
                y_errs = [p[3] for p in plot_data]
                
                plot!(p, x_vals, y_vals, 
                      label=method_labels[method_name], 
                      color=method_colors[method_name],
                      marker=method_markers[method_name], 
                      linewidth=2, markersize=5,
                      yerror=y_errs, capsize=3)
            end
        end
        
        push!(plots_array, p)
    end
    
    # Create legend plot
    legend_plot = plot(xlims=(0,1), ylims=(0,1), framestyle=:none, 
                      showaxis=false, grid=false, ticks=false)
    
    # Add invisible series to create legend
    for method_name in selected_methods
        plot!(legend_plot, [NaN], [NaN], 
              label=method_labels[method_name],
              color=method_colors[method_name],
              marker=method_markers[method_name],
              linewidth=2, markersize=5)
    end
    
    plot!(legend_plot, legend=:left)
    
    # Dynamic layout calculation
    n_plots = length(dimensions)
    plot_width = 0.8 / n_plots  # 80% for data plots
    legend_width = 0.2          # 20% for legend
    
    widths = [fill(plot_width, n_plots); legend_width]
    
    # Combine plots
    all_plots = [plots_array..., legend_plot]
    return plot(all_plots..., 
                layout=grid(1, n_plots + 1, widths=widths),
                size=(1400, 400), margin=7Plots.mm, bottom_margin=10Plots.mm, 
                plot_title="Total-Order Indices Error Comparison")
end

println("Starting distributed sensitivity study...")
println("Worker processes: $(nworkers())")

d_values = [3, 10, 30, 40]
cost_values = [5000, 50000, 100000]
selected_methods_all = ["PCE_deg2", "Sobol", "eFAST", "VARS"]
n_repeats = 20

results_distributed = run_sensitivity_study_distributed(d_values, cost_values; selected_methods=selected_methods_all, n_repeats=n_repeats)

try
    plot_distributed = plot_st_error_comparison_with_errorbars(results_distributed; selected_methods=selected_methods_all)
    display(plot_distributed)
    savefig("examples/sobol_g_different_dimensions.png")
    println("\nDistributed comparison plot generated with error bars (n=$n_repeats repeats)")
catch e
    println("Error in plotting: $e")
end
