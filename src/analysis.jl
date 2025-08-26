using Statistics, Combinatorics, Distributions

"""
Helper to get the min/max bounds for each parameter for normalization.
"""
function _get_param_bounds(parameters::OrderedDict)
    d = length(parameters)
    xmin = zeros(d)
    xmax = zeros(d)
    param_defs = collect(values(parameters))

    for i in 1:d
        p = param_defs[i]
        dist_type = p.dist
        if dist_type == "unif" || dist_type == "triangle"
            xmin[i], xmax[i] = p.p1, p.p2
        elseif dist_type == "norm"
            # Use 3-sigma rule as in the python code
            xmin[i] = p.p1 - 3 * p.p2
            xmax[i] = p.p1 + 3 * p.p2
        else
            # Fallback for other distributions: estimate from a large sample
            dist, _, _ = VariogramAnalysis._get_distribution_and_stats(p)
            q_low = quantile(dist, 0.001)
            q_high = quantile(dist, 0.999)
            xmin[i], xmax[i] = q_low, q_high
        end
    end
    return xmin, xmax
end


"""
    vars_analyse(...) - CORRECTED

Standard VARS analysis. Pairing logic now matches python's step-based approach.
"""
function vars_analyse(Y::Vector, info::Vector{NamedTuple{(:star_id, :dim_id, :step_id, :h), Tuple{Int, Int, Int, Float64}}}, N::Int, d::Int, delta_h::Float64)
    VY = var(Y)
    if VY < 1e-12
        return (ST = zeros(d),)
    end

    ST = zeros(d)
    for dim in 1:d
        gamma_sum, ecov_sum, stars_with_data = 0.0, 0.0, 0

        for star in 1:N
            ray_indices = findall(p -> p.star_id == star && (p.dim_id == dim || p.dim_id == 0), info)
            if length(ray_indices) < 2 continue end
            
            sort!(ray_indices, by = idx -> info[idx].step_id)
            
            one_step_pairs = Tuple{Float64, Float64}[]
            for k in 1:(length(ray_indices) - 1)
                idx1 = ray_indices[k]
                idx2 = ray_indices[k+1]
                if abs(info[idx1].step_id - info[idx2].step_id) == 1
                    push!(one_step_pairs, (Y[idx1], Y[idx2]))
                end
            end

            if isempty(one_step_pairs) continue end

            p1 = [p[1] for p in one_step_pairs]
            p2 = [p[2] for p in one_step_pairs]

            gamma_i = 0.5 * mean((p1 .- p2).^2)
            
            y_ray_values = Y[findall(p -> p.star_id == star && p.dim_id == dim, info)]
            mu_star = isempty(y_ray_values) ? 0.0 : mean(y_ray_values)

            ecov_i = mean((p1 .- mu_star) .* (p2 .- mu_star))

            gamma_sum += gamma_i
            ecov_sum += ecov_i
            stars_with_data += 1
        end

        ST[dim] = stars_with_data > 0 ? (gamma_sum / stars_with_data + ecov_sum / stars_with_data) / VY : NaN
    end
    return (ST=ST,)
end


"""
    gvars_analyse(...) - FINAL VERSION

Replicates the Python tool's complex binning logic for G-VARS analysis.
"""
function gvars_analyse(Y::Vector, X::Matrix, info::Vector{NamedTuple{(:star_id, :dim_id, :step_id, :h), Tuple{Int, Int, Int, Float64}}}, N::Int, d::Int, delta_h::Float64, parameters::OrderedDict)
    VY = var(Y)
    if VY < 1e-12
        return (ST = zeros(d),)
    end

    xmin, xmax = _get_param_bounds(parameters)
    param_ranges = xmax .- xmin

    ST = zeros(d)
    for dim in 1:d
        gamma_sum, ecov_sum, stars_with_data = 0.0, 0.0, 0

        for star in 1:N
            ray_indices = findall(p -> p.star_id == star && (p.dim_id == dim || p.dim_id == 0), info)
            if length(ray_indices) < 2 continue end

            # Collect pairs that fall into the delta_h bin based on normalized distance
            binned_pairs = Tuple{Float64, Float64}[]
            for (idx1, idx2) in combinations(ray_indices, 2)
                # Calculate normalized distance in the original parameter space
                dist = abs(X[dim, idx1] - X[dim, idx2])
                norm_dist = param_ranges[dim] > 1e-9 ? dist / param_ranges[dim] : 0.0
                
                # Python's binning logic: bin index is floor(norm_dist / delta_h)
                # The target bin for ST is the first one, corresponding to h=delta_h.
                # The python code's binning is a bit tricky, but this logic should be equivalent
                # for the first bin: 0 < norm_dist <= delta_h
                if 0 < norm_dist <= delta_h
                    push!(binned_pairs, (Y[idx1], Y[idx2]))
                end
            end

            if isempty(binned_pairs) continue end

            p1 = [p[1] for p in binned_pairs]
            p2 = [p[2] for p in binned_pairs]

            gamma_i = 0.5 * mean((p1 .- p2).^2)
            
            y_ray_values = Y[findall(p -> p.star_id == star && p.dim_id == dim, info)]
            mu_star = isempty(y_ray_values) ? 0.0 : mean(y_ray_values)

            ecov_i = mean((p1 .- mu_star) .* (p2 .- mu_star))

            gamma_sum += gamma_i
            ecov_sum += ecov_i
            stars_with_data += 1
        end

        ST[dim] = stars_with_data > 0 ? (gamma_sum / stars_with_data + ecov_sum / stars_with_data) / VY : NaN
    end
    return (ST=ST,)
end