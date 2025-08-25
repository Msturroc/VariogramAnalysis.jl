# src/analysis.jl

using Statistics, Combinatorics

"""
    vars_analyse(Y::Vector, info::Vector, N::Int, d::Int, delta_h::Float64)

Performs standard VARS analysis for independent, uniform parameters.
"""
function vars_analyse(Y::Vector, info::Vector, N::Int, d::Int, delta_h::Float64)
    centre_indices = findall(p -> p.dim_id == 0, info)
    if length(centre_indices) < 2 error("Not enough star centre outputs to compute variance.") end
    VY = var(Y[centre_indices])
    if VY < 1e-12 return (ST = zeros(d),) end

    ST = zeros(d)
    for dim in 1:d
        gamma_sum = 0.0
        ecov_sum = 0.0
        stars_with_data = 0
        for star in 1:N
            traj_indices = findall(p -> p.star_id == star && p.dim_id == dim, info)
            star_centre_idx = findfirst(p -> p.star_id == star && p.dim_id == 0, info)
            if isempty(traj_indices) || isnothing(star_centre_idx) continue end

            leg_points_Y = Y[traj_indices]
            leg_points_info = info[traj_indices]
            star_bin_pairs = Tuple{Float64, Float64}[]
            for (i, j) in combinations(1:length(leg_points_Y), 2)
                actual_h = abs(leg_points_info[i].h - leg_points_info[j].h)
                if isapprox(actual_h, delta_h, atol=1e-9)
                    push!(star_bin_pairs, (leg_points_Y[i], leg_points_Y[j]))
                end
            end

            if isempty(star_bin_pairs) continue end

            p1 = [p[1] for p in star_bin_pairs]
            p2 = [p[2] for p in star_bin_pairs]
            gamma_i = 0.5 * mean((p1 .- p2).^2)
            full_traj_Y = vcat(leg_points_Y, Y[star_centre_idx])
            mu_star = mean(full_traj_Y)
            ecov_i = mean((p1 .- mu_star) .* (p2 .- mu_star))

            if !isnan(gamma_i) && !isnan(ecov_i)
                gamma_sum += gamma_i
                ecov_sum += ecov_i
                stars_with_data += 1
            end
        end

        if stars_with_data > 0
            avg_gamma = gamma_sum / stars_with_data
            avg_ecov = ecov_sum / stars_with_data
            ST[dim] = (avg_gamma + avg_ecov) / VY
        else
            ST[dim] = NaN
        end
    end
    return (ST = ST,)
end

"""
    gvars_analyse(Y::Vector, X_norm::Matrix, info::Vector, N::Int, d::Int, delta_h::Float64)

Performs G-VARS analysis for correlated and/or non-uniform parameters.
"""
function gvars_analyse(Y::Vector, X_norm::Matrix, info::Vector, N::Int, d::Int, delta_h::Float64)
    centre_indices = findall(p -> p.dim_id == 0, info)
    if length(centre_indices) < 2 error("Not enough star centre outputs to compute variance.") end
    VY = var(Y[centre_indices])
    if VY < 1e-12 return (ST = zeros(d),) end

    ST = zeros(d)
    for dim in 1:d
        gamma_sum = 0.0
        ecov_sum = 0.0
        stars_with_data = 0
        for star in 1:N
            ray_indices = findall(p -> p.star_id == star && (p.dim_id == dim || p.dim_id == 0), info)
            if length(ray_indices) < 2 continue end

            star_bin_pairs = Tuple{Float64, Float64}[]
            for (i, j) in combinations(1:length(ray_indices), 2)
                idx1 = ray_indices[i]
                idx2 = ray_indices[j]
                
                if info[idx1].dim_id == 0 || info[idx2].dim_id == 0
                    continue
                end

                actual_h = abs(X_norm[dim, idx1] - X_norm[dim, idx2])
                if 0 < actual_h <= delta_h
                    push!(star_bin_pairs, (Y[idx1], Y[idx2]))
                end
            end

            if isempty(star_bin_pairs) continue end

            p1 = [p[1] for p in star_bin_pairs]
            p2 = [p[2] for p in star_bin_pairs]
            gamma_i = 0.5 * mean((p1 .- p2).^2)
            mu_star = mean(Y[ray_indices])
            ecov_i = mean((p1 .- mu_star) .* (p2 .- mu_star))

            if !isnan(gamma_i) && !isnan(ecov_i)
                gamma_sum += gamma_i
                ecov_sum += ecov_i
                stars_with_data += 1
            end
        end

        if stars_with_data == 0
            @warn "No valid pairs found in the first bin (h <= $delta_h) for dimension $dim. Sensitivity index will be NaN."
            ST[dim] = NaN
            continue
        end

        avg_gamma = gamma_sum / stars_with_data
        avg_ecov = ecov_sum / stars_with_data
        ST[dim] = (avg_gamma + avg_ecov) / VY
    end
    return (ST = ST,)
end