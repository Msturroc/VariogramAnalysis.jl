# src/analysis.jl

using Statistics, Combinatorics

"""
    vars_analyse(Y::Vector, info::Vector, N::Int, d::Int, delta_h::Float64)

Performs VARS analysis on a set of model outputs to compute sensitivity indices.

This function calculates the VARS Total-Order (VARS-TO) index, which is a robust
and efficient equivalent to the Sobol Total-Order index.

# Arguments
- `Y::Vector{Float64}`: A vector of model outputs, corresponding to the points in `X`
  from `generate_vars_samples`.
- `info::Vector`: The metadata vector returned by `generate_vars_samples`.
- `N::Int`: The number of star centres used.
- `d::Int`: The number of parameters (dimensions).
- `delta_h::Float64`: The sampling resolution used.

# Returns
- A `NamedTuple` containing the sensitivity indices. For now, it contains:
  - `ST::Vector{Float64}`: The Total-Order sensitivity indices for each parameter.
"""
function vars_analyse(Y::Vector, info::Vector, N::Int, d::Int, delta_h::Float64)
    # 1. Calculate the overall variance of the model output (VY) from the star centres.
    # This serves as the normalisation factor for the sensitivity indices.
    centre_indices = findall(p -> p.dim_id == 0, info)
    if length(centre_indices) < 2
        error("Not enough star centre outputs to compute variance. Need at least 2.")
    end
    VY = var(Y[centre_indices])

    # If variance is negligible, all sensitivities are zero.
    if VY < 1e-12
        return (ST = zeros(d),)
    end

    ST = zeros(d)

    # 2. Calculate sensitivity index for each parameter dimension.
    for dim in 1:d
        gamma_sum = 0.0
        ecov_sum = 0.0
        stars_with_data = 0

        # 3. Iterate through each star to compute sectional variograms and covariograms.
        for star in 1:N
            # Get all points related to the current star and dimension.
            # This includes the centre and all points along the `dim`-th axis.
            traj_indices = findall(p -> p.star_id == star && (p.dim_id == dim || p.dim_id == 0), info)
            if length(traj_indices) < 2
                continue
            end

            traj_Y = Y[traj_indices]
            traj_info = info[traj_indices]

            # Find pairs of points separated by approximately delta_h.
            pairs = Tuple{Float64, Float64}[]
            for (i, j) in combinations(1:length(traj_Y), 2)
                # The distance `h` is pre-calculated in the info struct.
                # We are interested in pairs with a separation of delta_h.
                h_dist = abs(traj_info[i].h - traj_info[j].h)
                if isapprox(h_dist, delta_h, atol=1e-9)
                    push!(pairs, (traj_Y[i], traj_Y[j]))
                end
            end

            if isempty(pairs)
                continue
            end

            p1 = [p[1] for p in pairs]
            p2 = [p[2] for p in pairs]

            # Calculate the sectional variogram (gamma_i) for this star.
            gamma_i = 0.5 * mean((p1 .- p2).^2)

            # Calculate the sectional expected covariogram (ecov_i) for this star.
            mu_star_traj = mean(traj_Y) # Mean of all outputs in this section.
            ecov_i = mean((p1 .- mu_star_traj) .* (p2 .- mu_star_traj))

            if !isnan(gamma_i) && !isnan(ecov_i)
                gamma_sum += gamma_i
                ecov_sum += ecov_i
                stars_with_data += 1
            end
        end

        # 4. Average across all stars and normalise to get the final index.
        if stars_with_data > 0
            avg_gamma = gamma_sum / stars_with_data
            avg_ecov = ecov_sum / stars_with_data
            ST[dim] = (avg_gamma + avg_ecov) / VY
        else
            ST[dim] = NaN # Indicates no valid pairs were found for this dimension.
        end
    end

    return (ST = ST,)
end

"""
    vars_analyse(Y::Vector, info::Vector, N::Int, d::Int, delta_h::Float64)

Performs standard VARS analysis for independent, uniform parameters.

This function calculates the VARS Total-Order (VARS-TO) index by averaging
sectional variograms and covariograms across all star centres.

# Arguments
- `Y::Vector{Float64}`: A vector of model outputs.
- `info::Vector`: The metadata vector from `generate_vars_samples`.
- `N::Int`: The number of star centres.
- `d::Int`: The number of parameters.
- `delta_h::Float64`: The sampling resolution.

# Returns
- A `NamedTuple` containing `(ST = [...],)`.
"""
function vars_analyse(Y::Vector, info::Vector, N::Int, d::Int, delta_h::Float64)
    # 1. Calculate overall variance from the star centres
    centre_indices = findall(p -> p.dim_id == 0, info)
    if length(centre_indices) < 2
        error("Not enough star centre outputs to compute variance.")
    end
    VY = var(Y[centre_indices])
    if VY < 1e-12 return (ST = zeros(d),) end

    ST = zeros(d)

    for dim in 1:d
        gamma_sum = 0.0
        ecov_sum = 0.0
        stars_with_data = 0

        # 2. Loop through each star to perform SECTIONAL analysis
        for star in 1:N
            # Get all points on the "leg" of the current star for the current dimension
            traj_indices = findall(p -> p.star_id == star && p.dim_id == dim, info)
            
            # Also need the centre point for this star to calculate the sectional mean
            star_centre_idx = findfirst(p -> p.star_id == star && p.dim_id == 0, info)
            
            if isempty(traj_indices) || isnothing(star_centre_idx)
                continue
            end

            # Find pairs of points on this leg separated by exactly delta_h
            leg_points_Y = Y[traj_indices]
            leg_points_info = info[traj_indices]
            
            star_bin_pairs = Tuple{Float64, Float64}[]
            for (i, j) in combinations(1:length(leg_points_Y), 2)
                # Check if the distance between points is approximately delta_h
                actual_h = abs(leg_points_info[i].h - leg_points_info[j].h)
                if isapprox(actual_h, delta_h, atol=1e-9)
                    push!(star_bin_pairs, (leg_points_Y[i], leg_points_Y[j]))
                end
            end

            if isempty(star_bin_pairs)
                continue
            end

            # 3. Calculate sectional variogram and covariogram for THIS STAR's leg
            p1 = [p[1] for p in star_bin_pairs]
            p2 = [p[2] for p in star_bin_pairs]

            gamma_i = 0.5 * mean((p1 .- p2).^2)
            
            # The mean for the sectional covariogram is the mean of the entire trajectory
            # (the leg points plus the star's centre point)
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

Performs G-VARS analysis on model outputs from non-uniform, correlated inputs.

This function correctly implements the sectional analysis by calculating variograms and
covariograms for each star centre's ray individually before averaging. This is the
definitive, correct implementation.

# Arguments
- `Y::Vector{Float64}`: A vector of model outputs.
- `X_norm::Matrix{Float64}`: The `d x n_points` matrix of samples, normalised to the [0,1] range.
- `info::Vector`: The metadata vector from `generate_gvars_samples`.
- `N::Int`: The number of star centres used.
- `d::Int`: The number of parameters.
- `delta_h::Float64`: The resolution for binning the pairs. The first bin will be [0, delta_h].

# Returns
- A `NamedTuple` containing the sensitivity indices:
  - `ST::Vector{Float64}`: The G-VARS Total-Order sensitivity indices.
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
                
                # --- THIS IS THE CRITICAL FIX ---
                # Ensure we are only pairing conditional points (dim_id > 0).
                # The star center (dim_id = 0) must be excluded from the pairs.
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