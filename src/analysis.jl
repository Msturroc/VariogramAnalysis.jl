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
    # 1. Calculate overall variance from the star centres for normalisation
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
            # Get all points for this star's "ray" (centre + conditional samples)
            ray_indices = findall(p -> p.star_id == star && (p.dim_id == dim || p.dim_id == 0), info)
            if length(ray_indices) < 2 continue end

            # Find pairs of CONDITIONAL points within this ray that fall into the first distance bin
            star_bin_pairs = Tuple{Float64, Float64}[]
            for (i, j) in combinations(1:length(ray_indices), 2)
                idx1 = ray_indices[i]
                idx2 = ray_indices[j]
                
                # Ensure we are only pairing conditional points (dim_id > 0)
                if info[idx1].dim_id == 0 || info[idx2].dim_id == 0 continue end

                actual_h = abs(X_norm[dim, idx1] - X_norm[dim, idx2])
                if 0 < actual_h <= delta_h
                    push!(star_bin_pairs, (Y[idx1], Y[idx2]))
                end
            end

            if isempty(star_bin_pairs)
                continue # No pairs in the first bin for this star
            end

            # 3. Calculate the sectional variogram and covariogram for THIS STAR
            p1 = [p[1] for p in star_bin_pairs]
            p2 = [p[2] for p in star_bin_pairs]

            gamma_i = 0.5 * mean((p1 .- p2).^2)

            # The mean for the sectional covariogram MUST be the mean of the current ray's Y values
            mu_star = mean(Y[ray_indices])
            ecov_i = mean((p1 .- mu_star) .* (p2 .- mu_star))

            # 4. Add the sectional results to the running sums
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

        # 5. Average the sectional results and normalise to get the final index
        avg_gamma = gamma_sum / stars_with_data
        avg_ecov = ecov_sum / stars_with_data
        ST[dim] = (avg_gamma + avg_ecov) / VY
    end

    return (ST = ST,)
end