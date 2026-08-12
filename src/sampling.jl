using QuasiMonteCarlo, Random, LinearAlgebra, OrderedCollections, Distributions

"""
    _ensure_pos_def(mat::AbstractMatrix, tol=1e-8)

Helper function to ensure a matrix is symmetric and positive definite.
"""
function _ensure_pos_def(mat::AbstractMatrix, tol=1e-8)

    E = eigen(Symmetric(mat))

    new_eigenvalues = max.(E.values, tol)

    # Reconstruct the matrix
    new_mat = E.vectors * Diagonal(new_eigenvalues) * E.vectors' # Use adjoint instead of inv for stability

    # Rescale to ensure the diagonal is all 1s
    D = Diagonal(1 ./ sqrt.(diag(new_mat)))
    corrected_mat_almost_symm = D * new_mat * D

    # Enforce perfect symmetry to avoid floating point errors
    corrected_mat = (corrected_mat_almost_symm + corrected_mat_almost_symm') / 2

    return corrected_mat
end


"""
    generate_vars_samples(parameters::OrderedDict, N::Int, delta_h::Float64;
                          sampler_type="lhs", ray_logic=:relative, seed=nothing)

Generate the star-based design of experiments for standard (uncorrelated) VARS.

`N` star centres are drawn in the unit hypercube with the chosen sampler
(`"lhs"` or `"sobol_shift"`), and from every centre a ray of points with pitch
`delta_h` is laid out along each of the `d = length(parameters)` dimensions.
`ray_logic` controls how the rays are constructed:

- `:relative` — steps of `delta_h` outwards from the centre value.
- `:shifted_grid` — points on a fixed grid of pitch `delta_h`, shifted so the
  centre lies on a grid line; all pairwise distances along a ray are then
  exact multiples of `delta_h`.

Pass `seed` for a reproducible design; the global RNG is left untouched.

Returns `(X_norm, info)` where `X_norm` is a `d × n` matrix of points in
`[0, 1]^d` and `info` is a vector of `(star_id, dim_id, step_id, h)` named
tuples, one per column. Centres have `dim_id == 0`; ray points record the
dimension they vary along and their signed step count from the centre.
"""
function generate_vars_samples(parameters::OrderedDict, N::Int, delta_h::Float64;
                               sampler_type::String="lhs",
                               ray_logic::Symbol=:relative,
                               seed::Union{Nothing, Int}=nothing)
    d = length(parameters)
    if N == 0
        # Handle the edge case where no star centers are requested.
        return (Matrix{Float64}(undef, d, 0), NamedTuple{(:star_id, :dim_id, :step_id, :h), Tuple{Int, Int, Int, Float64}}[])
    end

    local sampler
    if sampler_type == "lhs"
        if !isnothing(seed)
            rng = Random.MersenneTwister(seed)
            sampler = LatinHypercubeSample(rng)
        else
            sampler = LatinHypercubeSample()
        end
    elseif sampler_type == "sobol_shift"
        if !isnothing(seed)
            rng = Random.MersenneTwister(seed)
            sampler = SobolSample(QuasiMonteCarlo.Shift(rng))
        else
            sampler = SobolSample(QuasiMonteCarlo.Shift())
        end
    else
        error("Unsupported sampler_type: '$sampler_type'. Please use 'lhs' or 'sobol_shift'.")
    end

    centres = QuasiMonteCarlo.sample(N, d, sampler)

    point_vectors = Vector{Float64}[]
    point_info = NamedTuple{(:star_id, :dim_id, :step_id, :h), Tuple{Int, Int, Int, Float64}}[]

    for i in 1:N
        centre = centres[:, i]
        push!(point_vectors, centre)
        push!(point_info, (star_id=i, dim_id=0, step_id=0, h=0.0))

        for j in 1:d
            c_dim = centre[j]

            if ray_logic == :relative
                max_steps = floor(Int, 1 / delta_h)
                for step in 1:max_steps
                    h = step * delta_h
                    if c_dim + h <= 1.0
                        new_point = copy(centre); new_point[j] = c_dim + h
                        push!(point_vectors, new_point)
                        push!(point_info, (star_id=i, dim_id=j, step_id=step, h=h))
                    end
                    if c_dim - h >= 0.0
                        new_point = copy(centre); new_point[j] = c_dim - h
                        push!(point_vectors, new_point)
                        push!(point_info, (star_id=i, dim_id=j, step_id=-step, h=h))
                    end
                end
            elseif ray_logic == :shifted_grid
                # Rays follow a fixed grid of pitch delta_h, shifted so that the
                # centre lies on a grid line; this keeps cross-section spacing exact.
                traj_values = filter(x -> x != c_dim, unique(vcat(c_dim % delta_h : delta_h : 1.0, c_dim % delta_h : -delta_h : 0.0)))
                for traj_val in traj_values
                    new_point = copy(centre)
                    new_point[j] = clamp(traj_val, 0.0, 1.0)
                    h_dist = abs(traj_val - c_dim)
                    step = round(Int, h_dist / delta_h)
                    step_id = (traj_val > c_dim) ? step : -step
                    push!(point_vectors, new_point)
                    push!(point_info, (star_id=i, dim_id=j, step_id=step_id, h=h_dist))
                end
            else
                error("Unsupported ray_logic: '$ray_logic'. Please use :relative or :shifted_grid.")
            end
        end
    end

    X_norm = hcat(point_vectors...)
    return (X_norm, point_info)
end

"""
    generate_gvars_samples(parameters::OrderedDict, N::Int, corr_mat::AbstractMatrix,
                           num_dir_samples::Int, delta_h::Float64;
                           seed=nothing, use_fictive_corr=true, sampler_type="sobol")

Generate the design of experiments for G-VARS (correlated inputs), replicating
the conditional sampling logic of the original Python `varstool` implementation.

Star centres are drawn as correlated standard normals via a Cholesky factor of
the (fictive) correlation matrix and mapped to the target marginals by the
inverse-CDF transform. For each dimension, `num_dir_samples` ray points per
star are drawn from the conditional normal distribution of that dimension
given the others.

When `use_fictive_corr` is true, `corr_mat` is interpreted as the desired
correlation between the actual parameters and is first mapped to the
normal-space "fictive" correlation with [`map_to_fictive_corr`](@ref);
otherwise `corr_mat` is used directly in normal space. Either way the matrix
is nudged to the nearest positive-definite correlation matrix if needed.

Returns `(X, info)` where `X` is a `d × N * (1 + d * num_dir_samples)` matrix
of samples in the original parameter space. In `info`, `step_id` holds the
index of the directional sample (rays have no ordered steps here) and `h` is
`NaN`.
"""
function generate_gvars_samples(parameters::OrderedDict, N::Int, corr_mat::AbstractMatrix, num_dir_samples::Int, delta_h::Float64;
                                seed::Union{Nothing, Int}=nothing, use_fictive_corr::Bool=true, sampler_type::String="sobol")

    d = length(parameters)
    rng = isnothing(seed) ? Random.default_rng() : Random.MersenneTwister(seed)

    # 1. Compute Fictive Correlation Matrix
    fictive_corr_raw = use_fictive_corr ? map_to_fictive_corr(parameters, corr_mat) : corr_mat
    fictive_corr = _ensure_pos_def(fictive_corr_raw)

    # 2. Generate Correlated Standard Normal Star Centres
    sampler = if sampler_type == "lhs" || sampler_type == "plhs"
        LatinHypercubeSample(rng)
    elseif sampler_type == "sobol"
        SobolSample()
    else
        error("Unsupported sampler type: $sampler_type")
    end

    uniform_centres = QuasiMonteCarlo.sample(N, d, sampler)
    u_clamped = clamp.(uniform_centres, 1e-15, 1.0 - 1e-15)
    y_centres = quantile.(Normal(0, 1), u_clamped)

    C = cholesky(fictive_corr).L
    z_centres = C * y_centres # Result is d x N

    # 3. Pre-compute conditional mean/variance factors
    cond_mean_factors = [zeros(1, d - 1) for _ in 1:d]
    cond_std_devs = zeros(d)

    for i in 1:d
        noti = setdiff(1:d, i)
        Σ_ii = fictive_corr[i, i]
        Σ_inoti = fictive_corr[i:i, noti]
        Σ_notinoti_inv = inv(fictive_corr[noti, noti])

        cond_var = Σ_ii - (Σ_inoti * Σ_notinoti_inv * Σ_inoti')[1]
        cond_std_devs[i] = sqrt(max(0, cond_var))
        cond_mean_factors[i] = Σ_inoti * Σ_notinoti_inv
    end

    # 4. Generate Directional Samples (Star Rays) using conditional distributions
    all_points_z = Vector{Matrix{Float64}}()
    push!(all_points_z, z_centres) # Add the centres first

    info_list = [(star_id=i, dim_id=0, step_id=0, h=0.0) for i in 1:N]

    # For each dimension, generate all its rays across all star centres
    for i in 1:d
        noti = setdiff(1:d, i)

        # Calculate conditional mean for each star centre
        # z_centres[noti, :] gives a (d-1) x N matrix
        cond_means = (cond_mean_factors[i] * z_centres[noti, :])' # Result is N x 1

        # Generate random numbers for the rays
        stnrm_base = randn(rng, N, num_dir_samples) # N x num_dir_samples

        # Create the conditional samples for dimension i
        z_conditional_i = cond_means .+ stnrm_base .* cond_std_devs[i] # N x num_dir_samples

        # Assemble the full z vectors for these ray points
        # Each column is a point
        z_ray_points = similar(z_centres, d, N * num_dir_samples)
        for s in 1:num_dir_samples
            start_idx = (s - 1) * N + 1
            end_idx = s * N

            # Copy the centres as the base
            z_ray_points[:, start_idx:end_idx] = z_centres
            # Overwrite the i-th dimension with the conditional samples
            z_ray_points[i, start_idx:end_idx] = z_conditional_i[:, s]

            # Add info for these points
            for star_idx in 1:N
                # Note: step_id and h are not well-defined here as in VARS.
                # We use the sample index `s` as a proxy for step_id.
                push!(info_list, (star_id=star_idx, dim_id=i, step_id=s, h=NaN))
            end
        end
        push!(all_points_z, z_ray_points)
    end

    # 5. Combine all points and transform to original parameter space
    combined_z = hcat(all_points_z...) # Concatenate all matrices column-wise

    X = normal_to_original_dist(combined_z', parameters)'

    return (Matrix(X), info_list)
end
