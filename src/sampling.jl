# src/sampling.jl

using QuasiMonteCarlo

"""
    generate_vars_samples(N::Int, d::Int, delta_h::Float64; sampler=SobolSample())

Generates VARS star-based samples in the unit hypercube [0, 1]ᵈ.

This function creates a structured set of points for VARS analysis. It starts by
generating `N` star centres and then samples along the axis of each of the `d`
parameters from each centre.

# Arguments
- `N::Int`: The number of star centres to generate.
- `d::Int`: The number of parameters (dimensions).
- `delta_h::Float64`: The sampling resolution, i.e., the step size along each axis.
- `sampler`: A sampler from QuasiMonteCarlo.jl used to generate the star centres.
  Defaults to `SobolSample()`.

# Returns
- A `NamedTuple` with two fields:
  - `X::Matrix{Float64}`: A `d x n_points` matrix of all generated sample points.
  - `info::Vector`: A vector of `NamedTuple`s, where each element contains metadata
    for the corresponding point in `X`: `(star_id, dim_id, h)`.
    - `star_id`: The index of the star centre (1 to N).
    - `dim_id`: The dimension along which the point was perturbed. `0` indicates the point is a star centre itself.
    - `h`: The perturbation distance from the centre. `0.0` for centres.
"""
function generate_vars_samples(N::Int, d::Int, delta_h::Float64; sampler=SobolSample())
    # 1. Generate N star centres using a quasi-random sequence for good space-filling properties.
    centres = QuasiMonteCarlo.sample(N, zeros(d), ones(d), sampler)

    # Pre-allocate for efficiency, though dynamic pushing is fine for moderate sizes.
    point_vectors = Vector{Float64}[]
    point_info = NamedTuple{(:star_id, :dim_id, :h), Tuple{Int, Int, Float64}}[]

    for i in 1:N
        centre = centres[:, i]

        # Add the star centre itself to the sample set.
        push!(point_vectors, centre)
        push!(point_info, (star_id=i, dim_id=0, h=0.0))

        # 2. For each dimension, generate points along its axis.
        for j in 1:d
            c_dim = centre[j]

            # Create a unique, sorted set of points along the axis from 0 to 1,
            # ensuring the centre's coordinate is included.
            trajectory_values = sort(unique(vcat(c_dim, 0.0:delta_h:1.0)))

            for val in trajectory_values
                # We only need to add the perturbed points, not the centre again.
                if val != c_dim
                    new_point = copy(centre)
                    new_point[j] = val
                    push!(point_vectors, new_point)

                    # Record metadata: which star, which dimension, and how far.
                    h = abs(val - c_dim)
                    push!(point_info, (star_id=i, dim_id=j, h=h))
                end
            end
        end
    end

    # Convert the vector of vectors to a single matrix for efficient model evaluation.
    X = hcat(point_vectors...)

    return (X=X, info=point_info)
end

# src/sampling.jl

# Add this function to your existing sampling.jl file.
# Make sure you have `using LinearAlgebra, QuasiMonteCarlo, Random` at the top.

"""
    generate_gvars_samples(parameters::OrderedDict, N::Int, corr_mat::AbstractMatrix, num_dir_samples::Int;
                           sampler=SobolSample(), seed=nothing, use_fictive_corr=true)

Generates the full sample set for G-VARS analysis, accounting for non-uniform,
correlated parameter distributions.

# Arguments
- `parameters::OrderedDict`: Defines the marginal distribution for each parameter.
- `N::Int`: The number of star centres to generate.
- `corr_mat::AbstractMatrix`: The desired correlation matrix in the original parameter space.
- `num_dir_samples::Int`: The number of conditional samples to draw for each parameter at each star centre.
- `sampler`: A sampler from QuasiMonteCarlo.jl for generating the initial star centres.
- `seed`: An optional random seed for reproducibility.
- `use_fictive_corr::Bool`: If true, performs the Nataf transformation. If false, assumes `corr_mat` is already the fictive matrix.

# Returns
- A `NamedTuple` with two fields:
  - `X::Matrix{Float64}`: A `d x n_points` matrix of all generated sample points in their original distributions.
  - `info::Vector`: Metadata for each point: `(star_id, dim_id, sample_idx)`.
    - `star_id`: The index of the star centre.
    - `dim_id`: The dimension being sampled. `0` for star centres.
    - `sample_idx`: The index of the sample along a "ray". `0` for star centres.
"""
function generate_gvars_samples(parameters::OrderedDict, N::Int, corr_mat::AbstractMatrix, num_dir_samples::Int;
                                sampler=SobolSample(), seed::Union{Nothing, Int}=nothing, use_fictive_corr::Bool=true)
    
    d = length(parameters)
    rng = isnothing(seed) ? Random.default_rng() : Random.MersenneTwister(seed)

    # --- Step 1: Compute the Fictive Correlation Matrix ---
    fictive_corr = if use_fictive_corr
        map_to_fictive_corr(parameters, corr_mat)
    else
        corr_mat # Use the provided matrix directly
    end
    
    # --- Step 2: Generate Correlated Star Centres in Standard Normal Space ---
    # Generate N uniform samples and convert to standard normal
    uniform_samples = QuasiMonteCarlo.sample(N, d, sampler)
    y_centres = quantile.(Normal(0, 1), uniform_samples') # N x d matrix

    # Introduce correlation using the Cholesky decomposition of the fictive matrix
    C = cholesky(fictive_corr).L
    z_centres = Matrix((C * y_centres')') # N x d matrix of correlated normal samples

    # --- Step 3: Generate Conditional Samples in Standard Normal Space ---
    all_points_z = [z_centres] # A list to hold all batches of normal-space points
    
    # Pre-calculate conditional means and standard deviations for efficiency
    cond_means_factors = zeros(d, d - 1)
    cond_std_devs = zeros(d)

    for i in 1:d
        noti = setdiff(1:d, i) # Indices of all other dimensions
        Σ_ii = fictive_corr[i, i]
        Σ_inoti = fictive_corr[i:i, noti]
        Σ_notinoti_inv = inv(fictive_corr[noti, noti])
        
        cond_var = Σ_ii - (Σ_inoti * Σ_notinoti_inv * Σ_inoti')[1]
        cond_std_devs[i] = sqrt(max(0, cond_var))
        cond_means_factors[i, :] = (Σ_inoti * Σ_notinoti_inv)[:]
    end

    for i in 1:d
        noti = setdiff(1:d, i)
        
        # Calculate the conditional mean for each star centre
        # μ_{i|¬i} = z_¬i * (Σ_¬i,¬i)⁻¹ * Σ_¬i,i
        cond_means = z_centres[:, noti] * cond_means_factors[i, :]
        
        # Generate random draws and scale them
        stnrm_base = randn(rng, N, num_dir_samples)
        
        # Create the conditional samples for dimension `i`
        z_conditional_i = cond_means .+ stnrm_base .* cond_std_devs[i] # N x num_dir_samples
        
        # Assemble the full `d`-dimensional points for this conditional sample set
        z_points_for_dim_i = zeros(N * num_dir_samples, d)
        for s in 1:num_dir_samples
            start_idx = (s - 1) * N + 1
            end_idx = s * N
            
            z_points_for_dim_i[start_idx:end_idx, :] = z_centres
            z_points_for_dim_i[start_idx:end_idx, i] = z_conditional_i[:, s]
        end
        push!(all_points_z, z_points_for_dim_i)
    end

    # --- Step 4: Assemble Final Sample Matrix and Info ---
    # Combine all points (centres and conditional) into one large matrix
    combined_z = vcat(all_points_z...) # N_total x d matrix

    # Create the corresponding info vector
    info = vcat(
        [(star_id=i, dim_id=0, sample_idx=0) for i in 1:N], # Info for centres
        vcat([[(star_id=star, dim_id=dim, sample_idx=s) for star in 1:N for s in 1:num_dir_samples] for dim in 1:d]...)
    )

    # --- Step 5: Transform all points from Normal Space to Original Distributions ---
    final_samples_X = normal_to_original_dist(combined_z, parameters)

    return (X = Matrix(final_samples_X'), info = info)
end