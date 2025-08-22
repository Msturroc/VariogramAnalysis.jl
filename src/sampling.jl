# src/sampling.jl

using QuasiMonteCarlo, Random, LinearAlgebra, OrderedCollections, Distributions

"""
    _ensure_pos_def(mat::AbstractMatrix, tol=1e-8)

Helper function to ensure a matrix is symmetric and positive definite.
"""
function _ensure_pos_def(mat::AbstractMatrix, tol=1e-8)
    # --- CORRECTED EIGENDECOMPOSITION ---
    # Use Symmetric() to enforce symmetry and ensure real eigenvalues/vectors
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
    generate_vars_samples(...)
"""
function generate_vars_samples(parameters::OrderedDict, N::Int, delta_h::Float64; seed::Union{Nothing, Int}=nothing)
    d = length(parameters)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    sampler = SobolSample(R=OwenScramble(base=2))
    
    centres = QuasiMonteCarlo.sample(N, d, sampler)
    point_vectors = Vector{Float64}[]
    point_info = NamedTuple{(:star_id, :dim_id, :h), Tuple{Int, Int, Float64}}[]

    for i in 1:N
        centre = centres[:, i]
        push!(point_vectors, centre)
        push!(point_info, (star_id=i, dim_id=0, h=0.0))

        for j in 1:d
            c_dim = centre[j]
            forward_traj = c_dim:delta_h:1.0
            backward_traj = c_dim-delta_h:-delta_h:0.0
            trajectory_values = sort(unique(vcat(forward_traj, backward_traj)))
            
            for val in trajectory_values
                if val != c_dim
                    new_point = copy(centre)
                    new_point[j] = val
                    push!(point_vectors, new_point)
                    h = abs(val - c_dim)
                    push!(point_info, (star_id=i, dim_id=j, h=h))
                end
            end
        end
    end

    X = hcat(point_vectors...)
    return (X=X, info=point_info)
end


"""
    generate_gvars_samples(...)
"""
function generate_gvars_samples(parameters::OrderedDict, N::Int, corr_mat::AbstractMatrix, num_dir_samples::Int;
                                seed::Union{Nothing, Int}=nothing, use_fictive_corr::Bool=true)
    
    d = length(parameters)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    sampler = SobolSample(R=OwenScramble(base=2))

    fictive_corr_raw = use_fictive_corr ? map_to_fictive_corr(parameters, corr_mat) : corr_mat
    
    fictive_corr = _ensure_pos_def(fictive_corr_raw)
    
    uniform_samples = QuasiMonteCarlo.sample(N, d, sampler)
    y_centres = quantile.(Normal(0, 1), uniform_samples')

    C = cholesky(fictive_corr).L
    z_centres = Matrix((C * y_centres')')

    all_points_z = [z_centres]
    
    cond_means_factors = zeros(d, d - 1)
    cond_std_devs = zeros(d)

    for i in 1:d
        noti = setdiff(1:d, i)
        Σ_ii = fictive_corr[i, i]
        Σ_inoti = fictive_corr[i:i, noti]
        Σ_notinoti_inv = inv(fictive_corr[noti, noti])
        
        cond_var = Σ_ii - (Σ_inoti * Σ_notinoti_inv * Σ_inoti')[1]
        cond_std_devs[i] = sqrt(max(0, cond_var))
        cond_means_factors[i, :] = (Σ_inoti * Σ_notinoti_inv)[:]
    end

    for i in 1:d
        noti = setdiff(1:d, i)
        cond_means = z_centres[:, noti] * cond_means_factors[i, :]
        stnrm_base = randn(N, num_dir_samples)
        z_conditional_i = cond_means .+ stnrm_base .* cond_std_devs[i]
        
        z_points_for_dim_i = zeros(N * num_dir_samples, d)
        for s in 1:num_dir_samples
            start_idx = (s - 1) * N + 1
            end_idx = s * N
            z_points_for_dim_i[start_idx:end_idx, :] = z_centres
            z_points_for_dim_i[start_idx:end_idx, i] = z_conditional_i[:, s]
        end
        push!(all_points_z, z_points_for_dim_i)
    end

    combined_z = vcat(all_points_z...)

    info = vcat(
        [(star_id=i, dim_id=0, sample_idx=0) for i in 1:N],
        vcat([[(star_id=star, dim_id=dim, sample_idx=s) for star in 1:N for s in 1:num_dir_samples] for dim in 1:d]...)
    )

    final_samples_X = normal_to_original_dist(combined_z, parameters)

    return (X = Matrix(final_samples_X'), info = info)
end