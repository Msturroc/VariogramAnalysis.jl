# src/bootstrap.jl (MODIFIED)

module VARSBootstrap

using Statistics, Random

using ..VariogramAnalysis 

export bootstrap_st!, rank_from_bootstrap, group_factors

"""
    bootstrap_st!(compute_st, Y, X, X_norm, info, N, d, delta_h; ...)

Resample stars with replacement and recompute ST for each replicate.

`compute_st` must be a callable with the signature:
  `compute_st(Y_b, X_b, X_norm_b, info_b, N, d, delta_h) -> (ST=::Vector)`
"""
function bootstrap_st!(compute_st, Y::Vector, X::Matrix, X_norm::Matrix, info::Vector,
                       N::Int, d::Int, delta_h::Float64;
                       num_boot::Int=100, ci_level::Float64=0.90, seed::Int=1234)

    Random.seed!(seed)

    star_to_indices = Dict{Int, Vector{Int}}()
    for (idx, meta) in enumerate(info)
        push!(get!(star_to_indices, meta.star_id, Int[]), idx)
    end

    # Point estimate on original design
    st_point = compute_st(Y, X, X_norm, info, N, d, delta_h).ST

    st_boot = Matrix{Float64}(undef, num_boot, d)

    low_q = (1 - ci_level) / 2
    high_q = 1 - low_q

    for b in 1:num_boot
        boot_star_ids = rand(1:N, N)

        boot_indices = Int[]
        for sid in boot_star_ids
            append!(boot_indices, star_to_indices[sid])
        end

        boot_Y = Y[boot_indices]
        boot_X = X[:, boot_indices] # <-- ADDED THIS LINE
        boot_Xn = X_norm[:, boot_indices]

        boot_info = similar(info, length(boot_indices))
        cursor = 1
        for (new_sid, sid) in enumerate(boot_star_ids)
            idxs = star_to_indices[sid]
            for k in 1:length(idxs)
                old = info[idxs[k]]
                boot_info[cursor] = (star_id=new_sid, dim_id=old.dim_id, step_id=old.step_id, h=old.h)
                cursor += 1
            end
        end

        # Compute ST on bootstrap resample
        res = compute_st(boot_Y, boot_X, boot_Xn, boot_info, N, d, delta_h) # <-- MODIFIED THIS LINE
        st_boot[b, :] = res.ST
    end

    st_ci = Vector{Tuple{Float64,Float64}}(undef, d)
    for i in 1:d
        sorted_vals = sort(view(st_boot, :, i))
        l = quantile(sorted_vals, low_q)
        u = quantile(sorted_vals, high_q)
        st_ci[i] = (l, u)
    end

    return (st_point=st_point, st_boot=st_boot, st_ci=st_ci)
end

# --- rank_from_bootstrap and group_factors remain unchanged ---
# (You can keep your existing versions of these two functions)
"""
    rank_from_bootstrap(...)
"""
function rank_from_bootstrap(st_boot::Matrix{Float64}, param_names::Vector{String})
    B, d = size(st_boot)
    rank_counts = Dict{Int, Vector{Int}}()
    for r in 1:d
        rank_counts[r] = zeros(Int, d)
    end

    for b in 1:B
        vals = st_boot[b, :]
        order = sortperm(vals; rev=true)
        rank_of = invperm(order)
        for f in 1:d
            rank_counts[rank_of[f]][f] += 1
        end
    end

    rank_mode = zeros(Int, d)
    rank_agreement = zeros(Float64, d)
    for f in 1:d
        best_r, best_c = 0, -1
        for r in 1:d
            if rank_counts[r][f] > best_c
                best_c = rank_counts[r][f]
                best_r = r
            end
        end
        rank_mode[f] = best_r
        rank_agreement[f] = best_c / B
    end

    return (rank_mode=rank_mode, rank_agreement=rank_agreement, rank_counts=rank_counts)
end

"""
    group_factors(...)
"""
function group_factors(st_boot::Matrix{Float64}, param_names::Vector{String};
                       tol::Float64=1e-2, num_groups::Int=2)
    d = size(st_boot, 2)
    med = [median(view(st_boot, :, i)) for i in 1:d]
    order = sortperm(med; rev=true)
    groups = fill(1, d)

    g = 1
    for k in 2:d
        if abs(med[order[k-1]] - med[order[k]]) > tol
            g += 1
        end
        groups[order[k]] = g
    end

    if g > num_groups
        gaps = [(abs(med[order[k]] - med[order[k+1]]), k) for k in 1:(d-1)]
        sort!(gaps, by=x->x[1])
        merged_breaks = Set{Int}()
        for m in 1:(g - num_groups)
            push!(merged_breaks, gaps[m][2])
        end
        
        g2 = 1
        groups2 = similar(groups)
        groups2[order[1]] = g2
        for k in 2:d
            if !(k-1 in merged_breaks) && abs(med[order[k-1]] - med[order[k]]) > tol
                g2 += 1
            end
            groups2[order[k]] = g2
        end
        groups = groups2
    end

    group_map = Dict{Int, Vector{String}}()
    for i in 1:d
        push!(get!(group_map, groups[i], String[]), param_names[i])
    end

    return (groups=groups, group_map=group_map)
end


end # end module