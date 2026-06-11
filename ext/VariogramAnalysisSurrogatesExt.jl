module VariogramAnalysisSurrogatesExt

using VariogramAnalysis
using DataFrames
using Surrogates
using Statistics

import VariogramAnalysis: dvars_sensitivities, _dvars_integrated_variogram

function dvars_sensitivities(df::AbstractDataFrame, outvarname::Symbol; Hj::Float64=1.0, verbose::Bool=false)
    # --- Data Preparation ---
    df_norm = copy(df)
    for col in names(df_norm)
        min_val, max_val = minimum(df_norm[!, col]), maximum(df_norm[!, col])
        if max_val - min_val > 1e-9
            df_norm[!, col] = (df_norm[!, col] .- min_val) ./ (max_val - min_val)
        end
    end

    invar_names = [name for name in names(df) if name != String(outvarname)]
    X = [collect(row) for row in eachrow(Matrix(df_norm[!, invar_names]))]
    y = df_norm[!, outvarname]

    ninvars = length(invar_names)
    variance = var(y)

    # --- Surrogate Model Training ---
    verbose && println("Building and optimizing Anisotropic Kriging surrogate...")
    p_values = fill(2.0, ninvars)
    lower_bounds_theta = fill(1e-6, ninvars)
    upper_bounds_theta = fill(100.0, ninvars)

    surrogate = Kriging(X, y, lower_bounds_theta, upper_bounds_theta, p=p_values)
    phi_opt = surrogate.theta

    # --- Sensitivity Calculation ---
    sensitivities = [_dvars_integrated_variogram(variance, surrogate.theta[j], surrogate.p[j], Hj) for j in 1:ninvars]

    total_sensitivity = sum(sensitivities)
    ratios = total_sensitivity > 0 ? sensitivities ./ total_sensitivity : zeros(ninvars)

    return sensitivities, ratios, phi_opt, variance
end

end # module
