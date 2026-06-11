using PyCall

# This constant holds the Python code that defines the patch.
# It does NOT apply the patch, it just defines the necessary functions in the Python namespace.
const PATCH_DEFINITIONS = py"""
import numpy as np
# Backward-compat shim for removed NumPy aliases
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "bool"):
    np.bool = np.bool_

import pandas as pd
from varstool import VARS, GVARS, Model
from varstool.sensitivity_analysis import vars_funcs
from tqdm.auto import tqdm
from collections import OrderedDict

# --- ROBUST MONKEY-PATCH SETUP ---

# 1. Global storage for our results.
results_storage = {}

# 2. A complete, correct copy of the original bootstrapping function.
def bootstrapping_patched(
    num_stars, pair_df, df, cov_section_all, bootstrap_size, bootstrap_ci,
    delta_h, ivars_scales, parameters, st_factor_ranking, ivars_factor_ranking,
    grouping_flag, num_grps, progress=False
):
    result_bs_sobol = pd.DataFrame()
    
    bs = tqdm(range(0, bootstrap_size), desc='bootstrapping (patched)', disable=not progress, dynamic_ncols=True)
    for i in bs:
        bootstrap_rand = np.random.choice(list(range(0, num_stars)), size=len(range(0, num_stars)), replace=True).tolist()
        bootstrapped_pairdf = pd.concat([pair_df.loc[pd.IndexSlice[i, :, :, :], :] for i in bootstrap_rand])
        bootstrapped_df = pd.concat([df.loc[pd.IndexSlice[i, :, :], :] for i in bootstrap_rand])
        bootstrapped_cov_section_all = pd.concat([cov_section_all.loc[pd.IndexSlice[i, :]] for i in bootstrap_rand])
        
        bootstrapped_variogram = vars_funcs.variogram(bootstrapped_pairdf)
        bootstrapped_ecovariogram = vars_funcs.e_covariogram(bootstrapped_cov_section_all)
        bootstrapped_var = np.nanvar(bootstrapped_df.iloc[:, -1].unique(), ddof=1)
        bootstrapped_sobol = vars_funcs.sobol_eq(bootstrapped_variogram, bootstrapped_ecovariogram, bootstrapped_var, delta_h)
        
        bootstrapped_sobol_df = bootstrapped_sobol.to_frame().transpose()
        result_bs_sobol = pd.concat([result_bs_sobol, bootstrapped_sobol_df])

    # --- OUR ADDITION: Save the data to global storage ---
    results_storage['result_bs_sobol'] = result_bs_sobol
    
    # Return dummy values to satisfy the original function's return signature
    stlb = result_bs_sobol.quantile((1 - bootstrap_ci) / 2).rename('').to_frame().transpose()
    stub = result_bs_sobol.quantile(1 - ((1 - bootstrap_ci) / 2)).rename('').to_frame().transpose()
    
    dummy_df = pd.DataFrame()
    dummy_ranking = pd.DataFrame()
    if grouping_flag:
        return dummy_df, dummy_df, dummy_df, dummy_df, stlb, stub, dummy_df, dummy_df, dummy_ranking, dummy_ranking, dummy_df, dummy_df, dummy_df, dummy_df
    else:
        return dummy_df, dummy_df, dummy_df, dummy_df, stlb, stub, dummy_df, dummy_df, dummy_ranking, dummy_ranking

# 3. Define our wrapper function that performs the swap
def patched_run_online(self, original_run_func):
    original_bootstrap_in_module = vars_funcs.bootstrapping
    try:
        # Replace the function in the module with our patch
        vars_funcs.bootstrapping = bootstrapping_patched
        # Call the original run_online method
        original_run_func(self)
    finally:
        # ALWAYS restore the original function
        vars_funcs.bootstrapping = original_bootstrap_in_module
"""

"""
    with_patched_varstool(f::Function)

A helper function that monkey-patches the Python `varstool` library
for the duration of the function `f`. It guarantees the original methods
are restored, ensuring test isolation.
"""
function with_patched_varstool(f::Function)
    # Ensure the patch definitions exist in the Python main namespace
    py"$PATCH_DEFINITIONS"

    # Get Python object handles
    VARS = py"VARS"
    GVARS = py"GVARS"
    patched_run_online_py = py"patched_run_online"

    # 1. Store the original, unmodified methods
    original_vars_run_online = VARS.run_online
    original_gvars_run_online = GVARS.run_online

    println("[TEST HELPER] Patching Python methods...")
    try
        # 2. Apply the patch using the stored originals
        py"""
        VARS.run_online = lambda self: $patched_run_online_py(self, $original_vars_run_online)
        GVARS.run_online = lambda self: $patched_run_online_py(self, $original_gvars_run_online)
        """
        # 3. Run the user's test code
        f()

    finally
        # 4. This block is guaranteed to run, restoring the original state
        println("[TEST HELPER] Restoring original Python methods.")
        VARS.run_online = original_vars_run_online
        GVARS.run_online = original_gvars_run_online
    end
end