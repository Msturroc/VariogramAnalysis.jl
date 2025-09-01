using PyCall
using Conda

@info "VariogramAnalysis.jl build: Setting up Python dependencies..."

try
    # Step 1: Find the Python executable that PyCall is configured to use.
    python_exe = PyCall.python

    # Step 2: Determine the path to the corresponding 'pip' executable.
    # It lives in the same directory as the python executable.
    # We handle both Windows ('Scripts/pip.exe') and Unix-like ('bin/pip') cases.
    pip_exe = joinpath(dirname(python_exe), Sys.iswindows() ? "pip.exe" : "pip")

    if !isfile(pip_exe)
        error("Could not find pip executable at expected location: $pip_exe")
    end

    @info "Using pip executable at: $pip_exe"

    # Step 3: Use Julia's `run()` command to execute pip directly.
    @info "Installing 'varstool', 'numba', and other dependencies with pip..."
    run(`$pip_exe install varstool numba pandas numpy scipy tqdm joblib`)

    @info "Python dependencies configured successfully."

catch e
    @error "Error during build process. Python dependencies may not be installed correctly."
    @error "Please check your Conda and Python setup."
    rethrow(e)
end