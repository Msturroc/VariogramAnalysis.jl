# One-time setup for the Python validation suite: installs the Python
# packages (varstool etc.) into the Python environment used by PyCall.
using PyCall

@info "Setting up Python dependencies for the validation suite..."

python_exe = PyCall.python

# On Windows, pip is in the 'Scripts' subdirectory of the Python installation.
# On Unix-like systems, it's typically in the same 'bin' directory as the python executable.
pip_exe = if Sys.iswindows()
    joinpath(dirname(python_exe), "Scripts", "pip.exe")
else
    joinpath(dirname(python_exe), "pip")
end

if !isfile(pip_exe)
    error("Could not find pip executable at expected location: $pip_exe")
end

@info "Using pip executable at: $pip_exe"
@info "Installing 'varstool', 'numba', and other dependencies with pip..."
run(`$pip_exe install varstool numba pandas numpy scipy tqdm joblib`)

@info "Python dependencies configured successfully."
