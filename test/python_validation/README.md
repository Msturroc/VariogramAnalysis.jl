# Python validation suite

These tests validate `VariogramAnalysis.jl` against the original Python
implementation, [vars-tool](https://github.com/vars-tool/vars-tool). They need
a Python environment with `varstool` installed and are therefore not part of
the default `Pkg.test` run.

To run them, from the repository root:

```bash
# 1. Set up the Julia environment for this suite
julia --project=test/python_validation -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'

# 2. Install the Python dependencies (varstool, numba, ...) into PyCall's Python
julia --project=test/python_validation test/python_validation/setup.jl

# 3. Run the validation tests
julia --project=test/python_validation test/python_validation/runtests.jl
```
