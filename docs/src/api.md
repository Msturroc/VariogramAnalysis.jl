# API Reference

## Sampling

```@docs
VariogramAnalysis.sample
VariogramAnalysis.generate_vars_samples
VariogramAnalysis.generate_gvars_samples
```

## Analysis

```@docs
VariogramAnalysis.vars_analyse
VariogramAnalysis.gvars_analyse
```

## Bootstrap

```@docs
VariogramAnalysis.VARSBootstrap.bootstrap_st!
VariogramAnalysis.VARSBootstrap.rank_from_bootstrap
VariogramAnalysis.VARSBootstrap.group_factors
```

## G-VARS correlation utilities

```@docs
VariogramAnalysis.map_to_fictive_corr
VariogramAnalysis.rx_to_rn
VariogramAnalysis.rn_to_rx
VariogramAnalysis.normal_to_original_dist
```

## D-VARS (extensions)

```@docs
VariogramAnalysis.dvars_sensitivities
VariogramAnalysis.dvars_sensitivities_robust
```
