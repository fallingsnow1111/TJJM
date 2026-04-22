# Kaya Forecast Summary

- Base year: 2022
- Base CO2: 10612.201
- Forecast horizon: 2023-2035
- EI model: log(EI) = alpha + beta_A * log(A) + beta_S * log(S) + recent residual trend
- EI fit start year: 1990
- EI residual window: 4 years
- EI fit R^2: 0.9694

## baseline
- Peak year: 2030
- Peak CO2: 11167.063
- 2030 CO2: 11167.063
- 2035 CO2: 11109.014
- 2022-2035 cumulative change: 496.813

## high_growth
- Peak year: 2035
- Peak CO2: 11529.060
- 2030 CO2: 11415.805
- 2035 CO2: 11529.060
- 2022-2035 cumulative change: 916.859

## policy_lag
- Peak year: 2035
- Peak CO2: 11669.821
- 2030 CO2: 11483.825
- 2035 CO2: 11669.821
- 2022-2035 cumulative change: 1057.620

## policy_strengthening
- Peak year: 2030
- Peak CO2: 11071.419
- 2030 CO2: 11071.419
- 2035 CO2: 10945.700
- 2022-2035 cumulative change: 333.499

## Sensitivity
- Residual window 3 years: baseline peak year 2025
- Residual window 4 years: baseline peak year 2030
- Residual window 5 years: baseline peak year 2035

## Interpretation
This forecast is a deterministic Kaya-path projection under policy constraints. EI is not hand-set by a bridge; it is generated from historical structure plus a residual trend estimated from the most recent observations. If the implied peak year is still outside the target narrative, the first diagnostic should be the EI residual-window sensitivity rather than editing the historical panel.