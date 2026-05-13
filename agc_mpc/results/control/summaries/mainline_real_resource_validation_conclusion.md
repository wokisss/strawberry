# Mainline Real-Resource Validation Conclusion

This validation links the forecasting-model mainline to resource-aware greenhouse control using real AGC resource and economics data.

What was completed:

- The AGC Economics PDF rules were encoded for heat, electricity, CO2, crop maintenance, tomato Class A/B income, date-dependent prices, and Brix-dependent price interpolation.
- A compartment-level AGC baseline was generated from `Resources.csv`, `Production.csv`, `TomQuality.csv`, and `CropParameters.csv`.
- A positive-coefficient daily resource estimator was calibrated from recorded `GreenhouseClimate.csv`, `Weather/Weather.csv`, and `Resources.csv`.
- `current_hybrid_transformer` and `itransformer_co2_residual` were rerun for `GradientMPC`, `Reference`, 96 steps, starts `0`, `96`, `192`, `288`, and `384`, with both tracking-only `w=0.00` and low resource-aware `w=0.05` profiles.
- A narrow sensitivity check was completed for `w=0.02`, `w=0.05`, and `w=0.08`.

Main result:

| profile | predictor | objective | CO2 MAE | estimated resource cost |
| --- | --- | ---: | ---: | ---: |
| `real_resource_w000` | `current_hybrid_transformer` | 0.0660 | 29.472 | 0.0127 EUR/m2 |
| `real_resource_w000` | `itransformer_co2_residual` | 0.0695 | 10.168 | 0.0094 EUR/m2 |
| `real_resource_w005` | `current_hybrid_transformer` | 0.0841 | 29.929 | 0.0114 EUR/m2 |
| `real_resource_w005` | `itransformer_co2_residual` | 0.0879 | 10.980 | 0.0085 EUR/m2 |
| `real_resource_w008` | `current_hybrid_transformer` | 0.0941 | 30.180 | 0.0111 EUR/m2 |
| `real_resource_w008` | `itransformer_co2_residual` | 0.0931 | 11.660 | 0.0076 EUR/m2 |

Interpretation:

- `current_hybrid_transformer` remains the strongest overall tracking baseline by mean closed-loop objective under tracking-only control.
- `itransformer_co2_residual` remains the strongest CO2-aware closed-loop tracker and has lower estimated resource cost in this selected-model comparison.
- The low resource-aware setting `w=0.05` reduces estimated resource cost for both selected models, but it increases the optimized objective and slightly worsens CO2 tracking.
- `w=0.08` produces additional estimated cost reduction, but the CO2 tracking penalty is larger; `w=0.05` is the more defensible balanced setting.
- The result supports a thesis-facing claim that selected forecasting models can be evaluated under a real-AGC-resource-calibrated MPC framework.

Boundary:

- This is not a season-long net-profit claim.
- The MPC rollout does not include crop/yield/quality dynamics, so it cannot prove commercial profit improvement.
- The valid claim is resource-cost and tracking trade-off comparison for selected closed-loop MPC rollouts.
