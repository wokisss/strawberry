# Thesis Result Section: Real-Resource-Calibrated Closed-Loop Validation

To connect the forecasting model comparison with greenhouse control relevance, a final validation stage was conducted using real resource and economic fields from the AGC 2019 dataset. The purpose was not to claim season-long commercial profit improvement, but to evaluate whether selected forecasting models can be compared under a closed-loop MPC framework with resource-cost estimates calibrated from recorded greenhouse data.

The official AGC economics rules were encoded from `Economics.pdf`, including heat cost, peak/off-peak electricity cost, tiered CO2 cost, crop maintenance cost, and tomato income based on Class A/B production with date- and Brix-dependent prices. This produced a compartment-level reference baseline from the recorded AGC data. In parallel, a daily positive-coefficient ridge estimator was fitted to map climate setpoints, weather summaries, and derived drive terms to recorded heat, electricity, CO2, and irrigation consumption.

Two models were selected for final closed-loop validation: `current_hybrid_transformer`, the strongest overall tracking baseline, and `itransformer_co2_residual`, the strongest CO2-aware closed-loop tracker. Both were evaluated with `GradientMPC` on the `Reference` compartment for 96-step rollouts across starts `0`, `96`, `192`, `288`, and `384`. Four resource weights were compared: `0.00`, `0.02`, `0.05`, and `0.08`.

| resource weight | model | objective | CO2air MAE | estimated resource cost |
| ---: | --- | ---: | ---: | ---: |
| 0.00 | `current_hybrid_transformer` | 0.0660 | 29.472 | 0.0127 EUR/m2 |
| 0.00 | `itransformer_co2_residual` | 0.0695 | 10.168 | 0.0094 EUR/m2 |
| 0.02 | `current_hybrid_transformer` | 0.0743 | 29.808 | 0.0123 EUR/m2 |
| 0.02 | `itransformer_co2_residual` | 0.0778 | 10.297 | 0.0096 EUR/m2 |
| 0.05 | `current_hybrid_transformer` | 0.0841 | 29.929 | 0.0114 EUR/m2 |
| 0.05 | `itransformer_co2_residual` | 0.0879 | 10.980 | 0.0085 EUR/m2 |
| 0.08 | `current_hybrid_transformer` | 0.0941 | 30.180 | 0.0111 EUR/m2 |
| 0.08 | `itransformer_co2_residual` | 0.0931 | 11.660 | 0.0076 EUR/m2 |

The results show two complementary patterns. First, `current_hybrid_transformer` remains the best overall tracking baseline at zero resource weight. Second, `itransformer_co2_residual` provides substantially stronger CO2 tracking and lower estimated resource cost across the selected-model comparison. Increasing the resource weight reduces estimated resource cost, but it also increases the optimized objective and gradually worsens CO2 tracking. The `w=0.05` setting is a practical middle point: it gives a clear cost reduction while keeping the CO2 degradation moderate. The `w=0.08` setting further reduces estimated cost, especially for `itransformer_co2_residual`, but the CO2 tracking penalty becomes larger.

This final validation supports a bounded thesis claim: the proposed control-oriented forecasting workflow can evaluate selected predictors under closed-loop MPC with real-AGC-resource-calibrated cost estimates. It does not prove real commercial net-profit improvement, because the current closed-loop surrogate does not model crop growth, yield, or fruit quality dynamics.
