# Economic Resource MPC Sweep

Mean values are computed across the requested rollout starts. Objective values are not directly comparable across weights because the resource term changes the optimized objective; use tracking errors and resource proxy changes for the trade-off.

| predictor | weight | resource proxy | resource change | Tair MAE | Rhair MAE | CO2 MAE | CO2 change | action TV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_hybrid_transformer | 0.00 | 0.364 | +0.0% | 0.477 | 1.218 | 30.761 | +0.0% | 4.556 |
| current_hybrid_transformer | 0.05 | 0.328 | -9.8% | 0.511 | 1.178 | 31.396 | +2.1% | 4.067 |
| current_hybrid_transformer | 0.15 | 0.310 | -14.9% | 0.485 | 1.255 | 36.616 | +19.0% | 5.473 |
| current_hybrid_transformer | 0.30 | 0.265 | -27.0% | 0.609 | 1.520 | 35.970 | +16.9% | 5.500 |
| itransformer_co2_residual | 0.00 | 0.352 | +0.0% | 0.614 | 1.258 | 9.372 | +0.0% | 3.954 |
| itransformer_co2_residual | 0.05 | 0.326 | -7.3% | 0.585 | 1.244 | 9.778 | +4.3% | 3.608 |
| itransformer_co2_residual | 0.15 | 0.272 | -22.5% | 0.585 | 0.993 | 11.710 | +24.9% | 3.481 |
| itransformer_co2_residual | 0.30 | 0.270 | -23.2% | 0.817 | 1.586 | 15.892 | +69.6% | 3.223 |
| transformer_forecaster | 0.00 | 0.353 | +0.0% | 0.472 | 2.468 | 23.996 | +0.0% | 4.897 |
| transformer_forecaster | 0.05 | 0.332 | -5.9% | 0.451 | 2.373 | 24.793 | +3.3% | 4.944 |
| transformer_forecaster | 0.15 | 0.296 | -16.3% | 0.451 | 2.330 | 28.625 | +19.3% | 5.049 |
| transformer_forecaster | 0.30 | 0.273 | -22.7% | 0.446 | 2.627 | 33.531 | +39.7% | 5.431 |