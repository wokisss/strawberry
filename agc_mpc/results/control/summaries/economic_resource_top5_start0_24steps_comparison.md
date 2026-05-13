# Economic/Resource-Aware MPC Probe Comparison

Note: the reported objective values are not directly comparable because the economic run includes the resource term. Use target MAE and resource proxy changes to judge the trade-off.

| predictor | tracking objective | economic objective | tracking resource | economic resource | resource change | tracking Tair MAE | economic Tair MAE | tracking Rhair MAE | economic Rhair MAE | tracking CO2 MAE | economic CO2 MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_hybrid_transformer | 0.0366 | 0.0957 | 0.354 | 0.333 | -5.9% | 0.328 | 0.311 | 0.504 | 1.309 | 10.964 | 12.357 |
| itransformer_co2_residual | 0.0244 | 0.0763 | 0.377 | 0.357 | -5.3% | 0.136 | 0.213 | 1.002 | 0.902 | 2.938 | 4.899 |
| segrnn_forecaster | 0.0270 | 0.0662 | 0.294 | 0.286 | -3.0% | 0.256 | 0.296 | 1.351 | 1.436 | 12.891 | 14.519 |
| transformer_forecaster | 0.0229 | 0.0724 | 0.340 | 0.311 | -8.6% | 0.127 | 0.203 | 0.962 | 0.918 | 8.051 | 8.486 |
| transformer_hybrid_residual | 0.0229 | 0.0728 | 0.334 | 0.341 | +2.3% | 0.115 | 0.185 | 0.730 | 0.418 | 7.913 | 9.886 |