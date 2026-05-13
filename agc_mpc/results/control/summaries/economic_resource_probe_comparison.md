# Economic/Resource-Aware MPC Probe Comparison

Note: the reported objective values are not directly comparable because the economic run includes the resource term. Use target MAE and resource proxy changes to judge the trade-off.

| predictor | tracking objective | economic objective | tracking resource | economic resource | resource change | tracking Tair MAE | economic Tair MAE | tracking Rhair MAE | economic Rhair MAE | tracking CO2 MAE | economic CO2 MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_hybrid_transformer | 0.0366 | 0.0957 | 0.354 | 0.332 | -6.0% | 0.328 | 0.310 | 0.504 | 1.308 | 10.964 | 12.380 |
| itransformer_co2_residual | 0.0244 | 0.0763 | 0.377 | 0.357 | -5.3% | 0.136 | 0.213 | 1.002 | 0.902 | 2.938 | 4.899 |