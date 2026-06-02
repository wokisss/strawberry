# Same-Period AGC Resource Baselines

This comparison uses the same time window as the full-period anchored MPC experiment. AGC rows are real executed resource records. MPC rows are counterfactual resource estimates from generated control trajectories.

| source | case | heat | electricity | CO2 | irrigation | resource cost | vs Reference | CO2 MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| real_agc_executed_resource | Reference | 1.780 | 0.000 | 0.0697 | 5.900 | 0.0203 | 0.0% |  |
| real_agc_executed_resource | Automatoes | 1.232 | 0.304 | 0.0842 | 7.800 | 0.0291 | 43.1% |  |
| real_agc_executed_resource | AICU | 0.878 | 0.499 | 0.0988 | 4.700 | 0.0370 | 82.0% |  |
| counterfactual_estimated_mpc_resource | current_hybrid_transformer w=0.00 | 0.220 | 0.015 | 0.0098 | 0.759 | 0.0037 | -81.7% | 17.655 |
| counterfactual_estimated_mpc_resource | current_hybrid_transformer w=0.05 | 0.199 | 0.009 | 0.0097 | 0.746 | 0.0031 | -84.6% | 17.699 |
| counterfactual_estimated_mpc_resource | itransformer_co2_residual w=0.00 | 0.180 | 0.011 | 0.0098 | 0.758 | 0.0031 | -84.8% | 2.895 |
| counterfactual_estimated_mpc_resource | itransformer_co2_residual w=0.05 | 0.175 | 0.007 | 0.0096 | 0.751 | 0.0028 | -86.4% | 3.196 |

Boundary:

- AGC rows are real resource consumption over daily records intersecting the MPC window.
- MPC rows are estimated resource consumption over counterfactual anchored control trajectories.
- Real AGC production or net profit is not ranked against MPC because the MPC rollout has no crop/yield/quality dynamic model.