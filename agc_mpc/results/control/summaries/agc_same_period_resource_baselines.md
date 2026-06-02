# Same-Period AGC Resource Baselines

This comparison uses the same time window as the full-period anchored MPC experiment. AGC rows are real executed resource records. MPC rows are counterfactual resource estimates from generated control trajectories.

| source | case | heat | electricity | CO2 | irrigation | resource cost | vs Reference | CO2 MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| real_agc_executed_resource | Reference | 28.900 | 0.000 | 1.2833 | 109.400 | 0.3425 | 0.0% |  |
| real_agc_executed_resource | Automatoes | 30.796 | 0.844 | 0.9760 | 112.060 | 0.3674 | 7.3% |  |
| real_agc_executed_resource | AICU | 16.637 | 0.499 | 1.2249 | 100.800 | 0.2579 | -24.7% |  |
| counterfactual_estimated_mpc_resource | current_hybrid_transformer w=0.00 | 12.106 | 1.963 | 1.2825 | 109.675 | 0.3455 | 0.9% | 27.659 |
| counterfactual_estimated_mpc_resource | current_hybrid_transformer w=0.05 | 10.660 | 1.792 | 1.2743 | 109.253 | 0.3215 | -6.1% | 27.811 |
| counterfactual_estimated_mpc_resource | itransformer_co2_residual w=0.00 | 12.832 | 1.909 | 1.3286 | 109.610 | 0.3423 | -0.1% | 15.112 |
| counterfactual_estimated_mpc_resource | itransformer_co2_residual w=0.05 | 11.918 | 1.611 | 1.3060 | 108.723 | 0.3133 | -8.5% | 15.706 |

Boundary:

- AGC rows are real resource consumption over daily records intersecting the MPC window.
- MPC rows are estimated resource consumption over counterfactual anchored control trajectories.
- Real AGC production or net profit is not ranked against MPC because the MPC rollout has no crop/yield/quality dynamic model.