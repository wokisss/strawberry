# Same-Period All-Team Resource Baselines

This table directly sums recorded AGC resources over the full-period anchored MPC window. MPC rows are included only as counterfactual estimated-resource references.

| source | case | heat | electricity | CO2 | irrigation | resource cost | vs Reference | vs AICU | CO2 MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| real_agc_executed_resource | AICU | 16.637 | 0.499 | 1.2249 | 100.800 | 0.2579 | -24.7% | 0.0% |  |
| real_agc_executed_resource | Automatoes | 30.796 | 0.844 | 0.9760 | 112.060 | 0.3674 | 7.3% | 42.5% |  |
| real_agc_executed_resource | Digilog | 15.816 | 4.935 | 1.0197 | 188.951 | 0.4335 | 26.6% | 68.1% |  |
| real_agc_executed_resource | IUACAAS | 15.562 | 0.000 | 0.8738 | 163.075 | 0.1991 | -41.9% | -22.8% |  |
| real_agc_executed_resource | Reference | 28.900 | 0.000 | 1.2833 | 109.400 | 0.3425 | 0.0% | 32.8% |  |
| real_agc_executed_resource | TheAutomators | 16.920 | 1.800 | 1.6120 | 135.720 | 0.3654 | 6.7% | 41.7% |  |
| counterfactual_estimated_mpc_resource | current_hybrid_transformer w=0.00 | 12.106 | 1.963 | 1.2825 | 109.675 | 0.3455 | 0.9% | 33.9% | 27.659 |
| counterfactual_estimated_mpc_resource | current_hybrid_transformer w=0.05 | 10.660 | 1.792 | 1.2743 | 109.253 | 0.3215 | -6.1% | 24.7% | 27.811 |
| counterfactual_estimated_mpc_resource | itransformer_co2_residual w=0.00 | 12.832 | 1.909 | 1.3286 | 109.610 | 0.3423 | -0.1% | 32.7% | 15.112 |
| counterfactual_estimated_mpc_resource | itransformer_co2_residual w=0.05 | 11.918 | 1.611 | 1.3060 | 108.723 | 0.3133 | -8.5% | 21.5% | 15.706 |

Boundary:

- AGC rows are measured resources from recorded competition data.
- MPC rows are counterfactual resource estimates from generated control trajectories.
- This table is a resource baseline, not a net-profit ranking.