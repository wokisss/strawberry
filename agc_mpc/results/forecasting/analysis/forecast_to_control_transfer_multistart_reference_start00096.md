# Forecast-To-Control Transfer Analysis

Model count: `10`.

This report tests whether forecast-side validation metrics predict `GradientMPC` closed-loop outcomes.
For selection metrics, lower values are treated as better. Gradient metrics are diagnostic only.

## Metric Roles

| control_target | metric | role |
| --- | --- | --- |
| mpc_tair_mae | tair_first_step_mae | offline_or_diagnostic_only |
| mpc_tair_mae | tair_control_horizon_mae | offline_or_diagnostic_only |
| mpc_tair_mae | tair_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_tair_mae | tair_full_horizon_mae | offline_or_diagnostic_only |
| mpc_tair_mae | tair_final_step_mae | offline_or_diagnostic_only |
| mpc_tair_mae | tair_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_tair_mae | tair_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_tair_mae | rhair_first_step_mae | secondary_selection |
| mpc_tair_mae | rhair_control_horizon_mae | secondary_selection |
| mpc_tair_mae | rhair_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_tair_mae | rhair_full_horizon_mae | offline_or_diagnostic_only |
| mpc_tair_mae | rhair_final_step_mae | offline_or_diagnostic_only |
| mpc_tair_mae | rhair_control_horizon_abs_bias | secondary_selection |
| mpc_tair_mae | rhair_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_tair_mae | co2_first_step_mae | offline_or_diagnostic_only |
| mpc_tair_mae | co2_control_horizon_mae | weak_selection |
| mpc_tair_mae | co2_weighted_horizon_mae | weak_selection |
| mpc_tair_mae | co2_full_horizon_mae | weak_selection |
| mpc_tair_mae | co2_final_step_mae | offline_or_diagnostic_only |
| mpc_tair_mae | co2_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_tair_mae | co2_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_tair_mae | forecast_only_transfer_rank | offline_or_diagnostic_only |
| mpc_tair_mae | tair_transfer_selection_score | offline_or_diagnostic_only |
| mpc_tair_mae | rhair_transfer_selection_score | secondary_selection |
| mpc_tair_mae | co2_transfer_selection_score | offline_or_diagnostic_only |
| mpc_tair_mae | multiobjective_transfer_selection_score | weak_selection |
| mpc_tair_mae | cost_grad_mean_abs | diagnostic_only |
| mpc_tair_mae | tair_first_grad_mean_abs | diagnostic_only |
| mpc_tair_mae | tair_t_heat_sp_first_grad | diagnostic_only |
| mpc_tair_mae | tair_t_vent_sp_first_grad | diagnostic_only |
| mpc_tair_mae | tair_window_pos_lee_sp_first_grad | diagnostic_only |
| mpc_tair_mae | rhair_first_grad_mean_abs | diagnostic_only |
| mpc_tair_mae | rhair_dx_sp_first_grad | diagnostic_only |
| mpc_tair_mae | rhair_t_vent_sp_first_grad | diagnostic_only |
| mpc_tair_mae | rhair_window_pos_lee_sp_first_grad | diagnostic_only |
| mpc_tair_mae | rhair_water_sup_intervals_sp_min_first_grad | diagnostic_only |
| mpc_tair_mae | co2_first_grad_mean_abs | diagnostic_only |
| mpc_tair_mae | co2_sp_first_grad | diagnostic_only |
| mpc_tair_mae | co2_sp_first_grad_positive_fraction | diagnostic_only |
| mpc_tair_mae | co2_sp_first_grad_flat_fraction | diagnostic_only |
| mpc_tair_mae | t_vent_sp_first_grad | diagnostic_only |
| mpc_tair_mae | assim_sp_first_grad | diagnostic_only |
| mpc_rhair_mae | tair_first_step_mae | weak_selection |
| mpc_rhair_mae | tair_control_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | tair_weighted_horizon_mae | weak_selection |
| mpc_rhair_mae | tair_full_horizon_mae | secondary_selection |
| mpc_rhair_mae | tair_final_step_mae | secondary_selection |
| mpc_rhair_mae | tair_control_horizon_abs_bias | secondary_selection |
| mpc_rhair_mae | tair_constraint_near_mae_proxy | secondary_selection |
| mpc_rhair_mae | rhair_first_step_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_control_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_full_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_final_step_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_first_step_mae | secondary_selection |
| mpc_rhair_mae | co2_control_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_full_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_final_step_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_rhair_mae | forecast_only_transfer_rank | offline_or_diagnostic_only |
| mpc_rhair_mae | tair_transfer_selection_score | weak_selection |
| mpc_rhair_mae | rhair_transfer_selection_score | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_transfer_selection_score | offline_or_diagnostic_only |
| mpc_rhair_mae | multiobjective_transfer_selection_score | secondary_selection |
| mpc_rhair_mae | cost_grad_mean_abs | diagnostic_only |
| mpc_rhair_mae | tair_first_grad_mean_abs | diagnostic_only |
| mpc_rhair_mae | tair_t_heat_sp_first_grad | diagnostic_only |
| mpc_rhair_mae | tair_t_vent_sp_first_grad | diagnostic_only |
| mpc_rhair_mae | tair_window_pos_lee_sp_first_grad | diagnostic_only |
| mpc_rhair_mae | rhair_first_grad_mean_abs | diagnostic_only |
| mpc_rhair_mae | rhair_dx_sp_first_grad | diagnostic_only |
| mpc_rhair_mae | rhair_t_vent_sp_first_grad | diagnostic_only |
| mpc_rhair_mae | rhair_window_pos_lee_sp_first_grad | diagnostic_only |
| mpc_rhair_mae | rhair_water_sup_intervals_sp_min_first_grad | diagnostic_only |
| mpc_rhair_mae | co2_first_grad_mean_abs | diagnostic_only |
| mpc_rhair_mae | co2_sp_first_grad | diagnostic_only |
| mpc_rhair_mae | co2_sp_first_grad_positive_fraction | diagnostic_only |
| mpc_rhair_mae | co2_sp_first_grad_flat_fraction | diagnostic_only |
| mpc_rhair_mae | t_vent_sp_first_grad | diagnostic_only |
| mpc_rhair_mae | assim_sp_first_grad | diagnostic_only |
| mpc_co2_mae | tair_first_step_mae | offline_or_diagnostic_only |
| mpc_co2_mae | tair_control_horizon_mae | offline_or_diagnostic_only |
| mpc_co2_mae | tair_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_co2_mae | tair_full_horizon_mae | offline_or_diagnostic_only |
| mpc_co2_mae | tair_final_step_mae | offline_or_diagnostic_only |
| mpc_co2_mae | tair_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_co2_mae | tair_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_co2_mae | rhair_first_step_mae | weak_selection |
| mpc_co2_mae | rhair_control_horizon_mae | weak_selection |
| mpc_co2_mae | rhair_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_co2_mae | rhair_full_horizon_mae | offline_or_diagnostic_only |
| mpc_co2_mae | rhair_final_step_mae | offline_or_diagnostic_only |
| mpc_co2_mae | rhair_control_horizon_abs_bias | secondary_selection |
| mpc_co2_mae | rhair_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_co2_mae | co2_first_step_mae | offline_or_diagnostic_only |
| mpc_co2_mae | co2_control_horizon_mae | offline_or_diagnostic_only |
| mpc_co2_mae | co2_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_co2_mae | co2_full_horizon_mae | offline_or_diagnostic_only |
| mpc_co2_mae | co2_final_step_mae | offline_or_diagnostic_only |
| mpc_co2_mae | co2_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_co2_mae | co2_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_co2_mae | forecast_only_transfer_rank | offline_or_diagnostic_only |
| mpc_co2_mae | tair_transfer_selection_score | offline_or_diagnostic_only |
| mpc_co2_mae | rhair_transfer_selection_score | secondary_selection |
| mpc_co2_mae | co2_transfer_selection_score | offline_or_diagnostic_only |
| mpc_co2_mae | multiobjective_transfer_selection_score | offline_or_diagnostic_only |
| mpc_co2_mae | cost_grad_mean_abs | diagnostic_only |
| mpc_co2_mae | tair_first_grad_mean_abs | diagnostic_only |
| mpc_co2_mae | tair_t_heat_sp_first_grad | diagnostic_only |
| mpc_co2_mae | tair_t_vent_sp_first_grad | diagnostic_only |
| mpc_co2_mae | tair_window_pos_lee_sp_first_grad | diagnostic_only |
| mpc_co2_mae | rhair_first_grad_mean_abs | diagnostic_only |
| mpc_co2_mae | rhair_dx_sp_first_grad | diagnostic_only |
| mpc_co2_mae | rhair_t_vent_sp_first_grad | diagnostic_only |
| mpc_co2_mae | rhair_window_pos_lee_sp_first_grad | diagnostic_only |
| mpc_co2_mae | rhair_water_sup_intervals_sp_min_first_grad | diagnostic_only |
| mpc_co2_mae | co2_first_grad_mean_abs | diagnostic_only |
| mpc_co2_mae | co2_sp_first_grad | diagnostic_only |
| mpc_co2_mae | co2_sp_first_grad_positive_fraction | diagnostic_only |
| mpc_co2_mae | co2_sp_first_grad_flat_fraction | diagnostic_only |
| mpc_co2_mae | t_vent_sp_first_grad | diagnostic_only |
| mpc_co2_mae | assim_sp_first_grad | diagnostic_only |
| mpc_objective | tair_first_step_mae | offline_or_diagnostic_only |
| mpc_objective | tair_control_horizon_mae | offline_or_diagnostic_only |
| mpc_objective | tair_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_objective | tair_full_horizon_mae | offline_or_diagnostic_only |
| mpc_objective | tair_final_step_mae | offline_or_diagnostic_only |
| mpc_objective | tair_control_horizon_abs_bias | objective_secondary_selection |
| mpc_objective | tair_constraint_near_mae_proxy | weak_selection |
| mpc_objective | rhair_first_step_mae | objective_secondary_selection |
| mpc_objective | rhair_control_horizon_mae | objective_primary_selection |
| mpc_objective | rhair_weighted_horizon_mae | objective_secondary_selection |
| mpc_objective | rhair_full_horizon_mae | objective_secondary_selection |
| mpc_objective | rhair_final_step_mae | weak_selection |
| mpc_objective | rhair_control_horizon_abs_bias | weak_selection |
| mpc_objective | rhair_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_objective | co2_first_step_mae | offline_or_diagnostic_only |
| mpc_objective | co2_control_horizon_mae | offline_or_diagnostic_only |
| mpc_objective | co2_weighted_horizon_mae | objective_secondary_selection |
| mpc_objective | co2_full_horizon_mae | objective_secondary_selection |
| mpc_objective | co2_final_step_mae | objective_secondary_selection |
| mpc_objective | co2_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_objective | co2_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_objective | forecast_only_transfer_rank | weak_selection |
| mpc_objective | tair_transfer_selection_score | offline_or_diagnostic_only |
| mpc_objective | rhair_transfer_selection_score | objective_primary_selection |
| mpc_objective | co2_transfer_selection_score | offline_or_diagnostic_only |
| mpc_objective | multiobjective_transfer_selection_score | offline_or_diagnostic_only |
| mpc_objective | cost_grad_mean_abs | diagnostic_only |
| mpc_objective | tair_first_grad_mean_abs | diagnostic_only |
| mpc_objective | tair_t_heat_sp_first_grad | diagnostic_only |
| mpc_objective | tair_t_vent_sp_first_grad | diagnostic_only |
| mpc_objective | tair_window_pos_lee_sp_first_grad | diagnostic_only |
| mpc_objective | rhair_first_grad_mean_abs | diagnostic_only |
| mpc_objective | rhair_dx_sp_first_grad | diagnostic_only |
| mpc_objective | rhair_t_vent_sp_first_grad | diagnostic_only |
| mpc_objective | rhair_window_pos_lee_sp_first_grad | diagnostic_only |
| mpc_objective | rhair_water_sup_intervals_sp_min_first_grad | diagnostic_only |
| mpc_objective | co2_first_grad_mean_abs | diagnostic_only |
| mpc_objective | co2_sp_first_grad | diagnostic_only |
| mpc_objective | co2_sp_first_grad_positive_fraction | diagnostic_only |
| mpc_objective | co2_sp_first_grad_flat_fraction | diagnostic_only |
| mpc_objective | t_vent_sp_first_grad | diagnostic_only |
| mpc_objective | assim_sp_first_grad | diagnostic_only |

## FCTV Transfer Selection Scores

Each target-specific transfer score is a weighted average of forecast-only metric ranks. Lower is better.
`multiobjective_transfer_selection_score` averages the three target-specific scores.

| metric suffix | weight |
| --- | --- |
| first_step_mae | 3.0 |
| control_horizon_mae | 2.0 |
| constraint_near_mae_proxy | 1.5 |
| control_horizon_abs_bias | 1.5 |

Role definitions:

- `primary_selection`: stable enough for closed-loop target-specific model selection in the current pool.
- `secondary_selection`: useful supporting selection signal.
- `weak_selection`: directionally useful but not strong enough alone.
- `objective_primary_selection` / `objective_secondary_selection`: useful for whole-objective screening.
- `offline_or_diagnostic_only`: not suitable for control selection by itself.
- `diagnostic_only`: useful for interpretation, not direct ranking.

## Forecast-Only Transfer Rank

| rank | predictor | multiobjective_score | tair_score | rhair_score | co2_score | control_relevant_mean_rank | mpc_tair_mae | mpc_rhair_mae | mpc_co2_mae | mpc_objective |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | itransformer_co2_control_aware_fusion | 4.219 | 5.969 | 5.188 | 1.500 | 8.556 | 1.072 | 1.179 | 26.154 | 0.1072 |
| 2 | current_hybrid_transformer | 4.438 | 4.938 | 3.188 | 5.188 | 6.722 | 0.526 | 1.486 | 24.384 | 0.0517 |
| 3 | itransformer_co2_protected_expert | 4.521 | 2.562 | 6.250 | 4.750 | 9.278 | 0.829 | 0.749 | 50.390 | 0.1278 |
| 4 | itransformer_co2_late_frozen_expert | 4.552 | 5.406 | 6.188 | 2.062 | 9.944 | 1.122 | 1.282 | 25.366 | 0.1133 |
| 5 | transformer_hybrid_residual | 4.688 | 3.562 | 2.312 | 8.188 | 9.167 | 0.873 | 1.986 | 23.095 | 0.0666 |
| 6 | itransformer_co2_late_residual | 5.646 | 6.125 | 5.000 | 5.812 | 9.000 | 1.135 | 1.230 | 36.866 | 0.1007 |
| 7 | segrnn_forecaster | 5.708 | 9.000 | 4.375 | 3.750 | 9.389 | 0.340 | 3.209 | 16.886 | 0.0738 |
| 8 | itransformer_co2_horizon_mixture | 5.854 | 2.125 | 7.688 | 7.750 | 13.722 | 1.305 | 1.713 | 41.012 | 0.1234 |
| 9 | itransformer_co2_residual | 6.625 | 5.312 | 6.500 | 8.062 | 10.778 | 0.551 | 0.680 | 11.074 | 0.0654 |
| 10 | frequency_forecaster | 8.750 | 10.000 | 8.312 | 7.938 | 18.722 | 1.158 | 6.495 | 31.750 | 0.3964 |

## Metric Transfer Quality

### Target: `mpc_tair_mae`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rhair_first_step_mae | selection | 0.486 | 0.624 | 0.733 | no | yes | 0.667 |
| rhair_control_horizon_abs_bias | selection | 0.497 | 0.455 | 0.711 | no | no | 0.667 |
| rhair_transfer_selection_score | selection | 0.505 | 0.552 | 0.667 | no | yes | 0.667 |
| rhair_control_horizon_mae | selection | 0.405 | 0.394 | 0.622 | no | no | 0.333 |
| co2_weighted_horizon_mae | selection | 0.342 | 0.382 | 0.622 | yes | yes | 0.667 |
| multiobjective_transfer_selection_score | selection | 0.140 | 0.297 | 0.622 | no | no | 0.333 |
| co2_full_horizon_mae | selection | 0.378 | 0.358 | 0.600 | yes | yes | 0.667 |
| co2_constraint_near_mae_proxy | selection | 0.042 | 0.248 | 0.600 | yes | yes | 0.667 |
| co2_control_horizon_mae | selection | 0.224 | 0.255 | 0.591 | no | yes | 0.333 |
| co2_final_step_mae | selection | 0.172 | 0.212 | 0.578 | yes | yes | 0.333 |
| co2_transfer_selection_score | selection | 0.048 | 0.164 | 0.578 | no | yes | 0.333 |
| tair_control_horizon_abs_bias | selection | 0.071 | 0.127 | 0.578 | no | no | 0.667 |
| co2_first_step_mae | selection | 0.191 | 0.195 | 0.545 | no | no | 0.000 |
| rhair_weighted_horizon_mae | selection | 0.108 | 0.109 | 0.545 | no | no | 0.333 |
| rhair_full_horizon_mae | selection | 0.067 | 0.042 | 0.533 | no | no | 0.333 |
| tair_transfer_selection_score | selection | -0.194 | 0.006 | 0.533 | no | no | 0.000 |
| co2_control_horizon_abs_bias | selection | -0.052 | -0.030 | 0.511 | no | yes | 0.333 |
| tair_control_horizon_mae | selection | 0.111 | -0.073 | 0.500 | no | no | 0.000 |
| rhair_final_step_mae | selection | -0.078 | -0.049 | 0.477 | no | no | 0.333 |
| forecast_only_transfer_rank | selection | 0.042 | -0.036 | 0.477 | no | no | 0.333 |
| tair_first_step_mae | selection | 0.167 | -0.115 | 0.467 | no | no | 0.000 |
| tair_constraint_near_mae_proxy | selection | -0.175 | -0.170 | 0.432 | no | no | 0.333 |
| tair_full_horizon_mae | selection | -0.101 | -0.236 | 0.422 | no | no | 0.333 |
| tair_final_step_mae | selection | -0.170 | -0.200 | 0.422 | no | no | 0.333 |
| rhair_constraint_near_mae_proxy | selection | -0.223 | -0.261 | 0.400 | no | no | 0.000 |
| tair_weighted_horizon_mae | selection | -0.081 | -0.248 | 0.400 | no | no | 0.333 |
| assim_sp_first_grad | diagnostic | 0.877 | 0.778 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | 0.761 | 0.705 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | 0.705 | 0.644 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | 0.615 | 0.559 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | 0.549 | 0.547 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.508 | -0.511 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | 0.427 | 0.498 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | 0.628 | 0.462 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | 0.489 | 0.450 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | 0.514 | 0.365 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.311 | -0.348 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.311 | 0.348 |  |  |  |  |
| co2_sp_first_grad | diagnostic | 0.322 | 0.316 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | -0.071 | -0.231 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | -0.061 | 0.176 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | 0.402 | 0.061 |  |  |  |  |

### Target: `mpc_rhair_mae`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tair_constraint_near_mae_proxy | selection | 0.896 | 0.565 | 0.727 | no | yes | 0.333 |
| co2_first_step_mae | selection | 0.904 | 0.517 | 0.705 | no | no | 0.667 |
| tair_control_horizon_abs_bias | selection | 0.881 | 0.442 | 0.689 | no | yes | 0.333 |
| tair_final_step_mae | selection | 0.912 | 0.370 | 0.667 | no | yes | 0.333 |
| tair_full_horizon_mae | selection | 0.943 | 0.358 | 0.667 | no | yes | 0.333 |
| multiobjective_transfer_selection_score | selection | 0.770 | 0.370 | 0.644 | no | no | 0.667 |
| tair_weighted_horizon_mae | selection | 0.949 | 0.333 | 0.644 | no | yes | 0.333 |
| tair_first_step_mae | selection | 0.942 | 0.345 | 0.622 | no | no | 0.333 |
| co2_control_horizon_mae | selection | 0.746 | 0.249 | 0.614 | no | no | 0.333 |
| tair_transfer_selection_score | selection | 0.733 | 0.309 | 0.600 | no | no | 0.333 |
| rhair_constraint_near_mae_proxy | selection | 0.692 | 0.248 | 0.600 | no | no | 0.333 |
| co2_transfer_selection_score | selection | 0.291 | 0.212 | 0.600 | no | no | 0.333 |
| tair_control_horizon_mae | selection | 0.960 | 0.249 | 0.568 | no | no | 0.333 |
| co2_control_horizon_abs_bias | selection | -0.106 | 0.030 | 0.533 | no | no | 0.333 |
| rhair_weighted_horizon_mae | selection | 0.868 | 0.188 | 0.523 | no | no | 0.333 |
| rhair_full_horizon_mae | selection | 0.870 | 0.139 | 0.511 | no | no | 0.333 |
| forecast_only_transfer_rank | selection | 0.763 | 0.091 | 0.500 | no | no | 0.333 |
| rhair_final_step_mae | selection | 0.820 | 0.079 | 0.500 | no | no | 0.000 |
| rhair_control_horizon_abs_bias | selection | -0.367 | -0.200 | 0.467 | no | yes | 0.333 |
| co2_weighted_horizon_mae | selection | 0.422 | -0.103 | 0.467 | no | no | 0.333 |
| rhair_control_horizon_mae | selection | 0.727 | -0.030 | 0.467 | no | no | 0.000 |
| co2_full_horizon_mae | selection | 0.318 | -0.115 | 0.444 | no | no | 0.000 |
| rhair_first_step_mae | selection | 0.717 | -0.103 | 0.444 | no | no | 0.000 |
| co2_final_step_mae | selection | -0.102 | -0.236 | 0.422 | no | no | 0.333 |
| rhair_transfer_selection_score | selection | 0.333 | -0.115 | 0.422 | no | no | 0.000 |
| co2_constraint_near_mae_proxy | selection | -0.341 | -0.285 | 0.400 | no | no | 0.333 |
| rhair_dx_sp_first_grad | diagnostic | -0.731 | -0.894 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | -0.612 | -0.590 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | -0.497 | -0.578 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | -0.507 | -0.529 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | -0.610 | -0.517 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | -0.457 | -0.456 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | 0.481 | 0.285 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | 0.467 | 0.261 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | 0.016 | 0.261 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.080 | 0.261 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.080 | -0.261 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | 0.014 | 0.128 |  |  |  |  |
| co2_sp_first_grad | diagnostic | 0.348 | 0.103 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | -0.018 | -0.091 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | 0.799 | -0.055 |  |  |  |  |
| assim_sp_first_grad | diagnostic | -0.052 | -0.018 |  |  |  |  |

### Target: `mpc_co2_mae`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rhair_control_horizon_abs_bias | selection | 0.511 | 0.539 | 0.689 | no | yes | 0.333 |
| rhair_transfer_selection_score | selection | 0.354 | 0.430 | 0.644 | no | no | 0.667 |
| rhair_first_step_mae | selection | 0.256 | 0.345 | 0.622 | no | no | 0.667 |
| rhair_control_horizon_mae | selection | 0.215 | 0.309 | 0.600 | no | no | 0.333 |
| co2_weighted_horizon_mae | selection | 0.207 | 0.200 | 0.600 | no | no | 0.333 |
| rhair_full_horizon_mae | selection | 0.006 | 0.079 | 0.600 | no | no | 0.333 |
| rhair_weighted_horizon_mae | selection | 0.033 | 0.073 | 0.591 | no | no | 0.333 |
| co2_full_horizon_mae | selection | 0.269 | 0.176 | 0.578 | no | no | 0.333 |
| co2_constraint_near_mae_proxy | selection | 0.066 | 0.067 | 0.578 | no | no | 0.333 |
| co2_final_step_mae | selection | 0.314 | 0.188 | 0.556 | no | no | 0.333 |
| co2_control_horizon_abs_bias | selection | 0.238 | 0.127 | 0.533 | no | no | 0.333 |
| tair_weighted_horizon_mae | selection | -0.107 | 0.018 | 0.511 | no | yes | 0.667 |
| tair_full_horizon_mae | selection | -0.114 | 0.006 | 0.489 | no | yes | 0.667 |
| rhair_final_step_mae | selection | -0.116 | -0.122 | 0.477 | no | no | 0.333 |
| tair_constraint_near_mae_proxy | selection | -0.164 | -0.073 | 0.477 | no | yes | 0.667 |
| forecast_only_transfer_rank | selection | 0.029 | -0.061 | 0.477 | no | no | 0.333 |
| tair_transfer_selection_score | selection | -0.425 | -0.236 | 0.467 | no | no | 0.333 |
| tair_control_horizon_abs_bias | selection | -0.148 | -0.103 | 0.467 | no | yes | 0.333 |
| co2_control_horizon_mae | selection | 0.017 | -0.073 | 0.455 | no | no | 0.333 |
| tair_final_step_mae | selection | -0.145 | -0.067 | 0.444 | no | yes | 0.667 |
| co2_transfer_selection_score | selection | 0.019 | -0.176 | 0.422 | no | no | 0.333 |
| multiobjective_transfer_selection_score | selection | -0.086 | -0.127 | 0.422 | no | no | 0.000 |
| rhair_constraint_near_mae_proxy | selection | 0.025 | -0.067 | 0.422 | no | no | 0.000 |
| tair_control_horizon_mae | selection | -0.029 | -0.353 | 0.409 | no | no | 0.333 |
| co2_first_step_mae | selection | -0.028 | -0.146 | 0.409 | no | no | 0.000 |
| tair_first_step_mae | selection | -0.018 | -0.394 | 0.356 | no | no | 0.333 |
| rhair_t_vent_sp_first_grad | diagnostic | 0.415 | 0.717 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | 0.438 | 0.681 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | 0.533 | 0.559 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | 0.446 | 0.511 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | 0.354 | 0.401 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | 0.554 | 0.389 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | 0.371 | 0.353 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.225 | -0.348 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.225 | 0.348 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.230 | -0.292 |  |  |  |  |
| assim_sp_first_grad | diagnostic | 0.221 | 0.292 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | 0.113 | 0.255 |  |  |  |  |
| co2_sp_first_grad | diagnostic | -0.425 | -0.231 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | -0.067 | -0.182 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | -0.017 | -0.085 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | 0.039 | 0.079 |  |  |  |  |

### Target: `mpc_objective`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rhair_transfer_selection_score | selection | 0.666 | 0.697 | 0.822 | no | yes | 0.667 |
| rhair_control_horizon_mae | selection | 0.900 | 0.636 | 0.778 | no | yes | 0.667 |
| rhair_first_step_mae | selection | 0.925 | 0.600 | 0.756 | no | yes | 0.667 |
| co2_full_horizon_mae | selection | 0.609 | 0.442 | 0.711 | no | yes | 0.333 |
| rhair_full_horizon_mae | selection | 0.861 | 0.442 | 0.689 | no | yes | 0.667 |
| co2_weighted_horizon_mae | selection | 0.652 | 0.430 | 0.689 | no | yes | 0.333 |
| rhair_weighted_horizon_mae | selection | 0.880 | 0.426 | 0.659 | no | yes | 0.667 |
| tair_control_horizon_abs_bias | selection | 0.783 | 0.394 | 0.644 | yes | yes | 0.667 |
| rhair_control_horizon_abs_bias | selection | -0.271 | 0.285 | 0.644 | no | yes | 0.667 |
| co2_final_step_mae | selection | 0.218 | 0.358 | 0.600 | no | no | 0.000 |
| forecast_only_transfer_rank | selection | 0.818 | 0.286 | 0.591 | no | yes | 0.667 |
| rhair_final_step_mae | selection | 0.738 | 0.280 | 0.591 | yes | yes | 0.667 |
| tair_constraint_near_mae_proxy | selection | 0.654 | 0.255 | 0.591 | no | no | 0.667 |
| tair_final_step_mae | selection | 0.704 | 0.200 | 0.578 | no | no | 0.667 |
| multiobjective_transfer_selection_score | selection | 0.768 | 0.188 | 0.556 | no | yes | 0.333 |
| rhair_constraint_near_mae_proxy | selection | 0.651 | 0.152 | 0.556 | no | no | 0.000 |
| tair_full_horizon_mae | selection | 0.766 | 0.115 | 0.533 | no | no | 0.667 |
| tair_weighted_horizon_mae | selection | 0.782 | 0.103 | 0.511 | no | no | 0.667 |
| tair_transfer_selection_score | selection | 0.522 | 0.079 | 0.511 | no | no | 0.333 |
| co2_constraint_near_mae_proxy | selection | -0.157 | -0.018 | 0.489 | no | yes | 0.333 |
| tair_control_horizon_mae | selection | 0.898 | -0.024 | 0.477 | no | no | 0.333 |
| co2_first_step_mae | selection | 0.854 | 0.024 | 0.477 | no | no | 0.000 |
| co2_control_horizon_mae | selection | 0.766 | -0.024 | 0.477 | no | no | 0.000 |
| co2_transfer_selection_score | selection | 0.250 | -0.188 | 0.467 | no | no | 0.000 |
| co2_control_horizon_abs_bias | selection | -0.127 | -0.321 | 0.400 | no | no | 0.000 |
| tair_first_step_mae | selection | 0.915 | -0.176 | 0.400 | no | no | 0.333 |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.335 | -0.609 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.335 | 0.609 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | -0.142 | 0.535 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.390 | -0.462 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | -0.150 | 0.426 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | -0.075 | 0.401 |  |  |  |  |
| assim_sp_first_grad | diagnostic | 0.259 | 0.401 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | 0.425 | 0.394 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | 0.338 | 0.389 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | 0.620 | 0.353 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | -0.233 | -0.316 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | 0.312 | 0.304 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | -0.146 | 0.231 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | 0.911 | 0.219 |  |  |  |  |
| co2_sp_first_grad | diagnostic | 0.438 | 0.085 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | -0.300 | 0.073 |  |  |  |  |

## Robustness Summary

### Target: `mpc_tair_mae`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | offline_or_diagnostic_only | -0.115 | -0.400 .. 0.117 | -0.400 .. 0.117 | 0.361 |
| tair_control_horizon_mae | offline_or_diagnostic_only | -0.073 | -0.326 .. 0.276 | -0.326 .. 0.276 | 0.400 |
| tair_weighted_horizon_mae | offline_or_diagnostic_only | -0.248 | -0.567 .. 0.033 | -0.567 .. 0.033 | 0.278 |
| tair_full_horizon_mae | offline_or_diagnostic_only | -0.236 | -0.550 .. 0.050 | -0.550 .. 0.050 | 0.306 |
| tair_final_step_mae | offline_or_diagnostic_only | -0.200 | -0.517 .. 0.017 | -0.517 .. 0.017 | 0.306 |
| tair_control_horizon_abs_bias | offline_or_diagnostic_only | 0.127 | -0.067 .. 0.417 | -0.067 .. 0.417 | 0.500 |
| tair_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.170 | -0.510 .. 0.008 | -0.510 .. 0.008 | 0.314 |
| rhair_first_step_mae | secondary_selection | 0.624 | 0.500 .. 0.767 | 0.500 .. 0.833 | 0.694 |
| rhair_control_horizon_mae | secondary_selection | 0.394 | 0.183 .. 0.583 | 0.183 .. 0.595 | 0.556 |
| rhair_weighted_horizon_mae | offline_or_diagnostic_only | 0.109 | -0.176 .. 0.393 | -0.176 .. 0.419 | 0.457 |
| rhair_full_horizon_mae | offline_or_diagnostic_only | 0.042 | -0.250 .. 0.300 | -0.250 .. 0.429 | 0.444 |
| rhair_final_step_mae | offline_or_diagnostic_only | -0.049 | -0.310 .. 0.176 | -0.310 .. 0.176 | 0.371 |
| rhair_control_horizon_abs_bias | secondary_selection | 0.455 | 0.250 .. 0.850 | 0.250 .. 0.850 | 0.639 |
| rhair_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.261 | -0.583 .. 0.017 | -0.583 .. 0.017 | 0.278 |
| co2_first_step_mae | offline_or_diagnostic_only | 0.195 | -0.092 .. 0.300 | -0.092 .. 0.300 | 0.457 |
| co2_control_horizon_mae | weak_selection | 0.255 | 0.025 .. 0.427 | 0.025 .. 0.419 | 0.514 |
| co2_weighted_horizon_mae | weak_selection | 0.382 | 0.150 .. 0.517 | 0.150 .. 0.690 | 0.528 |
| co2_full_horizon_mae | weak_selection | 0.358 | 0.117 .. 0.483 | 0.117 .. 0.643 | 0.500 |
| co2_final_step_mae | offline_or_diagnostic_only | 0.212 | -0.083 .. 0.450 | -0.083 .. 0.450 | 0.472 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | -0.030 | -0.233 .. 0.067 | -0.233 .. 0.067 | 0.444 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.248 | -0.033 .. 0.450 | -0.033 .. 0.571 | 0.500 |
| forecast_only_transfer_rank | offline_or_diagnostic_only | -0.036 | -0.350 .. 0.159 | -0.350 .. 0.262 | 0.361 |
| tair_transfer_selection_score | offline_or_diagnostic_only | 0.006 | -0.217 .. 0.317 | -0.217 .. 0.317 | 0.444 |
| rhair_transfer_selection_score | secondary_selection | 0.552 | 0.400 .. 0.717 | 0.400 .. 0.838 | 0.611 |
| co2_transfer_selection_score | offline_or_diagnostic_only | 0.164 | 0.033 .. 0.483 | 0.033 .. 0.483 | 0.528 |
| multiobjective_transfer_selection_score | weak_selection | 0.297 | 0.067 .. 0.483 | 0.067 .. 0.418 | 0.556 |

### Target: `mpc_rhair_mae`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | weak_selection | 0.345 | 0.100 .. 0.517 | 0.100 .. 0.450 | 0.528 |
| tair_control_horizon_mae | offline_or_diagnostic_only | 0.249 | -0.033 .. 0.402 | -0.033 .. 0.402 | 0.457 |
| tair_weighted_horizon_mae | weak_selection | 0.333 | 0.083 .. 0.533 | 0.083 .. 0.533 | 0.556 |
| tair_full_horizon_mae | secondary_selection | 0.358 | 0.117 .. 0.567 | 0.117 .. 0.567 | 0.583 |
| tair_final_step_mae | secondary_selection | 0.370 | 0.133 .. 0.600 | 0.133 .. 0.600 | 0.583 |
| tair_control_horizon_abs_bias | secondary_selection | 0.442 | 0.233 .. 0.600 | 0.233 .. 0.600 | 0.611 |
| tair_constraint_near_mae_proxy | secondary_selection | 0.565 | 0.402 .. 0.686 | 0.402 .. 0.686 | 0.657 |
| rhair_first_step_mae | offline_or_diagnostic_only | -0.103 | -0.517 .. 0.067 | -0.517 .. 0.067 | 0.306 |
| rhair_control_horizon_mae | offline_or_diagnostic_only | -0.030 | -0.417 .. 0.133 | -0.417 .. 0.238 | 0.333 |
| rhair_weighted_horizon_mae | offline_or_diagnostic_only | 0.188 | -0.117 .. 0.351 | -0.117 .. 0.455 | 0.400 |
| rhair_full_horizon_mae | offline_or_diagnostic_only | 0.139 | -0.183 .. 0.283 | -0.183 .. 0.476 | 0.389 |
| rhair_final_step_mae | offline_or_diagnostic_only | 0.079 | -0.268 .. 0.201 | -0.268 .. 0.311 | 0.371 |
| rhair_control_horizon_abs_bias | offline_or_diagnostic_only | -0.200 | -0.500 .. 0.100 | -0.548 .. 0.100 | 0.361 |
| rhair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.248 | -0.033 .. 0.400 | -0.033 .. 0.595 | 0.500 |
| co2_first_step_mae | secondary_selection | 0.517 | 0.335 .. 0.837 | 0.335 .. 0.862 | 0.629 |
| co2_control_horizon_mae | offline_or_diagnostic_only | 0.249 | -0.033 .. 0.569 | -0.033 .. 0.647 | 0.514 |
| co2_weighted_horizon_mae | offline_or_diagnostic_only | -0.103 | -0.517 .. 0.083 | -0.517 .. 0.357 | 0.333 |
| co2_full_horizon_mae | offline_or_diagnostic_only | -0.115 | -0.533 .. 0.067 | -0.533 .. 0.333 | 0.306 |
| co2_final_step_mae | offline_or_diagnostic_only | -0.236 | -0.483 .. -0.050 | -0.483 .. 0.095 | 0.333 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | 0.030 | -0.167 .. 0.200 | -0.167 .. 0.214 | 0.472 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.285 | -0.417 .. -0.017 | -0.417 .. 0.143 | 0.333 |
| forecast_only_transfer_rank | offline_or_diagnostic_only | 0.091 | -0.267 .. 0.367 | -0.267 .. 0.595 | 0.361 |
| tair_transfer_selection_score | weak_selection | 0.309 | 0.050 .. 0.417 | 0.050 .. 0.417 | 0.500 |
| rhair_transfer_selection_score | offline_or_diagnostic_only | -0.115 | -0.533 .. 0.083 | -0.533 .. 0.084 | 0.278 |
| co2_transfer_selection_score | offline_or_diagnostic_only | 0.212 | 0.033 .. 0.550 | 0.033 .. 0.619 | 0.528 |
| multiobjective_transfer_selection_score | secondary_selection | 0.370 | 0.133 .. 0.800 | 0.133 .. 0.714 | 0.556 |

### Target: `mpc_co2_mae`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | offline_or_diagnostic_only | -0.394 | -0.600 .. -0.167 | -0.600 .. -0.143 | 0.278 |
| tair_control_horizon_mae | offline_or_diagnostic_only | -0.353 | -0.544 .. -0.126 | -0.544 .. -0.084 | 0.343 |
| tair_weighted_horizon_mae | offline_or_diagnostic_only | 0.018 | -0.183 .. 0.317 | -0.238 .. 0.317 | 0.444 |
| tair_full_horizon_mae | offline_or_diagnostic_only | 0.006 | -0.200 .. 0.300 | -0.262 .. 0.300 | 0.417 |
| tair_final_step_mae | offline_or_diagnostic_only | -0.067 | -0.300 .. 0.117 | -0.300 .. 0.117 | 0.361 |
| tair_control_horizon_abs_bias | offline_or_diagnostic_only | -0.103 | -0.333 .. 0.100 | -0.267 .. 0.100 | 0.389 |
| tair_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.073 | -0.360 .. 0.159 | -0.323 .. 0.159 | 0.371 |
| rhair_first_step_mae | weak_selection | 0.345 | 0.217 .. 0.617 | 0.217 .. 0.762 | 0.556 |
| rhair_control_horizon_mae | weak_selection | 0.309 | 0.133 .. 0.567 | 0.133 .. 0.483 | 0.556 |
| rhair_weighted_horizon_mae | offline_or_diagnostic_only | 0.073 | -0.075 .. 0.360 | -0.075 .. 0.371 | 0.543 |
| rhair_full_horizon_mae | offline_or_diagnostic_only | 0.079 | -0.067 .. 0.367 | -0.067 .. 0.367 | 0.556 |
| rhair_final_step_mae | offline_or_diagnostic_only | -0.122 | -0.293 .. 0.092 | -0.293 .. 0.092 | 0.429 |
| rhair_control_horizon_abs_bias | secondary_selection | 0.539 | 0.433 .. 0.750 | 0.433 .. 0.750 | 0.639 |
| rhair_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.067 | -0.283 .. 0.167 | -0.310 .. 0.167 | 0.333 |
| co2_first_step_mae | offline_or_diagnostic_only | -0.146 | -0.427 .. 0.025 | -0.427 .. 0.132 | 0.314 |
| co2_control_horizon_mae | offline_or_diagnostic_only | -0.073 | -0.259 .. 0.176 | -0.259 .. 0.275 | 0.400 |
| co2_weighted_horizon_mae | offline_or_diagnostic_only | 0.200 | 0.033 .. 0.550 | 0.033 .. 0.429 | 0.528 |
| co2_full_horizon_mae | offline_or_diagnostic_only | 0.176 | 0.000 .. 0.517 | 0.000 .. 0.381 | 0.500 |
| co2_final_step_mae | offline_or_diagnostic_only | 0.188 | -0.033 .. 0.333 | -0.017 .. 0.333 | 0.472 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | 0.127 | -0.017 .. 0.350 | -0.017 .. 0.350 | 0.500 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.067 | -0.133 .. 0.467 | -0.133 .. 0.429 | 0.500 |
| forecast_only_transfer_rank | offline_or_diagnostic_only | -0.061 | -0.233 .. 0.192 | -0.233 .. 0.192 | 0.417 |
| tair_transfer_selection_score | offline_or_diagnostic_only | -0.236 | -0.383 .. -0.033 | -0.383 .. -0.033 | 0.417 |
| rhair_transfer_selection_score | secondary_selection | 0.430 | 0.283 .. 0.650 | 0.283 .. 0.743 | 0.583 |
| co2_transfer_selection_score | offline_or_diagnostic_only | -0.176 | -0.367 .. 0.233 | -0.367 .. 0.233 | 0.361 |
| multiobjective_transfer_selection_score | offline_or_diagnostic_only | -0.127 | -0.333 .. 0.050 | -0.333 .. 0.214 | 0.361 |

### Target: `mpc_objective`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | offline_or_diagnostic_only | -0.176 | -0.617 .. -0.017 | -0.617 .. 0.024 | 0.250 |
| tair_control_horizon_mae | offline_or_diagnostic_only | -0.024 | -0.410 .. 0.176 | -0.410 .. 0.252 | 0.343 |
| tair_weighted_horizon_mae | offline_or_diagnostic_only | 0.103 | -0.233 .. 0.317 | -0.233 .. 0.317 | 0.389 |
| tair_full_horizon_mae | offline_or_diagnostic_only | 0.115 | -0.217 .. 0.333 | -0.217 .. 0.333 | 0.417 |
| tair_final_step_mae | offline_or_diagnostic_only | 0.200 | -0.100 .. 0.450 | -0.100 .. 0.450 | 0.472 |
| tair_control_horizon_abs_bias | objective_secondary_selection | 0.394 | 0.167 .. 0.600 | 0.167 .. 0.600 | 0.556 |
| tair_constraint_near_mae_proxy | weak_selection | 0.255 | -0.025 .. 0.527 | -0.025 .. 0.527 | 0.486 |
| rhair_first_step_mae | objective_secondary_selection | 0.600 | 0.450 .. 0.850 | 0.450 .. 0.929 | 0.694 |
| rhair_control_horizon_mae | objective_primary_selection | 0.636 | 0.500 .. 0.900 | 0.500 .. 0.881 | 0.722 |
| rhair_weighted_horizon_mae | objective_secondary_selection | 0.426 | 0.209 .. 0.661 | 0.209 .. 0.659 | 0.571 |
| rhair_full_horizon_mae | objective_secondary_selection | 0.442 | 0.233 .. 0.683 | 0.233 .. 0.667 | 0.611 |
| rhair_final_step_mae | weak_selection | 0.280 | 0.008 .. 0.444 | 0.008 .. 0.419 | 0.486 |
| rhair_control_horizon_abs_bias | weak_selection | 0.285 | 0.167 .. 0.767 | 0.167 .. 0.767 | 0.583 |
| rhair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.152 | -0.167 .. 0.317 | -0.167 .. 0.317 | 0.444 |
| co2_first_step_mae | offline_or_diagnostic_only | 0.024 | -0.343 .. 0.176 | -0.343 .. 0.252 | 0.343 |
| co2_control_horizon_mae | offline_or_diagnostic_only | -0.024 | -0.410 .. 0.126 | -0.410 .. 0.180 | 0.343 |
| co2_weighted_horizon_mae | objective_secondary_selection | 0.430 | 0.217 .. 0.700 | 0.217 .. 0.600 | 0.611 |
| co2_full_horizon_mae | objective_secondary_selection | 0.442 | 0.233 .. 0.717 | 0.233 .. 0.617 | 0.639 |
| co2_final_step_mae | objective_secondary_selection | 0.358 | 0.183 .. 0.517 | 0.183 .. 0.517 | 0.556 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | -0.321 | -0.417 .. -0.167 | -0.452 .. -0.167 | 0.361 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.018 | -0.267 .. 0.233 | -0.267 .. 0.233 | 0.417 |
| forecast_only_transfer_rank | weak_selection | 0.286 | 0.000 .. 0.517 | 0.000 .. 0.433 | 0.472 |
| tair_transfer_selection_score | offline_or_diagnostic_only | 0.079 | -0.267 .. 0.233 | -0.267 .. 0.262 | 0.389 |
| rhair_transfer_selection_score | objective_primary_selection | 0.697 | 0.583 .. 0.933 | 0.583 .. 0.970 | 0.778 |
| co2_transfer_selection_score | offline_or_diagnostic_only | -0.188 | -0.383 .. 0.183 | -0.383 .. 0.183 | 0.389 |
| multiobjective_transfer_selection_score | offline_or_diagnostic_only | 0.188 | -0.117 .. 0.317 | -0.117 .. 0.524 | 0.444 |
