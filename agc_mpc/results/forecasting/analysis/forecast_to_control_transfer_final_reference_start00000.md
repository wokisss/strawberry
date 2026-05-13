# Forecast-To-Control Transfer Analysis

Model count: `16`.

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
| mpc_tair_mae | rhair_first_step_mae | offline_or_diagnostic_only |
| mpc_tair_mae | rhair_control_horizon_mae | offline_or_diagnostic_only |
| mpc_tair_mae | rhair_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_tair_mae | rhair_full_horizon_mae | offline_or_diagnostic_only |
| mpc_tair_mae | rhair_final_step_mae | offline_or_diagnostic_only |
| mpc_tair_mae | rhair_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_tair_mae | rhair_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_tair_mae | co2_first_step_mae | offline_or_diagnostic_only |
| mpc_tair_mae | co2_control_horizon_mae | offline_or_diagnostic_only |
| mpc_tair_mae | co2_weighted_horizon_mae | weak_selection |
| mpc_tair_mae | co2_full_horizon_mae | secondary_selection |
| mpc_tair_mae | co2_final_step_mae | offline_or_diagnostic_only |
| mpc_tair_mae | co2_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_tair_mae | co2_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_tair_mae | forecast_only_transfer_rank | offline_or_diagnostic_only |
| mpc_tair_mae | tair_transfer_selection_score | offline_or_diagnostic_only |
| mpc_tair_mae | rhair_transfer_selection_score | offline_or_diagnostic_only |
| mpc_tair_mae | co2_transfer_selection_score | offline_or_diagnostic_only |
| mpc_tair_mae | multiobjective_transfer_selection_score | offline_or_diagnostic_only |
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
| mpc_rhair_mae | tair_first_step_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | tair_control_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | tair_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | tair_full_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | tair_final_step_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | tair_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_rhair_mae | tair_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_first_step_mae | secondary_selection |
| mpc_rhair_mae | rhair_control_horizon_mae | secondary_selection |
| mpc_rhair_mae | rhair_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_full_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_final_step_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_first_step_mae | secondary_selection |
| mpc_rhair_mae | co2_control_horizon_mae | weak_selection |
| mpc_rhair_mae | co2_weighted_horizon_mae | secondary_selection |
| mpc_rhair_mae | co2_full_horizon_mae | secondary_selection |
| mpc_rhair_mae | co2_final_step_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_rhair_mae | forecast_only_transfer_rank | offline_or_diagnostic_only |
| mpc_rhair_mae | tair_transfer_selection_score | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_transfer_selection_score | secondary_selection |
| mpc_rhair_mae | co2_transfer_selection_score | weak_selection |
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
| mpc_co2_mae | rhair_first_step_mae | offline_or_diagnostic_only |
| mpc_co2_mae | rhair_control_horizon_mae | offline_or_diagnostic_only |
| mpc_co2_mae | rhair_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_co2_mae | rhair_full_horizon_mae | offline_or_diagnostic_only |
| mpc_co2_mae | rhair_final_step_mae | offline_or_diagnostic_only |
| mpc_co2_mae | rhair_control_horizon_abs_bias | weak_selection |
| mpc_co2_mae | rhair_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_co2_mae | co2_first_step_mae | secondary_selection |
| mpc_co2_mae | co2_control_horizon_mae | weak_selection |
| mpc_co2_mae | co2_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_co2_mae | co2_full_horizon_mae | offline_or_diagnostic_only |
| mpc_co2_mae | co2_final_step_mae | offline_or_diagnostic_only |
| mpc_co2_mae | co2_control_horizon_abs_bias | secondary_selection |
| mpc_co2_mae | co2_constraint_near_mae_proxy | weak_selection |
| mpc_co2_mae | forecast_only_transfer_rank | secondary_selection |
| mpc_co2_mae | tair_transfer_selection_score | offline_or_diagnostic_only |
| mpc_co2_mae | rhair_transfer_selection_score | offline_or_diagnostic_only |
| mpc_co2_mae | co2_transfer_selection_score | secondary_selection |
| mpc_co2_mae | multiobjective_transfer_selection_score | secondary_selection |
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
| mpc_objective | tair_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_objective | tair_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_objective | rhair_first_step_mae | objective_secondary_selection |
| mpc_objective | rhair_control_horizon_mae | objective_secondary_selection |
| mpc_objective | rhair_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_objective | rhair_full_horizon_mae | offline_or_diagnostic_only |
| mpc_objective | rhair_final_step_mae | offline_or_diagnostic_only |
| mpc_objective | rhair_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_objective | rhair_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_objective | co2_first_step_mae | objective_secondary_selection |
| mpc_objective | co2_control_horizon_mae | objective_secondary_selection |
| mpc_objective | co2_weighted_horizon_mae | objective_secondary_selection |
| mpc_objective | co2_full_horizon_mae | objective_secondary_selection |
| mpc_objective | co2_final_step_mae | weak_selection |
| mpc_objective | co2_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_objective | co2_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_objective | forecast_only_transfer_rank | offline_or_diagnostic_only |
| mpc_objective | tair_transfer_selection_score | offline_or_diagnostic_only |
| mpc_objective | rhair_transfer_selection_score | objective_secondary_selection |
| mpc_objective | co2_transfer_selection_score | weak_selection |
| mpc_objective | multiobjective_transfer_selection_score | objective_secondary_selection |
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
| 1 | current_hybrid_transformer | 5.167 | 5.688 | 4.250 | 5.562 | 6.722 | 0.362 | 1.206 | 18.812 | 0.0442 |
| 2 | itransformer_co2_control_aware_fusion | 5.406 | 7.094 | 7.625 | 1.500 | 8.556 | 2.217 | 4.261 | 6.623 | 0.1505 |
| 3 | itransformer_residual | 5.583 | 4.062 | 5.688 | 7.000 | 9.167 | 2.216 | 5.675 | 11.532 | 0.1924 |
| 4 | itransformer_co2_late_frozen_expert | 5.740 | 6.531 | 8.625 | 2.062 | 9.944 | 2.202 | 4.302 | 6.442 | 0.1538 |
| 5 | transformer_hybrid_residual | 6.396 | 4.500 | 2.875 | 11.812 | 9.167 | 1.669 | 4.593 | 18.351 | 0.1060 |
| 6 | itransformer_co2_late_residual | 6.604 | 7.812 | 6.000 | 6.000 | 9.000 | 1.153 | 1.618 | 10.125 | 0.0705 |
| 7 | segrnn_forecaster | 7.979 | 14.188 | 6.000 | 3.750 | 9.389 | 0.391 | 2.195 | 14.425 | 0.0486 |
| 8 | dlinear_forecaster | 8.583 | 9.938 | 5.500 | 10.312 | 15.056 | 3.436 | 6.459 | 37.824 | 0.3962 |
| 9 | itransformer_co2_horizon_mixture | 8.646 | 2.875 | 12.125 | 10.938 | 13.722 | 3.329 | 5.668 | 29.380 | 0.3734 |
| 10 | transformer_forecaster | 9.250 | 6.188 | 11.125 | 10.438 | 13.056 | 1.039 | 4.072 | 16.448 | 0.0861 |
| 11 | itransformer_co2_residual | 9.312 | 6.062 | 9.938 | 11.938 | 10.778 | 0.939 | 1.497 | 6.304 | 0.0558 |
| 12 | gru_forecaster | 10.375 | 10.438 | 14.625 | 6.062 | 14.278 | 0.409 | 4.957 | 49.973 | 0.1108 |
| 13 | nlinear_forecaster | 10.729 | 13.125 | 4.812 | 14.250 | 15.500 | 1.867 | 4.182 | 25.236 | 0.1526 |
| 14 | patchtst_residual | 10.938 | 8.688 | 11.875 | 12.250 | 14.833 | 3.089 | 7.961 | 36.014 | 0.2628 |
| 15 | lstm_forecaster | 11.604 | 12.812 | 11.750 | 10.250 | 15.111 | 1.491 | 4.497 | 23.014 | 0.1780 |
| 16 | frequency_forecaster | 13.688 | 16.000 | 13.188 | 11.875 | 18.722 | 1.725 | 8.759 | 15.530 | 0.4338 |

## Metric Transfer Quality

### Target: `mpc_tair_mae`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| co2_full_horizon_mae | selection | 0.439 | 0.356 | 0.633 | no | yes | 0.667 |
| co2_weighted_horizon_mae | selection | 0.393 | 0.329 | 0.608 | no | yes | 0.667 |
| co2_final_step_mae | selection | 0.238 | 0.235 | 0.592 | no | no | 0.333 |
| co2_transfer_selection_score | selection | 0.272 | 0.221 | 0.583 | no | no | 0.333 |
| rhair_first_step_mae | selection | 0.170 | 0.244 | 0.575 | no | yes | 0.667 |
| co2_control_horizon_mae | selection | 0.202 | 0.182 | 0.571 | no | no | 0.333 |
| co2_first_step_mae | selection | 0.084 | 0.177 | 0.555 | no | no | 0.000 |
| co2_constraint_near_mae_proxy | selection | 0.261 | 0.185 | 0.550 | no | yes | 0.667 |
| rhair_transfer_selection_score | selection | 0.039 | 0.050 | 0.529 | no | yes | 0.333 |
| co2_control_horizon_abs_bias | selection | 0.128 | 0.059 | 0.525 | no | no | 0.333 |
| rhair_control_horizon_abs_bias | selection | 0.069 | 0.076 | 0.517 | no | no | 0.000 |
| rhair_control_horizon_mae | selection | 0.054 | 0.041 | 0.517 | no | no | 0.000 |
| tair_constraint_near_mae_proxy | selection | -0.215 | -0.024 | 0.496 | no | no | 0.000 |
| tair_control_horizon_abs_bias | selection | -0.097 | -0.038 | 0.492 | no | yes | 0.333 |
| multiobjective_transfer_selection_score | selection | 0.048 | 0.026 | 0.483 | yes | yes | 0.333 |
| tair_transfer_selection_score | selection | -0.217 | -0.138 | 0.467 | no | no | 0.000 |
| tair_control_horizon_mae | selection | -0.098 | -0.144 | 0.462 | no | no | 0.000 |
| forecast_only_transfer_rank | selection | -0.118 | -0.168 | 0.458 | no | yes | 0.333 |
| tair_first_step_mae | selection | -0.096 | -0.274 | 0.408 | no | no | 0.000 |
| rhair_weighted_horizon_mae | selection | -0.303 | -0.303 | 0.387 | no | yes | 0.333 |
| rhair_full_horizon_mae | selection | -0.332 | -0.312 | 0.375 | no | yes | 0.333 |
| rhair_final_step_mae | selection | -0.375 | -0.338 | 0.353 | yes | yes | 0.333 |
| tair_final_step_mae | selection | -0.340 | -0.438 | 0.350 | no | no | 0.000 |
| tair_weighted_horizon_mae | selection | -0.273 | -0.468 | 0.333 | no | no | 0.000 |
| tair_full_horizon_mae | selection | -0.290 | -0.474 | 0.325 | no | no | 0.000 |
| rhair_constraint_near_mae_proxy | selection | -0.511 | -0.526 | 0.308 | no | no | 0.000 |
| assim_sp_first_grad | diagnostic | 0.698 | 0.753 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | 0.406 | 0.612 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | 0.489 | 0.600 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | 0.562 | 0.597 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | 0.587 | 0.583 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | 0.446 | 0.568 |  |  |  |  |
| co2_sp_first_grad | diagnostic | 0.444 | 0.480 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | 0.447 | 0.439 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | 0.467 | 0.391 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | 0.334 | 0.259 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.111 | -0.241 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | 0.357 | 0.235 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | -0.331 | -0.226 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | 0.222 | 0.200 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.108 | 0.188 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.192 | -0.141 |  |  |  |  |

### Target: `mpc_rhair_mae`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rhair_first_step_mae | selection | 0.599 | 0.435 | 0.683 | no | yes | 0.333 |
| co2_weighted_horizon_mae | selection | 0.562 | 0.376 | 0.650 | no | yes | 0.333 |
| rhair_transfer_selection_score | selection | 0.439 | 0.384 | 0.647 | no | yes | 0.333 |
| rhair_control_horizon_mae | selection | 0.571 | 0.400 | 0.642 | no | no | 0.333 |
| multiobjective_transfer_selection_score | selection | 0.525 | 0.379 | 0.642 | yes | yes | 0.333 |
| co2_full_horizon_mae | selection | 0.552 | 0.376 | 0.642 | no | yes | 0.667 |
| co2_first_step_mae | selection | 0.592 | 0.381 | 0.639 | no | no | 0.333 |
| co2_transfer_selection_score | selection | 0.381 | 0.329 | 0.625 | no | no | 0.000 |
| co2_control_horizon_mae | selection | 0.571 | 0.325 | 0.622 | no | no | 0.000 |
| forecast_only_transfer_rank | selection | 0.436 | 0.215 | 0.600 | no | yes | 0.333 |
| co2_final_step_mae | selection | 0.236 | 0.224 | 0.583 | no | no | 0.000 |
| tair_constraint_near_mae_proxy | selection | 0.206 | 0.196 | 0.563 | no | no | 0.667 |
| rhair_weighted_horizon_mae | selection | 0.393 | 0.166 | 0.555 | no | yes | 0.333 |
| tair_control_horizon_abs_bias | selection | 0.329 | 0.182 | 0.550 | no | yes | 0.333 |
| rhair_full_horizon_mae | selection | 0.368 | 0.153 | 0.550 | no | yes | 0.333 |
| tair_transfer_selection_score | selection | 0.235 | 0.141 | 0.542 | no | no | 0.000 |
| co2_constraint_near_mae_proxy | selection | 0.086 | 0.138 | 0.542 | no | yes | 0.333 |
| rhair_control_horizon_abs_bias | selection | 0.076 | 0.094 | 0.542 | no | no | 0.333 |
| tair_first_step_mae | selection | 0.465 | 0.106 | 0.533 | no | no | 0.000 |
| co2_control_horizon_abs_bias | selection | 0.126 | 0.056 | 0.533 | no | no | 0.000 |
| rhair_constraint_near_mae_proxy | selection | 0.191 | 0.038 | 0.533 | no | no | 0.000 |
| rhair_final_step_mae | selection | 0.305 | 0.084 | 0.521 | yes | yes | 0.333 |
| tair_control_horizon_mae | selection | 0.420 | 0.046 | 0.513 | no | no | 0.000 |
| tair_final_step_mae | selection | 0.170 | -0.126 | 0.475 | no | no | 0.000 |
| tair_full_horizon_mae | selection | 0.256 | -0.129 | 0.450 | no | no | 0.000 |
| tair_weighted_horizon_mae | selection | 0.274 | -0.132 | 0.442 | no | no | 0.000 |
| assim_sp_first_grad | diagnostic | 0.343 | 0.431 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.192 | -0.311 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | -0.137 | -0.202 |  |  |  |  |
| co2_sp_first_grad | diagnostic | 0.257 | 0.193 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | 0.134 | 0.169 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | 0.209 | 0.155 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.013 | -0.150 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | 0.127 | 0.144 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | 0.146 | 0.110 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.010 | 0.094 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | -0.058 | 0.054 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | 0.010 | -0.043 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | 0.119 | -0.040 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | -0.124 | -0.031 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | 0.305 | 0.013 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | -0.058 | 0.004 |  |  |  |  |

### Target: `mpc_co2_mae`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| co2_control_horizon_abs_bias | selection | 0.465 | 0.556 | 0.683 | no | no | 0.667 |
| forecast_only_transfer_rank | selection | 0.373 | 0.418 | 0.633 | no | no | 0.333 |
| multiobjective_transfer_selection_score | selection | 0.446 | 0.421 | 0.625 | no | no | 0.333 |
| co2_transfer_selection_score | selection | 0.337 | 0.356 | 0.625 | no | no | 0.667 |
| co2_first_step_mae | selection | 0.109 | 0.364 | 0.613 | no | no | 0.667 |
| co2_constraint_near_mae_proxy | selection | 0.268 | 0.312 | 0.608 | no | no | 0.333 |
| co2_control_horizon_mae | selection | 0.132 | 0.260 | 0.597 | no | no | 0.667 |
| rhair_control_horizon_abs_bias | selection | 0.428 | 0.318 | 0.592 | no | yes | 0.333 |
| tair_first_step_mae | selection | 0.057 | 0.241 | 0.583 | no | no | 0.333 |
| tair_transfer_selection_score | selection | 0.200 | 0.209 | 0.575 | no | no | 0.000 |
| tair_weighted_horizon_mae | selection | -0.056 | 0.182 | 0.575 | no | no | 0.000 |
| rhair_full_horizon_mae | selection | 0.386 | 0.206 | 0.567 | no | no | 0.000 |
| tair_full_horizon_mae | selection | -0.057 | 0.179 | 0.567 | no | no | 0.000 |
| rhair_weighted_horizon_mae | selection | 0.380 | 0.157 | 0.563 | no | no | 0.000 |
| co2_weighted_horizon_mae | selection | 0.154 | 0.176 | 0.550 | no | no | 0.333 |
| rhair_constraint_near_mae_proxy | selection | 0.398 | 0.138 | 0.550 | no | no | 0.333 |
| tair_control_horizon_abs_bias | selection | 0.027 | 0.112 | 0.550 | no | no | 0.000 |
| rhair_transfer_selection_score | selection | 0.358 | 0.165 | 0.546 | no | no | 0.000 |
| co2_full_horizon_mae | selection | 0.134 | 0.188 | 0.542 | no | no | 0.000 |
| tair_control_horizon_mae | selection | -0.056 | 0.152 | 0.538 | no | no | 0.000 |
| rhair_first_step_mae | selection | 0.120 | 0.159 | 0.533 | no | no | 0.000 |
| rhair_final_step_mae | selection | 0.339 | 0.063 | 0.529 | no | no | 0.000 |
| rhair_control_horizon_mae | selection | 0.225 | 0.121 | 0.525 | no | no | 0.000 |
| tair_constraint_near_mae_proxy | selection | -0.150 | 0.043 | 0.521 | no | yes | 0.333 |
| co2_final_step_mae | selection | -0.115 | 0.041 | 0.500 | no | no | 0.333 |
| tair_final_step_mae | selection | -0.082 | -0.068 | 0.475 | no | no | 0.000 |
| rhair_dx_sp_first_grad | diagnostic | -0.402 | -0.469 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | -0.482 | -0.437 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | -0.418 | -0.402 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | -0.375 | -0.399 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | -0.455 | -0.352 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | -0.435 | -0.325 |  |  |  |  |
| assim_sp_first_grad | diagnostic | -0.335 | -0.284 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | -0.307 | -0.228 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | -0.096 | -0.219 |  |  |  |  |
| co2_sp_first_grad | diagnostic | -0.260 | -0.199 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | 0.094 | 0.188 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | 0.448 | 0.165 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | -0.328 | -0.102 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.243 | -0.102 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | -0.093 | 0.057 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | -0.399 | 0.040 |  |  |  |  |

### Target: `mpc_objective`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rhair_first_step_mae | selection | 0.681 | 0.535 | 0.700 | no | yes | 0.667 |
| co2_weighted_horizon_mae | selection | 0.590 | 0.479 | 0.683 | no | yes | 0.667 |
| co2_full_horizon_mae | selection | 0.579 | 0.482 | 0.675 | no | yes | 0.667 |
| co2_first_step_mae | selection | 0.606 | 0.456 | 0.655 | no | no | 0.000 |
| rhair_transfer_selection_score | selection | 0.342 | 0.371 | 0.655 | no | yes | 0.333 |
| multiobjective_transfer_selection_score | selection | 0.457 | 0.406 | 0.642 | yes | yes | 0.333 |
| rhair_control_horizon_mae | selection | 0.572 | 0.376 | 0.642 | no | no | 0.000 |
| co2_transfer_selection_score | selection | 0.379 | 0.341 | 0.625 | no | no | 0.333 |
| co2_control_horizon_mae | selection | 0.604 | 0.377 | 0.622 | no | no | 0.333 |
| co2_final_step_mae | selection | 0.166 | 0.282 | 0.600 | no | no | 0.333 |
| tair_constraint_near_mae_proxy | selection | 0.274 | 0.209 | 0.597 | no | no | 0.333 |
| forecast_only_transfer_rank | selection | 0.329 | 0.232 | 0.583 | no | yes | 0.333 |
| tair_control_horizon_abs_bias | selection | 0.308 | 0.197 | 0.583 | no | yes | 0.333 |
| tair_transfer_selection_score | selection | 0.195 | 0.156 | 0.558 | no | no | 0.000 |
| co2_constraint_near_mae_proxy | selection | 0.139 | 0.132 | 0.542 | no | yes | 0.667 |
| rhair_weighted_horizon_mae | selection | 0.224 | 0.094 | 0.538 | no | yes | 0.333 |
| rhair_full_horizon_mae | selection | 0.185 | 0.071 | 0.533 | no | yes | 0.333 |
| tair_control_horizon_mae | selection | 0.434 | 0.121 | 0.529 | no | no | 0.000 |
| rhair_control_horizon_abs_bias | selection | 0.027 | 0.059 | 0.525 | no | no | 0.333 |
| tair_first_step_mae | selection | 0.448 | 0.082 | 0.517 | no | no | 0.000 |
| co2_control_horizon_abs_bias | selection | 0.062 | -0.024 | 0.517 | no | no | 0.333 |
| rhair_final_step_mae | selection | 0.072 | -0.015 | 0.487 | yes | yes | 0.333 |
| tair_final_step_mae | selection | 0.195 | -0.079 | 0.475 | no | no | 0.000 |
| rhair_constraint_near_mae_proxy | selection | -0.089 | -0.141 | 0.467 | no | no | 0.000 |
| tair_full_horizon_mae | selection | 0.262 | -0.106 | 0.467 | no | no | 0.000 |
| tair_weighted_horizon_mae | selection | 0.280 | -0.112 | 0.458 | no | no | 0.000 |
| assim_sp_first_grad | diagnostic | 0.460 | 0.536 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | 0.367 | 0.400 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.337 | -0.386 |  |  |  |  |
| co2_sp_first_grad | diagnostic | 0.374 | 0.362 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | 0.315 | 0.353 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.243 | -0.332 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.241 | 0.282 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | 0.139 | 0.256 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | 0.238 | 0.233 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | 0.000 | 0.191 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | 0.182 | 0.182 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | 0.089 | 0.174 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | 0.336 | 0.135 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | 0.020 | 0.124 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | 0.009 | -0.035 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | 0.058 | -0.024 |  |  |  |  |

## Robustness Summary

### Target: `mpc_tair_mae`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | offline_or_diagnostic_only | -0.274 | -0.393 .. -0.146 | -0.407 .. -0.082 | 0.362 |
| tair_control_horizon_mae | offline_or_diagnostic_only | -0.144 | -0.275 .. 0.000 | -0.323 .. 0.000 | 0.413 |
| tair_weighted_horizon_mae | offline_or_diagnostic_only | -0.468 | -0.575 .. -0.382 | -0.575 .. -0.324 | 0.295 |
| tair_full_horizon_mae | offline_or_diagnostic_only | -0.474 | -0.582 .. -0.389 | -0.582 .. -0.335 | 0.286 |
| tair_final_step_mae | offline_or_diagnostic_only | -0.438 | -0.529 .. -0.346 | -0.529 .. -0.253 | 0.314 |
| tair_control_horizon_abs_bias | offline_or_diagnostic_only | -0.038 | -0.221 .. 0.079 | -0.221 .. 0.231 | 0.429 |
| tair_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.024 | -0.122 .. 0.150 | -0.122 .. 0.220 | 0.462 |
| rhair_first_step_mae | offline_or_diagnostic_only | 0.244 | 0.111 .. 0.350 | 0.111 .. 0.350 | 0.533 |
| rhair_control_horizon_mae | offline_or_diagnostic_only | 0.041 | -0.057 .. 0.179 | -0.066 .. 0.262 | 0.476 |
| rhair_weighted_horizon_mae | offline_or_diagnostic_only | -0.303 | -0.504 .. -0.182 | -0.504 .. -0.099 | 0.317 |
| rhair_full_horizon_mae | offline_or_diagnostic_only | -0.312 | -0.514 .. -0.193 | -0.514 .. -0.077 | 0.305 |
| rhair_final_step_mae | offline_or_diagnostic_only | -0.338 | -0.626 .. -0.257 | -0.626 .. -0.154 | 0.260 |
| rhair_control_horizon_abs_bias | offline_or_diagnostic_only | 0.076 | -0.057 .. 0.186 | -0.057 .. 0.253 | 0.467 |
| rhair_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.526 | -0.671 .. -0.429 | -0.671 .. -0.429 | 0.248 |
| co2_first_step_mae | offline_or_diagnostic_only | 0.177 | 0.097 .. 0.318 | 0.066 .. 0.318 | 0.529 |
| co2_control_horizon_mae | offline_or_diagnostic_only | 0.182 | 0.093 .. 0.329 | -0.011 .. 0.329 | 0.538 |
| co2_weighted_horizon_mae | weak_selection | 0.329 | 0.189 .. 0.432 | 0.011 .. 0.432 | 0.562 |
| co2_full_horizon_mae | secondary_selection | 0.356 | 0.221 .. 0.468 | 0.082 .. 0.468 | 0.590 |
| co2_final_step_mae | offline_or_diagnostic_only | 0.235 | 0.107 .. 0.375 | -0.011 .. 0.375 | 0.552 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | 0.059 | -0.086 .. 0.171 | -0.095 .. 0.171 | 0.486 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.185 | 0.018 .. 0.318 | -0.011 .. 0.318 | 0.495 |
| forecast_only_transfer_rank | offline_or_diagnostic_only | -0.168 | -0.336 .. -0.070 | -0.336 .. -0.036 | 0.400 |
| tair_transfer_selection_score | offline_or_diagnostic_only | -0.138 | -0.268 .. 0.011 | -0.288 .. 0.143 | 0.419 |
| rhair_transfer_selection_score | offline_or_diagnostic_only | 0.050 | -0.121 .. 0.179 | -0.121 .. 0.231 | 0.476 |
| co2_transfer_selection_score | offline_or_diagnostic_only | 0.221 | 0.107 .. 0.368 | -0.011 .. 0.368 | 0.548 |
| multiobjective_transfer_selection_score | offline_or_diagnostic_only | 0.026 | -0.175 .. 0.109 | -0.175 .. 0.109 | 0.423 |

### Target: `mpc_rhair_mae`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | offline_or_diagnostic_only | 0.106 | -0.086 .. 0.207 | -0.086 .. 0.207 | 0.467 |
| tair_control_horizon_mae | offline_or_diagnostic_only | 0.046 | -0.159 .. 0.141 | -0.159 .. 0.223 | 0.442 |
| tair_weighted_horizon_mae | offline_or_diagnostic_only | -0.132 | -0.375 .. -0.046 | -0.375 .. 0.143 | 0.362 |
| tair_full_horizon_mae | offline_or_diagnostic_only | -0.129 | -0.371 .. -0.043 | -0.371 .. 0.148 | 0.371 |
| tair_final_step_mae | offline_or_diagnostic_only | -0.126 | -0.368 .. 0.007 | -0.368 .. 0.159 | 0.400 |
| tair_control_horizon_abs_bias | offline_or_diagnostic_only | 0.182 | 0.007 .. 0.332 | 0.007 .. 0.401 | 0.486 |
| tair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.196 | 0.023 .. 0.359 | 0.023 .. 0.359 | 0.500 |
| rhair_first_step_mae | secondary_selection | 0.435 | 0.314 .. 0.579 | 0.314 .. 0.643 | 0.638 |
| rhair_control_horizon_mae | secondary_selection | 0.400 | 0.271 .. 0.521 | 0.271 .. 0.547 | 0.590 |
| rhair_weighted_horizon_mae | offline_or_diagnostic_only | 0.166 | -0.013 .. 0.306 | -0.013 .. 0.361 | 0.490 |
| rhair_full_horizon_mae | offline_or_diagnostic_only | 0.153 | -0.011 .. 0.293 | -0.011 .. 0.332 | 0.495 |
| rhair_final_step_mae | offline_or_diagnostic_only | 0.084 | -0.113 .. 0.163 | -0.113 .. 0.224 | 0.452 |
| rhair_control_horizon_abs_bias | offline_or_diagnostic_only | 0.094 | -0.021 .. 0.293 | -0.021 .. 0.293 | 0.505 |
| rhair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.038 | -0.150 .. 0.164 | -0.150 .. 0.191 | 0.476 |
| co2_first_step_mae | secondary_selection | 0.381 | 0.248 .. 0.466 | 0.248 .. 0.484 | 0.587 |
| co2_control_horizon_mae | weak_selection | 0.325 | 0.181 .. 0.506 | 0.181 .. 0.506 | 0.567 |
| co2_weighted_horizon_mae | secondary_selection | 0.376 | 0.243 .. 0.557 | 0.243 .. 0.557 | 0.600 |
| co2_full_horizon_mae | secondary_selection | 0.376 | 0.243 .. 0.557 | 0.243 .. 0.557 | 0.590 |
| co2_final_step_mae | offline_or_diagnostic_only | 0.224 | 0.121 .. 0.321 | 0.121 .. 0.286 | 0.543 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | 0.056 | -0.075 .. 0.154 | -0.126 .. 0.154 | 0.486 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.138 | -0.014 .. 0.336 | -0.014 .. 0.336 | 0.495 |
| forecast_only_transfer_rank | offline_or_diagnostic_only | 0.215 | 0.039 .. 0.361 | 0.039 .. 0.366 | 0.533 |
| tair_transfer_selection_score | offline_or_diagnostic_only | 0.141 | -0.043 .. 0.261 | -0.043 .. 0.302 | 0.476 |
| rhair_transfer_selection_score | secondary_selection | 0.384 | 0.270 .. 0.499 | 0.270 .. 0.531 | 0.606 |
| co2_transfer_selection_score | weak_selection | 0.329 | 0.211 .. 0.489 | 0.187 .. 0.489 | 0.581 |
| multiobjective_transfer_selection_score | secondary_selection | 0.379 | 0.238 .. 0.486 | 0.238 .. 0.470 | 0.587 |

### Target: `mpc_co2_mae`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | offline_or_diagnostic_only | 0.241 | 0.161 .. 0.400 | 0.148 .. 0.400 | 0.552 |
| tair_control_horizon_mae | offline_or_diagnostic_only | 0.152 | 0.084 .. 0.302 | 0.009 .. 0.302 | 0.519 |
| tair_weighted_horizon_mae | offline_or_diagnostic_only | 0.182 | 0.093 .. 0.354 | 0.093 .. 0.354 | 0.543 |
| tair_full_horizon_mae | offline_or_diagnostic_only | 0.179 | 0.089 .. 0.350 | 0.089 .. 0.350 | 0.533 |
| tair_final_step_mae | offline_or_diagnostic_only | -0.068 | -0.189 .. 0.071 | -0.275 .. 0.057 | 0.429 |
| tair_control_horizon_abs_bias | offline_or_diagnostic_only | 0.112 | 0.021 .. 0.221 | -0.044 .. 0.221 | 0.524 |
| tair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.043 | -0.152 .. 0.121 | -0.152 .. 0.123 | 0.462 |
| rhair_first_step_mae | offline_or_diagnostic_only | 0.159 | 0.021 .. 0.289 | -0.104 .. 0.289 | 0.486 |
| rhair_control_horizon_mae | offline_or_diagnostic_only | 0.121 | -0.036 .. 0.239 | -0.099 .. 0.354 | 0.467 |
| rhair_weighted_horizon_mae | offline_or_diagnostic_only | 0.157 | 0.009 .. 0.334 | -0.047 .. 0.339 | 0.510 |
| rhair_full_horizon_mae | offline_or_diagnostic_only | 0.206 | 0.036 .. 0.389 | 0.000 .. 0.354 | 0.505 |
| rhair_final_step_mae | offline_or_diagnostic_only | 0.063 | -0.138 .. 0.191 | -0.173 .. 0.181 | 0.462 |
| rhair_control_horizon_abs_bias | weak_selection | 0.318 | 0.211 .. 0.439 | 0.159 .. 0.516 | 0.552 |
| rhair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.138 | -0.046 .. 0.321 | -0.093 .. 0.301 | 0.486 |
| co2_first_step_mae | secondary_selection | 0.364 | 0.263 .. 0.517 | 0.263 .. 0.470 | 0.567 |
| co2_control_horizon_mae | weak_selection | 0.260 | 0.150 .. 0.470 | 0.150 .. 0.470 | 0.552 |
| co2_weighted_horizon_mae | offline_or_diagnostic_only | 0.176 | 0.104 .. 0.368 | 0.042 .. 0.368 | 0.524 |
| co2_full_horizon_mae | offline_or_diagnostic_only | 0.188 | 0.118 .. 0.382 | 0.046 .. 0.382 | 0.514 |
| co2_final_step_mae | offline_or_diagnostic_only | 0.041 | -0.050 .. 0.246 | -0.156 .. 0.246 | 0.467 |
| co2_control_horizon_abs_bias | secondary_selection | 0.556 | 0.486 .. 0.618 | 0.486 .. 0.676 | 0.657 |
| co2_constraint_near_mae_proxy | weak_selection | 0.312 | 0.236 .. 0.593 | 0.204 .. 0.593 | 0.571 |
| forecast_only_transfer_rank | secondary_selection | 0.418 | 0.286 .. 0.479 | 0.140 .. 0.479 | 0.583 |
| tair_transfer_selection_score | offline_or_diagnostic_only | 0.209 | 0.114 .. 0.361 | 0.110 .. 0.361 | 0.543 |
| rhair_transfer_selection_score | offline_or_diagnostic_only | 0.165 | -0.018 .. 0.307 | -0.115 .. 0.424 | 0.481 |
| co2_transfer_selection_score | secondary_selection | 0.356 | 0.268 .. 0.568 | 0.258 .. 0.568 | 0.590 |
| multiobjective_transfer_selection_score | secondary_selection | 0.421 | 0.368 .. 0.542 | 0.269 .. 0.542 | 0.600 |

### Target: `mpc_objective`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | offline_or_diagnostic_only | 0.082 | -0.114 .. 0.232 | -0.114 .. 0.232 | 0.448 |
| tair_control_horizon_mae | offline_or_diagnostic_only | 0.121 | -0.068 .. 0.268 | -0.068 .. 0.268 | 0.462 |
| tair_weighted_horizon_mae | offline_or_diagnostic_only | -0.112 | -0.350 .. 0.011 | -0.350 .. 0.038 | 0.381 |
| tair_full_horizon_mae | offline_or_diagnostic_only | -0.106 | -0.343 .. 0.018 | -0.343 .. 0.049 | 0.390 |
| tair_final_step_mae | offline_or_diagnostic_only | -0.079 | -0.311 .. 0.036 | -0.311 .. 0.033 | 0.400 |
| tair_control_horizon_abs_bias | offline_or_diagnostic_only | 0.197 | 0.025 .. 0.343 | 0.025 .. 0.374 | 0.524 |
| tair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.209 | 0.039 .. 0.408 | 0.039 .. 0.385 | 0.538 |
| rhair_first_step_mae | objective_secondary_selection | 0.535 | 0.436 .. 0.668 | 0.436 .. 0.668 | 0.657 |
| rhair_control_horizon_mae | objective_secondary_selection | 0.376 | 0.243 .. 0.525 | 0.243 .. 0.609 | 0.590 |
| rhair_weighted_horizon_mae | offline_or_diagnostic_only | 0.094 | -0.100 .. 0.243 | -0.100 .. 0.273 | 0.471 |
| rhair_full_horizon_mae | offline_or_diagnostic_only | 0.071 | -0.096 .. 0.218 | -0.096 .. 0.236 | 0.476 |
| rhair_final_step_mae | offline_or_diagnostic_only | -0.015 | -0.232 .. 0.089 | -0.232 .. 0.154 | 0.413 |
| rhair_control_horizon_abs_bias | offline_or_diagnostic_only | 0.059 | -0.039 .. 0.261 | -0.039 .. 0.261 | 0.486 |
| rhair_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.141 | -0.354 .. -0.011 | -0.354 .. -0.011 | 0.400 |
| co2_first_step_mae | objective_secondary_selection | 0.456 | 0.340 .. 0.532 | 0.340 .. 0.532 | 0.606 |
| co2_control_horizon_mae | objective_secondary_selection | 0.377 | 0.243 .. 0.518 | 0.243 .. 0.518 | 0.567 |
| co2_weighted_horizon_mae | objective_secondary_selection | 0.479 | 0.368 .. 0.625 | 0.368 .. 0.625 | 0.638 |
| co2_full_horizon_mae | objective_secondary_selection | 0.482 | 0.371 .. 0.629 | 0.371 .. 0.629 | 0.629 |
| co2_final_step_mae | weak_selection | 0.282 | 0.175 .. 0.404 | 0.160 .. 0.393 | 0.562 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | -0.024 | -0.161 .. 0.068 | -0.138 .. 0.068 | 0.476 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.132 | -0.046 .. 0.275 | -0.025 .. 0.275 | 0.486 |
| forecast_only_transfer_rank | offline_or_diagnostic_only | 0.232 | 0.082 .. 0.361 | 0.082 .. 0.314 | 0.533 |
| tair_transfer_selection_score | offline_or_diagnostic_only | 0.156 | -0.025 .. 0.318 | -0.025 .. 0.324 | 0.495 |
| rhair_transfer_selection_score | objective_secondary_selection | 0.371 | 0.261 .. 0.514 | 0.261 .. 0.569 | 0.615 |
| co2_transfer_selection_score | weak_selection | 0.341 | 0.236 .. 0.450 | 0.225 .. 0.446 | 0.596 |
| multiobjective_transfer_selection_score | objective_secondary_selection | 0.406 | 0.268 .. 0.500 | 0.268 .. 0.481 | 0.587 |
