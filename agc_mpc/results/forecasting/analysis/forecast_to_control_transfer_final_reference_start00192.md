# Forecast-To-Control Transfer Analysis

Model count: `16`.

This report tests whether forecast-side validation metrics predict `GradientMPC` closed-loop outcomes.
For selection metrics, lower values are treated as better. Gradient metrics are diagnostic only.

## Metric Roles

| control_target | metric | role |
| --- | --- | --- |
| mpc_tair_mae | tair_first_step_mae | secondary_selection |
| mpc_tair_mae | tair_control_horizon_mae | secondary_selection |
| mpc_tair_mae | tair_weighted_horizon_mae | secondary_selection |
| mpc_tair_mae | tair_full_horizon_mae | secondary_selection |
| mpc_tair_mae | tair_final_step_mae | secondary_selection |
| mpc_tair_mae | tair_control_horizon_abs_bias | secondary_selection |
| mpc_tair_mae | tair_constraint_near_mae_proxy | weak_selection |
| mpc_tair_mae | rhair_first_step_mae | offline_or_diagnostic_only |
| mpc_tair_mae | rhair_control_horizon_mae | offline_or_diagnostic_only |
| mpc_tair_mae | rhair_weighted_horizon_mae | secondary_selection |
| mpc_tair_mae | rhair_full_horizon_mae | secondary_selection |
| mpc_tair_mae | rhair_final_step_mae | secondary_selection |
| mpc_tair_mae | rhair_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_tair_mae | rhair_constraint_near_mae_proxy | secondary_selection |
| mpc_tair_mae | co2_first_step_mae | weak_selection |
| mpc_tair_mae | co2_control_horizon_mae | offline_or_diagnostic_only |
| mpc_tair_mae | co2_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_tair_mae | co2_full_horizon_mae | offline_or_diagnostic_only |
| mpc_tair_mae | co2_final_step_mae | offline_or_diagnostic_only |
| mpc_tair_mae | co2_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_tair_mae | co2_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_tair_mae | forecast_only_transfer_rank | secondary_selection |
| mpc_tair_mae | tair_transfer_selection_score | secondary_selection |
| mpc_tair_mae | rhair_transfer_selection_score | offline_or_diagnostic_only |
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
| mpc_rhair_mae | tair_first_step_mae | secondary_selection |
| mpc_rhair_mae | tair_control_horizon_mae | secondary_selection |
| mpc_rhair_mae | tair_weighted_horizon_mae | secondary_selection |
| mpc_rhair_mae | tair_full_horizon_mae | secondary_selection |
| mpc_rhair_mae | tair_final_step_mae | primary_selection |
| mpc_rhair_mae | tair_control_horizon_abs_bias | primary_selection |
| mpc_rhair_mae | tair_constraint_near_mae_proxy | secondary_selection |
| mpc_rhair_mae | rhair_first_step_mae | weak_selection |
| mpc_rhair_mae | rhair_control_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_weighted_horizon_mae | secondary_selection |
| mpc_rhair_mae | rhair_full_horizon_mae | secondary_selection |
| mpc_rhair_mae | rhair_final_step_mae | secondary_selection |
| mpc_rhair_mae | rhair_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_constraint_near_mae_proxy | secondary_selection |
| mpc_rhair_mae | co2_first_step_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_control_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_full_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_final_step_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_rhair_mae | forecast_only_transfer_rank | secondary_selection |
| mpc_rhair_mae | tair_transfer_selection_score | primary_selection |
| mpc_rhair_mae | rhair_transfer_selection_score | weak_selection |
| mpc_rhair_mae | co2_transfer_selection_score | offline_or_diagnostic_only |
| mpc_rhair_mae | multiobjective_transfer_selection_score | weak_selection |
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
| mpc_co2_mae | tair_first_step_mae | weak_selection |
| mpc_co2_mae | tair_control_horizon_mae | offline_or_diagnostic_only |
| mpc_co2_mae | tair_weighted_horizon_mae | secondary_selection |
| mpc_co2_mae | tair_full_horizon_mae | secondary_selection |
| mpc_co2_mae | tair_final_step_mae | offline_or_diagnostic_only |
| mpc_co2_mae | tair_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_co2_mae | tair_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_co2_mae | rhair_first_step_mae | offline_or_diagnostic_only |
| mpc_co2_mae | rhair_control_horizon_mae | offline_or_diagnostic_only |
| mpc_co2_mae | rhair_weighted_horizon_mae | secondary_selection |
| mpc_co2_mae | rhair_full_horizon_mae | secondary_selection |
| mpc_co2_mae | rhair_final_step_mae | weak_selection |
| mpc_co2_mae | rhair_control_horizon_abs_bias | secondary_selection |
| mpc_co2_mae | rhair_constraint_near_mae_proxy | secondary_selection |
| mpc_co2_mae | co2_first_step_mae | offline_or_diagnostic_only |
| mpc_co2_mae | co2_control_horizon_mae | offline_or_diagnostic_only |
| mpc_co2_mae | co2_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_co2_mae | co2_full_horizon_mae | offline_or_diagnostic_only |
| mpc_co2_mae | co2_final_step_mae | offline_or_diagnostic_only |
| mpc_co2_mae | co2_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_co2_mae | co2_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_co2_mae | forecast_only_transfer_rank | offline_or_diagnostic_only |
| mpc_co2_mae | tair_transfer_selection_score | weak_selection |
| mpc_co2_mae | rhair_transfer_selection_score | offline_or_diagnostic_only |
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
| mpc_objective | tair_first_step_mae | objective_secondary_selection |
| mpc_objective | tair_control_horizon_mae | weak_selection |
| mpc_objective | tair_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_objective | tair_full_horizon_mae | offline_or_diagnostic_only |
| mpc_objective | tair_final_step_mae | offline_or_diagnostic_only |
| mpc_objective | tair_control_horizon_abs_bias | weak_selection |
| mpc_objective | tair_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_objective | rhair_first_step_mae | weak_selection |
| mpc_objective | rhair_control_horizon_mae | objective_secondary_selection |
| mpc_objective | rhair_weighted_horizon_mae | objective_secondary_selection |
| mpc_objective | rhair_full_horizon_mae | objective_secondary_selection |
| mpc_objective | rhair_final_step_mae | objective_secondary_selection |
| mpc_objective | rhair_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_objective | rhair_constraint_near_mae_proxy | weak_selection |
| mpc_objective | co2_first_step_mae | offline_or_diagnostic_only |
| mpc_objective | co2_control_horizon_mae | offline_or_diagnostic_only |
| mpc_objective | co2_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_objective | co2_full_horizon_mae | offline_or_diagnostic_only |
| mpc_objective | co2_final_step_mae | offline_or_diagnostic_only |
| mpc_objective | co2_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_objective | co2_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_objective | forecast_only_transfer_rank | weak_selection |
| mpc_objective | tair_transfer_selection_score | objective_secondary_selection |
| mpc_objective | rhair_transfer_selection_score | objective_secondary_selection |
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
| 1 | current_hybrid_transformer | 5.167 | 5.688 | 4.250 | 5.562 | 6.722 | 0.543 | 0.967 | 49.084 | 0.0428 |
| 2 | itransformer_co2_control_aware_fusion | 5.406 | 7.094 | 7.625 | 1.500 | 8.556 | 0.126 | 2.372 | 20.161 | 0.0606 |
| 3 | itransformer_residual | 5.583 | 4.062 | 5.688 | 7.000 | 9.167 | 0.189 | 1.317 | 11.644 | 0.0360 |
| 4 | itransformer_co2_late_frozen_expert | 5.740 | 6.531 | 8.625 | 2.062 | 9.944 | 0.120 | 2.397 | 20.483 | 0.0616 |
| 5 | transformer_hybrid_residual | 6.396 | 4.500 | 2.875 | 11.812 | 9.167 | 0.129 | 0.546 | 20.698 | 0.0234 |
| 6 | itransformer_co2_late_residual | 6.604 | 7.812 | 6.000 | 6.000 | 9.000 | 0.243 | 1.269 | 47.742 | 0.1157 |
| 7 | segrnn_forecaster | 7.979 | 14.188 | 6.000 | 3.750 | 9.389 | 0.673 | 5.179 | 111.292 | 0.1164 |
| 8 | dlinear_forecaster | 8.583 | 9.938 | 5.500 | 10.312 | 15.056 | 0.261 | 2.010 | 11.316 | 0.0449 |
| 9 | itransformer_co2_horizon_mixture | 8.646 | 2.875 | 12.125 | 10.938 | 13.722 | 0.229 | 0.556 | 26.270 | 0.0678 |
| 10 | transformer_forecaster | 9.250 | 6.188 | 11.125 | 10.438 | 13.056 | 0.073 | 1.436 | 31.788 | 0.0389 |
| 11 | itransformer_co2_residual | 9.312 | 6.062 | 9.938 | 11.938 | 10.778 | 0.353 | 1.595 | 10.700 | 0.0465 |
| 12 | gru_forecaster | 10.375 | 10.438 | 14.625 | 6.062 | 14.278 | 0.489 | 4.057 | 176.336 | 0.1277 |
| 13 | nlinear_forecaster | 10.729 | 13.125 | 4.812 | 14.250 | 15.500 | 0.510 | 2.077 | 23.846 | 0.0452 |
| 14 | patchtst_residual | 10.938 | 8.688 | 11.875 | 12.250 | 14.833 | 0.296 | 1.211 | 57.069 | 0.0612 |
| 15 | lstm_forecaster | 11.604 | 12.812 | 11.750 | 10.250 | 15.111 | 0.396 | 2.434 | 39.634 | 0.0330 |
| 16 | frequency_forecaster | 13.688 | 16.000 | 13.188 | 11.875 | 18.722 | 0.343 | 2.140 | 12.041 | 0.0750 |

## Metric Transfer Quality

### Target: `mpc_tair_mae`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | selection | 0.319 | 0.659 | 0.742 | no | no | 0.333 |
| tair_transfer_selection_score | selection | 0.597 | 0.529 | 0.700 | no | no | 0.000 |
| tair_weighted_horizon_mae | selection | 0.457 | 0.597 | 0.683 | no | no | 0.000 |
| tair_full_horizon_mae | selection | 0.468 | 0.594 | 0.675 | no | no | 0.000 |
| forecast_only_transfer_rank | selection | 0.364 | 0.432 | 0.675 | no | no | 0.333 |
| rhair_full_horizon_mae | selection | 0.387 | 0.412 | 0.658 | no | no | 0.000 |
| rhair_weighted_horizon_mae | selection | 0.366 | 0.428 | 0.655 | no | no | 0.000 |
| tair_control_horizon_mae | selection | 0.315 | 0.478 | 0.647 | no | no | 0.000 |
| rhair_constraint_near_mae_proxy | selection | 0.445 | 0.376 | 0.642 | no | no | 0.333 |
| rhair_final_step_mae | selection | 0.402 | 0.372 | 0.639 | no | no | 0.000 |
| multiobjective_transfer_selection_score | selection | 0.303 | 0.338 | 0.633 | no | no | 0.333 |
| tair_control_horizon_abs_bias | selection | 0.380 | 0.365 | 0.625 | no | no | 0.000 |
| tair_final_step_mae | selection | 0.477 | 0.388 | 0.617 | no | no | 0.000 |
| co2_first_step_mae | selection | 0.181 | 0.255 | 0.605 | no | no | 0.667 |
| tair_constraint_near_mae_proxy | selection | 0.434 | 0.263 | 0.580 | no | yes | 0.333 |
| co2_control_horizon_mae | selection | 0.016 | 0.190 | 0.571 | no | no | 0.667 |
| co2_transfer_selection_score | selection | 0.017 | 0.124 | 0.533 | no | no | 0.667 |
| co2_control_horizon_abs_bias | selection | 0.014 | 0.059 | 0.525 | no | no | 0.667 |
| rhair_control_horizon_mae | selection | 0.084 | 0.021 | 0.500 | no | no | 0.000 |
| rhair_first_step_mae | selection | -0.002 | -0.026 | 0.492 | no | no | 0.000 |
| rhair_transfer_selection_score | selection | -0.020 | -0.022 | 0.487 | no | no | 0.000 |
| co2_final_step_mae | selection | -0.291 | -0.147 | 0.458 | no | no | 0.333 |
| co2_weighted_horizon_mae | selection | -0.232 | -0.118 | 0.458 | no | no | 0.333 |
| co2_full_horizon_mae | selection | -0.294 | -0.126 | 0.450 | no | no | 0.000 |
| co2_constraint_near_mae_proxy | selection | -0.269 | -0.185 | 0.433 | no | no | 0.333 |
| rhair_control_horizon_abs_bias | selection | -0.248 | -0.347 | 0.383 | no | no | 0.000 |
| assim_sp_first_grad | diagnostic | -0.724 | -0.690 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | -0.625 | -0.640 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | -0.669 | -0.623 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | -0.582 | -0.581 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | -0.603 | -0.549 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | -0.611 | -0.511 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | 0.497 | 0.500 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | -0.473 | -0.352 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | -0.372 | -0.340 |  |  |  |  |
| co2_sp_first_grad | diagnostic | -0.322 | -0.290 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | -0.355 | -0.249 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | -0.402 | -0.243 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | -0.351 | -0.222 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.141 | -0.187 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.174 | -0.169 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.173 | 0.157 |  |  |  |  |

### Target: `mpc_rhair_mae`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tair_final_step_mae | selection | 0.523 | 0.726 | 0.800 | no | yes | 0.667 |
| tair_transfer_selection_score | selection | 0.656 | 0.718 | 0.767 | no | yes | 0.667 |
| tair_control_horizon_abs_bias | selection | 0.550 | 0.738 | 0.758 | no | no | 0.667 |
| tair_full_horizon_mae | selection | 0.485 | 0.653 | 0.742 | no | yes | 0.667 |
| tair_weighted_horizon_mae | selection | 0.468 | 0.650 | 0.733 | no | yes | 0.667 |
| rhair_final_step_mae | selection | 0.608 | 0.596 | 0.714 | no | yes | 1.000 |
| tair_first_step_mae | selection | 0.253 | 0.550 | 0.708 | no | yes | 0.667 |
| tair_control_horizon_mae | selection | 0.264 | 0.543 | 0.706 | no | yes | 0.667 |
| rhair_full_horizon_mae | selection | 0.528 | 0.479 | 0.675 | yes | yes | 0.667 |
| rhair_weighted_horizon_mae | selection | 0.497 | 0.481 | 0.664 | yes | yes | 0.667 |
| forecast_only_transfer_rank | selection | 0.357 | 0.453 | 0.658 | no | no | 0.333 |
| rhair_constraint_near_mae_proxy | selection | 0.535 | 0.403 | 0.658 | no | no | 0.333 |
| tair_constraint_near_mae_proxy | selection | 0.513 | 0.434 | 0.655 | no | no | 0.000 |
| rhair_first_step_mae | selection | 0.015 | 0.250 | 0.625 | yes | yes | 0.667 |
| rhair_transfer_selection_score | selection | 0.189 | 0.313 | 0.605 | yes | yes | 0.667 |
| multiobjective_transfer_selection_score | selection | 0.198 | 0.259 | 0.600 | no | no | 0.333 |
| rhair_control_horizon_mae | selection | 0.131 | 0.244 | 0.583 | yes | yes | 0.333 |
| co2_final_step_mae | selection | -0.075 | -0.056 | 0.525 | no | no | 0.000 |
| co2_full_horizon_mae | selection | -0.396 | -0.094 | 0.500 | no | no | 0.333 |
| rhair_control_horizon_abs_bias | selection | 0.086 | 0.012 | 0.500 | no | no | 0.000 |
| co2_weighted_horizon_mae | selection | -0.396 | -0.165 | 0.475 | no | no | 0.333 |
| co2_first_step_mae | selection | -0.048 | -0.169 | 0.462 | no | no | 0.000 |
| co2_control_horizon_mae | selection | -0.296 | -0.325 | 0.429 | no | no | 0.000 |
| co2_transfer_selection_score | selection | -0.437 | -0.379 | 0.383 | no | no | 0.000 |
| co2_constraint_near_mae_proxy | selection | -0.425 | -0.385 | 0.367 | no | no | 0.333 |
| co2_control_horizon_abs_bias | selection | -0.483 | -0.582 | 0.308 | no | no | 0.000 |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.457 | 0.595 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.457 | -0.592 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | -0.610 | -0.584 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | 0.721 | 0.526 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | -0.419 | -0.393 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.190 | -0.361 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | -0.573 | -0.322 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | -0.483 | -0.322 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | -0.442 | -0.319 |  |  |  |  |
| assim_sp_first_grad | diagnostic | -0.568 | -0.311 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | -0.552 | -0.278 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | -0.432 | -0.263 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | -0.500 | -0.258 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | -0.323 | -0.169 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | -0.502 | -0.102 |  |  |  |  |
| co2_sp_first_grad | diagnostic | -0.338 | -0.087 |  |  |  |  |

### Target: `mpc_co2_mae`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rhair_constraint_near_mae_proxy | selection | 0.671 | 0.471 | 0.675 | no | no | 0.333 |
| rhair_full_horizon_mae | selection | 0.573 | 0.459 | 0.675 | no | no | 0.333 |
| rhair_control_horizon_abs_bias | selection | 0.409 | 0.500 | 0.667 | no | yes | 0.333 |
| rhair_weighted_horizon_mae | selection | 0.543 | 0.405 | 0.664 | no | no | 0.333 |
| tair_full_horizon_mae | selection | 0.127 | 0.359 | 0.625 | no | no | 0.333 |
| tair_weighted_horizon_mae | selection | 0.108 | 0.356 | 0.617 | no | no | 0.333 |
| tair_transfer_selection_score | selection | 0.293 | 0.250 | 0.583 | no | no | 0.333 |
| rhair_final_step_mae | selection | 0.581 | 0.278 | 0.580 | no | no | 0.000 |
| tair_first_step_mae | selection | 0.001 | 0.271 | 0.575 | no | no | 0.000 |
| forecast_only_transfer_rank | selection | 0.220 | 0.244 | 0.575 | no | no | 0.333 |
| rhair_transfer_selection_score | selection | 0.318 | 0.205 | 0.571 | no | no | 0.000 |
| tair_final_step_mae | selection | 0.172 | 0.191 | 0.567 | no | no | 0.000 |
| multiobjective_transfer_selection_score | selection | 0.141 | 0.121 | 0.550 | no | no | 0.333 |
| rhair_control_horizon_mae | selection | 0.131 | 0.076 | 0.550 | no | no | 0.000 |
| co2_control_horizon_abs_bias | selection | -0.056 | 0.185 | 0.542 | no | no | 0.000 |
| tair_control_horizon_mae | selection | -0.079 | 0.149 | 0.538 | no | no | 0.333 |
| tair_control_horizon_abs_bias | selection | 0.095 | 0.097 | 0.525 | no | no | 0.333 |
| tair_constraint_near_mae_proxy | selection | 0.099 | 0.028 | 0.504 | no | yes | 0.333 |
| co2_first_step_mae | selection | -0.195 | -0.149 | 0.445 | no | no | 0.000 |
| rhair_first_step_mae | selection | -0.096 | -0.141 | 0.442 | no | no | 0.000 |
| co2_transfer_selection_score | selection | -0.308 | -0.212 | 0.400 | no | no | 0.000 |
| co2_final_step_mae | selection | -0.288 | -0.350 | 0.392 | no | no | 0.000 |
| co2_control_horizon_mae | selection | -0.360 | -0.293 | 0.378 | no | no | 0.000 |
| co2_constraint_near_mae_proxy | selection | -0.281 | -0.341 | 0.367 | no | no | 0.000 |
| co2_full_horizon_mae | selection | -0.501 | -0.506 | 0.317 | no | no | 0.000 |
| co2_weighted_horizon_mae | selection | -0.479 | -0.535 | 0.308 | no | no | 0.000 |
| co2_sp_first_grad | diagnostic | -0.696 | -0.796 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | -0.734 | -0.711 |  |  |  |  |
| assim_sp_first_grad | diagnostic | -0.696 | -0.646 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | -0.511 | -0.505 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | -0.639 | -0.493 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | -0.510 | -0.475 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | -0.641 | -0.428 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | -0.485 | -0.425 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | -0.628 | -0.419 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | -0.487 | -0.369 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | 0.760 | 0.332 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.030 | -0.313 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | -0.489 | -0.293 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.031 | 0.287 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.133 | 0.275 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | -0.651 | -0.222 |  |  |  |  |

### Target: `mpc_objective`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rhair_full_horizon_mae | selection | 0.540 | 0.524 | 0.692 | yes | yes | 0.333 |
| rhair_transfer_selection_score | selection | 0.302 | 0.505 | 0.689 | yes | yes | 0.333 |
| rhair_weighted_horizon_mae | selection | 0.523 | 0.509 | 0.681 | yes | yes | 0.333 |
| rhair_final_step_mae | selection | 0.562 | 0.444 | 0.664 | no | yes | 0.333 |
| tair_transfer_selection_score | selection | 0.354 | 0.409 | 0.650 | no | yes | 0.667 |
| rhair_control_horizon_mae | selection | 0.292 | 0.400 | 0.650 | yes | yes | 0.333 |
| rhair_first_step_mae | selection | 0.166 | 0.347 | 0.642 | yes | yes | 0.333 |
| tair_control_horizon_abs_bias | selection | 0.269 | 0.338 | 0.625 | no | no | 0.333 |
| tair_first_step_mae | selection | 0.229 | 0.356 | 0.608 | no | yes | 0.333 |
| rhair_constraint_near_mae_proxy | selection | 0.518 | 0.279 | 0.592 | no | no | 0.000 |
| forecast_only_transfer_rank | selection | 0.186 | 0.279 | 0.592 | no | no | 0.333 |
| tair_control_horizon_mae | selection | 0.154 | 0.265 | 0.588 | no | yes | 0.667 |
| tair_full_horizon_mae | selection | 0.241 | 0.244 | 0.575 | no | yes | 0.667 |
| tair_weighted_horizon_mae | selection | 0.236 | 0.238 | 0.567 | no | yes | 0.667 |
| tair_final_step_mae | selection | 0.233 | 0.182 | 0.567 | no | yes | 0.333 |
| multiobjective_transfer_selection_score | selection | 0.105 | 0.174 | 0.567 | no | no | 0.333 |
| rhair_control_horizon_abs_bias | selection | 0.231 | 0.165 | 0.567 | no | no | 0.000 |
| co2_final_step_mae | selection | -0.156 | 0.059 | 0.542 | no | no | 0.333 |
| tair_constraint_near_mae_proxy | selection | 0.246 | 0.068 | 0.504 | no | no | 0.000 |
| co2_full_horizon_mae | selection | -0.385 | -0.162 | 0.450 | no | no | 0.000 |
| co2_transfer_selection_score | selection | -0.423 | -0.224 | 0.433 | no | no | 0.000 |
| co2_first_step_mae | selection | -0.090 | -0.265 | 0.429 | no | no | 0.000 |
| co2_control_horizon_abs_bias | selection | -0.229 | -0.262 | 0.425 | no | no | 0.000 |
| co2_weighted_horizon_mae | selection | -0.378 | -0.256 | 0.425 | no | no | 0.000 |
| co2_control_horizon_mae | selection | -0.260 | -0.271 | 0.412 | no | no | 0.000 |
| co2_constraint_near_mae_proxy | selection | -0.402 | -0.329 | 0.400 | no | no | 0.000 |
| cost_grad_mean_abs | diagnostic | 0.725 | 0.585 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.390 | -0.474 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.389 | 0.438 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | -0.485 | -0.356 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.309 | -0.294 |  |  |  |  |
| co2_sp_first_grad | diagnostic | -0.461 | -0.277 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | -0.359 | -0.182 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | -0.226 | 0.118 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | -0.105 | 0.109 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | -0.032 | 0.100 |  |  |  |  |
| assim_sp_first_grad | diagnostic | -0.344 | -0.085 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | -0.381 | -0.079 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | -0.024 | 0.047 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | -0.314 | -0.041 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | -0.074 | -0.009 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | -0.247 | 0.003 |  |  |  |  |

## Robustness Summary

### Target: `mpc_tair_mae`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | secondary_selection | 0.659 | 0.614 .. 0.807 | 0.549 .. 0.807 | 0.724 |
| tair_control_horizon_mae | secondary_selection | 0.478 | 0.395 .. 0.642 | 0.360 .. 0.642 | 0.615 |
| tair_weighted_horizon_mae | secondary_selection | 0.597 | 0.532 .. 0.675 | 0.374 .. 0.675 | 0.648 |
| tair_full_horizon_mae | secondary_selection | 0.594 | 0.529 .. 0.671 | 0.368 .. 0.671 | 0.638 |
| tair_final_step_mae | secondary_selection | 0.388 | 0.293 .. 0.461 | 0.033 .. 0.461 | 0.581 |
| tair_control_horizon_abs_bias | secondary_selection | 0.365 | 0.257 .. 0.575 | 0.038 .. 0.575 | 0.590 |
| tair_constraint_near_mae_proxy | weak_selection | 0.263 | 0.127 .. 0.345 | 0.041 .. 0.345 | 0.529 |
| rhair_first_step_mae | offline_or_diagnostic_only | -0.026 | -0.161 .. 0.139 | -0.319 .. 0.139 | 0.448 |
| rhair_control_horizon_mae | offline_or_diagnostic_only | 0.021 | -0.104 .. 0.171 | -0.159 .. 0.138 | 0.457 |
| rhair_weighted_horizon_mae | secondary_selection | 0.428 | 0.338 .. 0.624 | 0.212 .. 0.624 | 0.625 |
| rhair_full_horizon_mae | secondary_selection | 0.412 | 0.318 .. 0.604 | 0.214 .. 0.604 | 0.629 |
| rhair_final_step_mae | secondary_selection | 0.372 | 0.270 .. 0.620 | 0.135 .. 0.620 | 0.606 |
| rhair_control_horizon_abs_bias | offline_or_diagnostic_only | -0.347 | -0.443 .. -0.207 | -0.610 .. -0.207 | 0.343 |
| rhair_constraint_near_mae_proxy | secondary_selection | 0.376 | 0.314 .. 0.596 | 0.209 .. 0.596 | 0.619 |
| co2_first_step_mae | weak_selection | 0.255 | 0.125 .. 0.381 | 0.125 .. 0.465 | 0.562 |
| co2_control_horizon_mae | offline_or_diagnostic_only | 0.190 | 0.043 .. 0.349 | 0.043 .. 0.514 | 0.524 |
| co2_weighted_horizon_mae | offline_or_diagnostic_only | -0.118 | -0.239 .. 0.071 | -0.319 .. 0.264 | 0.410 |
| co2_full_horizon_mae | offline_or_diagnostic_only | -0.126 | -0.250 .. 0.061 | -0.336 .. 0.258 | 0.400 |
| co2_final_step_mae | offline_or_diagnostic_only | -0.147 | -0.264 .. 0.021 | -0.310 .. 0.220 | 0.410 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | 0.059 | -0.096 .. 0.200 | -0.096 .. 0.264 | 0.476 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.185 | -0.336 .. -0.011 | -0.336 .. 0.066 | 0.381 |
| forecast_only_transfer_rank | secondary_selection | 0.432 | 0.340 .. 0.629 | 0.234 .. 0.629 | 0.641 |
| tair_transfer_selection_score | secondary_selection | 0.529 | 0.443 .. 0.700 | 0.368 .. 0.700 | 0.657 |
| rhair_transfer_selection_score | offline_or_diagnostic_only | -0.022 | -0.152 .. 0.154 | -0.300 .. 0.154 | 0.442 |
| co2_transfer_selection_score | offline_or_diagnostic_only | 0.124 | -0.029 .. 0.302 | -0.029 .. 0.473 | 0.476 |
| multiobjective_transfer_selection_score | weak_selection | 0.338 | 0.257 .. 0.592 | 0.209 .. 0.592 | 0.600 |

### Target: `mpc_rhair_mae`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | secondary_selection | 0.550 | 0.457 .. 0.671 | 0.308 .. 0.671 | 0.676 |
| tair_control_horizon_mae | secondary_selection | 0.543 | 0.452 .. 0.614 | 0.432 .. 0.614 | 0.673 |
| tair_weighted_horizon_mae | secondary_selection | 0.650 | 0.582 .. 0.721 | 0.445 .. 0.721 | 0.705 |
| tair_full_horizon_mae | secondary_selection | 0.653 | 0.586 .. 0.725 | 0.451 .. 0.725 | 0.714 |
| tair_final_step_mae | primary_selection | 0.726 | 0.679 .. 0.886 | 0.637 .. 0.886 | 0.781 |
| tair_control_horizon_abs_bias | primary_selection | 0.738 | 0.682 .. 0.782 | 0.665 .. 0.775 | 0.724 |
| tair_constraint_near_mae_proxy | secondary_selection | 0.434 | 0.331 .. 0.574 | 0.289 .. 0.574 | 0.615 |
| rhair_first_step_mae | weak_selection | 0.250 | 0.089 .. 0.464 | 0.060 .. 0.432 | 0.571 |
| rhair_control_horizon_mae | offline_or_diagnostic_only | 0.244 | 0.082 .. 0.379 | 0.154 .. 0.379 | 0.524 |
| rhair_weighted_horizon_mae | secondary_selection | 0.481 | 0.370 .. 0.568 | 0.157 .. 0.568 | 0.615 |
| rhair_full_horizon_mae | secondary_selection | 0.479 | 0.368 .. 0.564 | 0.170 .. 0.564 | 0.629 |
| rhair_final_step_mae | secondary_selection | 0.596 | 0.517 .. 0.675 | 0.443 .. 0.675 | 0.683 |
| rhair_control_horizon_abs_bias | offline_or_diagnostic_only | 0.012 | -0.093 .. 0.175 | -0.280 .. 0.175 | 0.457 |
| rhair_constraint_near_mae_proxy | secondary_selection | 0.403 | 0.286 .. 0.514 | 0.077 .. 0.514 | 0.619 |
| co2_first_step_mae | offline_or_diagnostic_only | -0.169 | -0.331 .. -0.063 | -0.238 .. -0.071 | 0.413 |
| co2_control_horizon_mae | offline_or_diagnostic_only | -0.325 | -0.399 .. -0.206 | -0.405 .. -0.107 | 0.394 |
| co2_weighted_horizon_mae | offline_or_diagnostic_only | -0.165 | -0.304 .. 0.014 | -0.304 .. 0.253 | 0.429 |
| co2_full_horizon_mae | offline_or_diagnostic_only | -0.094 | -0.221 .. 0.100 | -0.221 .. 0.341 | 0.457 |
| co2_final_step_mae | offline_or_diagnostic_only | -0.056 | -0.150 .. 0.139 | -0.160 .. 0.489 | 0.486 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | -0.582 | -0.664 .. -0.518 | -0.692 .. -0.489 | 0.267 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.385 | -0.525 .. -0.254 | -0.525 .. -0.231 | 0.314 |
| forecast_only_transfer_rank | secondary_selection | 0.453 | 0.390 .. 0.604 | 0.349 .. 0.604 | 0.635 |
| tair_transfer_selection_score | primary_selection | 0.718 | 0.657 .. 0.771 | 0.599 .. 0.771 | 0.743 |
| rhair_transfer_selection_score | weak_selection | 0.313 | 0.166 .. 0.463 | 0.190 .. 0.463 | 0.548 |
| co2_transfer_selection_score | offline_or_diagnostic_only | -0.379 | -0.482 .. -0.271 | -0.468 .. -0.198 | 0.343 |
| multiobjective_transfer_selection_score | weak_selection | 0.259 | 0.159 .. 0.383 | 0.159 .. 0.383 | 0.562 |

### Target: `mpc_co2_mae`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | weak_selection | 0.271 | 0.189 .. 0.454 | -0.005 .. 0.454 | 0.543 |
| tair_control_horizon_mae | offline_or_diagnostic_only | 0.149 | 0.048 .. 0.298 | -0.047 .. 0.330 | 0.500 |
| tair_weighted_horizon_mae | secondary_selection | 0.356 | 0.271 .. 0.529 | 0.137 .. 0.529 | 0.581 |
| tair_full_horizon_mae | secondary_selection | 0.359 | 0.275 .. 0.532 | 0.143 .. 0.532 | 0.590 |
| tair_final_step_mae | offline_or_diagnostic_only | 0.191 | 0.086 .. 0.361 | -0.143 .. 0.341 | 0.533 |
| tair_control_horizon_abs_bias | offline_or_diagnostic_only | 0.097 | -0.054 .. 0.214 | -0.242 .. 0.214 | 0.476 |
| tair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.028 | -0.141 .. 0.141 | -0.184 .. 0.141 | 0.442 |
| rhair_first_step_mae | offline_or_diagnostic_only | -0.141 | -0.318 .. -0.032 | -0.363 .. -0.039 | 0.381 |
| rhair_control_horizon_mae | offline_or_diagnostic_only | 0.076 | -0.079 .. 0.218 | -0.082 .. 0.218 | 0.495 |
| rhair_weighted_horizon_mae | secondary_selection | 0.405 | 0.320 .. 0.599 | 0.146 .. 0.599 | 0.625 |
| rhair_full_horizon_mae | secondary_selection | 0.459 | 0.343 .. 0.618 | 0.187 .. 0.618 | 0.629 |
| rhair_final_step_mae | weak_selection | 0.278 | 0.123 .. 0.416 | -0.063 .. 0.416 | 0.519 |
| rhair_control_horizon_abs_bias | secondary_selection | 0.500 | 0.429 .. 0.586 | 0.401 .. 0.586 | 0.638 |
| rhair_constraint_near_mae_proxy | secondary_selection | 0.471 | 0.357 .. 0.632 | 0.220 .. 0.632 | 0.629 |
| co2_first_step_mae | offline_or_diagnostic_only | -0.149 | -0.257 .. -0.045 | -0.257 .. -0.003 | 0.410 |
| co2_control_horizon_mae | offline_or_diagnostic_only | -0.293 | -0.418 .. -0.177 | -0.418 .. -0.151 | 0.333 |
| co2_weighted_horizon_mae | offline_or_diagnostic_only | -0.535 | -0.639 .. -0.450 | -0.676 .. -0.385 | 0.267 |
| co2_full_horizon_mae | offline_or_diagnostic_only | -0.506 | -0.629 .. -0.418 | -0.731 .. -0.374 | 0.267 |
| co2_final_step_mae | offline_or_diagnostic_only | -0.350 | -0.461 .. -0.250 | -0.522 .. -0.148 | 0.362 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | 0.185 | 0.104 .. 0.343 | 0.099 .. 0.495 | 0.505 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.341 | -0.450 .. -0.200 | -0.450 .. -0.191 | 0.324 |
| forecast_only_transfer_rank | offline_or_diagnostic_only | 0.244 | 0.095 .. 0.386 | -0.011 .. 0.386 | 0.515 |
| tair_transfer_selection_score | weak_selection | 0.250 | 0.150 .. 0.411 | -0.088 .. 0.411 | 0.552 |
| rhair_transfer_selection_score | offline_or_diagnostic_only | 0.205 | 0.059 .. 0.325 | 0.019 .. 0.325 | 0.529 |
| co2_transfer_selection_score | offline_or_diagnostic_only | -0.212 | -0.346 .. -0.100 | -0.390 .. 0.016 | 0.352 |
| multiobjective_transfer_selection_score | offline_or_diagnostic_only | 0.121 | 0.029 .. 0.268 | -0.066 .. 0.268 | 0.510 |

### Target: `mpc_objective`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | objective_secondary_selection | 0.356 | 0.257 .. 0.471 | 0.242 .. 0.471 | 0.562 |
| tair_control_horizon_mae | weak_selection | 0.265 | 0.154 .. 0.386 | 0.047 .. 0.389 | 0.548 |
| tair_weighted_horizon_mae | offline_or_diagnostic_only | 0.238 | 0.096 .. 0.371 | 0.027 .. 0.358 | 0.524 |
| tair_full_horizon_mae | offline_or_diagnostic_only | 0.244 | 0.104 .. 0.379 | 0.033 .. 0.367 | 0.533 |
| tair_final_step_mae | offline_or_diagnostic_only | 0.182 | 0.039 .. 0.346 | 0.022 .. 0.264 | 0.514 |
| tair_control_horizon_abs_bias | weak_selection | 0.338 | 0.246 .. 0.493 | 0.254 .. 0.411 | 0.590 |
| tair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.068 | -0.093 .. 0.225 | -0.043 .. 0.225 | 0.452 |
| rhair_first_step_mae | weak_selection | 0.347 | 0.207 .. 0.539 | 0.159 .. 0.599 | 0.590 |
| rhair_control_horizon_mae | objective_secondary_selection | 0.400 | 0.271 .. 0.561 | 0.231 .. 0.561 | 0.600 |
| rhair_weighted_horizon_mae | objective_secondary_selection | 0.509 | 0.404 .. 0.615 | 0.349 .. 0.597 | 0.635 |
| rhair_full_horizon_mae | objective_secondary_selection | 0.524 | 0.421 .. 0.625 | 0.368 .. 0.607 | 0.648 |
| rhair_final_step_mae | objective_secondary_selection | 0.444 | 0.325 .. 0.554 | 0.237 .. 0.554 | 0.615 |
| rhair_control_horizon_abs_bias | offline_or_diagnostic_only | 0.165 | 0.043 .. 0.325 | 0.005 .. 0.325 | 0.524 |
| rhair_constraint_near_mae_proxy | weak_selection | 0.279 | 0.125 .. 0.393 | 0.066 .. 0.393 | 0.533 |
| co2_first_step_mae | offline_or_diagnostic_only | -0.265 | -0.411 .. -0.157 | -0.411 .. 0.088 | 0.375 |
| co2_control_horizon_mae | offline_or_diagnostic_only | -0.271 | -0.418 .. -0.161 | -0.418 .. 0.050 | 0.356 |
| co2_weighted_horizon_mae | offline_or_diagnostic_only | -0.256 | -0.389 .. -0.111 | -0.389 .. -0.005 | 0.371 |
| co2_full_horizon_mae | offline_or_diagnostic_only | -0.162 | -0.275 .. 0.000 | -0.275 .. 0.044 | 0.400 |
| co2_final_step_mae | offline_or_diagnostic_only | 0.059 | -0.114 .. 0.214 | -0.004 .. 0.154 | 0.486 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | -0.262 | -0.382 .. -0.157 | -0.330 .. -0.174 | 0.381 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.329 | -0.432 .. -0.218 | -0.432 .. -0.282 | 0.371 |
| forecast_only_transfer_rank | weak_selection | 0.279 | 0.146 .. 0.388 | 0.033 .. 0.364 | 0.552 |
| tair_transfer_selection_score | objective_secondary_selection | 0.409 | 0.286 .. 0.561 | 0.236 .. 0.539 | 0.610 |
| rhair_transfer_selection_score | objective_secondary_selection | 0.505 | 0.356 .. 0.611 | 0.286 .. 0.615 | 0.625 |
| co2_transfer_selection_score | offline_or_diagnostic_only | -0.224 | -0.346 .. -0.111 | -0.332 .. -0.022 | 0.390 |
| multiobjective_transfer_selection_score | offline_or_diagnostic_only | 0.174 | 0.039 .. 0.311 | -0.088 .. 0.341 | 0.524 |
