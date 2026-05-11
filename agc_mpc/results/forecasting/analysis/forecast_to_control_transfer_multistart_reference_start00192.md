# Forecast-To-Control Transfer Analysis

Model count: `10`.

This report tests whether forecast-side validation metrics predict `GradientMPC` closed-loop outcomes.
For selection metrics, lower values are treated as better. Gradient metrics are diagnostic only.

## Metric Roles

| control_target | metric | role |
| --- | --- | --- |
| mpc_tair_mae | tair_first_step_mae | weak_selection |
| mpc_tair_mae | tair_control_horizon_mae | offline_or_diagnostic_only |
| mpc_tair_mae | tair_weighted_horizon_mae | secondary_selection |
| mpc_tair_mae | tair_full_horizon_mae | secondary_selection |
| mpc_tair_mae | tair_final_step_mae | secondary_selection |
| mpc_tair_mae | tair_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_tair_mae | tair_constraint_near_mae_proxy | weak_selection |
| mpc_tair_mae | rhair_first_step_mae | offline_or_diagnostic_only |
| mpc_tair_mae | rhair_control_horizon_mae | offline_or_diagnostic_only |
| mpc_tair_mae | rhair_weighted_horizon_mae | secondary_selection |
| mpc_tair_mae | rhair_full_horizon_mae | secondary_selection |
| mpc_tair_mae | rhair_final_step_mae | weak_selection |
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
| mpc_rhair_mae | tair_first_step_mae | secondary_selection |
| mpc_rhair_mae | tair_control_horizon_mae | secondary_selection |
| mpc_rhair_mae | tair_weighted_horizon_mae | secondary_selection |
| mpc_rhair_mae | tair_full_horizon_mae | secondary_selection |
| mpc_rhair_mae | tair_final_step_mae | primary_selection |
| mpc_rhair_mae | tair_control_horizon_abs_bias | secondary_selection |
| mpc_rhair_mae | tair_constraint_near_mae_proxy | secondary_selection |
| mpc_rhair_mae | rhair_first_step_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_control_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_weighted_horizon_mae | secondary_selection |
| mpc_rhair_mae | rhair_full_horizon_mae | secondary_selection |
| mpc_rhair_mae | rhair_final_step_mae | secondary_selection |
| mpc_rhair_mae | rhair_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_constraint_near_mae_proxy | weak_selection |
| mpc_rhair_mae | co2_first_step_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_control_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_full_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_final_step_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_rhair_mae | forecast_only_transfer_rank | secondary_selection |
| mpc_rhair_mae | tair_transfer_selection_score | secondary_selection |
| mpc_rhair_mae | rhair_transfer_selection_score | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_transfer_selection_score | offline_or_diagnostic_only |
| mpc_rhair_mae | multiobjective_transfer_selection_score | offline_or_diagnostic_only |
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
| mpc_co2_mae | tair_weighted_horizon_mae | weak_selection |
| mpc_co2_mae | tair_full_horizon_mae | weak_selection |
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
| mpc_co2_mae | co2_first_step_mae | offline_or_diagnostic_only |
| mpc_co2_mae | co2_control_horizon_mae | offline_or_diagnostic_only |
| mpc_co2_mae | co2_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_co2_mae | co2_full_horizon_mae | offline_or_diagnostic_only |
| mpc_co2_mae | co2_final_step_mae | offline_or_diagnostic_only |
| mpc_co2_mae | co2_control_horizon_abs_bias | weak_selection |
| mpc_co2_mae | co2_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_co2_mae | forecast_only_transfer_rank | offline_or_diagnostic_only |
| mpc_co2_mae | tair_transfer_selection_score | offline_or_diagnostic_only |
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
| mpc_objective | tair_first_step_mae | weak_selection |
| mpc_objective | tair_control_horizon_mae | objective_secondary_selection |
| mpc_objective | tair_weighted_horizon_mae | objective_secondary_selection |
| mpc_objective | tair_full_horizon_mae | objective_secondary_selection |
| mpc_objective | tair_final_step_mae | weak_selection |
| mpc_objective | tair_control_horizon_abs_bias | objective_secondary_selection |
| mpc_objective | tair_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_objective | rhair_first_step_mae | offline_or_diagnostic_only |
| mpc_objective | rhair_control_horizon_mae | offline_or_diagnostic_only |
| mpc_objective | rhair_weighted_horizon_mae | objective_secondary_selection |
| mpc_objective | rhair_full_horizon_mae | objective_primary_selection |
| mpc_objective | rhair_final_step_mae | objective_secondary_selection |
| mpc_objective | rhair_control_horizon_abs_bias | weak_selection |
| mpc_objective | rhair_constraint_near_mae_proxy | objective_secondary_selection |
| mpc_objective | co2_first_step_mae | offline_or_diagnostic_only |
| mpc_objective | co2_control_horizon_mae | offline_or_diagnostic_only |
| mpc_objective | co2_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_objective | co2_full_horizon_mae | offline_or_diagnostic_only |
| mpc_objective | co2_final_step_mae | offline_or_diagnostic_only |
| mpc_objective | co2_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_objective | co2_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_objective | forecast_only_transfer_rank | objective_secondary_selection |
| mpc_objective | tair_transfer_selection_score | objective_secondary_selection |
| mpc_objective | rhair_transfer_selection_score | weak_selection |
| mpc_objective | co2_transfer_selection_score | offline_or_diagnostic_only |
| mpc_objective | multiobjective_transfer_selection_score | weak_selection |
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
| 1 | itransformer_co2_control_aware_fusion | 4.219 | 5.969 | 5.188 | 1.500 | 8.556 | 0.126 | 2.372 | 20.161 | 0.0606 |
| 2 | current_hybrid_transformer | 4.438 | 4.938 | 3.188 | 5.188 | 6.722 | 0.539 | 0.962 | 49.138 | 0.0429 |
| 3 | itransformer_co2_protected_expert | 4.521 | 2.562 | 6.250 | 4.750 | 9.278 | 0.511 | 1.381 | 51.831 | 0.0814 |
| 4 | itransformer_co2_late_frozen_expert | 4.552 | 5.406 | 6.188 | 2.062 | 9.944 | 0.120 | 2.397 | 20.483 | 0.0616 |
| 5 | transformer_hybrid_residual | 4.688 | 3.562 | 2.312 | 8.188 | 9.167 | 0.129 | 0.558 | 20.637 | 0.0235 |
| 6 | itransformer_co2_late_residual | 5.646 | 6.125 | 5.000 | 5.812 | 9.000 | 0.243 | 1.269 | 47.742 | 0.1157 |
| 7 | segrnn_forecaster | 5.708 | 9.000 | 4.375 | 3.750 | 9.389 | 0.673 | 5.179 | 111.292 | 0.1164 |
| 8 | itransformer_co2_horizon_mixture | 5.854 | 2.125 | 7.688 | 7.750 | 13.722 | 0.229 | 0.556 | 26.270 | 0.0678 |
| 9 | itransformer_co2_residual | 6.625 | 5.312 | 6.500 | 8.062 | 10.778 | 0.353 | 1.595 | 10.701 | 0.0465 |
| 10 | frequency_forecaster | 8.750 | 10.000 | 8.312 | 7.938 | 18.722 | 0.343 | 2.140 | 12.041 | 0.0750 |

## Metric Transfer Quality

### Target: `mpc_tair_mae`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rhair_constraint_near_mae_proxy | selection | 0.512 | 0.612 | 0.711 | no | yes | 0.667 |
| tair_weighted_horizon_mae | selection | 0.386 | 0.576 | 0.711 | no | no | 0.333 |
| tair_full_horizon_mae | selection | 0.405 | 0.564 | 0.689 | no | no | 0.333 |
| forecast_only_transfer_rank | selection | 0.291 | 0.480 | 0.682 | no | no | 0.667 |
| rhair_full_horizon_mae | selection | 0.308 | 0.479 | 0.667 | no | no | 0.667 |
| rhair_weighted_horizon_mae | selection | 0.281 | 0.438 | 0.659 | no | no | 0.667 |
| tair_final_step_mae | selection | 0.473 | 0.370 | 0.644 | no | no | 0.333 |
| co2_first_step_mae | selection | 0.075 | 0.292 | 0.614 | no | yes | 0.667 |
| tair_first_step_mae | selection | 0.143 | 0.358 | 0.600 | no | no | 0.333 |
| rhair_final_step_mae | selection | 0.341 | 0.316 | 0.591 | no | no | 0.333 |
| tair_constraint_near_mae_proxy | selection | 0.455 | 0.280 | 0.591 | no | no | 0.333 |
| multiobjective_transfer_selection_score | selection | 0.130 | 0.188 | 0.578 | no | no | 0.333 |
| tair_control_horizon_mae | selection | 0.201 | 0.219 | 0.568 | no | no | 0.333 |
| co2_control_horizon_abs_bias | selection | 0.063 | 0.212 | 0.556 | no | yes | 0.667 |
| tair_transfer_selection_score | selection | 0.274 | 0.091 | 0.533 | no | no | 0.333 |
| co2_control_horizon_mae | selection | -0.027 | 0.170 | 0.523 | no | yes | 0.667 |
| co2_transfer_selection_score | selection | 0.006 | 0.091 | 0.489 | no | yes | 0.667 |
| rhair_control_horizon_mae | selection | 0.020 | 0.018 | 0.489 | no | no | 0.333 |
| co2_final_step_mae | selection | -0.104 | -0.224 | 0.444 | no | no | 0.333 |
| co2_weighted_horizon_mae | selection | -0.203 | -0.139 | 0.444 | no | no | 0.333 |
| rhair_transfer_selection_score | selection | -0.088 | -0.091 | 0.444 | no | no | 0.333 |
| co2_full_horizon_mae | selection | -0.248 | -0.176 | 0.422 | no | no | 0.000 |
| co2_constraint_near_mae_proxy | selection | -0.211 | -0.152 | 0.422 | no | no | 0.333 |
| tair_control_horizon_abs_bias | selection | 0.123 | -0.200 | 0.400 | no | no | 0.000 |
| rhair_first_step_mae | selection | -0.073 | -0.261 | 0.378 | no | no | 0.333 |
| rhair_control_horizon_abs_bias | selection | -0.301 | -0.467 | 0.356 | no | no | 0.000 |
| assim_sp_first_grad | diagnostic | -0.904 | -0.900 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | -0.801 | -0.754 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | -0.645 | -0.669 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | -0.597 | -0.669 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | -0.629 | -0.547 |  |  |  |  |
| co2_sp_first_grad | diagnostic | -0.543 | -0.486 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | -0.525 | -0.438 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | -0.243 | -0.413 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | 0.418 | 0.345 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | -0.199 | -0.255 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | -0.199 | -0.231 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | -0.451 | -0.207 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | 0.020 | 0.087 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | -0.020 | -0.087 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | 0.338 | 0.061 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | -0.247 | -0.036 |  |  |  |  |

### Target: `mpc_rhair_mae`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tair_transfer_selection_score | selection | 0.702 | 0.709 | 0.778 | yes | yes | 0.667 |
| tair_final_step_mae | selection | 0.601 | 0.673 | 0.756 | no | yes | 0.667 |
| tair_control_horizon_mae | selection | 0.328 | 0.657 | 0.750 | yes | yes | 0.667 |
| tair_control_horizon_abs_bias | selection | 0.601 | 0.721 | 0.733 | no | yes | 0.667 |
| rhair_final_step_mae | selection | 0.566 | 0.620 | 0.727 | no | yes | 1.000 |
| rhair_full_horizon_mae | selection | 0.408 | 0.418 | 0.689 | no | no | 0.667 |
| tair_constraint_near_mae_proxy | selection | 0.657 | 0.523 | 0.682 | no | no | 0.333 |
| tair_full_horizon_mae | selection | 0.541 | 0.479 | 0.667 | yes | yes | 0.667 |
| tair_first_step_mae | selection | 0.254 | 0.467 | 0.667 | no | yes | 0.667 |
| rhair_weighted_horizon_mae | selection | 0.364 | 0.365 | 0.659 | no | no | 0.667 |
| forecast_only_transfer_rank | selection | 0.274 | 0.365 | 0.659 | no | no | 0.667 |
| tair_weighted_horizon_mae | selection | 0.523 | 0.467 | 0.644 | yes | yes | 0.667 |
| rhair_constraint_near_mae_proxy | selection | 0.416 | 0.333 | 0.644 | yes | yes | 0.333 |
| rhair_first_step_mae | selection | -0.062 | 0.091 | 0.578 | no | no | 0.667 |
| rhair_control_horizon_mae | selection | 0.013 | 0.152 | 0.556 | no | no | 0.667 |
| rhair_transfer_selection_score | selection | 0.013 | 0.091 | 0.556 | no | no | 0.667 |
| co2_final_step_mae | selection | -0.086 | 0.042 | 0.556 | no | yes | 0.333 |
| multiobjective_transfer_selection_score | selection | 0.123 | -0.018 | 0.511 | no | no | 0.333 |
| co2_full_horizon_mae | selection | -0.425 | -0.164 | 0.489 | no | no | 0.333 |
| co2_weighted_horizon_mae | selection | -0.414 | -0.212 | 0.467 | no | no | 0.333 |
| rhair_control_horizon_abs_bias | selection | -0.178 | -0.152 | 0.422 | no | no | 0.333 |
| co2_control_horizon_mae | selection | -0.237 | -0.498 | 0.341 | no | no | 0.000 |
| co2_first_step_mae | selection | 0.001 | -0.365 | 0.341 | no | no | 0.000 |
| co2_transfer_selection_score | selection | -0.514 | -0.612 | 0.289 | no | no | 0.000 |
| co2_constraint_near_mae_proxy | selection | -0.593 | -0.624 | 0.267 | no | no | 0.333 |
| co2_control_horizon_abs_bias | selection | -0.707 | -0.818 | 0.178 | no | no | 0.000 |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.421 | -0.522 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.421 | 0.522 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | -0.657 | -0.511 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | 0.645 | 0.309 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | -0.350 | -0.292 |  |  |  |  |
| co2_sp_first_grad | diagnostic | -0.137 | 0.280 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | 0.053 | 0.255 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | -0.513 | -0.255 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | -0.467 | 0.170 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | -0.361 | -0.122 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | -0.344 | -0.109 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | -0.476 | -0.085 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | -0.303 | 0.036 |  |  |  |  |
| assim_sp_first_grad | diagnostic | -0.453 | 0.036 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | -0.352 | -0.012 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | 0.186 | 0.000 |  |  |  |  |

### Target: `mpc_co2_mae`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tair_full_horizon_mae | selection | 0.229 | 0.309 | 0.622 | no | yes | 0.333 |
| rhair_control_horizon_abs_bias | selection | 0.094 | 0.309 | 0.600 | no | yes | 0.667 |
| tair_weighted_horizon_mae | selection | 0.204 | 0.297 | 0.600 | no | yes | 0.333 |
| rhair_constraint_near_mae_proxy | selection | 0.245 | 0.188 | 0.600 | no | no | 0.333 |
| co2_control_horizon_abs_bias | selection | -0.117 | 0.333 | 0.578 | no | no | 0.333 |
| tair_final_step_mae | selection | 0.306 | 0.127 | 0.578 | no | yes | 0.333 |
| tair_constraint_near_mae_proxy | selection | 0.346 | 0.122 | 0.545 | no | yes | 0.333 |
| forecast_only_transfer_rank | selection | -0.058 | -0.109 | 0.523 | no | no | 0.333 |
| rhair_full_horizon_mae | selection | 0.021 | -0.055 | 0.511 | no | no | 0.333 |
| rhair_weighted_horizon_mae | selection | -0.024 | -0.097 | 0.477 | no | no | 0.333 |
| co2_first_step_mae | selection | -0.278 | -0.243 | 0.432 | no | no | 0.333 |
| multiobjective_transfer_selection_score | selection | -0.197 | -0.321 | 0.422 | no | no | 0.333 |
| tair_transfer_selection_score | selection | 0.248 | -0.188 | 0.422 | no | no | 0.000 |
| rhair_final_step_mae | selection | 0.169 | -0.182 | 0.409 | no | no | 0.000 |
| tair_first_step_mae | selection | -0.120 | -0.176 | 0.400 | no | no | 0.000 |
| tair_control_horizon_mae | selection | -0.040 | -0.243 | 0.386 | no | no | 0.000 |
| rhair_control_horizon_mae | selection | -0.358 | -0.430 | 0.378 | no | no | 0.000 |
| co2_final_step_mae | selection | -0.261 | -0.333 | 0.378 | no | no | 0.333 |
| tair_control_horizon_abs_bias | selection | 0.067 | -0.200 | 0.378 | no | yes | 0.333 |
| co2_constraint_near_mae_proxy | selection | -0.476 | -0.345 | 0.356 | no | no | 0.333 |
| rhair_transfer_selection_score | selection | -0.351 | -0.479 | 0.333 | no | no | 0.000 |
| co2_transfer_selection_score | selection | -0.313 | -0.321 | 0.333 | no | no | 0.333 |
| co2_control_horizon_mae | selection | -0.431 | -0.401 | 0.295 | no | no | 0.333 |
| co2_weighted_horizon_mae | selection | -0.632 | -0.576 | 0.289 | no | no | 0.333 |
| rhair_first_step_mae | selection | -0.438 | -0.636 | 0.267 | no | no | 0.000 |
| co2_full_horizon_mae | selection | -0.659 | -0.624 | 0.267 | no | no | 0.000 |
| co2_sp_first_grad | diagnostic | -0.753 | -0.839 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | -0.470 | -0.815 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | -0.857 | -0.681 |  |  |  |  |
| assim_sp_first_grad | diagnostic | -0.766 | -0.620 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | 0.525 | 0.535 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | -0.356 | -0.280 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | -0.615 | -0.267 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | -0.364 | -0.195 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.037 | 0.174 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.037 | -0.174 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | -0.370 | -0.170 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | -0.551 | -0.122 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | -0.311 | -0.085 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | -0.424 | -0.012 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | 0.411 | -0.006 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | -0.342 | 0.000 |  |  |  |  |

### Target: `mpc_objective`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rhair_full_horizon_mae | selection | 0.382 | 0.673 | 0.778 | yes | yes | 0.667 |
| rhair_weighted_horizon_mae | selection | 0.352 | 0.644 | 0.750 | yes | yes | 0.667 |
| rhair_final_step_mae | selection | 0.477 | 0.608 | 0.727 | no | yes | 0.667 |
| tair_full_horizon_mae | selection | 0.368 | 0.515 | 0.711 | no | yes | 0.667 |
| forecast_only_transfer_rank | selection | 0.345 | 0.547 | 0.705 | no | yes | 0.667 |
| tair_weighted_horizon_mae | selection | 0.360 | 0.503 | 0.689 | no | yes | 0.667 |
| tair_transfer_selection_score | selection | 0.434 | 0.430 | 0.689 | no | yes | 0.333 |
| tair_control_horizon_mae | selection | 0.258 | 0.401 | 0.659 | no | yes | 0.333 |
| rhair_constraint_near_mae_proxy | selection | 0.467 | 0.539 | 0.644 | no | no | 0.000 |
| tair_control_horizon_abs_bias | selection | 0.332 | 0.370 | 0.644 | no | no | 0.667 |
| tair_final_step_mae | selection | 0.378 | 0.370 | 0.622 | yes | yes | 0.667 |
| rhair_control_horizon_abs_bias | selection | 0.204 | 0.333 | 0.600 | no | no | 0.667 |
| multiobjective_transfer_selection_score | selection | 0.217 | 0.285 | 0.600 | no | no | 0.333 |
| rhair_transfer_selection_score | selection | 0.252 | 0.273 | 0.600 | yes | yes | 0.667 |
| rhair_control_horizon_mae | selection | 0.100 | 0.200 | 0.600 | yes | yes | 0.667 |
| tair_first_step_mae | selection | 0.229 | 0.309 | 0.578 | no | yes | 0.333 |
| tair_constraint_near_mae_proxy | selection | 0.351 | 0.170 | 0.545 | no | yes | 0.667 |
| rhair_first_step_mae | selection | 0.054 | 0.091 | 0.533 | yes | yes | 0.667 |
| co2_first_step_mae | selection | -0.048 | -0.049 | 0.523 | no | no | 0.000 |
| co2_weighted_horizon_mae | selection | -0.330 | -0.139 | 0.467 | no | no | 0.333 |
| co2_final_step_mae | selection | -0.112 | -0.103 | 0.467 | no | no | 0.000 |
| co2_full_horizon_mae | selection | -0.346 | -0.200 | 0.444 | no | no | 0.333 |
| co2_control_horizon_abs_bias | selection | -0.256 | -0.200 | 0.444 | no | no | 0.000 |
| co2_control_horizon_mae | selection | -0.173 | -0.195 | 0.432 | no | no | 0.000 |
| co2_constraint_near_mae_proxy | selection | -0.471 | -0.321 | 0.400 | no | no | 0.333 |
| co2_transfer_selection_score | selection | -0.265 | -0.333 | 0.378 | no | no | 0.000 |
| rhair_window_pos_lee_sp_first_grad | diagnostic | -0.750 | -0.729 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | 0.744 | 0.709 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.635 | -0.696 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.635 | 0.696 |  |  |  |  |
| co2_sp_first_grad | diagnostic | -0.524 | -0.474 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.172 | -0.292 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | -0.183 | -0.255 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | 0.060 | 0.243 |  |  |  |  |
| assim_sp_first_grad | diagnostic | -0.306 | -0.207 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | 0.180 | 0.170 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | -0.358 | -0.158 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | 0.033 | 0.097 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | -0.167 | -0.061 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | -0.106 | 0.061 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | 0.127 | 0.036 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | 0.048 | 0.024 |  |  |  |  |

## Robustness Summary

### Target: `mpc_tair_mae`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | weak_selection | 0.358 | 0.183 .. 0.650 | 0.183 .. 0.667 | 0.528 |
| tair_control_horizon_mae | offline_or_diagnostic_only | 0.219 | -0.008 .. 0.427 | -0.008 .. 0.395 | 0.486 |
| tair_weighted_horizon_mae | secondary_selection | 0.576 | 0.483 .. 0.717 | 0.483 .. 0.714 | 0.667 |
| tair_full_horizon_mae | secondary_selection | 0.564 | 0.467 .. 0.700 | 0.467 .. 0.690 | 0.639 |
| tair_final_step_mae | secondary_selection | 0.370 | 0.200 .. 0.517 | 0.200 .. 0.548 | 0.583 |
| tair_control_horizon_abs_bias | offline_or_diagnostic_only | -0.200 | -0.583 .. -0.033 | -0.583 .. 0.024 | 0.278 |
| tair_constraint_near_mae_proxy | weak_selection | 0.280 | 0.075 .. 0.410 | 0.075 .. 0.491 | 0.514 |
| rhair_first_step_mae | offline_or_diagnostic_only | -0.261 | -0.550 .. -0.100 | -0.550 .. -0.100 | 0.278 |
| rhair_control_horizon_mae | offline_or_diagnostic_only | 0.018 | -0.200 .. 0.183 | -0.200 .. 0.183 | 0.417 |
| rhair_weighted_horizon_mae | secondary_selection | 0.438 | 0.293 .. 0.745 | 0.293 .. 0.745 | 0.600 |
| rhair_full_horizon_mae | secondary_selection | 0.479 | 0.350 .. 0.800 | 0.350 .. 0.800 | 0.611 |
| rhair_final_step_mae | weak_selection | 0.316 | 0.126 .. 0.678 | 0.126 .. 0.678 | 0.514 |
| rhair_control_horizon_abs_bias | offline_or_diagnostic_only | -0.467 | -0.583 .. -0.383 | -0.583 .. -0.383 | 0.333 |
| rhair_constraint_near_mae_proxy | secondary_selection | 0.612 | 0.533 .. 0.750 | 0.533 .. 0.750 | 0.667 |
| co2_first_step_mae | weak_selection | 0.292 | 0.033 .. 0.410 | 0.033 .. 0.467 | 0.528 |
| co2_control_horizon_mae | offline_or_diagnostic_only | 0.170 | -0.133 .. 0.326 | -0.133 .. 0.326 | 0.417 |
| co2_weighted_horizon_mae | offline_or_diagnostic_only | -0.139 | -0.250 .. 0.183 | -0.381 .. 0.183 | 0.389 |
| co2_full_horizon_mae | offline_or_diagnostic_only | -0.176 | -0.283 .. 0.133 | -0.452 .. 0.133 | 0.389 |
| co2_final_step_mae | offline_or_diagnostic_only | -0.224 | -0.400 .. 0.067 | -0.405 .. 0.067 | 0.361 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | 0.212 | -0.067 .. 0.383 | -0.067 .. 0.383 | 0.472 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.152 | -0.400 .. 0.167 | -0.400 .. 0.167 | 0.333 |
| forecast_only_transfer_rank | secondary_selection | 0.480 | 0.333 .. 0.750 | 0.333 .. 0.750 | 0.639 |
| tair_transfer_selection_score | offline_or_diagnostic_only | 0.091 | -0.183 .. 0.350 | -0.183 .. 0.333 | 0.444 |
| rhair_transfer_selection_score | offline_or_diagnostic_only | -0.091 | -0.450 .. 0.167 | -0.450 .. 0.167 | 0.333 |
| co2_transfer_selection_score | offline_or_diagnostic_only | 0.091 | -0.233 .. 0.300 | -0.233 .. 0.267 | 0.389 |
| multiobjective_transfer_selection_score | offline_or_diagnostic_only | 0.188 | 0.033 .. 0.460 | 0.033 .. 0.476 | 0.528 |

### Target: `mpc_rhair_mae`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | secondary_selection | 0.467 | 0.317 .. 0.567 | 0.317 .. 0.567 | 0.611 |
| tair_control_horizon_mae | secondary_selection | 0.657 | 0.527 .. 0.762 | 0.527 .. 0.762 | 0.686 |
| tair_weighted_horizon_mae | secondary_selection | 0.467 | 0.267 .. 0.583 | 0.267 .. 0.583 | 0.556 |
| tair_full_horizon_mae | secondary_selection | 0.479 | 0.283 .. 0.583 | 0.283 .. 0.583 | 0.583 |
| tair_final_step_mae | primary_selection | 0.673 | 0.567 .. 0.833 | 0.567 .. 0.833 | 0.722 |
| tair_control_horizon_abs_bias | secondary_selection | 0.721 | 0.650 .. 0.800 | 0.650 .. 0.800 | 0.694 |
| tair_constraint_near_mae_proxy | secondary_selection | 0.523 | 0.393 .. 0.695 | 0.393 .. 0.695 | 0.629 |
| rhair_first_step_mae | offline_or_diagnostic_only | 0.091 | -0.117 .. 0.400 | -0.117 .. 0.400 | 0.500 |
| rhair_control_horizon_mae | offline_or_diagnostic_only | 0.152 | -0.033 .. 0.483 | -0.033 .. 0.483 | 0.472 |
| rhair_weighted_horizon_mae | secondary_selection | 0.365 | 0.176 .. 0.544 | 0.176 .. 0.544 | 0.600 |
| rhair_full_horizon_mae | secondary_selection | 0.418 | 0.250 .. 0.583 | 0.250 .. 0.583 | 0.639 |
| rhair_final_step_mae | secondary_selection | 0.620 | 0.510 .. 0.733 | 0.510 .. 0.733 | 0.686 |
| rhair_control_horizon_abs_bias | offline_or_diagnostic_only | -0.152 | -0.283 .. 0.167 | -0.283 .. 0.167 | 0.361 |
| rhair_constraint_near_mae_proxy | weak_selection | 0.333 | 0.083 .. 0.550 | 0.083 .. 0.550 | 0.556 |
| co2_first_step_mae | offline_or_diagnostic_only | -0.365 | -0.527 .. -0.217 | -0.527 .. -0.217 | 0.257 |
| co2_control_horizon_mae | offline_or_diagnostic_only | -0.498 | -0.661 .. -0.350 | -0.661 .. -0.350 | 0.257 |
| co2_weighted_horizon_mae | offline_or_diagnostic_only | -0.212 | -0.367 .. 0.083 | -0.452 .. 0.083 | 0.417 |
| co2_full_horizon_mae | offline_or_diagnostic_only | -0.164 | -0.300 .. 0.150 | -0.405 .. 0.150 | 0.444 |
| co2_final_step_mae | offline_or_diagnostic_only | 0.042 | -0.167 .. 0.433 | -0.167 .. 0.433 | 0.472 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | -0.818 | -0.917 .. -0.750 | -0.917 .. -0.738 | 0.111 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.624 | -0.783 .. -0.483 | -0.786 .. -0.483 | 0.194 |
| forecast_only_transfer_rank | secondary_selection | 0.365 | 0.167 .. 0.567 | 0.167 .. 0.567 | 0.611 |
| tair_transfer_selection_score | secondary_selection | 0.709 | 0.550 .. 0.783 | 0.550 .. 0.783 | 0.694 |
| rhair_transfer_selection_score | offline_or_diagnostic_only | 0.091 | -0.050 .. 0.400 | -0.050 .. 0.400 | 0.500 |
| co2_transfer_selection_score | offline_or_diagnostic_only | -0.612 | -0.717 .. -0.483 | -0.717 .. -0.500 | 0.250 |
| multiobjective_transfer_selection_score | offline_or_diagnostic_only | -0.018 | -0.267 .. 0.267 | -0.267 .. 0.267 | 0.444 |

### Target: `mpc_co2_mae`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | offline_or_diagnostic_only | -0.176 | -0.483 .. 0.083 | -0.483 .. 0.083 | 0.278 |
| tair_control_horizon_mae | offline_or_diagnostic_only | -0.243 | -0.577 .. -0.008 | -0.577 .. -0.008 | 0.257 |
| tair_weighted_horizon_mae | weak_selection | 0.297 | 0.167 .. 0.667 | 0.095 .. 0.667 | 0.528 |
| tair_full_horizon_mae | weak_selection | 0.309 | 0.183 .. 0.683 | 0.119 .. 0.683 | 0.556 |
| tair_final_step_mae | offline_or_diagnostic_only | 0.127 | -0.067 .. 0.433 | -0.067 .. 0.433 | 0.500 |
| tair_control_horizon_abs_bias | offline_or_diagnostic_only | -0.200 | -0.517 .. -0.017 | -0.517 .. -0.017 | 0.250 |
| tair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.122 | -0.109 .. 0.410 | -0.108 .. 0.410 | 0.457 |
| rhair_first_step_mae | offline_or_diagnostic_only | -0.636 | -0.800 .. -0.533 | -0.800 .. -0.533 | 0.194 |
| rhair_control_horizon_mae | offline_or_diagnostic_only | -0.430 | -0.567 .. -0.250 | -0.550 .. -0.250 | 0.333 |
| rhair_weighted_horizon_mae | offline_or_diagnostic_only | -0.097 | -0.377 .. 0.209 | -0.377 .. 0.209 | 0.371 |
| rhair_full_horizon_mae | offline_or_diagnostic_only | -0.055 | -0.317 .. 0.267 | -0.317 .. 0.267 | 0.417 |
| rhair_final_step_mae | offline_or_diagnostic_only | -0.182 | -0.494 .. 0.092 | -0.494 .. 0.092 | 0.286 |
| rhair_control_horizon_abs_bias | weak_selection | 0.309 | 0.067 .. 0.433 | 0.067 .. 0.433 | 0.528 |
| rhair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.188 | 0.017 .. 0.567 | 0.017 .. 0.567 | 0.528 |
| co2_first_step_mae | offline_or_diagnostic_only | -0.243 | -0.433 .. 0.008 | -0.433 .. 0.008 | 0.361 |
| co2_control_horizon_mae | offline_or_diagnostic_only | -0.401 | -0.633 .. -0.192 | -0.633 .. -0.156 | 0.194 |
| co2_weighted_horizon_mae | offline_or_diagnostic_only | -0.576 | -0.767 .. -0.417 | -0.690 .. -0.417 | 0.194 |
| co2_full_horizon_mae | offline_or_diagnostic_only | -0.624 | -0.833 .. -0.483 | -0.786 .. -0.483 | 0.167 |
| co2_final_step_mae | offline_or_diagnostic_only | -0.333 | -0.600 .. -0.083 | -0.524 .. -0.083 | 0.278 |
| co2_control_horizon_abs_bias | weak_selection | 0.333 | 0.200 .. 0.617 | 0.200 .. 0.617 | 0.528 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.345 | -0.517 .. -0.100 | -0.517 .. -0.100 | 0.278 |
| forecast_only_transfer_rank | offline_or_diagnostic_only | -0.109 | -0.267 .. 0.233 | -0.243 .. 0.233 | 0.444 |
| tair_transfer_selection_score | offline_or_diagnostic_only | -0.188 | -0.500 .. 0.033 | -0.500 .. 0.033 | 0.306 |
| rhair_transfer_selection_score | offline_or_diagnostic_only | -0.479 | -0.683 .. -0.317 | -0.683 .. -0.317 | 0.250 |
| co2_transfer_selection_score | offline_or_diagnostic_only | -0.321 | -0.567 .. -0.133 | -0.567 .. -0.071 | 0.222 |
| multiobjective_transfer_selection_score | offline_or_diagnostic_only | -0.321 | -0.477 .. -0.083 | -0.477 .. 0.048 | 0.343 |

### Target: `mpc_objective`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | weak_selection | 0.309 | 0.100 .. 0.550 | 0.100 .. 0.595 | 0.500 |
| tair_control_horizon_mae | objective_secondary_selection | 0.401 | 0.226 .. 0.644 | 0.226 .. 0.683 | 0.600 |
| tair_weighted_horizon_mae | objective_secondary_selection | 0.503 | 0.367 .. 0.717 | 0.367 .. 0.717 | 0.639 |
| tair_full_horizon_mae | objective_secondary_selection | 0.515 | 0.383 .. 0.733 | 0.383 .. 0.733 | 0.667 |
| tair_final_step_mae | weak_selection | 0.370 | 0.133 .. 0.517 | 0.133 .. 0.517 | 0.528 |
| tair_control_horizon_abs_bias | objective_secondary_selection | 0.370 | 0.183 .. 0.517 | 0.183 .. 0.517 | 0.583 |
| tair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.170 | -0.092 .. 0.477 | -0.092 .. 0.477 | 0.457 |
| rhair_first_step_mae | offline_or_diagnostic_only | 0.091 | -0.250 .. 0.350 | -0.250 .. 0.350 | 0.417 |
| rhair_control_horizon_mae | offline_or_diagnostic_only | 0.200 | -0.100 .. 0.467 | -0.100 .. 0.467 | 0.500 |
| rhair_weighted_horizon_mae | objective_secondary_selection | 0.644 | 0.510 .. 0.795 | 0.510 .. 0.874 | 0.686 |
| rhair_full_horizon_mae | objective_primary_selection | 0.673 | 0.550 .. 0.833 | 0.550 .. 0.881 | 0.722 |
| rhair_final_step_mae | objective_secondary_selection | 0.608 | 0.494 .. 0.762 | 0.494 .. 0.731 | 0.686 |
| rhair_control_horizon_abs_bias | weak_selection | 0.333 | 0.217 .. 0.550 | 0.217 .. 0.550 | 0.556 |
| rhair_constraint_near_mae_proxy | objective_secondary_selection | 0.539 | 0.417 .. 0.617 | 0.417 .. 0.617 | 0.583 |
| co2_first_step_mae | offline_or_diagnostic_only | -0.049 | -0.159 .. 0.092 | -0.159 .. 0.156 | 0.486 |
| co2_control_horizon_mae | offline_or_diagnostic_only | -0.195 | -0.360 .. -0.025 | -0.360 .. 0.012 | 0.371 |
| co2_weighted_horizon_mae | offline_or_diagnostic_only | -0.139 | -0.333 .. 0.183 | -0.333 .. 0.183 | 0.389 |
| co2_full_horizon_mae | offline_or_diagnostic_only | -0.200 | -0.417 .. 0.100 | -0.417 .. 0.100 | 0.361 |
| co2_final_step_mae | offline_or_diagnostic_only | -0.103 | -0.233 .. 0.233 | -0.238 .. 0.233 | 0.417 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | -0.200 | -0.400 .. 0.100 | -0.400 .. 0.100 | 0.361 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.321 | -0.517 .. -0.067 | -0.517 .. -0.067 | 0.333 |
| forecast_only_transfer_rank | objective_secondary_selection | 0.547 | 0.433 .. 0.800 | 0.433 .. 0.810 | 0.667 |
| tair_transfer_selection_score | objective_secondary_selection | 0.430 | 0.233 .. 0.650 | 0.233 .. 0.571 | 0.611 |
| rhair_transfer_selection_score | weak_selection | 0.273 | -0.050 .. 0.550 | -0.050 .. 0.550 | 0.472 |
| co2_transfer_selection_score | offline_or_diagnostic_only | -0.333 | -0.483 .. 0.017 | -0.483 .. 0.017 | 0.306 |
| multiobjective_transfer_selection_score | weak_selection | 0.285 | 0.117 .. 0.500 | 0.117 .. 0.786 | 0.556 |
