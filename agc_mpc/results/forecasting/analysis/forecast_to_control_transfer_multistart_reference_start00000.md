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
| mpc_tair_mae | co2_control_horizon_mae | offline_or_diagnostic_only |
| mpc_tair_mae | co2_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_tair_mae | co2_full_horizon_mae | weak_selection |
| mpc_tair_mae | co2_final_step_mae | offline_or_diagnostic_only |
| mpc_tair_mae | co2_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_tair_mae | co2_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_tair_mae | forecast_only_transfer_rank | offline_or_diagnostic_only |
| mpc_tair_mae | tair_transfer_selection_score | offline_or_diagnostic_only |
| mpc_tair_mae | rhair_transfer_selection_score | secondary_selection |
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
| mpc_rhair_mae | tair_control_horizon_abs_bias | secondary_selection |
| mpc_rhair_mae | tair_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_first_step_mae | secondary_selection |
| mpc_rhair_mae | rhair_control_horizon_mae | weak_selection |
| mpc_rhair_mae | rhair_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_full_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_final_step_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_first_step_mae | secondary_selection |
| mpc_rhair_mae | co2_control_horizon_mae | weak_selection |
| mpc_rhair_mae | co2_weighted_horizon_mae | weak_selection |
| mpc_rhair_mae | co2_full_horizon_mae | weak_selection |
| mpc_rhair_mae | co2_final_step_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_rhair_mae | forecast_only_transfer_rank | offline_or_diagnostic_only |
| mpc_rhair_mae | tair_transfer_selection_score | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_transfer_selection_score | secondary_selection |
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
| mpc_co2_mae | tair_constraint_near_mae_proxy | weak_selection |
| mpc_co2_mae | rhair_first_step_mae | offline_or_diagnostic_only |
| mpc_co2_mae | rhair_control_horizon_mae | offline_or_diagnostic_only |
| mpc_co2_mae | rhair_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_co2_mae | rhair_full_horizon_mae | offline_or_diagnostic_only |
| mpc_co2_mae | rhair_final_step_mae | offline_or_diagnostic_only |
| mpc_co2_mae | rhair_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_co2_mae | rhair_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_co2_mae | co2_first_step_mae | secondary_selection |
| mpc_co2_mae | co2_control_horizon_mae | weak_selection |
| mpc_co2_mae | co2_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_co2_mae | co2_full_horizon_mae | offline_or_diagnostic_only |
| mpc_co2_mae | co2_final_step_mae | offline_or_diagnostic_only |
| mpc_co2_mae | co2_control_horizon_abs_bias | secondary_selection |
| mpc_co2_mae | co2_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_co2_mae | forecast_only_transfer_rank | offline_or_diagnostic_only |
| mpc_co2_mae | tair_transfer_selection_score | offline_or_diagnostic_only |
| mpc_co2_mae | rhair_transfer_selection_score | offline_or_diagnostic_only |
| mpc_co2_mae | co2_transfer_selection_score | weak_selection |
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
| mpc_objective | tair_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_objective | rhair_first_step_mae | objective_secondary_selection |
| mpc_objective | rhair_control_horizon_mae | objective_secondary_selection |
| mpc_objective | rhair_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_objective | rhair_full_horizon_mae | offline_or_diagnostic_only |
| mpc_objective | rhair_final_step_mae | offline_or_diagnostic_only |
| mpc_objective | rhair_control_horizon_abs_bias | weak_selection |
| mpc_objective | rhair_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_objective | co2_first_step_mae | offline_or_diagnostic_only |
| mpc_objective | co2_control_horizon_mae | offline_or_diagnostic_only |
| mpc_objective | co2_weighted_horizon_mae | objective_secondary_selection |
| mpc_objective | co2_full_horizon_mae | objective_secondary_selection |
| mpc_objective | co2_final_step_mae | weak_selection |
| mpc_objective | co2_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_objective | co2_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_objective | forecast_only_transfer_rank | offline_or_diagnostic_only |
| mpc_objective | tair_transfer_selection_score | offline_or_diagnostic_only |
| mpc_objective | rhair_transfer_selection_score | objective_secondary_selection |
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
| 1 | itransformer_co2_control_aware_fusion | 4.219 | 5.969 | 5.188 | 1.500 | 8.556 | 2.217 | 4.261 | 6.623 | 0.1505 |
| 2 | current_hybrid_transformer | 4.438 | 4.938 | 3.188 | 5.188 | 6.722 | 0.362 | 1.206 | 18.818 | 0.0442 |
| 3 | itransformer_co2_protected_expert | 4.521 | 2.562 | 6.250 | 4.750 | 9.278 | 0.880 | 1.441 | 14.206 | 0.0606 |
| 4 | itransformer_co2_late_frozen_expert | 4.552 | 5.406 | 6.188 | 2.062 | 9.944 | 2.202 | 4.302 | 6.442 | 0.1538 |
| 5 | transformer_hybrid_residual | 4.688 | 3.562 | 2.312 | 8.188 | 9.167 | 1.672 | 4.584 | 18.168 | 0.1062 |
| 6 | itransformer_co2_late_residual | 5.646 | 6.125 | 5.000 | 5.812 | 9.000 | 1.153 | 1.618 | 10.125 | 0.0705 |
| 7 | segrnn_forecaster | 5.708 | 9.000 | 4.375 | 3.750 | 9.389 | 0.391 | 2.195 | 14.425 | 0.0486 |
| 8 | itransformer_co2_horizon_mixture | 5.854 | 2.125 | 7.688 | 7.750 | 13.722 | 3.329 | 5.668 | 29.380 | 0.3734 |
| 9 | itransformer_co2_residual | 6.625 | 5.312 | 6.500 | 8.062 | 10.778 | 0.938 | 1.500 | 6.331 | 0.0558 |
| 10 | frequency_forecaster | 8.750 | 10.000 | 8.312 | 7.938 | 18.722 | 1.725 | 8.759 | 15.530 | 0.4338 |

## Metric Transfer Quality

### Target: `mpc_tair_mae`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rhair_first_step_mae | selection | 0.382 | 0.576 | 0.689 | no | yes | 0.667 |
| rhair_control_horizon_abs_bias | selection | 0.659 | 0.479 | 0.667 | no | yes | 0.333 |
| rhair_transfer_selection_score | selection | 0.468 | 0.455 | 0.667 | no | yes | 0.667 |
| rhair_control_horizon_mae | selection | 0.339 | 0.370 | 0.622 | no | yes | 0.333 |
| tair_control_horizon_abs_bias | selection | -0.022 | 0.236 | 0.622 | yes | yes | 0.333 |
| co2_full_horizon_mae | selection | 0.333 | 0.297 | 0.600 | no | yes | 0.667 |
| co2_weighted_horizon_mae | selection | 0.294 | 0.248 | 0.578 | no | yes | 0.667 |
| multiobjective_transfer_selection_score | selection | 0.048 | 0.115 | 0.533 | no | yes | 0.667 |
| co2_final_step_mae | selection | 0.022 | 0.042 | 0.533 | no | no | 0.333 |
| co2_first_step_mae | selection | 0.132 | 0.036 | 0.523 | no | no | 0.333 |
| co2_control_horizon_mae | selection | 0.143 | 0.000 | 0.523 | no | no | 0.333 |
| co2_constraint_near_mae_proxy | selection | 0.178 | 0.152 | 0.511 | no | yes | 0.667 |
| tair_transfer_selection_score | selection | -0.320 | -0.067 | 0.489 | no | no | 0.333 |
| co2_transfer_selection_score | selection | 0.051 | -0.030 | 0.489 | no | no | 0.333 |
| rhair_weighted_horizon_mae | selection | -0.004 | -0.024 | 0.477 | no | yes | 0.333 |
| tair_constraint_near_mae_proxy | selection | -0.194 | -0.122 | 0.455 | no | no | 0.000 |
| rhair_full_horizon_mae | selection | -0.051 | -0.115 | 0.444 | no | yes | 0.333 |
| tair_control_horizon_mae | selection | -0.069 | -0.182 | 0.432 | no | no | 0.333 |
| tair_first_step_mae | selection | -0.031 | -0.273 | 0.422 | no | no | 0.333 |
| rhair_final_step_mae | selection | -0.210 | -0.170 | 0.409 | yes | yes | 0.333 |
| forecast_only_transfer_rank | selection | -0.148 | -0.280 | 0.386 | no | yes | 0.333 |
| co2_control_horizon_abs_bias | selection | -0.199 | -0.358 | 0.378 | no | no | 0.333 |
| tair_final_step_mae | selection | -0.262 | -0.285 | 0.378 | no | no | 0.000 |
| tair_weighted_horizon_mae | selection | -0.208 | -0.515 | 0.311 | no | no | 0.000 |
| tair_full_horizon_mae | selection | -0.222 | -0.527 | 0.289 | no | no | 0.000 |
| rhair_constraint_near_mae_proxy | selection | -0.488 | -0.552 | 0.267 | no | no | 0.000 |
| assim_sp_first_grad | diagnostic | 0.831 | 0.912 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | 0.663 | 0.790 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | 0.562 | 0.620 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | 0.520 | 0.608 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | 0.349 | 0.462 |  |  |  |  |
| co2_sp_first_grad | diagnostic | 0.356 | 0.462 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | 0.249 | 0.450 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | 0.487 | 0.413 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.413 | -0.353 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.266 | -0.348 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.266 | 0.348 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | 0.213 | 0.267 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | 0.292 | 0.255 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | 0.031 | -0.097 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | -0.049 | 0.079 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | 0.350 | 0.036 |  |  |  |  |

### Target: `mpc_rhair_mae`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tair_control_horizon_abs_bias | selection | 0.632 | 0.527 | 0.689 | yes | yes | 0.667 |
| multiobjective_transfer_selection_score | selection | 0.556 | 0.455 | 0.689 | no | yes | 0.667 |
| rhair_first_step_mae | selection | 0.767 | 0.418 | 0.667 | no | yes | 0.333 |
| co2_full_horizon_mae | selection | 0.534 | 0.345 | 0.667 | no | yes | 0.333 |
| co2_first_step_mae | selection | 0.755 | 0.426 | 0.659 | no | no | 0.333 |
| rhair_transfer_selection_score | selection | 0.498 | 0.358 | 0.644 | no | yes | 0.333 |
| rhair_control_horizon_mae | selection | 0.731 | 0.333 | 0.644 | no | yes | 0.333 |
| co2_weighted_horizon_mae | selection | 0.575 | 0.321 | 0.644 | no | yes | 0.333 |
| co2_control_horizon_mae | selection | 0.667 | 0.280 | 0.568 | no | no | 0.000 |
| rhair_weighted_horizon_mae | selection | 0.596 | 0.207 | 0.568 | no | yes | 0.333 |
| co2_transfer_selection_score | selection | 0.248 | 0.236 | 0.556 | no | no | 0.000 |
| tair_transfer_selection_score | selection | 0.298 | 0.176 | 0.556 | no | no | 0.333 |
| rhair_full_horizon_mae | selection | 0.566 | 0.127 | 0.556 | no | yes | 0.333 |
| rhair_control_horizon_abs_bias | selection | 0.055 | 0.115 | 0.556 | no | yes | 0.667 |
| tair_constraint_near_mae_proxy | selection | 0.493 | 0.182 | 0.545 | no | no | 0.333 |
| co2_constraint_near_mae_proxy | selection | -0.075 | 0.091 | 0.533 | no | yes | 0.333 |
| tair_control_horizon_mae | selection | 0.653 | 0.097 | 0.523 | no | no | 0.333 |
| co2_final_step_mae | selection | 0.032 | 0.030 | 0.511 | no | no | 0.000 |
| rhair_final_step_mae | selection | 0.431 | 0.097 | 0.500 | yes | yes | 0.333 |
| forecast_only_transfer_rank | selection | 0.442 | 0.024 | 0.500 | no | yes | 0.333 |
| tair_first_step_mae | selection | 0.667 | 0.042 | 0.489 | no | no | 0.333 |
| tair_final_step_mae | selection | 0.461 | -0.042 | 0.489 | no | no | 0.333 |
| rhair_constraint_near_mae_proxy | selection | 0.177 | -0.115 | 0.467 | no | no | 0.000 |
| co2_control_horizon_abs_bias | selection | -0.190 | -0.248 | 0.400 | no | no | 0.000 |
| tair_full_horizon_mae | selection | 0.518 | -0.200 | 0.400 | no | no | 0.333 |
| tair_weighted_horizon_mae | selection | 0.533 | -0.212 | 0.378 | no | no | 0.333 |
| assim_sp_first_grad | diagnostic | 0.550 | 0.620 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | 0.484 | 0.486 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | -0.302 | -0.462 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | 0.781 | 0.450 |  |  |  |  |
| co2_sp_first_grad | diagnostic | 0.590 | 0.426 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | 0.194 | 0.321 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | -0.182 | -0.267 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.233 | -0.243 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | 0.531 | 0.219 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.142 | -0.174 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.142 | 0.174 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | -0.062 | 0.134 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | 0.229 | 0.122 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | -0.054 | 0.109 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | -0.137 | -0.049 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | -0.122 | -0.049 |  |  |  |  |

### Target: `mpc_co2_mae`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| co2_first_step_mae | selection | 0.309 | 0.498 | 0.705 | no | no | 0.667 |
| co2_control_horizon_abs_bias | selection | 0.474 | 0.467 | 0.644 | no | no | 0.667 |
| co2_transfer_selection_score | selection | 0.505 | 0.321 | 0.622 | no | no | 0.667 |
| co2_control_horizon_mae | selection | 0.311 | 0.304 | 0.614 | no | no | 0.667 |
| tair_constraint_near_mae_proxy | selection | 0.089 | 0.255 | 0.591 | no | yes | 0.333 |
| tair_weighted_horizon_mae | selection | 0.033 | 0.042 | 0.578 | no | yes | 0.333 |
| tair_full_horizon_mae | selection | 0.035 | 0.030 | 0.556 | no | yes | 0.333 |
| multiobjective_transfer_selection_score | selection | 0.103 | 0.079 | 0.533 | no | no | 0.333 |
| rhair_weighted_horizon_mae | selection | 0.043 | -0.085 | 0.523 | no | no | 0.333 |
| tair_final_step_mae | selection | 0.041 | -0.030 | 0.511 | no | yes | 0.333 |
| co2_constraint_near_mae_proxy | selection | 0.168 | 0.006 | 0.511 | no | no | 0.333 |
| rhair_full_horizon_mae | selection | 0.013 | -0.152 | 0.489 | no | no | 0.333 |
| rhair_constraint_near_mae_proxy | selection | -0.174 | -0.079 | 0.489 | no | no | 0.667 |
| tair_first_step_mae | selection | 0.013 | -0.164 | 0.467 | no | no | 0.000 |
| forecast_only_transfer_rank | selection | -0.078 | -0.255 | 0.455 | no | no | 0.333 |
| tair_transfer_selection_score | selection | -0.360 | -0.345 | 0.444 | no | no | 0.000 |
| tair_control_horizon_abs_bias | selection | -0.189 | -0.248 | 0.444 | no | yes | 0.333 |
| co2_weighted_horizon_mae | selection | 0.188 | -0.103 | 0.444 | no | no | 0.333 |
| rhair_control_horizon_mae | selection | 0.206 | -0.067 | 0.444 | no | no | 0.000 |
| rhair_control_horizon_abs_bias | selection | 0.476 | -0.018 | 0.444 | no | yes | 0.333 |
| co2_full_horizon_mae | selection | 0.124 | -0.115 | 0.422 | no | no | 0.000 |
| rhair_transfer_selection_score | selection | 0.038 | -0.152 | 0.400 | no | no | 0.000 |
| tair_control_horizon_mae | selection | 0.021 | -0.389 | 0.386 | no | no | 0.000 |
| rhair_first_step_mae | selection | 0.183 | -0.176 | 0.378 | no | no | 0.000 |
| rhair_final_step_mae | selection | -0.156 | -0.450 | 0.364 | no | no | 0.000 |
| co2_final_step_mae | selection | -0.327 | -0.345 | 0.356 | no | no | 0.333 |
| rhair_dx_sp_first_grad | diagnostic | -0.335 | -0.584 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | 0.329 | 0.522 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | -0.329 | -0.522 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | -0.120 | -0.511 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | -0.242 | -0.474 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | -0.357 | -0.413 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | 0.252 | 0.316 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | -0.214 | -0.292 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | -0.289 | -0.280 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | 0.194 | 0.255 |  |  |  |  |
| co2_sp_first_grad | diagnostic | -0.293 | -0.243 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | 0.162 | 0.207 |  |  |  |  |
| assim_sp_first_grad | diagnostic | -0.065 | -0.182 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | -0.197 | -0.170 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | -0.006 | -0.006 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | -0.065 | 0.000 |  |  |  |  |

### Target: `mpc_objective`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rhair_first_step_mae | selection | 0.872 | 0.648 | 0.756 | no | yes | 0.667 |
| rhair_transfer_selection_score | selection | 0.702 | 0.576 | 0.733 | no | yes | 0.667 |
| rhair_control_horizon_mae | selection | 0.854 | 0.491 | 0.689 | no | yes | 0.333 |
| tair_control_horizon_abs_bias | selection | 0.478 | 0.394 | 0.689 | yes | yes | 0.667 |
| co2_full_horizon_mae | selection | 0.558 | 0.479 | 0.667 | no | yes | 0.667 |
| co2_weighted_horizon_mae | selection | 0.599 | 0.455 | 0.644 | no | yes | 0.667 |
| rhair_control_horizon_abs_bias | selection | 0.222 | 0.273 | 0.644 | no | yes | 0.667 |
| multiobjective_transfer_selection_score | selection | 0.625 | 0.285 | 0.600 | no | yes | 0.333 |
| co2_final_step_mae | selection | -0.017 | 0.273 | 0.600 | no | no | 0.333 |
| co2_first_step_mae | selection | 0.752 | 0.195 | 0.568 | no | no | 0.000 |
| co2_control_horizon_mae | selection | 0.683 | 0.182 | 0.568 | no | no | 0.333 |
| co2_transfer_selection_score | selection | 0.339 | 0.091 | 0.556 | no | no | 0.333 |
| tair_constraint_near_mae_proxy | selection | 0.409 | 0.073 | 0.545 | no | no | 0.333 |
| co2_constraint_near_mae_proxy | selection | 0.006 | 0.152 | 0.533 | no | yes | 0.667 |
| rhair_weighted_horizon_mae | selection | 0.636 | 0.146 | 0.523 | no | yes | 0.333 |
| rhair_full_horizon_mae | selection | 0.592 | 0.091 | 0.511 | no | yes | 0.333 |
| tair_transfer_selection_score | selection | 0.171 | 0.079 | 0.511 | no | no | 0.000 |
| tair_final_step_mae | selection | 0.397 | -0.067 | 0.489 | no | no | 0.333 |
| tair_control_horizon_mae | selection | 0.593 | -0.024 | 0.477 | no | no | 0.000 |
| rhair_final_step_mae | selection | 0.379 | 0.036 | 0.455 | yes | yes | 0.333 |
| forecast_only_transfer_rank | selection | 0.477 | -0.012 | 0.455 | no | yes | 0.333 |
| co2_control_horizon_abs_bias | selection | -0.107 | -0.285 | 0.444 | no | no | 0.333 |
| tair_first_step_mae | selection | 0.624 | -0.127 | 0.444 | no | no | 0.000 |
| tair_full_horizon_mae | selection | 0.446 | -0.236 | 0.400 | no | no | 0.333 |
| tair_weighted_horizon_mae | selection | 0.462 | -0.248 | 0.378 | no | no | 0.333 |
| rhair_constraint_near_mae_proxy | selection | 0.143 | -0.224 | 0.378 | no | no | 0.000 |
| assim_sp_first_grad | diagnostic | 0.534 | 0.815 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | 0.494 | 0.729 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | 0.711 | 0.523 |  |  |  |  |
| co2_sp_first_grad | diagnostic | 0.458 | 0.498 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | 0.067 | 0.474 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | 0.070 | 0.474 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | 0.521 | 0.426 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.407 | -0.426 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | 0.352 | 0.365 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.280 | -0.348 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.280 | 0.348 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | 0.013 | 0.316 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | -0.116 | 0.292 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | -0.176 | -0.195 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | 0.302 | 0.188 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | -0.173 | -0.122 |  |  |  |  |

## Robustness Summary

### Target: `mpc_tair_mae`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | offline_or_diagnostic_only | -0.273 | -0.533 .. -0.117 | -0.548 .. -0.117 | 0.333 |
| tair_control_horizon_mae | offline_or_diagnostic_only | -0.182 | -0.360 .. 0.126 | -0.360 .. 0.126 | 0.371 |
| tair_weighted_horizon_mae | offline_or_diagnostic_only | -0.515 | -0.750 .. -0.333 | -0.750 .. -0.333 | 0.222 |
| tair_full_horizon_mae | offline_or_diagnostic_only | -0.527 | -0.767 .. -0.350 | -0.767 .. -0.350 | 0.194 |
| tair_final_step_mae | offline_or_diagnostic_only | -0.285 | -0.517 .. -0.083 | -0.517 .. -0.083 | 0.306 |
| tair_control_horizon_abs_bias | offline_or_diagnostic_only | 0.236 | -0.050 .. 0.550 | -0.050 .. 0.550 | 0.528 |
| tair_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.122 | -0.326 .. 0.092 | -0.326 .. 0.092 | 0.400 |
| rhair_first_step_mae | secondary_selection | 0.576 | 0.467 .. 0.700 | 0.467 .. 0.700 | 0.639 |
| rhair_control_horizon_mae | secondary_selection | 0.370 | 0.183 .. 0.467 | 0.183 .. 0.619 | 0.556 |
| rhair_weighted_horizon_mae | offline_or_diagnostic_only | -0.024 | -0.326 .. 0.126 | -0.326 .. 0.156 | 0.371 |
| rhair_full_horizon_mae | offline_or_diagnostic_only | -0.115 | -0.450 .. 0.000 | -0.450 .. 0.143 | 0.333 |
| rhair_final_step_mae | offline_or_diagnostic_only | -0.170 | -0.611 .. -0.008 | -0.611 .. -0.008 | 0.257 |
| rhair_control_horizon_abs_bias | secondary_selection | 0.479 | 0.283 .. 0.667 | 0.283 .. 0.667 | 0.583 |
| rhair_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.552 | -0.733 .. -0.383 | -0.733 .. -0.383 | 0.167 |
| co2_first_step_mae | offline_or_diagnostic_only | 0.036 | -0.276 .. 0.283 | -0.276 .. 0.283 | 0.429 |
| co2_control_horizon_mae | offline_or_diagnostic_only | 0.000 | -0.159 .. 0.267 | -0.159 .. 0.267 | 0.486 |
| co2_weighted_horizon_mae | offline_or_diagnostic_only | 0.248 | -0.017 .. 0.383 | -0.017 .. 0.500 | 0.500 |
| co2_full_horizon_mae | weak_selection | 0.297 | 0.050 .. 0.417 | 0.050 .. 0.595 | 0.528 |
| co2_final_step_mae | offline_or_diagnostic_only | 0.042 | -0.267 .. 0.283 | -0.267 .. 0.283 | 0.444 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | -0.358 | -0.533 .. -0.200 | -0.533 .. -0.200 | 0.306 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.152 | -0.133 .. 0.350 | -0.133 .. 0.350 | 0.417 |
| forecast_only_transfer_rank | offline_or_diagnostic_only | -0.280 | -0.583 .. 0.000 | -0.583 .. 0.000 | 0.278 |
| tair_transfer_selection_score | offline_or_diagnostic_only | -0.067 | -0.233 .. 0.217 | -0.200 .. 0.217 | 0.417 |
| rhair_transfer_selection_score | secondary_selection | 0.455 | 0.267 .. 0.633 | 0.267 .. 0.647 | 0.583 |
| co2_transfer_selection_score | offline_or_diagnostic_only | -0.030 | -0.233 .. 0.233 | -0.233 .. 0.233 | 0.417 |
| multiobjective_transfer_selection_score | offline_or_diagnostic_only | 0.115 | -0.167 .. 0.333 | -0.167 .. 0.333 | 0.444 |

### Target: `mpc_rhair_mae`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | offline_or_diagnostic_only | 0.042 | -0.317 .. 0.167 | -0.317 .. 0.167 | 0.361 |
| tair_control_horizon_mae | offline_or_diagnostic_only | 0.097 | -0.243 .. 0.360 | -0.243 .. 0.360 | 0.400 |
| tair_weighted_horizon_mae | offline_or_diagnostic_only | -0.212 | -0.667 .. -0.067 | -0.667 .. -0.067 | 0.222 |
| tair_full_horizon_mae | offline_or_diagnostic_only | -0.200 | -0.650 .. -0.050 | -0.650 .. -0.050 | 0.250 |
| tair_final_step_mae | offline_or_diagnostic_only | -0.042 | -0.433 .. 0.150 | -0.433 .. 0.150 | 0.361 |
| tair_control_horizon_abs_bias | secondary_selection | 0.527 | 0.350 .. 0.817 | 0.350 .. 0.817 | 0.611 |
| tair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.182 | -0.126 .. 0.410 | -0.126 .. 0.410 | 0.429 |
| rhair_first_step_mae | secondary_selection | 0.418 | 0.200 .. 0.667 | 0.200 .. 0.667 | 0.583 |
| rhair_control_horizon_mae | weak_selection | 0.333 | 0.083 .. 0.550 | 0.083 .. 0.619 | 0.556 |
| rhair_weighted_horizon_mae | offline_or_diagnostic_only | 0.207 | -0.092 .. 0.410 | -0.092 .. 0.410 | 0.457 |
| rhair_full_horizon_mae | offline_or_diagnostic_only | 0.127 | -0.200 .. 0.317 | -0.200 .. 0.357 | 0.444 |
| rhair_final_step_mae | offline_or_diagnostic_only | 0.097 | -0.243 .. 0.226 | -0.243 .. 0.226 | 0.371 |
| rhair_control_horizon_abs_bias | offline_or_diagnostic_only | 0.115 | -0.067 .. 0.533 | -0.067 .. 0.533 | 0.472 |
| rhair_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.115 | -0.533 .. 0.067 | -0.533 .. 0.067 | 0.333 |
| co2_first_step_mae | secondary_selection | 0.426 | 0.209 .. 0.567 | 0.209 .. 0.611 | 0.571 |
| co2_control_horizon_mae | weak_selection | 0.280 | 0.008 .. 0.460 | 0.008 .. 0.539 | 0.457 |
| co2_weighted_horizon_mae | weak_selection | 0.321 | 0.067 .. 0.533 | 0.067 .. 0.857 | 0.556 |
| co2_full_horizon_mae | weak_selection | 0.345 | 0.100 .. 0.567 | 0.100 .. 0.905 | 0.583 |
| co2_final_step_mae | offline_or_diagnostic_only | 0.030 | -0.150 .. 0.233 | -0.150 .. 0.381 | 0.444 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | -0.248 | -0.550 .. -0.133 | -0.550 .. -0.095 | 0.306 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.091 | -0.100 .. 0.333 | -0.100 .. 0.500 | 0.472 |
| forecast_only_transfer_rank | offline_or_diagnostic_only | 0.024 | -0.400 .. 0.200 | -0.400 .. 0.286 | 0.361 |
| tair_transfer_selection_score | offline_or_diagnostic_only | 0.176 | -0.133 .. 0.400 | -0.133 .. 0.400 | 0.444 |
| rhair_transfer_selection_score | secondary_selection | 0.358 | 0.117 .. 0.667 | 0.117 .. 0.667 | 0.556 |
| co2_transfer_selection_score | offline_or_diagnostic_only | 0.236 | 0.017 .. 0.433 | 0.017 .. 0.500 | 0.472 |
| multiobjective_transfer_selection_score | secondary_selection | 0.455 | 0.150 .. 0.617 | 0.150 .. 0.550 | 0.556 |

### Target: `mpc_co2_mae`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | offline_or_diagnostic_only | -0.164 | -0.283 .. 0.067 | -0.283 .. 0.067 | 0.417 |
| tair_control_horizon_mae | offline_or_diagnostic_only | -0.389 | -0.544 .. -0.159 | -0.544 .. -0.159 | 0.314 |
| tair_weighted_horizon_mae | offline_or_diagnostic_only | 0.042 | -0.083 .. 0.433 | -0.143 .. 0.433 | 0.528 |
| tair_full_horizon_mae | offline_or_diagnostic_only | 0.030 | -0.100 .. 0.417 | -0.167 .. 0.417 | 0.500 |
| tair_final_step_mae | offline_or_diagnostic_only | -0.030 | -0.200 .. 0.300 | -0.200 .. 0.300 | 0.444 |
| tair_control_horizon_abs_bias | offline_or_diagnostic_only | -0.248 | -0.433 .. 0.017 | -0.500 .. 0.017 | 0.361 |
| tair_constraint_near_mae_proxy | weak_selection | 0.255 | 0.025 .. 0.444 | 0.012 .. 0.444 | 0.514 |
| rhair_first_step_mae | offline_or_diagnostic_only | -0.176 | -0.567 .. -0.033 | -0.567 .. -0.033 | 0.250 |
| rhair_control_horizon_mae | offline_or_diagnostic_only | -0.067 | -0.417 .. 0.117 | -0.417 .. 0.117 | 0.333 |
| rhair_weighted_horizon_mae | offline_or_diagnostic_only | -0.085 | -0.226 .. 0.142 | -0.226 .. 0.142 | 0.486 |
| rhair_full_horizon_mae | offline_or_diagnostic_only | -0.152 | -0.250 .. 0.067 | -0.250 .. 0.067 | 0.444 |
| rhair_final_step_mae | offline_or_diagnostic_only | -0.450 | -0.594 .. -0.243 | -0.594 .. -0.243 | 0.286 |
| rhair_control_horizon_abs_bias | offline_or_diagnostic_only | -0.018 | -0.400 .. 0.117 | -0.400 .. 0.117 | 0.306 |
| rhair_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.079 | -0.250 .. 0.267 | -0.250 .. 0.267 | 0.444 |
| co2_first_step_mae | secondary_selection | 0.498 | 0.360 .. 0.812 | 0.360 .. 0.755 | 0.657 |
| co2_control_horizon_mae | weak_selection | 0.304 | 0.183 .. 0.695 | 0.183 .. 0.659 | 0.556 |
| co2_weighted_horizon_mae | offline_or_diagnostic_only | -0.103 | -0.267 .. 0.133 | -0.267 .. 0.214 | 0.389 |
| co2_full_horizon_mae | offline_or_diagnostic_only | -0.115 | -0.283 .. 0.117 | -0.283 .. 0.190 | 0.361 |
| co2_final_step_mae | offline_or_diagnostic_only | -0.345 | -0.550 .. -0.150 | -0.550 .. -0.150 | 0.278 |
| co2_control_horizon_abs_bias | secondary_selection | 0.467 | 0.350 .. 0.583 | 0.350 .. 0.595 | 0.611 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.006 | -0.183 .. 0.383 | -0.183 .. 0.429 | 0.444 |
| forecast_only_transfer_rank | offline_or_diagnostic_only | -0.255 | -0.417 .. -0.017 | -0.417 .. -0.017 | 0.389 |
| tair_transfer_selection_score | offline_or_diagnostic_only | -0.345 | -0.483 .. -0.117 | -0.524 .. -0.117 | 0.389 |
| rhair_transfer_selection_score | offline_or_diagnostic_only | -0.152 | -0.533 .. -0.033 | -0.533 .. -0.033 | 0.278 |
| co2_transfer_selection_score | weak_selection | 0.321 | 0.183 .. 0.700 | 0.183 .. 0.667 | 0.583 |
| multiobjective_transfer_selection_score | offline_or_diagnostic_only | 0.079 | -0.167 .. 0.450 | -0.167 .. 0.267 | 0.444 |

### Target: `mpc_objective`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | offline_or_diagnostic_only | -0.127 | -0.550 .. 0.000 | -0.550 .. 0.000 | 0.306 |
| tair_control_horizon_mae | offline_or_diagnostic_only | -0.024 | -0.410 .. 0.192 | -0.410 .. 0.192 | 0.343 |
| tair_weighted_horizon_mae | offline_or_diagnostic_only | -0.248 | -0.717 .. -0.117 | -0.717 .. -0.117 | 0.222 |
| tair_full_horizon_mae | offline_or_diagnostic_only | -0.236 | -0.700 .. -0.100 | -0.700 .. -0.100 | 0.250 |
| tair_final_step_mae | offline_or_diagnostic_only | -0.067 | -0.467 .. 0.117 | -0.467 .. 0.117 | 0.361 |
| tair_control_horizon_abs_bias | objective_secondary_selection | 0.394 | 0.167 .. 0.633 | 0.167 .. 0.633 | 0.611 |
| tair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.073 | -0.276 .. 0.310 | -0.276 .. 0.310 | 0.429 |
| rhair_first_step_mae | objective_secondary_selection | 0.648 | 0.517 .. 0.800 | 0.517 .. 0.833 | 0.694 |
| rhair_control_horizon_mae | objective_secondary_selection | 0.491 | 0.300 .. 0.617 | 0.300 .. 0.786 | 0.611 |
| rhair_weighted_horizon_mae | offline_or_diagnostic_only | 0.146 | -0.176 .. 0.310 | -0.176 .. 0.419 | 0.400 |
| rhair_full_horizon_mae | offline_or_diagnostic_only | 0.091 | -0.250 .. 0.233 | -0.250 .. 0.429 | 0.389 |
| rhair_final_step_mae | offline_or_diagnostic_only | 0.036 | -0.326 .. 0.142 | -0.326 .. 0.252 | 0.314 |
| rhair_control_horizon_abs_bias | weak_selection | 0.273 | 0.150 .. 0.750 | 0.143 .. 0.750 | 0.583 |
| rhair_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.224 | -0.683 .. -0.083 | -0.683 .. -0.083 | 0.222 |
| co2_first_step_mae | offline_or_diagnostic_only | 0.195 | -0.109 .. 0.350 | -0.109 .. 0.350 | 0.457 |
| co2_control_horizon_mae | offline_or_diagnostic_only | 0.182 | -0.126 .. 0.367 | -0.126 .. 0.367 | 0.457 |
| co2_weighted_horizon_mae | objective_secondary_selection | 0.455 | 0.250 .. 0.600 | 0.250 .. 0.786 | 0.556 |
| co2_full_horizon_mae | objective_secondary_selection | 0.479 | 0.283 .. 0.633 | 0.283 .. 0.833 | 0.583 |
| co2_final_step_mae | weak_selection | 0.273 | 0.050 .. 0.483 | 0.050 .. 0.483 | 0.528 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | -0.285 | -0.467 .. -0.167 | -0.467 .. -0.167 | 0.389 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.152 | -0.133 .. 0.317 | -0.133 .. 0.381 | 0.444 |
| forecast_only_transfer_rank | offline_or_diagnostic_only | -0.012 | -0.433 .. 0.150 | -0.433 .. 0.262 | 0.306 |
| tair_transfer_selection_score | offline_or_diagnostic_only | 0.079 | -0.267 .. 0.250 | -0.267 .. 0.250 | 0.389 |
| rhair_transfer_selection_score | objective_secondary_selection | 0.576 | 0.417 .. 0.783 | 0.417 .. 0.838 | 0.667 |
| co2_transfer_selection_score | offline_or_diagnostic_only | 0.091 | -0.117 .. 0.283 | -0.117 .. 0.283 | 0.472 |
| multiobjective_transfer_selection_score | weak_selection | 0.285 | 0.017 .. 0.433 | 0.017 .. 0.400 | 0.500 |
