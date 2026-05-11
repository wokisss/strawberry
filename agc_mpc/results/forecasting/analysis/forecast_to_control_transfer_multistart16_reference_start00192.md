# Forecast-To-Control Transfer Analysis

Model count: `16`.

This report tests whether forecast-side validation metrics predict `GradientMPC` closed-loop outcomes.
For selection metrics, lower values are treated as better. Gradient metrics are diagnostic only.

## Metric Roles

| control_target | metric | role |
| --- | --- | --- |
| mpc_tair_mae | tair_first_step_mae | weak_selection |
| mpc_tair_mae | tair_control_horizon_mae | offline_or_diagnostic_only |
| mpc_tair_mae | tair_weighted_horizon_mae | secondary_selection |
| mpc_tair_mae | tair_full_horizon_mae | secondary_selection |
| mpc_tair_mae | tair_final_step_mae | offline_or_diagnostic_only |
| mpc_tair_mae | tair_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_tair_mae | tair_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_tair_mae | rhair_first_step_mae | offline_or_diagnostic_only |
| mpc_tair_mae | rhair_control_horizon_mae | offline_or_diagnostic_only |
| mpc_tair_mae | rhair_weighted_horizon_mae | weak_selection |
| mpc_tair_mae | rhair_full_horizon_mae | secondary_selection |
| mpc_tair_mae | rhair_final_step_mae | offline_or_diagnostic_only |
| mpc_tair_mae | rhair_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_tair_mae | rhair_constraint_near_mae_proxy | weak_selection |
| mpc_tair_mae | co2_first_step_mae | offline_or_diagnostic_only |
| mpc_tair_mae | co2_control_horizon_mae | offline_or_diagnostic_only |
| mpc_tair_mae | co2_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_tair_mae | co2_full_horizon_mae | offline_or_diagnostic_only |
| mpc_tair_mae | co2_final_step_mae | offline_or_diagnostic_only |
| mpc_tair_mae | co2_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_tair_mae | co2_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_tair_mae | forecast_only_transfer_rank | weak_selection |
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
| mpc_rhair_mae | tair_first_step_mae | weak_selection |
| mpc_rhair_mae | tair_control_horizon_mae | secondary_selection |
| mpc_rhair_mae | tair_weighted_horizon_mae | secondary_selection |
| mpc_rhair_mae | tair_full_horizon_mae | secondary_selection |
| mpc_rhair_mae | tair_final_step_mae | primary_selection |
| mpc_rhair_mae | tair_control_horizon_abs_bias | secondary_selection |
| mpc_rhair_mae | tair_constraint_near_mae_proxy | weak_selection |
| mpc_rhair_mae | rhair_first_step_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_control_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_weighted_horizon_mae | weak_selection |
| mpc_rhair_mae | rhair_full_horizon_mae | weak_selection |
| mpc_rhair_mae | rhair_final_step_mae | secondary_selection |
| mpc_rhair_mae | rhair_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_first_step_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_control_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_full_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_final_step_mae | weak_selection |
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
| mpc_co2_mae | rhair_full_horizon_mae | weak_selection |
| mpc_co2_mae | rhair_final_step_mae | offline_or_diagnostic_only |
| mpc_co2_mae | rhair_control_horizon_abs_bias | secondary_selection |
| mpc_co2_mae | rhair_constraint_near_mae_proxy | weak_selection |
| mpc_co2_mae | co2_first_step_mae | offline_or_diagnostic_only |
| mpc_co2_mae | co2_control_horizon_mae | offline_or_diagnostic_only |
| mpc_co2_mae | co2_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_co2_mae | co2_full_horizon_mae | offline_or_diagnostic_only |
| mpc_co2_mae | co2_final_step_mae | offline_or_diagnostic_only |
| mpc_co2_mae | co2_control_horizon_abs_bias | offline_or_diagnostic_only |
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
| mpc_objective | tair_first_step_mae | offline_or_diagnostic_only |
| mpc_objective | tair_control_horizon_mae | offline_or_diagnostic_only |
| mpc_objective | tair_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_objective | tair_full_horizon_mae | weak_selection |
| mpc_objective | tair_final_step_mae | offline_or_diagnostic_only |
| mpc_objective | tair_control_horizon_abs_bias | objective_secondary_selection |
| mpc_objective | tair_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_objective | rhair_first_step_mae | weak_selection |
| mpc_objective | rhair_control_horizon_mae | weak_selection |
| mpc_objective | rhair_weighted_horizon_mae | objective_secondary_selection |
| mpc_objective | rhair_full_horizon_mae | objective_secondary_selection |
| mpc_objective | rhair_final_step_mae | weak_selection |
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
| mpc_objective | tair_transfer_selection_score | weak_selection |
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
| 1 | current_hybrid_transformer | 5.521 | 6.750 | 4.438 | 5.375 | 6.722 | 0.539 | 0.962 | 49.138 | 0.0429 |
| 2 | itransformer_residual | 5.646 | 4.625 | 5.688 | 6.625 | 9.167 | 0.189 | 1.317 | 11.644 | 0.0360 |
| 3 | itransformer_co2_control_aware_fusion | 6.135 | 8.531 | 8.375 | 1.500 | 8.556 | 0.126 | 2.372 | 20.161 | 0.0606 |
| 4 | itransformer_co2_late_frozen_expert | 6.469 | 7.969 | 9.375 | 2.062 | 9.944 | 0.120 | 2.397 | 20.483 | 0.0616 |
| 5 | itransformer_co2_protected_expert | 6.479 | 4.312 | 9.812 | 5.312 | 9.278 | 0.511 | 1.381 | 51.831 | 0.0814 |
| 6 | transformer_hybrid_residual | 6.667 | 5.750 | 3.062 | 11.188 | 9.167 | 0.129 | 0.558 | 20.637 | 0.0235 |
| 7 | itransformer_co2_late_residual | 7.312 | 9.375 | 6.562 | 6.000 | 9.000 | 0.243 | 1.269 | 47.742 | 0.1157 |
| 8 | segrnn_forecaster | 8.104 | 14.188 | 6.375 | 3.750 | 9.389 | 0.673 | 5.179 | 111.292 | 0.1164 |
| 9 | itransformer_co2_horizon_mixture | 8.729 | 3.250 | 12.375 | 10.562 | 13.722 | 0.229 | 0.556 | 26.270 | 0.0678 |
| 10 | dlinear_forecaster | 9.042 | 11.500 | 5.688 | 9.938 | 15.056 | 0.261 | 2.010 | 11.316 | 0.0449 |
| 11 | itransformer_co2_residual | 9.708 | 7.688 | 10.125 | 11.312 | 10.778 | 0.353 | 1.595 | 10.701 | 0.0465 |
| 12 | transformer_forecaster | 9.729 | 7.562 | 11.562 | 10.062 | 13.056 | 0.073 | 1.436 | 31.788 | 0.0389 |
| 13 | itransformer_co2_wavelet_residual | 10.625 | 4.938 | 10.938 | 16.000 | 13.611 | 0.410 | 2.294 | 12.502 | 0.0553 |
| 14 | nlinear_forecaster | 10.708 | 13.312 | 5.188 | 13.625 | 15.500 | 0.510 | 2.077 | 23.846 | 0.0452 |
| 15 | patchtst_residual | 11.646 | 10.250 | 13.062 | 11.625 | 14.833 | 0.296 | 1.211 | 57.069 | 0.0612 |
| 16 | frequency_forecaster | 13.479 | 16.000 | 13.375 | 11.062 | 18.722 | 0.343 | 2.140 | 12.041 | 0.0750 |

## Metric Transfer Quality

### Target: `mpc_tair_mae`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rhair_full_horizon_mae | selection | 0.242 | 0.353 | 0.625 | no | no | 0.000 |
| tair_full_horizon_mae | selection | 0.392 | 0.409 | 0.617 | no | no | 0.000 |
| tair_weighted_horizon_mae | selection | 0.381 | 0.388 | 0.617 | no | no | 0.000 |
| rhair_constraint_near_mae_proxy | selection | 0.374 | 0.318 | 0.617 | no | no | 0.333 |
| forecast_only_transfer_rank | selection | 0.286 | 0.286 | 0.610 | no | no | 0.000 |
| tair_first_step_mae | selection | 0.225 | 0.332 | 0.608 | no | no | 0.000 |
| rhair_weighted_horizon_mae | selection | 0.218 | 0.322 | 0.605 | no | no | 0.000 |
| co2_first_step_mae | selection | 0.182 | 0.210 | 0.597 | no | no | 0.667 |
| co2_control_horizon_mae | selection | 0.120 | 0.196 | 0.588 | no | no | 0.667 |
| rhair_final_step_mae | selection | 0.272 | 0.243 | 0.580 | no | no | 0.000 |
| tair_transfer_selection_score | selection | 0.316 | 0.191 | 0.575 | no | no | 0.000 |
| multiobjective_transfer_selection_score | selection | 0.135 | 0.168 | 0.575 | no | no | 0.333 |
| tair_final_step_mae | selection | 0.404 | 0.168 | 0.567 | no | no | 0.000 |
| co2_control_horizon_abs_bias | selection | 0.154 | 0.159 | 0.567 | no | no | 0.667 |
| co2_transfer_selection_score | selection | 0.060 | 0.159 | 0.567 | no | no | 0.667 |
| co2_final_step_mae | selection | 0.072 | 0.121 | 0.567 | no | no | 0.333 |
| tair_control_horizon_mae | selection | 0.262 | 0.213 | 0.555 | no | no | 0.000 |
| co2_full_horizon_mae | selection | 0.062 | 0.135 | 0.550 | no | no | 0.000 |
| tair_constraint_near_mae_proxy | selection | 0.388 | 0.152 | 0.546 | no | no | 0.000 |
| co2_weighted_horizon_mae | selection | 0.073 | 0.082 | 0.542 | no | no | 0.333 |
| tair_control_horizon_abs_bias | selection | 0.282 | 0.062 | 0.508 | no | no | 0.000 |
| co2_constraint_near_mae_proxy | selection | -0.021 | -0.068 | 0.475 | no | no | 0.333 |
| rhair_control_horizon_mae | selection | 0.002 | -0.029 | 0.475 | no | no | 0.000 |
| rhair_first_step_mae | selection | -0.039 | -0.097 | 0.467 | no | no | 0.000 |
| rhair_transfer_selection_score | selection | -0.148 | -0.119 | 0.445 | no | no | 0.000 |
| rhair_control_horizon_abs_bias | selection | -0.464 | -0.506 | 0.325 | no | no | 0.000 |
| assim_sp_first_grad | diagnostic | -0.658 | -0.552 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | -0.582 | -0.519 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | -0.475 | -0.449 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | 0.391 | 0.441 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | -0.295 | -0.352 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | -0.359 | -0.246 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | -0.381 | -0.231 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | -0.389 | -0.199 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | -0.307 | -0.199 |  |  |  |  |
| co2_sp_first_grad | diagnostic | -0.238 | -0.172 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.182 | -0.169 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.182 | 0.157 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | -0.186 | -0.113 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | -0.322 | -0.104 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.054 | -0.090 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | -0.106 | 0.040 |  |  |  |  |

### Target: `mpc_rhair_mae`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tair_final_step_mae | selection | 0.573 | 0.615 | 0.750 | no | yes | 0.667 |
| tair_transfer_selection_score | selection | 0.588 | 0.526 | 0.708 | yes | yes | 0.333 |
| tair_control_horizon_abs_bias | selection | 0.563 | 0.597 | 0.692 | no | yes | 0.667 |
| tair_control_horizon_mae | selection | 0.331 | 0.446 | 0.681 | yes | yes | 0.333 |
| tair_full_horizon_mae | selection | 0.529 | 0.429 | 0.667 | yes | yes | 0.667 |
| rhair_final_step_mae | selection | 0.474 | 0.458 | 0.655 | no | yes | 1.000 |
| tair_weighted_horizon_mae | selection | 0.513 | 0.394 | 0.650 | yes | yes | 0.333 |
| forecast_only_transfer_rank | selection | 0.296 | 0.395 | 0.644 | no | no | 0.667 |
| co2_final_step_mae | selection | 0.026 | 0.256 | 0.633 | no | yes | 0.333 |
| tair_first_step_mae | selection | 0.257 | 0.347 | 0.625 | no | yes | 0.667 |
| co2_full_horizon_mae | selection | -0.053 | 0.168 | 0.617 | no | no | 0.333 |
| rhair_full_horizon_mae | selection | 0.338 | 0.297 | 0.608 | no | no | 0.667 |
| co2_weighted_horizon_mae | selection | -0.053 | 0.124 | 0.608 | no | no | 0.333 |
| rhair_weighted_horizon_mae | selection | 0.300 | 0.278 | 0.597 | no | no | 0.667 |
| tair_constraint_near_mae_proxy | selection | 0.608 | 0.281 | 0.588 | no | no | 0.000 |
| rhair_constraint_near_mae_proxy | selection | 0.302 | 0.200 | 0.583 | yes | yes | 0.333 |
| rhair_first_step_mae | selection | -0.043 | 0.174 | 0.583 | no | no | 0.667 |
| multiobjective_transfer_selection_score | selection | 0.114 | 0.147 | 0.558 | no | no | 0.333 |
| rhair_transfer_selection_score | selection | -0.044 | 0.088 | 0.529 | no | no | 0.667 |
| rhair_control_horizon_mae | selection | -0.001 | 0.062 | 0.525 | no | no | 0.333 |
| co2_control_horizon_mae | selection | -0.031 | -0.166 | 0.496 | no | no | 0.000 |
| co2_first_step_mae | selection | 0.094 | -0.131 | 0.471 | no | no | 0.000 |
| co2_transfer_selection_score | selection | -0.297 | -0.250 | 0.450 | no | no | 0.000 |
| co2_constraint_near_mae_proxy | selection | -0.231 | -0.218 | 0.442 | no | no | 0.333 |
| rhair_control_horizon_abs_bias | selection | -0.202 | -0.256 | 0.408 | no | no | 0.000 |
| co2_control_horizon_abs_bias | selection | -0.161 | -0.385 | 0.383 | no | no | 0.000 |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.416 | 0.532 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.416 | -0.526 |  |  |  |  |
| co2_sp_first_grad | diagnostic | -0.013 | 0.384 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | -0.252 | 0.328 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | 0.492 | 0.312 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | -0.436 | -0.246 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | 0.008 | 0.210 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | 0.018 | -0.157 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | -0.377 | -0.072 |  |  |  |  |
| assim_sp_first_grad | diagnostic | -0.327 | 0.069 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | -0.451 | -0.066 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | -0.311 | 0.057 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | -0.299 | -0.037 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | -0.283 | 0.031 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | -0.241 | -0.031 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | -0.421 | 0.004 |  |  |  |  |

### Target: `mpc_co2_mae`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rhair_control_horizon_abs_bias | selection | 0.184 | 0.479 | 0.667 | no | yes | 0.333 |
| rhair_constraint_near_mae_proxy | selection | 0.299 | 0.344 | 0.625 | no | no | 0.333 |
| rhair_full_horizon_mae | selection | 0.121 | 0.259 | 0.600 | no | no | 0.333 |
| tair_weighted_horizon_mae | selection | 0.227 | 0.288 | 0.592 | no | no | 0.333 |
| tair_full_horizon_mae | selection | 0.249 | 0.282 | 0.592 | no | no | 0.333 |
| rhair_weighted_horizon_mae | selection | 0.087 | 0.210 | 0.571 | no | no | 0.333 |
| tair_constraint_near_mae_proxy | selection | 0.356 | 0.102 | 0.529 | no | yes | 0.333 |
| co2_control_horizon_abs_bias | selection | -0.221 | 0.150 | 0.525 | no | no | 0.000 |
| tair_final_step_mae | selection | 0.309 | 0.050 | 0.525 | no | no | 0.000 |
| tair_control_horizon_mae | selection | 0.006 | 0.043 | 0.521 | no | no | 0.333 |
| forecast_only_transfer_rank | selection | -0.001 | 0.047 | 0.517 | no | no | 0.333 |
| tair_transfer_selection_score | selection | 0.255 | 0.015 | 0.517 | no | no | 0.333 |
| rhair_final_step_mae | selection | 0.196 | 0.093 | 0.513 | no | no | 0.000 |
| tair_control_horizon_abs_bias | selection | 0.106 | 0.000 | 0.500 | no | no | 0.333 |
| rhair_transfer_selection_score | selection | -0.104 | 0.004 | 0.496 | no | no | 0.000 |
| rhair_control_horizon_mae | selection | -0.224 | -0.091 | 0.483 | no | no | 0.000 |
| multiobjective_transfer_selection_score | selection | -0.139 | -0.082 | 0.483 | no | no | 0.333 |
| tair_first_step_mae | selection | -0.074 | -0.024 | 0.467 | no | no | 0.000 |
| co2_final_step_mae | selection | -0.265 | -0.229 | 0.425 | no | no | 0.000 |
| co2_first_step_mae | selection | -0.303 | -0.243 | 0.412 | no | no | 0.000 |
| co2_transfer_selection_score | selection | -0.379 | -0.238 | 0.392 | no | no | 0.000 |
| rhair_first_step_mae | selection | -0.342 | -0.347 | 0.375 | no | no | 0.000 |
| co2_full_horizon_mae | selection | -0.423 | -0.429 | 0.358 | no | no | 0.000 |
| co2_control_horizon_mae | selection | -0.382 | -0.358 | 0.353 | no | no | 0.000 |
| co2_weighted_horizon_mae | selection | -0.418 | -0.485 | 0.333 | no | no | 0.000 |
| co2_constraint_near_mae_proxy | selection | -0.463 | -0.415 | 0.333 | no | no | 0.000 |
| co2_sp_first_grad | diagnostic | -0.758 | -0.787 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | -0.829 | -0.693 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | 0.406 | 0.673 |  |  |  |  |
| assim_sp_first_grad | diagnostic | -0.672 | -0.617 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | -0.164 | -0.407 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | 0.163 | 0.379 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | -0.329 | -0.287 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | -0.355 | -0.228 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | -0.537 | -0.216 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | -0.373 | -0.208 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | -0.353 | -0.184 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | -0.318 | -0.160 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | -0.260 | -0.063 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | -0.290 | -0.046 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | 0.103 | 0.044 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | -0.219 | 0.007 |  |  |  |  |

### Target: `mpc_objective`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rhair_weighted_horizon_mae | selection | 0.322 | 0.450 | 0.655 | yes | yes | 0.333 |
| rhair_transfer_selection_score | selection | 0.186 | 0.449 | 0.655 | yes | yes | 0.333 |
| rhair_full_horizon_mae | selection | 0.335 | 0.450 | 0.650 | yes | yes | 0.333 |
| tair_control_horizon_abs_bias | selection | 0.334 | 0.362 | 0.633 | no | no | 0.333 |
| rhair_control_horizon_mae | selection | 0.164 | 0.329 | 0.633 | yes | yes | 0.333 |
| rhair_first_step_mae | selection | 0.118 | 0.300 | 0.625 | yes | yes | 0.333 |
| tair_transfer_selection_score | selection | 0.345 | 0.291 | 0.617 | no | no | 0.333 |
| rhair_final_step_mae | selection | 0.334 | 0.321 | 0.613 | no | yes | 0.333 |
| tair_full_horizon_mae | selection | 0.400 | 0.271 | 0.608 | no | yes | 0.667 |
| forecast_only_transfer_rank | selection | 0.145 | 0.265 | 0.602 | no | yes | 0.667 |
| tair_weighted_horizon_mae | selection | 0.389 | 0.244 | 0.592 | no | no | 0.333 |
| tair_control_horizon_mae | selection | 0.262 | 0.244 | 0.588 | no | no | 0.333 |
| rhair_control_horizon_abs_bias | selection | 0.122 | 0.244 | 0.583 | no | no | 0.000 |
| rhair_constraint_near_mae_proxy | selection | 0.331 | 0.256 | 0.575 | no | no | 0.000 |
| tair_final_step_mae | selection | 0.431 | 0.203 | 0.575 | no | yes | 0.333 |
| multiobjective_transfer_selection_score | selection | 0.041 | 0.144 | 0.567 | no | no | 0.333 |
| tair_constraint_near_mae_proxy | selection | 0.439 | 0.235 | 0.563 | no | no | 0.000 |
| tair_first_step_mae | selection | 0.208 | 0.176 | 0.550 | no | yes | 0.333 |
| co2_final_step_mae | selection | -0.078 | 0.012 | 0.525 | no | no | 0.000 |
| co2_first_step_mae | selection | -0.075 | -0.168 | 0.479 | no | no | 0.000 |
| co2_full_horizon_mae | selection | -0.206 | -0.103 | 0.475 | no | no | 0.000 |
| co2_weighted_horizon_mae | selection | -0.207 | -0.191 | 0.450 | no | no | 0.000 |
| co2_control_horizon_mae | selection | -0.165 | -0.235 | 0.437 | no | no | 0.000 |
| co2_transfer_selection_score | selection | -0.383 | -0.312 | 0.408 | no | no | 0.000 |
| co2_control_horizon_abs_bias | selection | -0.229 | -0.379 | 0.392 | no | no | 0.000 |
| co2_constraint_near_mae_proxy | selection | -0.388 | -0.444 | 0.367 | no | no | 0.000 |
| cost_grad_mean_abs | diagnostic | 0.395 | 0.579 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.437 | -0.540 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.436 | 0.501 |  |  |  |  |
| co2_sp_first_grad | diagnostic | -0.563 | -0.500 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | -0.502 | -0.406 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | 0.151 | 0.274 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | -0.448 | -0.268 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | -0.286 | -0.265 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | 0.173 | 0.253 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | -0.091 | 0.241 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.193 | -0.238 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | 0.039 | 0.233 |  |  |  |  |
| assim_sp_first_grad | diagnostic | -0.311 | -0.180 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | -0.057 | -0.062 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | 0.091 | 0.059 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | -0.207 | -0.003 |  |  |  |  |

## Robustness Summary

### Target: `mpc_tair_mae`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | weak_selection | 0.332 | 0.221 .. 0.554 | 0.221 .. 0.626 | 0.571 |
| tair_control_horizon_mae | offline_or_diagnostic_only | 0.213 | 0.077 .. 0.349 | 0.077 .. 0.514 | 0.510 |
| tair_weighted_horizon_mae | secondary_selection | 0.388 | 0.279 .. 0.507 | 0.279 .. 0.571 | 0.571 |
| tair_full_horizon_mae | secondary_selection | 0.409 | 0.304 .. 0.532 | 0.304 .. 0.566 | 0.571 |
| tair_final_step_mae | offline_or_diagnostic_only | 0.168 | 0.011 .. 0.275 | 0.011 .. 0.275 | 0.514 |
| tair_control_horizon_abs_bias | offline_or_diagnostic_only | 0.062 | -0.107 .. 0.207 | -0.107 .. 0.264 | 0.457 |
| tair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.152 | -0.009 .. 0.252 | -0.009 .. 0.349 | 0.490 |
| rhair_first_step_mae | offline_or_diagnostic_only | -0.097 | -0.239 .. 0.054 | -0.396 .. 0.054 | 0.419 |
| rhair_control_horizon_mae | offline_or_diagnostic_only | -0.029 | -0.161 .. 0.082 | -0.214 .. 0.061 | 0.429 |
| rhair_weighted_horizon_mae | weak_selection | 0.322 | 0.198 .. 0.492 | 0.157 .. 0.492 | 0.558 |
| rhair_full_horizon_mae | secondary_selection | 0.353 | 0.236 .. 0.529 | 0.209 .. 0.529 | 0.581 |
| rhair_final_step_mae | offline_or_diagnostic_only | 0.243 | 0.102 .. 0.459 | 0.102 .. 0.459 | 0.529 |
| rhair_control_horizon_abs_bias | offline_or_diagnostic_only | -0.506 | -0.611 .. -0.400 | -0.670 .. -0.400 | 0.295 |
| rhair_constraint_near_mae_proxy | weak_selection | 0.318 | 0.246 .. 0.568 | 0.246 .. 0.568 | 0.581 |
| co2_first_step_mae | offline_or_diagnostic_only | 0.210 | 0.071 .. 0.356 | 0.071 .. 0.331 | 0.552 |
| co2_control_horizon_mae | offline_or_diagnostic_only | 0.196 | 0.050 .. 0.356 | 0.050 .. 0.356 | 0.543 |
| co2_weighted_horizon_mae | offline_or_diagnostic_only | 0.082 | 0.018 .. 0.314 | -0.088 .. 0.314 | 0.514 |
| co2_full_horizon_mae | offline_or_diagnostic_only | 0.135 | 0.061 .. 0.379 | -0.093 .. 0.379 | 0.514 |
| co2_final_step_mae | offline_or_diagnostic_only | 0.121 | 0.043 .. 0.361 | -0.071 .. 0.361 | 0.533 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | 0.159 | 0.021 .. 0.311 | 0.021 .. 0.311 | 0.524 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.068 | -0.196 .. 0.132 | -0.280 .. 0.132 | 0.429 |
| forecast_only_transfer_rank | weak_selection | 0.286 | 0.216 .. 0.506 | 0.198 .. 0.506 | 0.577 |
| tair_transfer_selection_score | offline_or_diagnostic_only | 0.191 | 0.039 .. 0.382 | 0.039 .. 0.555 | 0.524 |
| rhair_transfer_selection_score | offline_or_diagnostic_only | -0.119 | -0.263 .. 0.018 | -0.429 .. 0.018 | 0.394 |
| co2_transfer_selection_score | offline_or_diagnostic_only | 0.159 | 0.011 .. 0.329 | 0.011 .. 0.329 | 0.524 |
| multiobjective_transfer_selection_score | offline_or_diagnostic_only | 0.168 | 0.089 .. 0.429 | 0.005 .. 0.429 | 0.548 |

### Target: `mpc_rhair_mae`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | weak_selection | 0.347 | 0.229 .. 0.464 | 0.229 .. 0.464 | 0.581 |
| tair_control_horizon_mae | secondary_selection | 0.446 | 0.327 .. 0.574 | 0.327 .. 0.591 | 0.635 |
| tair_weighted_horizon_mae | secondary_selection | 0.394 | 0.264 .. 0.529 | 0.176 .. 0.555 | 0.600 |
| tair_full_horizon_mae | secondary_selection | 0.429 | 0.307 .. 0.532 | 0.181 .. 0.560 | 0.619 |
| tair_final_step_mae | primary_selection | 0.615 | 0.546 .. 0.786 | 0.495 .. 0.786 | 0.724 |
| tair_control_horizon_abs_bias | secondary_selection | 0.597 | 0.536 .. 0.707 | 0.525 .. 0.725 | 0.657 |
| tair_constraint_near_mae_proxy | weak_selection | 0.281 | 0.141 .. 0.424 | 0.141 .. 0.443 | 0.538 |
| rhair_first_step_mae | offline_or_diagnostic_only | 0.174 | 0.046 .. 0.386 | 0.049 .. 0.386 | 0.533 |
| rhair_control_horizon_mae | offline_or_diagnostic_only | 0.062 | -0.089 .. 0.250 | -0.039 .. 0.250 | 0.467 |
| rhair_weighted_horizon_mae | weak_selection | 0.278 | 0.138 .. 0.393 | 0.138 .. 0.393 | 0.548 |
| rhair_full_horizon_mae | weak_selection | 0.297 | 0.161 .. 0.414 | 0.161 .. 0.414 | 0.562 |
| rhair_final_step_mae | secondary_selection | 0.458 | 0.349 .. 0.561 | 0.349 .. 0.580 | 0.615 |
| rhair_control_horizon_abs_bias | offline_or_diagnostic_only | -0.256 | -0.346 .. -0.121 | -0.346 .. -0.121 | 0.371 |
| rhair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.200 | 0.029 .. 0.339 | 0.029 .. 0.339 | 0.524 |
| co2_first_step_mae | offline_or_diagnostic_only | -0.131 | -0.234 .. 0.036 | -0.305 .. 0.036 | 0.423 |
| co2_control_horizon_mae | offline_or_diagnostic_only | -0.166 | -0.266 .. 0.004 | -0.366 .. 0.004 | 0.452 |
| co2_weighted_horizon_mae | offline_or_diagnostic_only | 0.124 | 0.032 .. 0.364 | -0.055 .. 0.364 | 0.581 |
| co2_full_horizon_mae | offline_or_diagnostic_only | 0.168 | 0.082 .. 0.418 | 0.016 .. 0.418 | 0.590 |
| co2_final_step_mae | weak_selection | 0.256 | 0.157 .. 0.525 | 0.157 .. 0.525 | 0.600 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | -0.385 | -0.532 .. -0.264 | -0.552 .. -0.264 | 0.324 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.218 | -0.332 .. -0.050 | -0.451 .. -0.050 | 0.390 |
| forecast_only_transfer_rank | secondary_selection | 0.395 | 0.268 .. 0.539 | 0.275 .. 0.539 | 0.610 |
| tair_transfer_selection_score | secondary_selection | 0.526 | 0.425 .. 0.654 | 0.425 .. 0.654 | 0.667 |
| rhair_transfer_selection_score | offline_or_diagnostic_only | 0.088 | -0.061 .. 0.243 | -0.060 .. 0.243 | 0.471 |
| co2_transfer_selection_score | offline_or_diagnostic_only | -0.250 | -0.389 .. -0.100 | -0.451 .. -0.100 | 0.400 |
| multiobjective_transfer_selection_score | offline_or_diagnostic_only | 0.147 | 0.014 .. 0.289 | 0.014 .. 0.289 | 0.524 |

### Target: `mpc_co2_mae`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | offline_or_diagnostic_only | -0.024 | -0.175 .. 0.114 | -0.203 .. 0.154 | 0.410 |
| tair_control_horizon_mae | offline_or_diagnostic_only | 0.043 | -0.095 .. 0.181 | -0.184 .. 0.202 | 0.471 |
| tair_weighted_horizon_mae | weak_selection | 0.288 | 0.179 .. 0.457 | 0.170 .. 0.457 | 0.543 |
| tair_full_horizon_mae | weak_selection | 0.282 | 0.171 .. 0.450 | 0.171 .. 0.450 | 0.543 |
| tair_final_step_mae | offline_or_diagnostic_only | 0.050 | -0.111 .. 0.225 | -0.111 .. 0.214 | 0.467 |
| tair_control_horizon_abs_bias | offline_or_diagnostic_only | 0.000 | -0.161 .. 0.107 | -0.165 .. 0.125 | 0.448 |
| tair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.102 | -0.048 .. 0.241 | -0.184 .. 0.241 | 0.471 |
| rhair_first_step_mae | offline_or_diagnostic_only | -0.347 | -0.443 .. -0.236 | -0.621 .. -0.236 | 0.333 |
| rhair_control_horizon_mae | offline_or_diagnostic_only | -0.091 | -0.232 .. 0.014 | -0.313 .. 0.014 | 0.438 |
| rhair_weighted_horizon_mae | offline_or_diagnostic_only | 0.210 | 0.084 .. 0.370 | -0.003 .. 0.388 | 0.519 |
| rhair_full_horizon_mae | weak_selection | 0.259 | 0.143 .. 0.425 | 0.071 .. 0.425 | 0.552 |
| rhair_final_step_mae | offline_or_diagnostic_only | 0.093 | -0.059 .. 0.238 | -0.059 .. 0.238 | 0.452 |
| rhair_control_horizon_abs_bias | secondary_selection | 0.479 | 0.407 .. 0.550 | 0.346 .. 0.578 | 0.638 |
| rhair_constraint_near_mae_proxy | weak_selection | 0.344 | 0.261 .. 0.525 | 0.244 .. 0.525 | 0.590 |
| co2_first_step_mae | offline_or_diagnostic_only | -0.243 | -0.359 .. -0.163 | -0.393 .. -0.047 | 0.375 |
| co2_control_horizon_mae | offline_or_diagnostic_only | -0.358 | -0.456 .. -0.273 | -0.509 .. -0.113 | 0.308 |
| co2_weighted_horizon_mae | offline_or_diagnostic_only | -0.485 | -0.596 .. -0.375 | -0.659 .. -0.375 | 0.295 |
| co2_full_horizon_mae | offline_or_diagnostic_only | -0.429 | -0.543 .. -0.307 | -0.615 .. -0.307 | 0.314 |
| co2_final_step_mae | offline_or_diagnostic_only | -0.229 | -0.361 .. -0.064 | -0.363 .. -0.064 | 0.381 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | 0.150 | 0.071 .. 0.329 | 0.005 .. 0.329 | 0.486 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.415 | -0.507 .. -0.289 | -0.522 .. -0.269 | 0.295 |
| forecast_only_transfer_rank | offline_or_diagnostic_only | 0.047 | -0.079 .. 0.207 | -0.214 .. 0.207 | 0.476 |
| tair_transfer_selection_score | offline_or_diagnostic_only | 0.015 | -0.154 .. 0.146 | -0.198 .. 0.146 | 0.457 |
| rhair_transfer_selection_score | offline_or_diagnostic_only | 0.004 | -0.088 .. 0.120 | -0.291 .. 0.120 | 0.452 |
| co2_transfer_selection_score | offline_or_diagnostic_only | -0.238 | -0.375 .. -0.143 | -0.440 .. 0.022 | 0.343 |
| multiobjective_transfer_selection_score | offline_or_diagnostic_only | -0.082 | -0.268 .. 0.000 | -0.478 .. 0.077 | 0.419 |

### Target: `mpc_objective`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | offline_or_diagnostic_only | 0.176 | 0.046 .. 0.343 | 0.027 .. 0.379 | 0.505 |
| tair_control_horizon_mae | offline_or_diagnostic_only | 0.244 | 0.129 .. 0.375 | 0.025 .. 0.468 | 0.548 |
| tair_weighted_horizon_mae | offline_or_diagnostic_only | 0.244 | 0.093 .. 0.346 | -0.005 .. 0.420 | 0.543 |
| tair_full_horizon_mae | weak_selection | 0.271 | 0.125 .. 0.379 | 0.000 .. 0.451 | 0.562 |
| tair_final_step_mae | offline_or_diagnostic_only | 0.203 | 0.043 .. 0.286 | 0.043 .. 0.310 | 0.524 |
| tair_control_horizon_abs_bias | objective_secondary_selection | 0.362 | 0.257 .. 0.461 | 0.268 .. 0.486 | 0.590 |
| tair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.235 | 0.082 .. 0.425 | 0.082 .. 0.425 | 0.510 |
| rhair_first_step_mae | weak_selection | 0.300 | 0.150 .. 0.525 | 0.066 .. 0.525 | 0.571 |
| rhair_control_horizon_mae | weak_selection | 0.329 | 0.186 .. 0.514 | 0.121 .. 0.514 | 0.581 |
| rhair_weighted_horizon_mae | objective_secondary_selection | 0.450 | 0.332 .. 0.565 | 0.245 .. 0.578 | 0.606 |
| rhair_full_horizon_mae | objective_secondary_selection | 0.450 | 0.332 .. 0.564 | 0.242 .. 0.582 | 0.600 |
| rhair_final_step_mae | weak_selection | 0.321 | 0.186 .. 0.436 | 0.186 .. 0.436 | 0.567 |
| rhair_control_horizon_abs_bias | offline_or_diagnostic_only | 0.244 | 0.175 .. 0.436 | 0.104 .. 0.436 | 0.562 |
| rhair_constraint_near_mae_proxy | weak_selection | 0.256 | 0.143 .. 0.421 | 0.143 .. 0.421 | 0.533 |
| co2_first_step_mae | offline_or_diagnostic_only | -0.168 | -0.275 .. -0.046 | -0.275 .. -0.046 | 0.442 |
| co2_control_horizon_mae | offline_or_diagnostic_only | -0.235 | -0.357 .. -0.114 | -0.357 .. -0.114 | 0.394 |
| co2_weighted_horizon_mae | offline_or_diagnostic_only | -0.191 | -0.325 .. -0.018 | -0.325 .. -0.018 | 0.410 |
| co2_full_horizon_mae | offline_or_diagnostic_only | -0.103 | -0.229 .. 0.089 | -0.229 .. 0.089 | 0.438 |
| co2_final_step_mae | offline_or_diagnostic_only | 0.012 | -0.079 .. 0.229 | -0.137 .. 0.229 | 0.495 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | -0.379 | -0.486 .. -0.289 | -0.484 .. -0.289 | 0.352 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.444 | -0.589 .. -0.325 | -0.589 .. -0.325 | 0.324 |
| forecast_only_transfer_rank | weak_selection | 0.265 | 0.114 .. 0.391 | -0.077 .. 0.391 | 0.552 |
| tair_transfer_selection_score | weak_selection | 0.291 | 0.150 .. 0.461 | 0.137 .. 0.533 | 0.571 |
| rhair_transfer_selection_score | objective_secondary_selection | 0.449 | 0.331 .. 0.572 | 0.187 .. 0.572 | 0.606 |
| co2_transfer_selection_score | offline_or_diagnostic_only | -0.312 | -0.379 .. -0.171 | -0.350 .. -0.171 | 0.390 |
| multiobjective_transfer_selection_score | offline_or_diagnostic_only | 0.144 | -0.011 .. 0.250 | -0.148 .. 0.308 | 0.524 |
