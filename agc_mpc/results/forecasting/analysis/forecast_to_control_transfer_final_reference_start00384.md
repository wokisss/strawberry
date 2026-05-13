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
| mpc_tair_mae | tair_control_horizon_abs_bias | secondary_selection |
| mpc_tair_mae | tair_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_tair_mae | rhair_first_step_mae | offline_or_diagnostic_only |
| mpc_tair_mae | rhair_control_horizon_mae | offline_or_diagnostic_only |
| mpc_tair_mae | rhair_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_tair_mae | rhair_full_horizon_mae | offline_or_diagnostic_only |
| mpc_tair_mae | rhair_final_step_mae | offline_or_diagnostic_only |
| mpc_tair_mae | rhair_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_tair_mae | rhair_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_tair_mae | co2_first_step_mae | offline_or_diagnostic_only |
| mpc_tair_mae | co2_control_horizon_mae | weak_selection |
| mpc_tair_mae | co2_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_tair_mae | co2_full_horizon_mae | offline_or_diagnostic_only |
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
| mpc_rhair_mae | tair_first_step_mae | primary_selection |
| mpc_rhair_mae | tair_control_horizon_mae | secondary_selection |
| mpc_rhair_mae | tair_weighted_horizon_mae | primary_selection |
| mpc_rhair_mae | tair_full_horizon_mae | secondary_selection |
| mpc_rhair_mae | tair_final_step_mae | secondary_selection |
| mpc_rhair_mae | tair_control_horizon_abs_bias | secondary_selection |
| mpc_rhair_mae | tair_constraint_near_mae_proxy | weak_selection |
| mpc_rhair_mae | rhair_first_step_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_control_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_weighted_horizon_mae | secondary_selection |
| mpc_rhair_mae | rhair_full_horizon_mae | secondary_selection |
| mpc_rhair_mae | rhair_final_step_mae | primary_selection |
| mpc_rhair_mae | rhair_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_constraint_near_mae_proxy | primary_selection |
| mpc_rhair_mae | co2_first_step_mae | secondary_selection |
| mpc_rhair_mae | co2_control_horizon_mae | weak_selection |
| mpc_rhair_mae | co2_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_full_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_final_step_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_rhair_mae | forecast_only_transfer_rank | secondary_selection |
| mpc_rhair_mae | tair_transfer_selection_score | secondary_selection |
| mpc_rhair_mae | rhair_transfer_selection_score | offline_or_diagnostic_only |
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
| mpc_co2_mae | rhair_control_horizon_mae | weak_selection |
| mpc_co2_mae | rhair_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_co2_mae | rhair_full_horizon_mae | offline_or_diagnostic_only |
| mpc_co2_mae | rhair_final_step_mae | offline_or_diagnostic_only |
| mpc_co2_mae | rhair_control_horizon_abs_bias | primary_selection |
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
| mpc_objective | tair_control_horizon_mae | weak_selection |
| mpc_objective | tair_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_objective | tair_full_horizon_mae | offline_or_diagnostic_only |
| mpc_objective | tair_final_step_mae | offline_or_diagnostic_only |
| mpc_objective | tair_control_horizon_abs_bias | objective_secondary_selection |
| mpc_objective | tair_constraint_near_mae_proxy | objective_secondary_selection |
| mpc_objective | rhair_first_step_mae | weak_selection |
| mpc_objective | rhair_control_horizon_mae | offline_or_diagnostic_only |
| mpc_objective | rhair_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_objective | rhair_full_horizon_mae | offline_or_diagnostic_only |
| mpc_objective | rhair_final_step_mae | offline_or_diagnostic_only |
| mpc_objective | rhair_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_objective | rhair_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_objective | co2_first_step_mae | offline_or_diagnostic_only |
| mpc_objective | co2_control_horizon_mae | offline_or_diagnostic_only |
| mpc_objective | co2_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_objective | co2_full_horizon_mae | offline_or_diagnostic_only |
| mpc_objective | co2_final_step_mae | offline_or_diagnostic_only |
| mpc_objective | co2_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_objective | co2_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_objective | forecast_only_transfer_rank | offline_or_diagnostic_only |
| mpc_objective | tair_transfer_selection_score | weak_selection |
| mpc_objective | rhair_transfer_selection_score | offline_or_diagnostic_only |
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
| 1 | current_hybrid_transformer | 5.167 | 5.688 | 4.250 | 5.562 | 6.722 | 0.535 | 1.530 | 30.872 | 0.0787 |
| 2 | itransformer_co2_control_aware_fusion | 5.406 | 7.094 | 7.625 | 1.500 | 8.556 | 0.931 | 1.338 | 42.784 | 0.1535 |
| 3 | itransformer_residual | 5.583 | 4.062 | 5.688 | 7.000 | 9.167 | 1.125 | 2.010 | 32.863 | 0.1368 |
| 4 | itransformer_co2_late_frozen_expert | 5.740 | 6.531 | 8.625 | 2.062 | 9.944 | 0.841 | 1.074 | 41.607 | 0.1485 |
| 5 | transformer_hybrid_residual | 6.396 | 4.500 | 2.875 | 11.812 | 9.167 | 1.325 | 2.385 | 26.551 | 0.1162 |
| 6 | itransformer_co2_late_residual | 6.604 | 7.812 | 6.000 | 6.000 | 9.000 | 0.885 | 1.241 | 49.925 | 0.1501 |
| 7 | segrnn_forecaster | 7.979 | 14.188 | 6.000 | 3.750 | 9.389 | 0.604 | 4.952 | 33.332 | 0.1215 |
| 8 | dlinear_forecaster | 8.583 | 9.938 | 5.500 | 10.312 | 15.056 | 0.868 | 1.884 | 21.373 | 0.1452 |
| 9 | itransformer_co2_horizon_mixture | 8.646 | 2.875 | 12.125 | 10.938 | 13.722 | 0.812 | 0.997 | 49.239 | 0.1459 |
| 10 | transformer_forecaster | 9.250 | 6.188 | 11.125 | 10.438 | 13.056 | 0.370 | 2.457 | 32.867 | 0.1016 |
| 11 | itransformer_co2_residual | 9.312 | 6.062 | 9.938 | 11.938 | 10.778 | 0.524 | 1.839 | 12.225 | 0.1144 |
| 12 | gru_forecaster | 10.375 | 10.438 | 14.625 | 6.062 | 14.278 | 0.642 | 4.642 | 59.054 | 0.1069 |
| 13 | nlinear_forecaster | 10.729 | 13.125 | 4.812 | 14.250 | 15.500 | 1.244 | 2.222 | 15.711 | 0.1299 |
| 14 | patchtst_residual | 10.938 | 8.688 | 11.875 | 12.250 | 14.833 | 0.655 | 2.034 | 36.695 | 0.1203 |
| 15 | lstm_forecaster | 11.604 | 12.812 | 11.750 | 10.250 | 15.111 | 0.878 | 11.374 | 63.140 | 0.3009 |
| 16 | frequency_forecaster | 13.688 | 16.000 | 13.188 | 11.875 | 18.722 | 1.509 | 6.704 | 25.958 | 0.6076 |

## Metric Transfer Quality

### Target: `mpc_tair_mae`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tair_control_horizon_abs_bias | selection | 0.483 | 0.385 | 0.633 | no | no | 0.333 |
| co2_control_horizon_mae | selection | 0.501 | 0.291 | 0.597 | no | no | 0.000 |
| co2_weighted_horizon_mae | selection | 0.394 | 0.247 | 0.583 | no | no | 0.333 |
| co2_first_step_mae | selection | 0.527 | 0.221 | 0.580 | no | no | 0.000 |
| tair_constraint_near_mae_proxy | selection | 0.284 | 0.197 | 0.571 | no | yes | 0.667 |
| co2_full_horizon_mae | selection | 0.368 | 0.191 | 0.558 | no | no | 0.333 |
| tair_transfer_selection_score | selection | 0.264 | 0.182 | 0.558 | no | no | 0.000 |
| tair_control_horizon_mae | selection | 0.539 | 0.141 | 0.555 | no | no | 0.000 |
| co2_final_step_mae | selection | 0.151 | 0.129 | 0.550 | no | no | 0.000 |
| co2_transfer_selection_score | selection | 0.262 | 0.147 | 0.542 | no | no | 0.000 |
| tair_first_step_mae | selection | 0.538 | 0.097 | 0.533 | no | no | 0.000 |
| multiobjective_transfer_selection_score | selection | 0.165 | 0.029 | 0.508 | no | no | 0.333 |
| tair_weighted_horizon_mae | selection | 0.375 | 0.006 | 0.508 | no | no | 0.000 |
| tair_full_horizon_mae | selection | 0.354 | -0.006 | 0.500 | no | no | 0.000 |
| tair_final_step_mae | selection | 0.249 | -0.015 | 0.492 | no | no | 0.000 |
| co2_control_horizon_abs_bias | selection | 0.016 | -0.021 | 0.483 | no | no | 0.000 |
| rhair_final_step_mae | selection | 0.074 | -0.068 | 0.479 | no | no | 0.333 |
| co2_constraint_near_mae_proxy | selection | -0.048 | -0.121 | 0.475 | no | no | 0.333 |
| rhair_constraint_near_mae_proxy | selection | -0.031 | -0.129 | 0.450 | no | no | 0.000 |
| rhair_first_step_mae | selection | 0.336 | -0.085 | 0.450 | no | no | 0.333 |
| forecast_only_transfer_rank | selection | 0.136 | -0.071 | 0.450 | no | no | 0.333 |
| rhair_weighted_horizon_mae | selection | 0.045 | -0.209 | 0.412 | no | no | 0.333 |
| rhair_full_horizon_mae | selection | 0.037 | -0.238 | 0.400 | no | no | 0.333 |
| rhair_transfer_selection_score | selection | -0.222 | -0.234 | 0.387 | no | no | 0.333 |
| rhair_control_horizon_mae | selection | 0.163 | -0.279 | 0.375 | no | no | 0.000 |
| rhair_control_horizon_abs_bias | selection | -0.480 | -0.426 | 0.342 | no | no | 0.333 |
| co2_first_grad_mean_abs | diagnostic | 0.412 | 0.447 |  |  |  |  |
| assim_sp_first_grad | diagnostic | 0.306 | 0.377 |  |  |  |  |
| co2_sp_first_grad | diagnostic | 0.432 | 0.365 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | 0.182 | 0.312 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | 0.218 | 0.286 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.262 | 0.282 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.262 | -0.278 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | 0.146 | 0.268 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | -0.108 | -0.218 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | 0.337 | 0.194 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | 0.154 | 0.180 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | 0.034 | 0.171 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.265 | -0.115 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | 0.165 | 0.082 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | -0.060 | -0.032 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | -0.101 | -0.024 |  |  |  |  |

### Target: `mpc_rhair_mae`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rhair_constraint_near_mae_proxy | selection | 0.533 | 0.788 | 0.792 | yes | yes | 0.333 |
| tair_first_step_mae | selection | 0.461 | 0.685 | 0.775 | yes | yes | 0.667 |
| rhair_final_step_mae | selection | 0.445 | 0.680 | 0.756 | no | yes | 0.333 |
| tair_weighted_horizon_mae | selection | 0.738 | 0.662 | 0.750 | yes | yes | 0.333 |
| tair_full_horizon_mae | selection | 0.754 | 0.656 | 0.742 | yes | yes | 0.333 |
| tair_transfer_selection_score | selection | 0.668 | 0.618 | 0.733 | yes | yes | 0.333 |
| forecast_only_transfer_rank | selection | 0.604 | 0.665 | 0.725 | no | no | 0.000 |
| tair_control_horizon_mae | selection | 0.549 | 0.595 | 0.714 | yes | yes | 0.333 |
| rhair_weighted_horizon_mae | selection | 0.545 | 0.618 | 0.706 | no | no | 0.000 |
| multiobjective_transfer_selection_score | selection | 0.617 | 0.597 | 0.700 | no | no | 0.000 |
| rhair_full_horizon_mae | selection | 0.542 | 0.609 | 0.692 | no | no | 0.000 |
| tair_control_horizon_abs_bias | selection | 0.576 | 0.556 | 0.692 | no | yes | 0.333 |
| tair_final_step_mae | selection | 0.811 | 0.474 | 0.683 | no | yes | 0.333 |
| co2_first_step_mae | selection | 0.523 | 0.471 | 0.672 | no | no | 0.667 |
| co2_control_horizon_mae | selection | 0.351 | 0.336 | 0.605 | no | no | 0.333 |
| tair_constraint_near_mae_proxy | selection | 0.618 | 0.280 | 0.597 | no | no | 0.333 |
| co2_transfer_selection_score | selection | 0.161 | 0.268 | 0.583 | no | no | 0.333 |
| co2_weighted_horizon_mae | selection | 0.048 | 0.224 | 0.558 | no | no | 0.000 |
| rhair_transfer_selection_score | selection | 0.407 | 0.210 | 0.555 | no | no | 0.000 |
| co2_full_horizon_mae | selection | -0.037 | 0.182 | 0.550 | no | no | 0.333 |
| rhair_control_horizon_mae | selection | 0.383 | 0.179 | 0.550 | no | no | 0.333 |
| co2_control_horizon_abs_bias | selection | -0.272 | 0.059 | 0.508 | no | no | 0.333 |
| rhair_first_step_mae | selection | 0.327 | 0.050 | 0.508 | no | no | 0.000 |
| co2_constraint_near_mae_proxy | selection | -0.281 | -0.003 | 0.500 | no | no | 0.000 |
| co2_final_step_mae | selection | -0.266 | -0.074 | 0.475 | no | no | 0.000 |
| rhair_control_horizon_abs_bias | selection | 0.147 | -0.079 | 0.450 | no | no | 0.000 |
| rhair_dx_sp_first_grad | diagnostic | -0.726 | -0.857 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | -0.648 | -0.854 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | -0.767 | -0.851 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | -0.631 | -0.792 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | -0.703 | -0.730 |  |  |  |  |
| assim_sp_first_grad | diagnostic | -0.581 | -0.571 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | -0.666 | -0.506 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | -0.640 | -0.486 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | 0.476 | 0.485 |  |  |  |  |
| co2_sp_first_grad | diagnostic | -0.482 | -0.253 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | -0.584 | -0.247 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | -0.560 | -0.109 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.205 | 0.095 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.205 | -0.094 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | -0.187 | -0.041 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.231 | -0.032 |  |  |  |  |

### Target: `mpc_co2_mae`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rhair_control_horizon_abs_bias | selection | 0.725 | 0.779 | 0.783 | no | yes | 0.667 |
| rhair_transfer_selection_score | selection | 0.456 | 0.472 | 0.681 | no | no | 0.333 |
| rhair_control_horizon_mae | selection | 0.176 | 0.306 | 0.633 | no | no | 0.333 |
| rhair_full_horizon_mae | selection | 0.234 | 0.226 | 0.592 | no | no | 0.333 |
| rhair_first_step_mae | selection | 0.043 | 0.221 | 0.592 | no | no | 0.000 |
| rhair_weighted_horizon_mae | selection | 0.244 | 0.216 | 0.588 | no | no | 0.333 |
| tair_final_step_mae | selection | 0.150 | 0.168 | 0.567 | no | no | 0.000 |
| tair_transfer_selection_score | selection | -0.020 | 0.029 | 0.533 | no | no | 0.000 |
| tair_constraint_near_mae_proxy | selection | 0.009 | 0.099 | 0.529 | no | yes | 0.333 |
| tair_control_horizon_abs_bias | selection | -0.095 | 0.029 | 0.525 | no | no | 0.000 |
| tair_weighted_horizon_mae | selection | -0.005 | 0.012 | 0.517 | no | no | 0.000 |
| rhair_constraint_near_mae_proxy | selection | 0.206 | 0.091 | 0.508 | no | no | 0.333 |
| tair_full_horizon_mae | selection | 0.018 | 0.009 | 0.508 | no | no | 0.000 |
| rhair_final_step_mae | selection | 0.069 | 0.007 | 0.504 | no | no | 0.000 |
| multiobjective_transfer_selection_score | selection | -0.008 | -0.021 | 0.500 | no | no | 0.000 |
| tair_first_step_mae | selection | -0.207 | -0.094 | 0.475 | no | no | 0.000 |
| forecast_only_transfer_rank | selection | -0.068 | -0.056 | 0.475 | no | no | 0.000 |
| tair_control_horizon_mae | selection | -0.197 | -0.137 | 0.462 | no | no | 0.000 |
| co2_first_step_mae | selection | -0.219 | -0.319 | 0.387 | no | no | 0.000 |
| co2_control_horizon_abs_bias | selection | -0.389 | -0.344 | 0.375 | no | no | 0.000 |
| co2_constraint_near_mae_proxy | selection | -0.466 | -0.429 | 0.350 | no | no | 0.000 |
| co2_control_horizon_mae | selection | -0.379 | -0.519 | 0.319 | no | no | 0.000 |
| co2_transfer_selection_score | selection | -0.415 | -0.518 | 0.317 | no | no | 0.000 |
| co2_final_step_mae | selection | -0.354 | -0.553 | 0.308 | no | no | 0.000 |
| co2_full_horizon_mae | selection | -0.524 | -0.574 | 0.300 | no | no | 0.000 |
| co2_weighted_horizon_mae | selection | -0.514 | -0.606 | 0.292 | no | no | 0.000 |
| co2_sp_first_grad | diagnostic | -0.754 | -0.711 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | -0.636 | -0.617 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | -0.567 | -0.499 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | -0.556 | -0.458 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | -0.585 | -0.440 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | -0.481 | -0.313 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | 0.408 | 0.303 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.278 | -0.181 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.171 | -0.177 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.170 | 0.157 |  |  |  |  |
| assim_sp_first_grad | diagnostic | -0.237 | -0.131 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | -0.303 | -0.063 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | -0.126 | -0.031 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | 0.041 | 0.031 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | -0.067 | -0.010 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | -0.058 | -0.001 |  |  |  |  |

### Target: `mpc_objective`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tair_control_horizon_abs_bias | selection | 0.723 | 0.426 | 0.683 | no | yes | 0.333 |
| tair_constraint_near_mae_proxy | selection | 0.727 | 0.389 | 0.672 | no | no | 0.333 |
| rhair_first_step_mae | selection | 0.834 | 0.306 | 0.617 | no | yes | 0.333 |
| tair_transfer_selection_score | selection | 0.587 | 0.326 | 0.608 | no | no | 0.000 |
| tair_control_horizon_mae | selection | 0.895 | 0.286 | 0.605 | no | no | 0.000 |
| rhair_transfer_selection_score | selection | 0.401 | 0.212 | 0.588 | no | yes | 0.333 |
| tair_final_step_mae | selection | 0.769 | 0.235 | 0.575 | no | no | 0.000 |
| tair_first_step_mae | selection | 0.854 | 0.121 | 0.567 | no | no | 0.000 |
| co2_first_step_mae | selection | 0.870 | 0.130 | 0.563 | no | no | 0.000 |
| rhair_control_horizon_mae | selection | 0.739 | 0.118 | 0.558 | no | no | 0.000 |
| multiobjective_transfer_selection_score | selection | 0.605 | 0.141 | 0.542 | yes | yes | 0.333 |
| tair_weighted_horizon_mae | selection | 0.834 | 0.118 | 0.542 | no | no | 0.000 |
| tair_full_horizon_mae | selection | 0.822 | 0.112 | 0.533 | no | no | 0.000 |
| co2_control_horizon_mae | selection | 0.728 | 0.038 | 0.513 | no | no | 0.000 |
| co2_full_horizon_mae | selection | 0.422 | 0.053 | 0.508 | no | yes | 0.333 |
| rhair_control_horizon_abs_bias | selection | -0.224 | -0.026 | 0.508 | no | no | 0.000 |
| co2_weighted_horizon_mae | selection | 0.481 | 0.059 | 0.500 | no | yes | 0.333 |
| rhair_weighted_horizon_mae | selection | 0.561 | 0.038 | 0.496 | no | yes | 0.333 |
| rhair_final_step_mae | selection | 0.419 | -0.038 | 0.496 | yes | yes | 0.333 |
| co2_final_step_mae | selection | 0.112 | -0.082 | 0.483 | no | no | 0.000 |
| rhair_full_horizon_mae | selection | 0.530 | -0.029 | 0.483 | no | yes | 0.333 |
| forecast_only_transfer_rank | selection | 0.598 | 0.021 | 0.483 | no | yes | 0.333 |
| co2_transfer_selection_score | selection | 0.225 | -0.159 | 0.458 | no | no | 0.000 |
| rhair_constraint_near_mae_proxy | selection | 0.334 | -0.188 | 0.433 | no | no | 0.000 |
| co2_constraint_near_mae_proxy | selection | -0.250 | -0.379 | 0.342 | no | yes | 0.333 |
| co2_control_horizon_abs_bias | selection | -0.252 | -0.509 | 0.333 | no | no | 0.000 |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.297 | -0.641 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.297 | 0.626 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | -0.263 | -0.477 |  |  |  |  |
| assim_sp_first_grad | diagnostic | 0.049 | 0.442 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | -0.142 | 0.433 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.283 | -0.400 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | 0.034 | 0.350 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | -0.170 | 0.347 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | -0.219 | 0.303 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | 0.018 | 0.297 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | -0.222 | 0.294 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | 0.210 | 0.221 |  |  |  |  |
| co2_sp_first_grad | diagnostic | 0.126 | 0.141 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | -0.353 | 0.121 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | 0.472 | 0.115 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | 0.069 | -0.012 |  |  |  |  |

## Robustness Summary

### Target: `mpc_tair_mae`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | offline_or_diagnostic_only | 0.097 | -0.096 .. 0.250 | -0.096 .. 0.379 | 0.467 |
| tair_control_horizon_mae | offline_or_diagnostic_only | 0.141 | -0.043 .. 0.297 | -0.053 .. 0.462 | 0.490 |
| tair_weighted_horizon_mae | offline_or_diagnostic_only | 0.006 | -0.207 .. 0.132 | -0.207 .. 0.264 | 0.438 |
| tair_full_horizon_mae | offline_or_diagnostic_only | -0.006 | -0.221 .. 0.118 | -0.221 .. 0.242 | 0.429 |
| tair_final_step_mae | offline_or_diagnostic_only | -0.015 | -0.232 .. 0.111 | -0.232 .. 0.165 | 0.419 |
| tair_control_horizon_abs_bias | secondary_selection | 0.385 | 0.254 .. 0.554 | 0.254 .. 0.637 | 0.581 |
| tair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.197 | 0.025 .. 0.306 | 0.025 .. 0.322 | 0.510 |
| rhair_first_step_mae | offline_or_diagnostic_only | -0.085 | -0.318 .. 0.057 | -0.318 .. 0.148 | 0.371 |
| rhair_control_horizon_mae | offline_or_diagnostic_only | -0.279 | -0.554 .. -0.179 | -0.554 .. -0.121 | 0.286 |
| rhair_weighted_horizon_mae | offline_or_diagnostic_only | -0.209 | -0.468 .. -0.093 | -0.468 .. -0.011 | 0.327 |
| rhair_full_horizon_mae | offline_or_diagnostic_only | -0.238 | -0.464 .. -0.125 | -0.464 .. -0.055 | 0.324 |
| rhair_final_step_mae | offline_or_diagnostic_only | -0.068 | -0.257 .. 0.036 | -0.257 .. 0.146 | 0.413 |
| rhair_control_horizon_abs_bias | offline_or_diagnostic_only | -0.426 | -0.586 .. -0.304 | -0.586 .. -0.304 | 0.276 |
| rhair_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.129 | -0.332 .. -0.011 | -0.332 .. 0.071 | 0.381 |
| co2_first_step_mae | offline_or_diagnostic_only | 0.221 | 0.054 .. 0.321 | 0.054 .. 0.321 | 0.519 |
| co2_control_horizon_mae | weak_selection | 0.291 | 0.139 .. 0.447 | 0.132 .. 0.447 | 0.538 |
| co2_weighted_horizon_mae | offline_or_diagnostic_only | 0.247 | 0.086 .. 0.407 | 0.086 .. 0.407 | 0.524 |
| co2_full_horizon_mae | offline_or_diagnostic_only | 0.191 | 0.018 .. 0.339 | 0.018 .. 0.339 | 0.495 |
| co2_final_step_mae | offline_or_diagnostic_only | 0.129 | -0.021 .. 0.204 | -0.029 .. 0.196 | 0.505 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | -0.021 | -0.193 .. 0.082 | -0.222 .. 0.082 | 0.429 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.121 | -0.254 .. 0.061 | -0.242 .. 0.061 | 0.429 |
| forecast_only_transfer_rank | offline_or_diagnostic_only | -0.071 | -0.282 .. 0.016 | -0.282 .. 0.187 | 0.381 |
| tair_transfer_selection_score | offline_or_diagnostic_only | 0.182 | 0.007 .. 0.321 | 0.007 .. 0.511 | 0.495 |
| rhair_transfer_selection_score | offline_or_diagnostic_only | -0.234 | -0.459 .. -0.120 | -0.459 .. -0.041 | 0.308 |
| co2_transfer_selection_score | offline_or_diagnostic_only | 0.147 | -0.014 .. 0.311 | -0.007 .. 0.311 | 0.486 |
| multiobjective_transfer_selection_score | offline_or_diagnostic_only | 0.029 | -0.179 .. 0.125 | -0.179 .. 0.192 | 0.438 |

### Target: `mpc_rhair_mae`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | primary_selection | 0.685 | 0.618 .. 0.807 | 0.516 .. 0.791 | 0.743 |
| tair_control_horizon_mae | secondary_selection | 0.595 | 0.508 .. 0.693 | 0.446 .. 0.759 | 0.673 |
| tair_weighted_horizon_mae | primary_selection | 0.662 | 0.589 .. 0.750 | 0.401 .. 0.874 | 0.714 |
| tair_full_horizon_mae | secondary_selection | 0.656 | 0.582 .. 0.743 | 0.390 .. 0.863 | 0.705 |
| tair_final_step_mae | secondary_selection | 0.474 | 0.361 .. 0.564 | 0.066 .. 0.681 | 0.638 |
| tair_control_horizon_abs_bias | secondary_selection | 0.556 | 0.471 .. 0.629 | 0.258 .. 0.643 | 0.657 |
| tair_constraint_near_mae_proxy | weak_selection | 0.280 | 0.132 .. 0.375 | -0.066 .. 0.407 | 0.548 |
| rhair_first_step_mae | offline_or_diagnostic_only | 0.050 | -0.139 .. 0.225 | -0.139 .. 0.225 | 0.448 |
| rhair_control_horizon_mae | offline_or_diagnostic_only | 0.179 | 0.021 .. 0.339 | -0.022 .. 0.339 | 0.495 |
| rhair_weighted_horizon_mae | secondary_selection | 0.618 | 0.550 .. 0.740 | 0.363 .. 0.737 | 0.673 |
| rhair_full_horizon_mae | secondary_selection | 0.609 | 0.550 .. 0.729 | 0.379 .. 0.709 | 0.667 |
| rhair_final_step_mae | primary_selection | 0.680 | 0.626 .. 0.765 | 0.600 .. 0.721 | 0.731 |
| rhair_control_horizon_abs_bias | offline_or_diagnostic_only | -0.079 | -0.232 .. 0.079 | -0.352 .. 0.079 | 0.400 |
| rhair_constraint_near_mae_proxy | primary_selection | 0.788 | 0.743 .. 0.832 | 0.725 .. 0.832 | 0.762 |
| co2_first_step_mae | secondary_selection | 0.471 | 0.361 .. 0.651 | 0.361 .. 0.651 | 0.635 |
| co2_control_horizon_mae | weak_selection | 0.336 | 0.211 .. 0.454 | 0.211 .. 0.616 | 0.558 |
| co2_weighted_horizon_mae | offline_or_diagnostic_only | 0.224 | 0.086 .. 0.407 | 0.086 .. 0.643 | 0.505 |
| co2_full_horizon_mae | offline_or_diagnostic_only | 0.182 | 0.039 .. 0.361 | 0.039 .. 0.615 | 0.495 |
| co2_final_step_mae | offline_or_diagnostic_only | -0.074 | -0.193 .. 0.125 | -0.193 .. 0.363 | 0.429 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | 0.059 | -0.111 .. 0.186 | -0.111 .. 0.434 | 0.457 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.003 | -0.121 .. 0.179 | -0.121 .. 0.346 | 0.467 |
| forecast_only_transfer_rank | secondary_selection | 0.665 | 0.611 .. 0.758 | 0.503 .. 0.806 | 0.705 |
| tair_transfer_selection_score | secondary_selection | 0.618 | 0.529 .. 0.746 | 0.319 .. 0.802 | 0.695 |
| rhair_transfer_selection_score | offline_or_diagnostic_only | 0.210 | 0.063 .. 0.377 | 0.033 .. 0.377 | 0.510 |
| co2_transfer_selection_score | weak_selection | 0.268 | 0.154 .. 0.396 | 0.154 .. 0.577 | 0.543 |
| multiobjective_transfer_selection_score | secondary_selection | 0.597 | 0.514 .. 0.693 | 0.514 .. 0.703 | 0.667 |

### Target: `mpc_co2_mae`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | offline_or_diagnostic_only | -0.094 | -0.207 .. 0.064 | -0.401 .. 0.059 | 0.438 |
| tair_control_horizon_mae | offline_or_diagnostic_only | -0.137 | -0.263 .. 0.009 | -0.360 .. 0.119 | 0.413 |
| tair_weighted_horizon_mae | offline_or_diagnostic_only | 0.012 | -0.132 .. 0.150 | -0.313 .. 0.150 | 0.467 |
| tair_full_horizon_mae | offline_or_diagnostic_only | 0.009 | -0.136 .. 0.146 | -0.319 .. 0.146 | 0.457 |
| tair_final_step_mae | offline_or_diagnostic_only | 0.168 | 0.032 .. 0.314 | -0.099 .. 0.314 | 0.514 |
| tair_control_horizon_abs_bias | offline_or_diagnostic_only | 0.029 | -0.071 .. 0.204 | -0.220 .. 0.222 | 0.486 |
| tair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.099 | -0.048 .. 0.266 | -0.048 .. 0.266 | 0.471 |
| rhair_first_step_mae | offline_or_diagnostic_only | 0.221 | 0.114 .. 0.404 | 0.033 .. 0.404 | 0.562 |
| rhair_control_horizon_mae | weak_selection | 0.306 | 0.207 .. 0.479 | 0.125 .. 0.479 | 0.600 |
| rhair_weighted_horizon_mae | offline_or_diagnostic_only | 0.216 | 0.102 .. 0.381 | -0.047 .. 0.381 | 0.548 |
| rhair_full_horizon_mae | offline_or_diagnostic_only | 0.226 | 0.079 .. 0.354 | -0.044 .. 0.354 | 0.543 |
| rhair_final_step_mae | offline_or_diagnostic_only | 0.007 | -0.181 .. 0.123 | -0.267 .. 0.109 | 0.442 |
| rhair_control_horizon_abs_bias | primary_selection | 0.779 | 0.739 .. 0.864 | 0.742 .. 0.864 | 0.762 |
| rhair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.091 | -0.086 .. 0.207 | -0.247 .. 0.207 | 0.448 |
| co2_first_step_mae | offline_or_diagnostic_only | -0.319 | -0.559 .. -0.227 | -0.476 .. -0.227 | 0.308 |
| co2_control_horizon_mae | offline_or_diagnostic_only | -0.519 | -0.638 .. -0.431 | -0.624 .. -0.431 | 0.269 |
| co2_weighted_horizon_mae | offline_or_diagnostic_only | -0.606 | -0.711 .. -0.532 | -0.711 .. -0.437 | 0.248 |
| co2_full_horizon_mae | offline_or_diagnostic_only | -0.574 | -0.675 .. -0.493 | -0.675 .. -0.389 | 0.257 |
| co2_final_step_mae | offline_or_diagnostic_only | -0.553 | -0.682 .. -0.457 | -0.682 .. -0.451 | 0.257 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | -0.344 | -0.443 .. -0.236 | -0.443 .. -0.156 | 0.343 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.429 | -0.514 .. -0.307 | -0.533 .. -0.191 | 0.324 |
| forecast_only_transfer_rank | offline_or_diagnostic_only | -0.056 | -0.207 .. 0.100 | -0.369 .. 0.121 | 0.413 |
| tair_transfer_selection_score | offline_or_diagnostic_only | 0.029 | -0.086 .. 0.186 | -0.286 .. 0.218 | 0.486 |
| rhair_transfer_selection_score | secondary_selection | 0.472 | 0.370 .. 0.638 | 0.269 .. 0.638 | 0.644 |
| co2_transfer_selection_score | offline_or_diagnostic_only | -0.518 | -0.586 .. -0.400 | -0.586 .. -0.398 | 0.295 |
| multiobjective_transfer_selection_score | offline_or_diagnostic_only | -0.021 | -0.207 .. 0.132 | -0.379 .. 0.132 | 0.438 |

### Target: `mpc_objective`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | offline_or_diagnostic_only | 0.121 | -0.068 .. 0.218 | -0.068 .. 0.218 | 0.505 |
| tair_control_horizon_mae | weak_selection | 0.286 | 0.132 .. 0.397 | 0.132 .. 0.397 | 0.548 |
| tair_weighted_horizon_mae | offline_or_diagnostic_only | 0.118 | -0.071 .. 0.200 | -0.071 .. 0.200 | 0.476 |
| tair_full_horizon_mae | offline_or_diagnostic_only | 0.112 | -0.079 .. 0.193 | -0.079 .. 0.193 | 0.467 |
| tair_final_step_mae | offline_or_diagnostic_only | 0.235 | 0.071 .. 0.375 | 0.071 .. 0.375 | 0.514 |
| tair_control_horizon_abs_bias | objective_secondary_selection | 0.426 | 0.304 .. 0.518 | 0.304 .. 0.549 | 0.638 |
| tair_constraint_near_mae_proxy | objective_secondary_selection | 0.389 | 0.257 .. 0.561 | 0.257 .. 0.561 | 0.625 |
| rhair_first_step_mae | weak_selection | 0.306 | 0.157 .. 0.436 | 0.157 .. 0.436 | 0.562 |
| rhair_control_horizon_mae | offline_or_diagnostic_only | 0.118 | -0.071 .. 0.250 | -0.071 .. 0.218 | 0.495 |
| rhair_weighted_horizon_mae | offline_or_diagnostic_only | 0.038 | -0.168 .. 0.150 | -0.168 .. 0.134 | 0.423 |
| rhair_full_horizon_mae | offline_or_diagnostic_only | -0.029 | -0.204 .. 0.118 | -0.204 .. 0.071 | 0.419 |
| rhair_final_step_mae | offline_or_diagnostic_only | -0.038 | -0.261 .. 0.100 | -0.261 .. 0.132 | 0.423 |
| rhair_control_horizon_abs_bias | offline_or_diagnostic_only | -0.026 | -0.129 .. 0.154 | -0.129 .. 0.154 | 0.476 |
| rhair_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.188 | -0.396 .. -0.061 | -0.396 .. -0.107 | 0.362 |
| co2_first_step_mae | offline_or_diagnostic_only | 0.130 | -0.057 .. 0.264 | -0.116 .. 0.264 | 0.500 |
| co2_control_horizon_mae | offline_or_diagnostic_only | 0.038 | -0.168 .. 0.168 | -0.168 .. 0.176 | 0.442 |
| co2_weighted_horizon_mae | offline_or_diagnostic_only | 0.059 | -0.143 .. 0.136 | -0.143 .. 0.181 | 0.429 |
| co2_full_horizon_mae | offline_or_diagnostic_only | 0.053 | -0.150 .. 0.125 | -0.150 .. 0.170 | 0.438 |
| co2_final_step_mae | offline_or_diagnostic_only | -0.082 | -0.271 .. 0.068 | -0.271 .. 0.043 | 0.429 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | -0.509 | -0.575 .. -0.429 | -0.587 .. -0.429 | 0.314 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.379 | -0.607 .. -0.304 | -0.607 .. -0.275 | 0.267 |
| forecast_only_transfer_rank | offline_or_diagnostic_only | 0.021 | -0.146 .. 0.154 | -0.146 .. 0.125 | 0.419 |
| tair_transfer_selection_score | weak_selection | 0.326 | 0.182 .. 0.439 | 0.182 .. 0.439 | 0.552 |
| rhair_transfer_selection_score | offline_or_diagnostic_only | 0.212 | 0.050 .. 0.400 | 0.050 .. 0.340 | 0.538 |
| co2_transfer_selection_score | offline_or_diagnostic_only | -0.159 | -0.259 .. -0.002 | -0.286 .. -0.002 | 0.423 |
| multiobjective_transfer_selection_score | offline_or_diagnostic_only | 0.141 | -0.054 .. 0.206 | -0.043 .. 0.206 | 0.457 |
