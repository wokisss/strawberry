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
| mpc_tair_mae | rhair_first_step_mae | weak_selection |
| mpc_tair_mae | rhair_control_horizon_mae | offline_or_diagnostic_only |
| mpc_tair_mae | rhair_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_tair_mae | rhair_full_horizon_mae | offline_or_diagnostic_only |
| mpc_tair_mae | rhair_final_step_mae | offline_or_diagnostic_only |
| mpc_tair_mae | rhair_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_tair_mae | rhair_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_tair_mae | co2_first_step_mae | offline_or_diagnostic_only |
| mpc_tair_mae | co2_control_horizon_mae | offline_or_diagnostic_only |
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
| mpc_rhair_mae | tair_first_step_mae | secondary_selection |
| mpc_rhair_mae | tair_control_horizon_mae | secondary_selection |
| mpc_rhair_mae | tair_weighted_horizon_mae | secondary_selection |
| mpc_rhair_mae | tair_full_horizon_mae | secondary_selection |
| mpc_rhair_mae | tair_final_step_mae | secondary_selection |
| mpc_rhair_mae | tair_control_horizon_abs_bias | secondary_selection |
| mpc_rhair_mae | tair_constraint_near_mae_proxy | secondary_selection |
| mpc_rhair_mae | rhair_first_step_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_control_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_weighted_horizon_mae | secondary_selection |
| mpc_rhair_mae | rhair_full_horizon_mae | secondary_selection |
| mpc_rhair_mae | rhair_final_step_mae | secondary_selection |
| mpc_rhair_mae | rhair_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_constraint_near_mae_proxy | secondary_selection |
| mpc_rhair_mae | co2_first_step_mae | secondary_selection |
| mpc_rhair_mae | co2_control_horizon_mae | weak_selection |
| mpc_rhair_mae | co2_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_full_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_final_step_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_rhair_mae | forecast_only_transfer_rank | secondary_selection |
| mpc_rhair_mae | tair_transfer_selection_score | secondary_selection |
| mpc_rhair_mae | rhair_transfer_selection_score | weak_selection |
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
| mpc_co2_mae | rhair_first_step_mae | secondary_selection |
| mpc_co2_mae | rhair_control_horizon_mae | secondary_selection |
| mpc_co2_mae | rhair_weighted_horizon_mae | secondary_selection |
| mpc_co2_mae | rhair_full_horizon_mae | secondary_selection |
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
| mpc_co2_mae | multiobjective_transfer_selection_score | weak_selection |
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
| mpc_objective | tair_final_step_mae | weak_selection |
| mpc_objective | tair_control_horizon_abs_bias | weak_selection |
| mpc_objective | tair_constraint_near_mae_proxy | weak_selection |
| mpc_objective | rhair_first_step_mae | objective_secondary_selection |
| mpc_objective | rhair_control_horizon_mae | objective_secondary_selection |
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
| mpc_objective | tair_transfer_selection_score | offline_or_diagnostic_only |
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
| 1 | current_hybrid_transformer | 5.167 | 5.688 | 4.250 | 5.562 | 6.722 | 0.525 | 1.486 | 24.360 | 0.0517 |
| 2 | itransformer_co2_control_aware_fusion | 5.406 | 7.094 | 7.625 | 1.500 | 8.556 | 1.072 | 1.179 | 26.154 | 0.1072 |
| 3 | itransformer_residual | 5.583 | 4.062 | 5.688 | 7.000 | 9.167 | 2.392 | 4.199 | 18.052 | 0.1666 |
| 4 | itransformer_co2_late_frozen_expert | 5.740 | 6.531 | 8.625 | 2.062 | 9.944 | 1.122 | 1.282 | 25.366 | 0.1133 |
| 5 | transformer_hybrid_residual | 6.396 | 4.500 | 2.875 | 11.812 | 9.167 | 0.873 | 1.988 | 23.098 | 0.0666 |
| 6 | itransformer_co2_late_residual | 6.604 | 7.812 | 6.000 | 6.000 | 9.000 | 1.135 | 1.230 | 36.866 | 0.1007 |
| 7 | segrnn_forecaster | 7.979 | 14.188 | 6.000 | 3.750 | 9.389 | 0.340 | 3.209 | 16.886 | 0.0738 |
| 8 | dlinear_forecaster | 8.583 | 9.938 | 5.500 | 10.312 | 15.056 | 1.018 | 2.195 | 16.177 | 0.0977 |
| 9 | itransformer_co2_horizon_mixture | 8.646 | 2.875 | 12.125 | 10.938 | 13.722 | 1.305 | 1.713 | 41.012 | 0.1234 |
| 10 | transformer_forecaster | 9.250 | 6.188 | 11.125 | 10.438 | 13.056 | 0.306 | 1.899 | 23.676 | 0.0589 |
| 11 | itransformer_co2_residual | 9.312 | 6.062 | 9.938 | 11.938 | 10.778 | 0.636 | 0.884 | 11.311 | 0.0683 |
| 12 | gru_forecaster | 10.375 | 10.438 | 14.625 | 6.062 | 14.278 | 0.707 | 5.867 | 63.234 | 0.1095 |
| 13 | nlinear_forecaster | 10.729 | 13.125 | 4.812 | 14.250 | 15.500 | 1.025 | 2.695 | 18.772 | 0.0958 |
| 14 | patchtst_residual | 10.938 | 8.688 | 11.875 | 12.250 | 14.833 | 0.851 | 2.598 | 30.908 | 0.0952 |
| 15 | lstm_forecaster | 11.604 | 12.812 | 11.750 | 10.250 | 15.111 | 1.539 | 9.494 | 43.074 | 0.1654 |
| 16 | frequency_forecaster | 13.688 | 16.000 | 13.188 | 11.875 | 18.722 | 1.158 | 6.495 | 31.750 | 0.3964 |

## Metric Transfer Quality

### Target: `mpc_tair_mae`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rhair_first_step_mae | selection | 0.135 | 0.282 | 0.592 | no | no | 0.667 |
| tair_constraint_near_mae_proxy | selection | -0.079 | 0.169 | 0.571 | no | yes | 0.333 |
| co2_first_step_mae | selection | 0.122 | 0.208 | 0.563 | no | no | 0.000 |
| co2_control_horizon_mae | selection | 0.107 | 0.137 | 0.546 | no | no | 0.333 |
| rhair_transfer_selection_score | selection | -0.041 | 0.135 | 0.546 | no | no | 0.333 |
| co2_weighted_horizon_mae | selection | -0.012 | 0.076 | 0.542 | no | no | 0.667 |
| tair_control_horizon_abs_bias | selection | -0.117 | 0.059 | 0.542 | no | no | 0.333 |
| rhair_control_horizon_mae | selection | 0.062 | 0.103 | 0.533 | no | no | 0.000 |
| multiobjective_transfer_selection_score | selection | -0.095 | 0.053 | 0.533 | no | no | 0.333 |
| rhair_control_horizon_abs_bias | selection | -0.144 | -0.044 | 0.500 | no | no | 0.000 |
| co2_transfer_selection_score | selection | 0.012 | -0.015 | 0.500 | no | no | 0.333 |
| tair_final_step_mae | selection | -0.021 | -0.012 | 0.500 | no | no | 0.000 |
| co2_full_horizon_mae | selection | -0.033 | 0.006 | 0.500 | no | no | 0.667 |
| co2_final_step_mae | selection | -0.101 | -0.126 | 0.492 | no | no | 0.333 |
| tair_control_horizon_mae | selection | 0.008 | -0.090 | 0.487 | no | no | 0.000 |
| tair_transfer_selection_score | selection | -0.160 | -0.056 | 0.483 | no | no | 0.000 |
| rhair_weighted_horizon_mae | selection | -0.141 | -0.107 | 0.479 | no | no | 0.333 |
| rhair_full_horizon_mae | selection | -0.161 | -0.156 | 0.475 | no | no | 0.333 |
| tair_first_step_mae | selection | 0.007 | -0.091 | 0.475 | no | no | 0.000 |
| co2_constraint_near_mae_proxy | selection | -0.115 | -0.174 | 0.450 | no | no | 0.667 |
| rhair_final_step_mae | selection | -0.217 | -0.222 | 0.445 | no | no | 0.333 |
| tair_full_horizon_mae | selection | -0.050 | -0.206 | 0.442 | no | no | 0.000 |
| forecast_only_transfer_rank | selection | -0.224 | -0.156 | 0.442 | no | no | 0.333 |
| tair_weighted_horizon_mae | selection | -0.047 | -0.209 | 0.433 | no | no | 0.000 |
| co2_control_horizon_abs_bias | selection | -0.210 | -0.321 | 0.408 | no | no | 0.333 |
| rhair_constraint_near_mae_proxy | selection | -0.239 | -0.297 | 0.392 | no | no | 0.000 |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.430 | -0.542 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.429 | 0.532 |  |  |  |  |
| assim_sp_first_grad | diagnostic | 0.499 | 0.493 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | 0.411 | 0.472 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.478 | -0.440 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | -0.265 | -0.437 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | 0.298 | 0.390 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | 0.340 | 0.390 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | 0.285 | 0.349 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | 0.317 | 0.328 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | 0.122 | 0.272 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | 0.290 | 0.178 |  |  |  |  |
| co2_sp_first_grad | diagnostic | 0.225 | 0.166 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | -0.082 | 0.138 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | -0.059 | -0.116 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | -0.056 | -0.063 |  |  |  |  |

### Target: `mpc_rhair_mae`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | selection | 0.491 | 0.629 | 0.717 | no | no | 0.000 |
| forecast_only_transfer_rank | selection | 0.590 | 0.568 | 0.717 | no | no | 0.333 |
| rhair_constraint_near_mae_proxy | selection | 0.594 | 0.582 | 0.700 | no | no | 0.333 |
| tair_full_horizon_mae | selection | 0.668 | 0.526 | 0.700 | no | no | 0.000 |
| multiobjective_transfer_selection_score | selection | 0.621 | 0.553 | 0.692 | no | no | 0.333 |
| tair_transfer_selection_score | selection | 0.600 | 0.550 | 0.692 | no | no | 0.000 |
| tair_weighted_horizon_mae | selection | 0.657 | 0.521 | 0.692 | no | no | 0.000 |
| rhair_weighted_horizon_mae | selection | 0.623 | 0.545 | 0.689 | no | no | 0.000 |
| rhair_final_step_mae | selection | 0.512 | 0.521 | 0.689 | no | no | 0.000 |
| rhair_full_horizon_mae | selection | 0.618 | 0.547 | 0.683 | no | no | 0.000 |
| co2_first_step_mae | selection | 0.552 | 0.483 | 0.672 | no | no | 0.667 |
| tair_control_horizon_abs_bias | selection | 0.496 | 0.476 | 0.667 | no | no | 0.000 |
| tair_control_horizon_mae | selection | 0.520 | 0.462 | 0.647 | no | no | 0.000 |
| tair_final_step_mae | selection | 0.713 | 0.424 | 0.642 | no | no | 0.000 |
| tair_constraint_near_mae_proxy | selection | 0.504 | 0.374 | 0.630 | no | yes | 0.667 |
| co2_control_horizon_mae | selection | 0.378 | 0.286 | 0.622 | no | no | 0.333 |
| rhair_control_horizon_mae | selection | 0.479 | 0.238 | 0.592 | no | no | 0.333 |
| co2_transfer_selection_score | selection | 0.181 | 0.224 | 0.592 | no | no | 0.333 |
| rhair_transfer_selection_score | selection | 0.469 | 0.258 | 0.580 | no | no | 0.000 |
| co2_weighted_horizon_mae | selection | 0.068 | 0.147 | 0.567 | no | no | 0.333 |
| co2_full_horizon_mae | selection | -0.017 | 0.085 | 0.558 | no | no | 0.333 |
| rhair_first_step_mae | selection | 0.412 | 0.153 | 0.533 | no | no | 0.000 |
| co2_control_horizon_abs_bias | selection | -0.201 | 0.024 | 0.533 | no | no | 0.333 |
| rhair_control_horizon_abs_bias | selection | 0.151 | -0.041 | 0.492 | no | yes | 0.333 |
| co2_final_step_mae | selection | -0.247 | -0.085 | 0.483 | no | no | 0.333 |
| co2_constraint_near_mae_proxy | selection | -0.259 | -0.094 | 0.458 | no | no | 0.333 |
| rhair_dx_sp_first_grad | diagnostic | -0.684 | -0.792 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | -0.630 | -0.703 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | -0.712 | -0.698 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | -0.559 | -0.603 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | -0.631 | -0.547 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | 0.549 | 0.538 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | -0.622 | -0.447 |  |  |  |  |
| assim_sp_first_grad | diagnostic | -0.457 | -0.309 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | -0.606 | -0.274 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | -0.246 | -0.265 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | -0.525 | -0.262 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.368 | -0.197 |  |  |  |  |
| co2_sp_first_grad | diagnostic | -0.395 | -0.159 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.259 | -0.144 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.259 | 0.125 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | -0.574 | -0.103 |  |  |  |  |

### Target: `mpc_co2_mae`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rhair_control_horizon_abs_bias | selection | 0.646 | 0.591 | 0.717 | no | yes | 0.333 |
| rhair_transfer_selection_score | selection | 0.639 | 0.636 | 0.714 | no | no | 0.000 |
| rhair_first_step_mae | selection | 0.331 | 0.506 | 0.692 | no | no | 0.333 |
| rhair_control_horizon_mae | selection | 0.467 | 0.506 | 0.683 | no | no | 0.000 |
| rhair_full_horizon_mae | selection | 0.561 | 0.350 | 0.642 | no | no | 0.333 |
| rhair_weighted_horizon_mae | selection | 0.570 | 0.352 | 0.639 | no | no | 0.333 |
| forecast_only_transfer_rank | selection | 0.257 | 0.232 | 0.592 | no | no | 0.000 |
| multiobjective_transfer_selection_score | selection | 0.315 | 0.294 | 0.583 | no | no | 0.000 |
| tair_final_step_mae | selection | 0.157 | 0.215 | 0.583 | no | no | 0.000 |
| tair_transfer_selection_score | selection | 0.122 | 0.144 | 0.583 | no | no | 0.000 |
| rhair_constraint_near_mae_proxy | selection | 0.520 | 0.226 | 0.558 | no | no | 0.333 |
| tair_control_horizon_abs_bias | selection | 0.071 | 0.165 | 0.558 | no | no | 0.000 |
| tair_weighted_horizon_mae | selection | 0.099 | 0.150 | 0.550 | no | no | 0.000 |
| tair_constraint_near_mae_proxy | selection | -0.008 | 0.143 | 0.546 | no | yes | 0.333 |
| tair_full_horizon_mae | selection | 0.109 | 0.147 | 0.542 | no | no | 0.000 |
| rhair_final_step_mae | selection | 0.392 | 0.104 | 0.538 | no | no | 0.000 |
| tair_control_horizon_mae | selection | 0.019 | -0.001 | 0.529 | no | no | 0.000 |
| tair_first_step_mae | selection | 0.104 | 0.071 | 0.525 | no | no | 0.000 |
| co2_first_step_mae | selection | 0.075 | 0.037 | 0.504 | no | no | 0.000 |
| co2_control_horizon_mae | selection | -0.045 | -0.099 | 0.454 | no | no | 0.333 |
| co2_control_horizon_abs_bias | selection | -0.074 | -0.132 | 0.442 | no | no | 0.333 |
| co2_full_horizon_mae | selection | -0.184 | -0.179 | 0.433 | no | no | 0.333 |
| co2_transfer_selection_score | selection | -0.097 | -0.144 | 0.433 | no | no | 0.333 |
| co2_final_step_mae | selection | -0.232 | -0.241 | 0.425 | no | no | 0.333 |
| co2_weighted_horizon_mae | selection | -0.161 | -0.221 | 0.425 | no | no | 0.333 |
| co2_constraint_near_mae_proxy | selection | -0.201 | -0.309 | 0.417 | no | no | 0.333 |
| co2_sp_first_grad | diagnostic | -0.666 | -0.567 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | -0.521 | -0.467 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | -0.528 | -0.378 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | 0.613 | 0.376 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.408 | -0.281 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | -0.495 | -0.243 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | -0.298 | -0.202 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | -0.622 | -0.196 |  |  |  |  |
| assim_sp_first_grad | diagnostic | -0.384 | -0.172 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | -0.249 | -0.155 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | -0.382 | -0.125 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | -0.221 | -0.104 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | -0.512 | -0.093 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.123 | -0.085 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.122 | 0.063 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | -0.153 | -0.043 |  |  |  |  |

### Target: `mpc_objective`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rhair_first_step_mae | selection | 0.861 | 0.547 | 0.708 | no | yes | 0.667 |
| rhair_transfer_selection_score | selection | 0.421 | 0.477 | 0.681 | no | yes | 0.667 |
| rhair_control_horizon_mae | selection | 0.784 | 0.444 | 0.667 | no | no | 0.333 |
| tair_control_horizon_abs_bias | selection | 0.625 | 0.279 | 0.625 | no | yes | 0.333 |
| tair_constraint_near_mae_proxy | selection | 0.641 | 0.315 | 0.622 | no | no | 0.333 |
| tair_final_step_mae | selection | 0.671 | 0.262 | 0.600 | no | no | 0.333 |
| multiobjective_transfer_selection_score | selection | 0.539 | 0.235 | 0.583 | yes | yes | 0.333 |
| rhair_weighted_horizon_mae | selection | 0.589 | 0.247 | 0.580 | no | yes | 0.667 |
| rhair_full_horizon_mae | selection | 0.558 | 0.209 | 0.575 | no | yes | 0.667 |
| tair_transfer_selection_score | selection | 0.491 | 0.200 | 0.567 | no | no | 0.333 |
| co2_first_step_mae | selection | 0.837 | 0.144 | 0.563 | no | no | 0.000 |
| tair_first_step_mae | selection | 0.831 | 0.147 | 0.558 | no | no | 0.333 |
| rhair_control_horizon_abs_bias | selection | -0.237 | 0.062 | 0.550 | no | no | 0.000 |
| rhair_final_step_mae | selection | 0.452 | 0.127 | 0.546 | yes | yes | 0.667 |
| forecast_only_transfer_rank | selection | 0.523 | 0.115 | 0.542 | no | yes | 0.333 |
| co2_weighted_horizon_mae | selection | 0.437 | 0.038 | 0.525 | no | yes | 0.333 |
| tair_control_horizon_mae | selection | 0.832 | 0.068 | 0.521 | no | no | 0.333 |
| co2_full_horizon_mae | selection | 0.382 | 0.018 | 0.517 | no | yes | 0.333 |
| co2_control_horizon_mae | selection | 0.685 | 0.009 | 0.513 | no | no | 0.000 |
| tair_full_horizon_mae | selection | 0.729 | 0.032 | 0.508 | no | no | 0.333 |
| tair_weighted_horizon_mae | selection | 0.742 | 0.026 | 0.500 | no | no | 0.333 |
| co2_final_step_mae | selection | 0.114 | -0.088 | 0.492 | no | no | 0.000 |
| rhair_constraint_near_mae_proxy | selection | 0.344 | -0.009 | 0.492 | no | no | 0.000 |
| co2_transfer_selection_score | selection | 0.173 | -0.112 | 0.483 | no | no | 0.000 |
| co2_constraint_near_mae_proxy | selection | -0.260 | -0.268 | 0.400 | no | yes | 0.333 |
| co2_control_horizon_abs_bias | selection | -0.259 | -0.515 | 0.325 | no | no | 0.000 |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.368 | -0.707 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.367 | 0.689 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | -0.289 | -0.621 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.371 | -0.618 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | 0.268 | 0.438 |  |  |  |  |
| assim_sp_first_grad | diagnostic | 0.171 | 0.371 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | -0.050 | 0.271 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | 0.132 | 0.265 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | 0.076 | -0.177 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | 0.081 | 0.159 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | -0.098 | 0.159 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | -0.136 | 0.150 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | 0.432 | -0.127 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | -0.137 | 0.097 |  |  |  |  |
| co2_sp_first_grad | diagnostic | 0.196 | 0.088 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | -0.261 | -0.006 |  |  |  |  |

## Robustness Summary

### Target: `mpc_tair_mae`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | offline_or_diagnostic_only | -0.091 | -0.221 .. 0.046 | -0.221 .. 0.046 | 0.429 |
| tair_control_horizon_mae | offline_or_diagnostic_only | -0.090 | -0.209 .. 0.098 | -0.209 .. 0.059 | 0.442 |
| tair_weighted_horizon_mae | offline_or_diagnostic_only | -0.209 | -0.364 .. -0.046 | -0.357 .. -0.088 | 0.381 |
| tair_full_horizon_mae | offline_or_diagnostic_only | -0.206 | -0.361 .. -0.043 | -0.354 .. -0.082 | 0.390 |
| tair_final_step_mae | offline_or_diagnostic_only | -0.012 | -0.186 .. 0.089 | -0.143 .. 0.089 | 0.448 |
| tair_control_horizon_abs_bias | offline_or_diagnostic_only | 0.059 | -0.039 .. 0.286 | -0.039 .. 0.253 | 0.505 |
| tair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.169 | 0.041 .. 0.334 | 0.041 .. 0.291 | 0.529 |
| rhair_first_step_mae | weak_selection | 0.282 | 0.171 .. 0.414 | 0.171 .. 0.505 | 0.552 |
| rhair_control_horizon_mae | offline_or_diagnostic_only | 0.103 | -0.032 .. 0.200 | -0.032 .. 0.225 | 0.495 |
| rhair_weighted_horizon_mae | offline_or_diagnostic_only | -0.107 | -0.266 .. 0.016 | -0.266 .. 0.008 | 0.433 |
| rhair_full_horizon_mae | offline_or_diagnostic_only | -0.156 | -0.293 .. -0.039 | -0.293 .. 0.033 | 0.438 |
| rhair_final_step_mae | offline_or_diagnostic_only | -0.222 | -0.399 .. -0.120 | -0.399 .. -0.014 | 0.385 |
| rhair_control_horizon_abs_bias | offline_or_diagnostic_only | -0.044 | -0.168 .. 0.161 | -0.168 .. 0.161 | 0.457 |
| rhair_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.297 | -0.443 .. -0.204 | -0.443 .. -0.204 | 0.343 |
| co2_first_step_mae | offline_or_diagnostic_only | 0.208 | 0.077 .. 0.289 | 0.003 .. 0.289 | 0.519 |
| co2_control_horizon_mae | offline_or_diagnostic_only | 0.137 | 0.023 .. 0.218 | -0.063 .. 0.294 | 0.510 |
| co2_weighted_horizon_mae | offline_or_diagnostic_only | 0.076 | -0.082 .. 0.200 | -0.126 .. 0.225 | 0.486 |
| co2_full_horizon_mae | offline_or_diagnostic_only | 0.006 | -0.168 .. 0.114 | -0.170 .. 0.192 | 0.438 |
| co2_final_step_mae | offline_or_diagnostic_only | -0.126 | -0.289 .. 0.046 | -0.211 .. -0.011 | 0.438 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | -0.321 | -0.500 .. -0.243 | -0.478 .. -0.243 | 0.352 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.174 | -0.379 .. -0.032 | -0.368 .. -0.032 | 0.381 |
| forecast_only_transfer_rank | offline_or_diagnostic_only | -0.156 | -0.293 .. 0.025 | -0.275 .. 0.050 | 0.394 |
| tair_transfer_selection_score | offline_or_diagnostic_only | -0.056 | -0.168 .. 0.139 | -0.168 .. 0.100 | 0.438 |
| rhair_transfer_selection_score | offline_or_diagnostic_only | 0.135 | -0.025 .. 0.247 | -0.025 .. 0.358 | 0.495 |
| co2_transfer_selection_score | offline_or_diagnostic_only | -0.015 | -0.139 .. 0.152 | -0.187 .. 0.100 | 0.457 |
| multiobjective_transfer_selection_score | offline_or_diagnostic_only | 0.053 | -0.129 .. 0.211 | -0.137 .. 0.225 | 0.467 |

### Target: `mpc_rhair_mae`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | secondary_selection | 0.629 | 0.564 .. 0.700 | 0.445 .. 0.693 | 0.686 |
| tair_control_horizon_mae | secondary_selection | 0.462 | 0.357 .. 0.629 | 0.292 .. 0.660 | 0.606 |
| tair_weighted_horizon_mae | secondary_selection | 0.521 | 0.425 .. 0.682 | 0.220 .. 0.830 | 0.657 |
| tair_full_horizon_mae | secondary_selection | 0.526 | 0.432 .. 0.689 | 0.231 .. 0.841 | 0.667 |
| tair_final_step_mae | secondary_selection | 0.424 | 0.304 .. 0.507 | 0.027 .. 0.637 | 0.600 |
| tair_control_horizon_abs_bias | secondary_selection | 0.476 | 0.375 .. 0.657 | 0.176 .. 0.632 | 0.629 |
| tair_constraint_near_mae_proxy | secondary_selection | 0.374 | 0.247 .. 0.471 | 0.182 .. 0.506 | 0.587 |
| rhair_first_step_mae | offline_or_diagnostic_only | 0.153 | -0.014 .. 0.286 | -0.014 .. 0.286 | 0.476 |
| rhair_control_horizon_mae | offline_or_diagnostic_only | 0.238 | 0.093 .. 0.357 | 0.027 .. 0.357 | 0.543 |
| rhair_weighted_horizon_mae | secondary_selection | 0.545 | 0.461 .. 0.633 | 0.259 .. 0.633 | 0.654 |
| rhair_full_horizon_mae | secondary_selection | 0.547 | 0.471 .. 0.607 | 0.319 .. 0.643 | 0.657 |
| rhair_final_step_mae | secondary_selection | 0.521 | 0.447 .. 0.622 | 0.374 .. 0.622 | 0.663 |
| rhair_control_horizon_abs_bias | offline_or_diagnostic_only | -0.041 | -0.189 .. 0.107 | -0.319 .. 0.107 | 0.438 |
| rhair_constraint_near_mae_proxy | secondary_selection | 0.582 | 0.514 .. 0.636 | 0.407 .. 0.636 | 0.676 |
| co2_first_step_mae | secondary_selection | 0.483 | 0.375 .. 0.611 | 0.375 .. 0.611 | 0.635 |
| co2_control_horizon_mae | weak_selection | 0.286 | 0.150 .. 0.475 | 0.150 .. 0.528 | 0.577 |
| co2_weighted_horizon_mae | offline_or_diagnostic_only | 0.147 | -0.007 .. 0.307 | -0.007 .. 0.462 | 0.514 |
| co2_full_horizon_mae | offline_or_diagnostic_only | 0.085 | -0.079 .. 0.232 | -0.079 .. 0.412 | 0.505 |
| co2_final_step_mae | offline_or_diagnostic_only | -0.085 | -0.211 .. 0.111 | -0.211 .. 0.330 | 0.438 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | 0.024 | -0.164 .. 0.136 | -0.164 .. 0.341 | 0.476 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.094 | -0.239 .. 0.100 | -0.239 .. 0.115 | 0.400 |
| forecast_only_transfer_rank | secondary_selection | 0.568 | 0.463 .. 0.761 | 0.303 .. 0.787 | 0.680 |
| tair_transfer_selection_score | secondary_selection | 0.550 | 0.457 .. 0.721 | 0.269 .. 0.720 | 0.657 |
| rhair_transfer_selection_score | weak_selection | 0.258 | 0.116 .. 0.363 | 0.044 .. 0.363 | 0.538 |
| co2_transfer_selection_score | offline_or_diagnostic_only | 0.224 | 0.104 .. 0.393 | 0.104 .. 0.478 | 0.543 |
| multiobjective_transfer_selection_score | secondary_selection | 0.553 | 0.461 .. 0.683 | 0.379 .. 0.720 | 0.657 |

### Target: `mpc_co2_mae`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | offline_or_diagnostic_only | 0.071 | -0.029 .. 0.218 | -0.066 .. 0.218 | 0.486 |
| tair_control_horizon_mae | offline_or_diagnostic_only | -0.001 | -0.105 .. 0.145 | -0.151 .. 0.202 | 0.500 |
| tair_weighted_horizon_mae | offline_or_diagnostic_only | 0.150 | 0.036 .. 0.311 | 0.022 .. 0.311 | 0.514 |
| tair_full_horizon_mae | offline_or_diagnostic_only | 0.147 | 0.032 .. 0.307 | 0.016 .. 0.307 | 0.505 |
| tair_final_step_mae | offline_or_diagnostic_only | 0.215 | 0.071 .. 0.346 | 0.049 .. 0.346 | 0.543 |
| tair_control_horizon_abs_bias | offline_or_diagnostic_only | 0.165 | 0.075 .. 0.311 | 0.089 .. 0.301 | 0.524 |
| tair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.143 | 0.002 .. 0.291 | 0.002 .. 0.291 | 0.490 |
| rhair_first_step_mae | secondary_selection | 0.506 | 0.421 .. 0.689 | 0.319 .. 0.689 | 0.667 |
| rhair_control_horizon_mae | secondary_selection | 0.506 | 0.414 .. 0.654 | 0.401 .. 0.654 | 0.648 |
| rhair_weighted_horizon_mae | secondary_selection | 0.352 | 0.227 .. 0.492 | 0.278 .. 0.434 | 0.596 |
| rhair_full_horizon_mae | secondary_selection | 0.350 | 0.211 .. 0.486 | 0.279 .. 0.418 | 0.590 |
| rhair_final_step_mae | offline_or_diagnostic_only | 0.104 | -0.088 .. 0.238 | -0.025 .. 0.238 | 0.471 |
| rhair_control_horizon_abs_bias | secondary_selection | 0.591 | 0.543 .. 0.732 | 0.500 .. 0.732 | 0.695 |
| rhair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.226 | 0.061 .. 0.400 | 0.033 .. 0.400 | 0.495 |
| co2_first_step_mae | offline_or_diagnostic_only | 0.037 | -0.116 .. 0.202 | -0.084 .. 0.130 | 0.452 |
| co2_control_horizon_mae | offline_or_diagnostic_only | -0.099 | -0.216 .. 0.041 | -0.216 .. 0.041 | 0.413 |
| co2_weighted_horizon_mae | offline_or_diagnostic_only | -0.221 | -0.393 .. -0.107 | -0.346 .. -0.042 | 0.362 |
| co2_full_horizon_mae | offline_or_diagnostic_only | -0.179 | -0.343 .. -0.057 | -0.293 .. 0.011 | 0.371 |
| co2_final_step_mae | offline_or_diagnostic_only | -0.241 | -0.386 .. -0.104 | -0.346 .. -0.112 | 0.371 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | -0.132 | -0.268 .. -0.054 | -0.209 .. 0.037 | 0.400 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.309 | -0.486 .. -0.161 | -0.577 .. -0.064 | 0.352 |
| forecast_only_transfer_rank | offline_or_diagnostic_only | 0.232 | 0.100 .. 0.350 | 0.022 .. 0.420 | 0.538 |
| tair_transfer_selection_score | offline_or_diagnostic_only | 0.144 | 0.043 .. 0.336 | 0.022 .. 0.314 | 0.552 |
| rhair_transfer_selection_score | secondary_selection | 0.636 | 0.550 .. 0.758 | 0.500 .. 0.758 | 0.673 |
| co2_transfer_selection_score | offline_or_diagnostic_only | -0.144 | -0.268 .. 0.021 | -0.203 .. -0.009 | 0.390 |
| multiobjective_transfer_selection_score | weak_selection | 0.294 | 0.146 .. 0.375 | 0.055 .. 0.385 | 0.543 |

### Target: `mpc_objective`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | offline_or_diagnostic_only | 0.147 | -0.036 .. 0.282 | -0.036 .. 0.282 | 0.495 |
| tair_control_horizon_mae | offline_or_diagnostic_only | 0.068 | -0.132 .. 0.236 | -0.132 .. 0.197 | 0.452 |
| tair_weighted_horizon_mae | offline_or_diagnostic_only | 0.026 | -0.182 .. 0.186 | -0.182 .. 0.143 | 0.429 |
| tair_full_horizon_mae | offline_or_diagnostic_only | 0.032 | -0.175 .. 0.193 | -0.175 .. 0.150 | 0.438 |
| tair_final_step_mae | weak_selection | 0.262 | 0.104 .. 0.418 | 0.104 .. 0.418 | 0.543 |
| tair_control_horizon_abs_bias | weak_selection | 0.279 | 0.125 .. 0.500 | 0.125 .. 0.456 | 0.571 |
| tair_constraint_near_mae_proxy | weak_selection | 0.315 | 0.168 .. 0.483 | 0.168 .. 0.483 | 0.567 |
| rhair_first_step_mae | objective_secondary_selection | 0.547 | 0.450 .. 0.675 | 0.450 .. 0.692 | 0.667 |
| rhair_control_horizon_mae | objective_secondary_selection | 0.444 | 0.325 .. 0.532 | 0.325 .. 0.532 | 0.619 |
| rhair_weighted_horizon_mae | offline_or_diagnostic_only | 0.247 | 0.086 .. 0.318 | 0.086 .. 0.323 | 0.519 |
| rhair_full_horizon_mae | offline_or_diagnostic_only | 0.209 | 0.057 .. 0.314 | 0.057 .. 0.314 | 0.524 |
| rhair_final_step_mae | offline_or_diagnostic_only | 0.127 | -0.061 .. 0.225 | -0.061 .. 0.225 | 0.481 |
| rhair_control_horizon_abs_bias | offline_or_diagnostic_only | 0.062 | -0.004 .. 0.257 | -0.099 .. 0.257 | 0.524 |
| rhair_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.009 | -0.207 .. 0.096 | -0.207 .. 0.096 | 0.429 |
| co2_first_step_mae | offline_or_diagnostic_only | 0.144 | -0.039 .. 0.218 | -0.039 .. 0.244 | 0.500 |
| co2_control_horizon_mae | offline_or_diagnostic_only | 0.009 | -0.204 .. 0.089 | -0.204 .. 0.154 | 0.442 |
| co2_weighted_horizon_mae | offline_or_diagnostic_only | 0.038 | -0.168 .. 0.125 | -0.168 .. 0.209 | 0.457 |
| co2_full_horizon_mae | offline_or_diagnostic_only | 0.018 | -0.193 .. 0.096 | -0.193 .. 0.203 | 0.448 |
| co2_final_step_mae | offline_or_diagnostic_only | -0.088 | -0.275 .. 0.046 | -0.275 .. 0.007 | 0.438 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | -0.515 | -0.654 .. -0.468 | -0.604 .. -0.456 | 0.286 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.268 | -0.493 .. -0.175 | -0.493 .. -0.175 | 0.333 |
| forecast_only_transfer_rank | offline_or_diagnostic_only | 0.115 | -0.064 .. 0.300 | -0.064 .. 0.234 | 0.486 |
| tair_transfer_selection_score | offline_or_diagnostic_only | 0.200 | 0.029 .. 0.396 | 0.029 .. 0.357 | 0.505 |
| rhair_transfer_selection_score | objective_secondary_selection | 0.477 | 0.339 .. 0.597 | 0.339 .. 0.633 | 0.629 |
| co2_transfer_selection_score | offline_or_diagnostic_only | -0.112 | -0.222 .. 0.009 | -0.222 .. 0.009 | 0.452 |
| multiobjective_transfer_selection_score | offline_or_diagnostic_only | 0.235 | 0.043 .. 0.379 | 0.022 .. 0.352 | 0.510 |
