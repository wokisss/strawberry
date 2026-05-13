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
| mpc_tair_mae | co2_first_step_mae | weak_selection |
| mpc_tair_mae | co2_control_horizon_mae | secondary_selection |
| mpc_tair_mae | co2_weighted_horizon_mae | secondary_selection |
| mpc_tair_mae | co2_full_horizon_mae | secondary_selection |
| mpc_tair_mae | co2_final_step_mae | secondary_selection |
| mpc_tair_mae | co2_control_horizon_abs_bias | weak_selection |
| mpc_tair_mae | co2_constraint_near_mae_proxy | weak_selection |
| mpc_tair_mae | forecast_only_transfer_rank | offline_or_diagnostic_only |
| mpc_tair_mae | tair_transfer_selection_score | offline_or_diagnostic_only |
| mpc_tair_mae | rhair_transfer_selection_score | offline_or_diagnostic_only |
| mpc_tair_mae | co2_transfer_selection_score | secondary_selection |
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
| mpc_rhair_mae | rhair_first_step_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_control_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_full_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_final_step_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_first_step_mae | secondary_selection |
| mpc_rhair_mae | co2_control_horizon_mae | secondary_selection |
| mpc_rhair_mae | co2_weighted_horizon_mae | secondary_selection |
| mpc_rhair_mae | co2_full_horizon_mae | secondary_selection |
| mpc_rhair_mae | co2_final_step_mae | secondary_selection |
| mpc_rhair_mae | co2_control_horizon_abs_bias | secondary_selection |
| mpc_rhair_mae | co2_constraint_near_mae_proxy | weak_selection |
| mpc_rhair_mae | forecast_only_transfer_rank | offline_or_diagnostic_only |
| mpc_rhair_mae | tair_transfer_selection_score | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_transfer_selection_score | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_transfer_selection_score | secondary_selection |
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
| mpc_co2_mae | tair_first_step_mae | secondary_selection |
| mpc_co2_mae | tair_control_horizon_mae | weak_selection |
| mpc_co2_mae | tair_weighted_horizon_mae | secondary_selection |
| mpc_co2_mae | tair_full_horizon_mae | secondary_selection |
| mpc_co2_mae | tair_final_step_mae | offline_or_diagnostic_only |
| mpc_co2_mae | tair_control_horizon_abs_bias | secondary_selection |
| mpc_co2_mae | tair_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_co2_mae | rhair_first_step_mae | offline_or_diagnostic_only |
| mpc_co2_mae | rhair_control_horizon_mae | offline_or_diagnostic_only |
| mpc_co2_mae | rhair_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_co2_mae | rhair_full_horizon_mae | offline_or_diagnostic_only |
| mpc_co2_mae | rhair_final_step_mae | offline_or_diagnostic_only |
| mpc_co2_mae | rhair_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_co2_mae | rhair_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_co2_mae | co2_first_step_mae | offline_or_diagnostic_only |
| mpc_co2_mae | co2_control_horizon_mae | offline_or_diagnostic_only |
| mpc_co2_mae | co2_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_co2_mae | co2_full_horizon_mae | offline_or_diagnostic_only |
| mpc_co2_mae | co2_final_step_mae | offline_or_diagnostic_only |
| mpc_co2_mae | co2_control_horizon_abs_bias | secondary_selection |
| mpc_co2_mae | co2_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_co2_mae | forecast_only_transfer_rank | secondary_selection |
| mpc_co2_mae | tair_transfer_selection_score | secondary_selection |
| mpc_co2_mae | rhair_transfer_selection_score | offline_or_diagnostic_only |
| mpc_co2_mae | co2_transfer_selection_score | offline_or_diagnostic_only |
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
| mpc_objective | tair_constraint_near_mae_proxy | weak_selection |
| mpc_objective | rhair_first_step_mae | weak_selection |
| mpc_objective | rhair_control_horizon_mae | offline_or_diagnostic_only |
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
| mpc_objective | rhair_transfer_selection_score | offline_or_diagnostic_only |
| mpc_objective | co2_transfer_selection_score | objective_secondary_selection |
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
| 1 | current_hybrid_transformer | 5.167 | 5.688 | 4.250 | 5.562 | 6.722 | 0.874 | 3.371 | 24.307 | 0.1133 |
| 2 | itransformer_co2_control_aware_fusion | 5.406 | 7.094 | 7.625 | 1.500 | 8.556 | 2.233 | 3.595 | 12.214 | 0.1774 |
| 3 | itransformer_residual | 5.583 | 4.062 | 5.688 | 7.000 | 9.167 | 2.039 | 4.124 | 9.779 | 0.2065 |
| 4 | itransformer_co2_late_frozen_expert | 5.740 | 6.531 | 8.625 | 2.062 | 9.944 | 2.258 | 3.556 | 12.142 | 0.1806 |
| 5 | transformer_hybrid_residual | 6.396 | 4.500 | 2.875 | 11.812 | 9.167 | 2.260 | 5.763 | 27.211 | 0.2204 |
| 6 | itransformer_co2_late_residual | 6.604 | 7.812 | 6.000 | 6.000 | 9.000 | 1.356 | 1.718 | 20.652 | 0.0937 |
| 7 | segrnn_forecaster | 7.979 | 14.188 | 6.000 | 3.750 | 9.389 | 0.396 | 2.459 | 17.815 | 0.0667 |
| 8 | dlinear_forecaster | 8.583 | 9.938 | 5.500 | 10.312 | 15.056 | 3.354 | 6.343 | 30.560 | 0.4287 |
| 9 | itransformer_co2_horizon_mixture | 8.646 | 2.875 | 12.125 | 10.938 | 13.722 | 3.532 | 5.218 | 19.116 | 0.4645 |
| 10 | transformer_forecaster | 9.250 | 6.188 | 11.125 | 10.438 | 13.056 | 1.417 | 5.362 | 16.428 | 0.1599 |
| 11 | itransformer_co2_residual | 9.312 | 6.062 | 9.938 | 11.938 | 10.778 | 0.830 | 1.178 | 10.536 | 0.0658 |
| 12 | gru_forecaster | 10.375 | 10.438 | 14.625 | 6.062 | 14.278 | 0.415 | 3.213 | 58.764 | 0.0892 |
| 13 | nlinear_forecaster | 10.729 | 13.125 | 4.812 | 14.250 | 15.500 | 2.447 | 4.723 | 29.146 | 0.2409 |
| 14 | patchtst_residual | 10.938 | 8.688 | 11.875 | 12.250 | 14.833 | 2.932 | 9.861 | 25.866 | 0.3089 |
| 15 | lstm_forecaster | 11.604 | 12.812 | 11.750 | 10.250 | 15.111 | 1.984 | 3.535 | 32.284 | 0.2234 |
| 16 | frequency_forecaster | 13.688 | 16.000 | 13.188 | 11.875 | 18.722 | 2.224 | 11.978 | 17.692 | 1.0029 |

## Metric Transfer Quality

### Target: `mpc_tair_mae`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| co2_full_horizon_mae | selection | 0.521 | 0.485 | 0.692 | yes | yes | 0.333 |
| co2_weighted_horizon_mae | selection | 0.485 | 0.465 | 0.667 | yes | yes | 0.333 |
| co2_transfer_selection_score | selection | 0.383 | 0.400 | 0.642 | no | yes | 0.333 |
| co2_control_horizon_mae | selection | 0.321 | 0.352 | 0.639 | no | yes | 0.333 |
| co2_final_step_mae | selection | 0.242 | 0.362 | 0.617 | no | yes | 0.333 |
| co2_first_step_mae | selection | 0.203 | 0.337 | 0.605 | no | no | 0.000 |
| co2_control_horizon_abs_bias | selection | 0.208 | 0.312 | 0.600 | no | yes | 0.333 |
| co2_constraint_near_mae_proxy | selection | 0.272 | 0.288 | 0.592 | yes | yes | 0.333 |
| rhair_first_step_mae | selection | 0.214 | 0.147 | 0.567 | no | yes | 0.333 |
| multiobjective_transfer_selection_score | selection | 0.111 | 0.082 | 0.542 | no | no | 0.000 |
| rhair_control_horizon_abs_bias | selection | 0.051 | 0.038 | 0.508 | no | no | 0.333 |
| rhair_transfer_selection_score | selection | -0.005 | -0.077 | 0.487 | no | no | 0.000 |
| forecast_only_transfer_rank | selection | -0.051 | -0.071 | 0.483 | no | no | 0.000 |
| tair_constraint_near_mae_proxy | selection | -0.157 | -0.010 | 0.479 | no | no | 0.333 |
| rhair_control_horizon_mae | selection | 0.059 | -0.085 | 0.475 | no | no | 0.000 |
| tair_control_horizon_abs_bias | selection | -0.011 | -0.018 | 0.467 | no | no | 0.000 |
| tair_control_horizon_mae | selection | 0.012 | -0.137 | 0.445 | no | no | 0.000 |
| tair_transfer_selection_score | selection | -0.165 | -0.171 | 0.442 | no | no | 0.000 |
| rhair_weighted_horizon_mae | selection | -0.307 | -0.381 | 0.370 | no | no | 0.000 |
| rhair_full_horizon_mae | selection | -0.339 | -0.374 | 0.367 | no | no | 0.000 |
| tair_full_horizon_mae | selection | -0.195 | -0.412 | 0.350 | no | no | 0.000 |
| tair_first_step_mae | selection | -0.003 | -0.315 | 0.350 | no | no | 0.000 |
| tair_final_step_mae | selection | -0.259 | -0.476 | 0.342 | no | no | 0.000 |
| tair_weighted_horizon_mae | selection | -0.176 | -0.415 | 0.342 | no | no | 0.000 |
| rhair_final_step_mae | selection | -0.396 | -0.422 | 0.319 | no | no | 0.000 |
| rhair_constraint_near_mae_proxy | selection | -0.511 | -0.529 | 0.317 | no | no | 0.000 |
| assim_sp_first_grad | diagnostic | 0.632 | 0.572 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | 0.566 | 0.517 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | 0.529 | 0.502 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | 0.363 | 0.490 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | 0.432 | 0.461 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | 0.477 | 0.452 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | 0.351 | 0.446 |  |  |  |  |
| co2_sp_first_grad | diagnostic | 0.450 | 0.440 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | 0.352 | 0.422 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | 0.386 | 0.352 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | -0.405 | -0.326 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | 0.294 | 0.243 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | 0.343 | 0.243 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.024 | -0.063 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.096 | 0.060 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.026 | 0.023 |  |  |  |  |

### Target: `mpc_rhair_mae`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| co2_full_horizon_mae | selection | 0.636 | 0.591 | 0.742 | no | no | 0.667 |
| co2_weighted_horizon_mae | selection | 0.664 | 0.582 | 0.733 | no | no | 0.333 |
| co2_control_horizon_mae | selection | 0.728 | 0.472 | 0.681 | no | no | 0.333 |
| co2_transfer_selection_score | selection | 0.484 | 0.500 | 0.675 | no | no | 0.333 |
| co2_first_step_mae | selection | 0.736 | 0.484 | 0.664 | no | no | 0.333 |
| co2_final_step_mae | selection | 0.268 | 0.415 | 0.633 | no | no | 0.333 |
| co2_control_horizon_abs_bias | selection | 0.293 | 0.374 | 0.617 | no | no | 0.333 |
| multiobjective_transfer_selection_score | selection | 0.550 | 0.276 | 0.608 | no | no | 0.000 |
| co2_constraint_near_mae_proxy | selection | 0.043 | 0.256 | 0.608 | no | no | 0.333 |
| forecast_only_transfer_rank | selection | 0.502 | 0.191 | 0.567 | no | no | 0.000 |
| rhair_first_step_mae | selection | 0.618 | 0.159 | 0.550 | no | no | 0.333 |
| tair_control_horizon_abs_bias | selection | 0.396 | 0.082 | 0.533 | no | no | 0.000 |
| rhair_control_horizon_mae | selection | 0.560 | 0.103 | 0.525 | no | no | 0.333 |
| tair_control_horizon_mae | selection | 0.590 | 0.081 | 0.521 | no | no | 0.000 |
| rhair_transfer_selection_score | selection | 0.310 | 0.072 | 0.513 | no | no | 0.000 |
| tair_transfer_selection_score | selection | 0.301 | 0.032 | 0.508 | no | no | 0.000 |
| tair_constraint_near_mae_proxy | selection | 0.342 | 0.122 | 0.504 | no | yes | 0.667 |
| rhair_full_horizon_mae | selection | 0.340 | -0.050 | 0.500 | no | no | 0.000 |
| tair_first_step_mae | selection | 0.609 | -0.035 | 0.500 | no | no | 0.000 |
| rhair_weighted_horizon_mae | selection | 0.366 | -0.057 | 0.496 | no | no | 0.000 |
| rhair_control_horizon_abs_bias | selection | -0.114 | -0.012 | 0.492 | no | yes | 0.333 |
| rhair_constraint_near_mae_proxy | selection | 0.195 | -0.047 | 0.483 | no | no | 0.000 |
| rhair_final_step_mae | selection | 0.287 | -0.069 | 0.479 | no | no | 0.000 |
| tair_weighted_horizon_mae | selection | 0.415 | -0.091 | 0.458 | no | no | 0.000 |
| tair_full_horizon_mae | selection | 0.393 | -0.094 | 0.450 | no | no | 0.000 |
| tair_final_step_mae | selection | 0.273 | -0.206 | 0.392 | no | no | 0.000 |
| tair_first_grad_mean_abs | diagnostic | 0.411 | 0.599 |  |  |  |  |
| assim_sp_first_grad | diagnostic | 0.277 | 0.434 |  |  |  |  |
| co2_sp_first_grad | diagnostic | 0.305 | 0.428 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | 0.328 | 0.422 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | 0.323 | 0.399 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | 0.550 | 0.387 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | -0.284 | -0.344 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | 0.282 | 0.313 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | 0.222 | 0.299 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | -0.067 | -0.232 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | 0.115 | 0.225 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | -0.275 | -0.208 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | -0.088 | 0.143 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | -0.176 | 0.054 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | -0.043 | 0.022 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | -0.234 | -0.007 |  |  |  |  |

### Target: `mpc_co2_mae`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| co2_control_horizon_abs_bias | selection | 0.326 | 0.488 | 0.675 | no | no | 0.333 |
| tair_transfer_selection_score | selection | 0.318 | 0.406 | 0.650 | no | yes | 0.333 |
| tair_weighted_horizon_mae | selection | 0.050 | 0.379 | 0.650 | no | yes | 0.333 |
| forecast_only_transfer_rank | selection | 0.369 | 0.415 | 0.642 | yes | yes | 0.333 |
| tair_full_horizon_mae | selection | 0.056 | 0.376 | 0.642 | no | yes | 0.333 |
| tair_control_horizon_abs_bias | selection | 0.150 | 0.359 | 0.642 | yes | yes | 0.333 |
| tair_first_step_mae | selection | 0.092 | 0.359 | 0.625 | no | no | 0.333 |
| multiobjective_transfer_selection_score | selection | 0.370 | 0.388 | 0.617 | no | yes | 0.333 |
| tair_control_horizon_mae | selection | -0.015 | 0.343 | 0.613 | no | yes | 0.333 |
| co2_first_step_mae | selection | 0.019 | 0.243 | 0.588 | no | no | 0.333 |
| rhair_control_horizon_abs_bias | selection | 0.376 | 0.229 | 0.583 | no | no | 0.333 |
| co2_control_horizon_mae | selection | -0.006 | 0.199 | 0.571 | no | no | 0.333 |
| co2_transfer_selection_score | selection | 0.157 | 0.206 | 0.567 | no | no | 0.333 |
| co2_constraint_near_mae_proxy | selection | 0.113 | 0.185 | 0.567 | no | no | 0.000 |
| rhair_constraint_near_mae_proxy | selection | 0.570 | 0.188 | 0.558 | no | no | 0.000 |
| rhair_final_step_mae | selection | 0.432 | 0.084 | 0.538 | no | no | 0.000 |
| tair_constraint_near_mae_proxy | selection | -0.094 | 0.084 | 0.529 | no | no | 0.333 |
| rhair_full_horizon_mae | selection | 0.458 | 0.112 | 0.525 | no | no | 0.000 |
| co2_weighted_horizon_mae | selection | -0.043 | 0.062 | 0.525 | no | no | 0.000 |
| rhair_weighted_horizon_mae | selection | 0.437 | 0.081 | 0.521 | no | no | 0.000 |
| tair_final_step_mae | selection | 0.056 | 0.068 | 0.517 | no | no | 0.000 |
| co2_full_horizon_mae | selection | -0.069 | 0.062 | 0.517 | no | no | 0.000 |
| co2_final_step_mae | selection | -0.150 | -0.018 | 0.508 | no | no | 0.000 |
| rhair_transfer_selection_score | selection | 0.267 | -0.015 | 0.496 | no | no | 0.000 |
| rhair_first_step_mae | selection | 0.008 | -0.029 | 0.492 | no | no | 0.000 |
| rhair_control_horizon_mae | selection | 0.120 | -0.138 | 0.450 | no | no | 0.000 |
| assim_sp_first_grad | diagnostic | -0.609 | -0.508 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | -0.551 | -0.496 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | -0.599 | -0.496 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | -0.576 | -0.490 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | -0.514 | -0.458 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | -0.622 | -0.399 |  |  |  |  |
| co2_sp_first_grad | diagnostic | -0.446 | -0.378 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | -0.510 | -0.372 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | -0.493 | -0.358 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | -0.578 | -0.275 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | -0.644 | -0.181 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | 0.550 | 0.168 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.300 | -0.163 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | -0.357 | -0.131 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | -0.033 | -0.125 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | 0.033 | 0.113 |  |  |  |  |

### Target: `mpc_objective`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| co2_weighted_horizon_mae | selection | 0.688 | 0.544 | 0.717 | no | no | 0.333 |
| co2_full_horizon_mae | selection | 0.651 | 0.529 | 0.708 | no | no | 0.333 |
| co2_first_step_mae | selection | 0.862 | 0.611 | 0.706 | no | no | 0.000 |
| co2_control_horizon_mae | selection | 0.808 | 0.531 | 0.706 | no | no | 0.333 |
| co2_transfer_selection_score | selection | 0.414 | 0.476 | 0.675 | no | no | 0.333 |
| rhair_first_step_mae | selection | 0.844 | 0.312 | 0.633 | no | no | 0.333 |
| multiobjective_transfer_selection_score | selection | 0.556 | 0.362 | 0.625 | no | no | 0.000 |
| co2_final_step_mae | selection | 0.207 | 0.288 | 0.583 | no | no | 0.333 |
| co2_control_horizon_abs_bias | selection | 0.041 | 0.212 | 0.583 | no | no | 0.333 |
| tair_constraint_near_mae_proxy | selection | 0.526 | 0.258 | 0.580 | no | yes | 0.333 |
| forecast_only_transfer_rank | selection | 0.496 | 0.212 | 0.567 | no | no | 0.000 |
| co2_constraint_near_mae_proxy | selection | 0.018 | 0.144 | 0.558 | no | no | 0.333 |
| rhair_control_horizon_mae | selection | 0.716 | 0.138 | 0.558 | no | no | 0.000 |
| rhair_transfer_selection_score | selection | 0.333 | 0.127 | 0.555 | no | no | 0.000 |
| tair_control_horizon_abs_bias | selection | 0.535 | 0.171 | 0.550 | no | no | 0.000 |
| tair_control_horizon_mae | selection | 0.751 | 0.128 | 0.529 | no | no | 0.000 |
| tair_transfer_selection_score | selection | 0.362 | 0.097 | 0.525 | no | no | 0.000 |
| rhair_control_horizon_abs_bias | selection | -0.179 | -0.038 | 0.492 | no | yes | 0.333 |
| rhair_weighted_horizon_mae | selection | 0.403 | -0.057 | 0.487 | no | no | 0.000 |
| rhair_full_horizon_mae | selection | 0.364 | -0.076 | 0.483 | no | no | 0.000 |
| tair_first_step_mae | selection | 0.749 | 0.012 | 0.467 | no | no | 0.000 |
| tair_full_horizon_mae | selection | 0.561 | -0.062 | 0.450 | no | no | 0.000 |
| tair_final_step_mae | selection | 0.462 | -0.129 | 0.442 | no | no | 0.000 |
| tair_weighted_horizon_mae | selection | 0.583 | -0.065 | 0.442 | no | no | 0.000 |
| rhair_final_step_mae | selection | 0.258 | -0.181 | 0.437 | no | no | 0.000 |
| rhair_constraint_near_mae_proxy | selection | 0.105 | -0.215 | 0.433 | no | no | 0.000 |
| tair_t_vent_sp_first_grad | diagnostic | 0.335 | 0.455 |  |  |  |  |
| assim_sp_first_grad | diagnostic | 0.324 | 0.428 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | 0.345 | 0.405 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | 0.316 | 0.372 |  |  |  |  |
| co2_sp_first_grad | diagnostic | 0.380 | 0.361 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | 0.577 | 0.202 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | -0.017 | 0.169 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | 0.048 | 0.166 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | -0.200 | -0.160 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | -0.092 | 0.143 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | 0.032 | 0.137 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | -0.057 | 0.107 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.179 | -0.072 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | 0.025 | -0.047 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.135 | -0.035 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.134 | 0.000 |  |  |  |  |

## Robustness Summary

### Target: `mpc_tair_mae`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | offline_or_diagnostic_only | -0.315 | -0.486 .. -0.168 | -0.560 .. -0.115 | 0.295 |
| tair_control_horizon_mae | offline_or_diagnostic_only | -0.137 | -0.277 .. 0.048 | -0.400 .. 0.048 | 0.404 |
| tair_weighted_horizon_mae | offline_or_diagnostic_only | -0.415 | -0.532 .. -0.289 | -0.569 .. -0.214 | 0.295 |
| tair_full_horizon_mae | offline_or_diagnostic_only | -0.412 | -0.529 .. -0.286 | -0.565 .. -0.209 | 0.305 |
| tair_final_step_mae | offline_or_diagnostic_only | -0.476 | -0.589 .. -0.386 | -0.589 .. -0.330 | 0.295 |
| tair_control_horizon_abs_bias | offline_or_diagnostic_only | -0.018 | -0.121 .. 0.121 | -0.130 .. 0.341 | 0.429 |
| tair_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.010 | -0.148 .. 0.173 | -0.148 .. 0.212 | 0.433 |
| rhair_first_step_mae | offline_or_diagnostic_only | 0.147 | -0.011 .. 0.268 | -0.011 .. 0.268 | 0.514 |
| rhair_control_horizon_mae | offline_or_diagnostic_only | -0.085 | -0.243 .. 0.050 | -0.243 .. 0.138 | 0.419 |
| rhair_weighted_horizon_mae | offline_or_diagnostic_only | -0.381 | -0.492 .. -0.281 | -0.481 .. -0.118 | 0.327 |
| rhair_full_horizon_mae | offline_or_diagnostic_only | -0.374 | -0.482 .. -0.246 | -0.471 .. -0.066 | 0.324 |
| rhair_final_step_mae | offline_or_diagnostic_only | -0.422 | -0.595 .. -0.306 | -0.595 .. -0.173 | 0.250 |
| rhair_control_horizon_abs_bias | offline_or_diagnostic_only | 0.038 | -0.132 .. 0.164 | -0.132 .. 0.236 | 0.448 |
| rhair_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.529 | -0.618 .. -0.429 | -0.610 .. -0.363 | 0.276 |
| co2_first_step_mae | weak_selection | 0.337 | 0.263 .. 0.436 | 0.212 .. 0.436 | 0.577 |
| co2_control_horizon_mae | secondary_selection | 0.352 | 0.281 .. 0.495 | 0.223 .. 0.495 | 0.606 |
| co2_weighted_horizon_mae | secondary_selection | 0.465 | 0.350 .. 0.629 | 0.220 .. 0.629 | 0.619 |
| co2_full_horizon_mae | secondary_selection | 0.485 | 0.375 .. 0.657 | 0.264 .. 0.657 | 0.648 |
| co2_final_step_mae | secondary_selection | 0.362 | 0.246 .. 0.546 | 0.181 .. 0.546 | 0.571 |
| co2_control_horizon_abs_bias | weak_selection | 0.312 | 0.207 .. 0.375 | 0.108 .. 0.375 | 0.562 |
| co2_constraint_near_mae_proxy | weak_selection | 0.288 | 0.136 .. 0.489 | 0.134 .. 0.489 | 0.533 |
| forecast_only_transfer_rank | offline_or_diagnostic_only | -0.071 | -0.221 .. 0.057 | -0.211 .. 0.132 | 0.448 |
| tair_transfer_selection_score | offline_or_diagnostic_only | -0.171 | -0.271 .. 0.007 | -0.398 .. 0.143 | 0.410 |
| rhair_transfer_selection_score | offline_or_diagnostic_only | -0.077 | -0.232 .. 0.048 | -0.232 .. 0.116 | 0.433 |
| co2_transfer_selection_score | secondary_selection | 0.400 | 0.329 .. 0.554 | 0.247 .. 0.554 | 0.610 |
| multiobjective_transfer_selection_score | offline_or_diagnostic_only | 0.082 | -0.018 .. 0.168 | -0.073 .. 0.242 | 0.514 |

### Target: `mpc_rhair_mae`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | offline_or_diagnostic_only | -0.035 | -0.257 .. 0.050 | -0.257 .. 0.269 | 0.429 |
| tair_control_horizon_mae | offline_or_diagnostic_only | 0.081 | -0.116 .. 0.191 | -0.116 .. 0.305 | 0.452 |
| tair_weighted_horizon_mae | offline_or_diagnostic_only | -0.091 | -0.325 .. -0.014 | -0.325 .. 0.148 | 0.381 |
| tair_full_horizon_mae | offline_or_diagnostic_only | -0.094 | -0.329 .. -0.018 | -0.329 .. 0.143 | 0.371 |
| tair_final_step_mae | offline_or_diagnostic_only | -0.206 | -0.464 .. -0.089 | -0.464 .. 0.066 | 0.305 |
| tair_control_horizon_abs_bias | offline_or_diagnostic_only | 0.082 | -0.114 .. 0.168 | -0.114 .. 0.346 | 0.467 |
| tair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.122 | -0.066 .. 0.213 | -0.066 .. 0.267 | 0.433 |
| rhair_first_step_mae | offline_or_diagnostic_only | 0.159 | -0.021 .. 0.307 | -0.021 .. 0.307 | 0.486 |
| rhair_control_horizon_mae | offline_or_diagnostic_only | 0.103 | -0.089 .. 0.232 | -0.089 .. 0.266 | 0.457 |
| rhair_weighted_horizon_mae | offline_or_diagnostic_only | -0.057 | -0.284 .. 0.045 | -0.284 .. 0.239 | 0.423 |
| rhair_full_horizon_mae | offline_or_diagnostic_only | -0.050 | -0.232 .. 0.089 | -0.232 .. 0.269 | 0.438 |
| rhair_final_step_mae | offline_or_diagnostic_only | -0.069 | -0.256 .. 0.073 | -0.256 .. 0.245 | 0.413 |
| rhair_control_horizon_abs_bias | offline_or_diagnostic_only | -0.012 | -0.143 .. 0.179 | -0.143 .. 0.179 | 0.438 |
| rhair_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.047 | -0.229 .. 0.093 | -0.229 .. 0.253 | 0.419 |
| co2_first_step_mae | secondary_selection | 0.484 | 0.374 .. 0.613 | 0.374 .. 0.613 | 0.615 |
| co2_control_horizon_mae | secondary_selection | 0.472 | 0.359 .. 0.702 | 0.322 .. 0.702 | 0.635 |
| co2_weighted_horizon_mae | secondary_selection | 0.582 | 0.493 .. 0.836 | 0.493 .. 0.836 | 0.695 |
| co2_full_horizon_mae | secondary_selection | 0.591 | 0.504 .. 0.846 | 0.504 .. 0.846 | 0.705 |
| co2_final_step_mae | secondary_selection | 0.415 | 0.343 .. 0.514 | 0.286 .. 0.514 | 0.600 |
| co2_control_horizon_abs_bias | secondary_selection | 0.374 | 0.296 .. 0.525 | 0.203 .. 0.525 | 0.581 |
| co2_constraint_near_mae_proxy | weak_selection | 0.256 | 0.171 .. 0.525 | 0.152 .. 0.525 | 0.571 |
| forecast_only_transfer_rank | offline_or_diagnostic_only | 0.191 | 0.025 .. 0.297 | 0.025 .. 0.421 | 0.514 |
| tair_transfer_selection_score | offline_or_diagnostic_only | 0.032 | -0.175 .. 0.129 | -0.175 .. 0.308 | 0.438 |
| rhair_transfer_selection_score | offline_or_diagnostic_only | 0.072 | -0.084 .. 0.209 | -0.084 .. 0.291 | 0.452 |
| co2_transfer_selection_score | secondary_selection | 0.500 | 0.433 .. 0.729 | 0.374 .. 0.729 | 0.635 |
| multiobjective_transfer_selection_score | weak_selection | 0.276 | 0.121 .. 0.411 | 0.121 .. 0.527 | 0.552 |

### Target: `mpc_co2_mae`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | secondary_selection | 0.359 | 0.296 .. 0.475 | 0.235 .. 0.464 | 0.600 |
| tair_control_horizon_mae | weak_selection | 0.343 | 0.227 .. 0.449 | 0.211 .. 0.449 | 0.567 |
| tair_weighted_horizon_mae | secondary_selection | 0.379 | 0.271 .. 0.493 | 0.305 .. 0.493 | 0.610 |
| tair_full_horizon_mae | secondary_selection | 0.376 | 0.268 .. 0.489 | 0.301 .. 0.489 | 0.600 |
| tair_final_step_mae | offline_or_diagnostic_only | 0.068 | -0.086 .. 0.157 | -0.214 .. 0.181 | 0.467 |
| tair_control_horizon_abs_bias | secondary_selection | 0.359 | 0.221 .. 0.475 | 0.187 .. 0.475 | 0.590 |
| tair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.084 | -0.070 .. 0.152 | -0.070 .. 0.152 | 0.481 |
| rhair_first_step_mae | offline_or_diagnostic_only | -0.029 | -0.186 .. 0.054 | -0.352 .. 0.054 | 0.438 |
| rhair_control_horizon_mae | offline_or_diagnostic_only | -0.138 | -0.346 .. -0.054 | -0.473 .. 0.046 | 0.381 |
| rhair_weighted_horizon_mae | offline_or_diagnostic_only | 0.081 | -0.080 .. 0.216 | -0.234 .. 0.202 | 0.462 |
| rhair_full_horizon_mae | offline_or_diagnostic_only | 0.112 | -0.079 .. 0.254 | -0.181 .. 0.200 | 0.457 |
| rhair_final_step_mae | offline_or_diagnostic_only | 0.084 | -0.113 .. 0.191 | -0.135 .. 0.138 | 0.471 |
| rhair_control_horizon_abs_bias | offline_or_diagnostic_only | 0.229 | 0.129 .. 0.379 | -0.016 .. 0.420 | 0.543 |
| rhair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.188 | 0.014 .. 0.332 | -0.033 .. 0.297 | 0.495 |
| co2_first_step_mae | offline_or_diagnostic_only | 0.243 | 0.139 .. 0.395 | 0.139 .. 0.331 | 0.548 |
| co2_control_horizon_mae | offline_or_diagnostic_only | 0.199 | 0.096 .. 0.370 | 0.096 .. 0.370 | 0.533 |
| co2_weighted_horizon_mae | offline_or_diagnostic_only | 0.062 | -0.036 .. 0.200 | -0.121 .. 0.231 | 0.495 |
| co2_full_horizon_mae | offline_or_diagnostic_only | 0.062 | -0.057 .. 0.193 | -0.154 .. 0.220 | 0.476 |
| co2_final_step_mae | offline_or_diagnostic_only | -0.018 | -0.129 .. 0.168 | -0.220 .. 0.253 | 0.476 |
| co2_control_horizon_abs_bias | secondary_selection | 0.488 | 0.421 .. 0.668 | 0.421 .. 0.758 | 0.648 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.185 | 0.111 .. 0.411 | 0.055 .. 0.411 | 0.533 |
| forecast_only_transfer_rank | secondary_selection | 0.415 | 0.289 .. 0.536 | 0.206 .. 0.536 | 0.590 |
| tair_transfer_selection_score | secondary_selection | 0.406 | 0.304 .. 0.529 | 0.301 .. 0.521 | 0.610 |
| rhair_transfer_selection_score | offline_or_diagnostic_only | -0.015 | -0.197 .. 0.086 | -0.396 .. 0.209 | 0.442 |
| co2_transfer_selection_score | offline_or_diagnostic_only | 0.206 | 0.107 .. 0.382 | 0.071 .. 0.382 | 0.533 |
| multiobjective_transfer_selection_score | secondary_selection | 0.388 | 0.300 .. 0.507 | 0.269 .. 0.507 | 0.581 |

### Target: `mpc_objective`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | offline_or_diagnostic_only | 0.012 | -0.200 .. 0.175 | -0.200 .. 0.175 | 0.390 |
| tair_control_horizon_mae | offline_or_diagnostic_only | 0.128 | -0.059 .. 0.316 | -0.092 .. 0.316 | 0.462 |
| tair_weighted_horizon_mae | offline_or_diagnostic_only | -0.065 | -0.293 .. 0.082 | -0.293 .. 0.088 | 0.362 |
| tair_full_horizon_mae | offline_or_diagnostic_only | -0.062 | -0.289 .. 0.086 | -0.289 .. 0.093 | 0.371 |
| tair_final_step_mae | offline_or_diagnostic_only | -0.129 | -0.371 .. -0.021 | -0.371 .. 0.044 | 0.362 |
| tair_control_horizon_abs_bias | offline_or_diagnostic_only | 0.171 | -0.007 .. 0.311 | -0.007 .. 0.385 | 0.486 |
| tair_constraint_near_mae_proxy | weak_selection | 0.258 | 0.098 .. 0.427 | 0.098 .. 0.426 | 0.519 |
| rhair_first_step_mae | weak_selection | 0.312 | 0.164 .. 0.482 | 0.164 .. 0.482 | 0.581 |
| rhair_control_horizon_mae | offline_or_diagnostic_only | 0.138 | -0.046 .. 0.286 | -0.046 .. 0.336 | 0.495 |
| rhair_weighted_horizon_mae | offline_or_diagnostic_only | -0.057 | -0.284 .. 0.070 | -0.284 .. 0.151 | 0.413 |
| rhair_full_horizon_mae | offline_or_diagnostic_only | -0.076 | -0.261 .. 0.089 | -0.261 .. 0.198 | 0.419 |
| rhair_final_step_mae | offline_or_diagnostic_only | -0.181 | -0.388 .. -0.034 | -0.388 .. 0.074 | 0.365 |
| rhair_control_horizon_abs_bias | offline_or_diagnostic_only | -0.038 | -0.179 .. 0.154 | -0.179 .. 0.154 | 0.438 |
| rhair_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.215 | -0.429 .. -0.086 | -0.429 .. -0.049 | 0.362 |
| co2_first_step_mae | objective_secondary_selection | 0.611 | 0.527 .. 0.735 | 0.481 .. 0.735 | 0.663 |
| co2_control_horizon_mae | objective_secondary_selection | 0.531 | 0.431 .. 0.767 | 0.431 .. 0.767 | 0.663 |
| co2_weighted_horizon_mae | objective_secondary_selection | 0.544 | 0.446 .. 0.782 | 0.440 .. 0.782 | 0.676 |
| co2_full_horizon_mae | objective_secondary_selection | 0.529 | 0.429 .. 0.764 | 0.424 .. 0.764 | 0.667 |
| co2_final_step_mae | weak_selection | 0.288 | 0.179 .. 0.429 | 0.116 .. 0.429 | 0.543 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | 0.212 | 0.104 .. 0.343 | 0.051 .. 0.343 | 0.552 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.144 | 0.014 .. 0.389 | 0.020 .. 0.389 | 0.505 |
| forecast_only_transfer_rank | offline_or_diagnostic_only | 0.212 | 0.071 .. 0.340 | 0.071 .. 0.300 | 0.514 |
| tair_transfer_selection_score | offline_or_diagnostic_only | 0.097 | -0.096 .. 0.279 | -0.096 .. 0.279 | 0.457 |
| rhair_transfer_selection_score | offline_or_diagnostic_only | 0.127 | -0.014 .. 0.291 | -0.014 .. 0.305 | 0.500 |
| co2_transfer_selection_score | objective_secondary_selection | 0.476 | 0.404 .. 0.689 | 0.368 .. 0.689 | 0.648 |
| multiobjective_transfer_selection_score | objective_secondary_selection | 0.362 | 0.225 .. 0.485 | 0.225 .. 0.485 | 0.571 |
