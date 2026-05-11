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
| mpc_rhair_mae | tair_control_horizon_mae | weak_selection |
| mpc_rhair_mae | tair_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | tair_full_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | tair_final_step_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | tair_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_rhair_mae | tair_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_first_step_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_control_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_weighted_horizon_mae | secondary_selection |
| mpc_rhair_mae | rhair_full_horizon_mae | secondary_selection |
| mpc_rhair_mae | rhair_final_step_mae | secondary_selection |
| mpc_rhair_mae | rhair_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_constraint_near_mae_proxy | secondary_selection |
| mpc_rhair_mae | co2_first_step_mae | secondary_selection |
| mpc_rhair_mae | co2_control_horizon_mae | secondary_selection |
| mpc_rhair_mae | co2_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_full_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_final_step_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_rhair_mae | forecast_only_transfer_rank | weak_selection |
| mpc_rhair_mae | tair_transfer_selection_score | secondary_selection |
| mpc_rhair_mae | rhair_transfer_selection_score | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_transfer_selection_score | secondary_selection |
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
| mpc_objective | tair_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_objective | tair_constraint_near_mae_proxy | weak_selection |
| mpc_objective | rhair_first_step_mae | objective_secondary_selection |
| mpc_objective | rhair_control_horizon_mae | weak_selection |
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
| mpc_objective | rhair_transfer_selection_score | weak_selection |
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
| 1 | current_hybrid_transformer | 5.521 | 6.750 | 4.438 | 5.375 | 6.722 | 0.526 | 1.486 | 24.384 | 0.0517 |
| 2 | itransformer_residual | 5.646 | 4.625 | 5.688 | 6.625 | 9.167 | 2.392 | 4.199 | 18.052 | 0.1666 |
| 3 | itransformer_co2_control_aware_fusion | 6.135 | 8.531 | 8.375 | 1.500 | 8.556 | 1.072 | 1.179 | 26.154 | 0.1072 |
| 4 | itransformer_co2_late_frozen_expert | 6.469 | 7.969 | 9.375 | 2.062 | 9.944 | 1.122 | 1.282 | 25.366 | 0.1133 |
| 5 | itransformer_co2_protected_expert | 6.479 | 4.312 | 9.812 | 5.312 | 9.278 | 0.829 | 0.749 | 50.390 | 0.1278 |
| 6 | transformer_hybrid_residual | 6.667 | 5.750 | 3.062 | 11.188 | 9.167 | 0.873 | 1.986 | 23.095 | 0.0666 |
| 7 | itransformer_co2_late_residual | 7.312 | 9.375 | 6.562 | 6.000 | 9.000 | 1.135 | 1.230 | 36.866 | 0.1007 |
| 8 | segrnn_forecaster | 8.104 | 14.188 | 6.375 | 3.750 | 9.389 | 0.340 | 3.209 | 16.886 | 0.0738 |
| 9 | itransformer_co2_horizon_mixture | 8.729 | 3.250 | 12.375 | 10.562 | 13.722 | 1.305 | 1.713 | 41.012 | 0.1234 |
| 10 | dlinear_forecaster | 9.042 | 11.500 | 5.688 | 9.938 | 15.056 | 1.018 | 2.195 | 16.177 | 0.0977 |
| 11 | itransformer_co2_residual | 9.708 | 7.688 | 10.125 | 11.312 | 10.778 | 0.551 | 0.680 | 11.074 | 0.0654 |
| 12 | transformer_forecaster | 9.729 | 7.562 | 11.562 | 10.062 | 13.056 | 0.305 | 1.898 | 23.648 | 0.0589 |
| 13 | itransformer_co2_wavelet_residual | 10.625 | 4.938 | 10.938 | 16.000 | 13.611 | 0.997 | 2.260 | 12.312 | 0.0902 |
| 14 | nlinear_forecaster | 10.708 | 13.312 | 5.188 | 13.625 | 15.500 | 1.025 | 2.695 | 18.772 | 0.0958 |
| 15 | patchtst_residual | 11.646 | 10.250 | 13.062 | 11.625 | 14.833 | 0.851 | 2.598 | 30.908 | 0.0952 |
| 16 | frequency_forecaster | 13.479 | 16.000 | 13.375 | 11.062 | 18.722 | 1.158 | 6.495 | 31.750 | 0.3964 |

## Metric Transfer Quality

### Target: `mpc_tair_mae`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rhair_first_step_mae | selection | 0.157 | 0.291 | 0.600 | no | no | 0.667 |
| tair_control_horizon_abs_bias | selection | -0.159 | 0.068 | 0.558 | no | no | 0.333 |
| co2_control_horizon_mae | selection | 0.052 | 0.081 | 0.529 | no | no | 0.333 |
| rhair_control_horizon_mae | selection | 0.103 | 0.079 | 0.525 | no | no | 0.000 |
| rhair_transfer_selection_score | selection | -0.085 | 0.075 | 0.521 | no | no | 0.333 |
| co2_first_step_mae | selection | 0.067 | 0.072 | 0.521 | no | no | 0.000 |
| tair_constraint_near_mae_proxy | selection | -0.156 | 0.057 | 0.521 | no | no | 0.000 |
| co2_final_step_mae | selection | -0.056 | -0.006 | 0.517 | no | no | 0.333 |
| co2_weighted_horizon_mae | selection | -0.002 | 0.012 | 0.508 | no | no | 0.667 |
| tair_final_step_mae | selection | -0.155 | -0.085 | 0.500 | no | no | 0.000 |
| co2_transfer_selection_score | selection | -0.035 | -0.021 | 0.500 | no | no | 0.333 |
| rhair_control_horizon_abs_bias | selection | -0.188 | -0.085 | 0.492 | no | no | 0.000 |
| tair_first_step_mae | selection | 0.030 | -0.056 | 0.492 | no | no | 0.000 |
| tair_transfer_selection_score | selection | -0.247 | -0.044 | 0.492 | no | no | 0.000 |
| co2_full_horizon_mae | selection | -0.008 | -0.012 | 0.483 | no | no | 0.667 |
| multiobjective_transfer_selection_score | selection | -0.191 | -0.079 | 0.475 | no | no | 0.333 |
| tair_control_horizon_mae | selection | -0.027 | -0.146 | 0.462 | no | no | 0.000 |
| co2_constraint_near_mae_proxy | selection | -0.054 | -0.150 | 0.458 | no | no | 0.667 |
| rhair_final_step_mae | selection | -0.151 | -0.152 | 0.454 | no | no | 0.333 |
| rhair_full_horizon_mae | selection | -0.126 | -0.209 | 0.442 | no | no | 0.333 |
| co2_control_horizon_abs_bias | selection | -0.062 | -0.262 | 0.433 | no | no | 0.333 |
| rhair_weighted_horizon_mae | selection | -0.105 | -0.205 | 0.429 | no | no | 0.333 |
| forecast_only_transfer_rank | selection | -0.324 | -0.256 | 0.407 | no | no | 0.333 |
| tair_full_horizon_mae | selection | -0.146 | -0.318 | 0.400 | no | no | 0.000 |
| tair_weighted_horizon_mae | selection | -0.137 | -0.324 | 0.383 | no | no | 0.000 |
| rhair_constraint_near_mae_proxy | selection | -0.263 | -0.362 | 0.367 | no | no | 0.000 |
| assim_sp_first_grad | diagnostic | 0.711 | 0.670 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | 0.586 | 0.614 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | 0.511 | 0.611 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.409 | -0.542 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.408 | 0.532 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | 0.471 | 0.508 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.526 | -0.481 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | -0.268 | -0.481 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | 0.437 | 0.443 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | 0.362 | 0.437 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | 0.345 | 0.422 |  |  |  |  |
| co2_sp_first_grad | diagnostic | 0.386 | 0.358 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | 0.343 | 0.208 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | -0.011 | 0.132 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | -0.038 | 0.099 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | 0.022 | 0.081 |  |  |  |  |

### Target: `mpc_rhair_mae`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | selection | 0.809 | 0.571 | 0.717 | no | no | 0.333 |
| co2_first_step_mae | selection | 0.550 | 0.539 | 0.697 | no | no | 0.667 |
| co2_control_horizon_mae | selection | 0.394 | 0.421 | 0.672 | no | no | 0.333 |
| multiobjective_transfer_selection_score | selection | 0.510 | 0.424 | 0.667 | no | no | 0.333 |
| rhair_final_step_mae | selection | 0.702 | 0.456 | 0.664 | no | no | 0.000 |
| rhair_constraint_near_mae_proxy | selection | 0.548 | 0.388 | 0.642 | no | no | 0.333 |
| co2_transfer_selection_score | selection | 0.246 | 0.365 | 0.642 | no | no | 0.333 |
| rhair_full_horizon_mae | selection | 0.717 | 0.397 | 0.633 | no | no | 0.000 |
| tair_transfer_selection_score | selection | 0.545 | 0.382 | 0.633 | no | no | 0.333 |
| rhair_weighted_horizon_mae | selection | 0.713 | 0.368 | 0.622 | no | no | 0.000 |
| tair_control_horizon_mae | selection | 0.806 | 0.347 | 0.622 | no | no | 0.333 |
| forecast_only_transfer_rank | selection | 0.507 | 0.342 | 0.610 | no | no | 0.000 |
| tair_full_horizon_mae | selection | 0.770 | 0.235 | 0.608 | no | no | 0.000 |
| co2_control_horizon_abs_bias | selection | -0.012 | 0.191 | 0.608 | no | no | 0.333 |
| tair_control_horizon_abs_bias | selection | 0.593 | 0.241 | 0.600 | no | no | 0.000 |
| tair_weighted_horizon_mae | selection | 0.777 | 0.224 | 0.592 | no | no | 0.000 |
| co2_weighted_horizon_mae | selection | 0.148 | 0.191 | 0.583 | no | no | 0.333 |
| tair_constraint_near_mae_proxy | selection | 0.667 | 0.165 | 0.563 | no | yes | 0.333 |
| tair_final_step_mae | selection | 0.720 | 0.171 | 0.558 | no | no | 0.000 |
| co2_full_horizon_mae | selection | 0.099 | 0.121 | 0.558 | no | no | 0.000 |
| co2_final_step_mae | selection | -0.130 | 0.015 | 0.525 | no | no | 0.333 |
| rhair_control_horizon_mae | selection | 0.594 | 0.032 | 0.517 | no | no | 0.000 |
| rhair_transfer_selection_score | selection | 0.161 | -0.004 | 0.496 | no | no | 0.000 |
| co2_constraint_near_mae_proxy | selection | -0.188 | -0.056 | 0.483 | no | no | 0.333 |
| rhair_first_step_mae | selection | 0.596 | -0.068 | 0.458 | no | no | 0.000 |
| rhair_control_horizon_abs_bias | selection | -0.368 | -0.371 | 0.383 | no | yes | 0.333 |
| rhair_dx_sp_first_grad | diagnostic | -0.586 | -0.627 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | -0.511 | -0.503 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | -0.484 | -0.492 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | -0.414 | -0.474 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | -0.374 | -0.353 |  |  |  |  |
| co2_sp_first_grad | diagnostic | 0.350 | 0.347 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | 0.283 | 0.318 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | 0.311 | 0.259 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | 0.133 | 0.218 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | -0.294 | -0.174 |  |  |  |  |
| assim_sp_first_grad | diagnostic | 0.113 | 0.130 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | 0.591 | 0.127 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.062 | 0.091 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | 0.066 | -0.056 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.102 | -0.031 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.102 | 0.006 |  |  |  |  |

### Target: `mpc_co2_mae`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rhair_control_horizon_abs_bias | selection | 0.444 | 0.515 | 0.675 | no | yes | 0.333 |
| rhair_transfer_selection_score | selection | 0.323 | 0.353 | 0.605 | no | no | 0.000 |
| rhair_control_horizon_mae | selection | 0.251 | 0.253 | 0.592 | no | no | 0.000 |
| rhair_first_step_mae | selection | 0.264 | 0.206 | 0.583 | no | no | 0.000 |
| tair_constraint_near_mae_proxy | selection | 0.022 | 0.237 | 0.580 | no | yes | 0.667 |
| rhair_full_horizon_mae | selection | 0.079 | 0.038 | 0.542 | no | no | 0.333 |
| tair_final_step_mae | selection | -0.001 | 0.021 | 0.533 | no | no | 0.000 |
| rhair_weighted_horizon_mae | selection | 0.106 | 0.010 | 0.521 | no | no | 0.333 |
| tair_weighted_horizon_mae | selection | 0.002 | 0.059 | 0.517 | no | no | 0.333 |
| tair_control_horizon_abs_bias | selection | -0.042 | 0.053 | 0.508 | no | no | 0.000 |
| tair_full_horizon_mae | selection | -0.000 | 0.041 | 0.500 | no | no | 0.000 |
| tair_transfer_selection_score | selection | -0.218 | -0.129 | 0.492 | no | no | 0.000 |
| forecast_only_transfer_rank | selection | -0.096 | -0.108 | 0.492 | no | no | 0.000 |
| tair_control_horizon_mae | selection | 0.027 | -0.137 | 0.487 | no | no | 0.000 |
| co2_final_step_mae | selection | 0.054 | -0.062 | 0.483 | no | no | 0.000 |
| rhair_constraint_near_mae_proxy | selection | 0.076 | 0.038 | 0.483 | no | no | 0.333 |
| multiobjective_transfer_selection_score | selection | -0.129 | -0.124 | 0.458 | no | no | 0.000 |
| co2_full_horizon_mae | selection | -0.220 | -0.174 | 0.450 | no | no | 0.000 |
| rhair_final_step_mae | selection | -0.073 | -0.140 | 0.445 | no | no | 0.000 |
| co2_weighted_horizon_mae | selection | -0.245 | -0.259 | 0.425 | no | no | 0.000 |
| co2_control_horizon_abs_bias | selection | -0.275 | -0.194 | 0.417 | no | no | 0.000 |
| co2_first_step_mae | selection | -0.265 | -0.263 | 0.395 | no | no | 0.000 |
| tair_first_step_mae | selection | 0.009 | -0.282 | 0.392 | no | no | 0.000 |
| co2_control_horizon_mae | selection | -0.285 | -0.290 | 0.387 | no | no | 0.000 |
| co2_constraint_near_mae_proxy | selection | -0.309 | -0.429 | 0.375 | no | no | 0.000 |
| co2_transfer_selection_score | selection | -0.274 | -0.321 | 0.367 | no | no | 0.000 |
| co2_sp_first_grad | diagnostic | -0.622 | -0.534 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | 0.434 | 0.458 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | 0.403 | 0.366 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | 0.423 | 0.355 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | -0.280 | -0.308 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | 0.220 | 0.255 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | 0.246 | 0.234 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | -0.118 | -0.228 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | 0.286 | 0.216 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.028 | 0.157 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | 0.210 | 0.104 |  |  |  |  |
| assim_sp_first_grad | diagnostic | -0.077 | -0.066 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | -0.005 | -0.063 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | -0.204 | -0.041 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | 0.004 | 0.039 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | -0.005 | -0.022 |  |  |  |  |

### Target: `mpc_objective`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rhair_first_step_mae | selection | 0.870 | 0.379 | 0.633 | no | yes | 0.333 |
| rhair_control_horizon_mae | selection | 0.816 | 0.268 | 0.625 | no | no | 0.000 |
| rhair_transfer_selection_score | selection | 0.408 | 0.271 | 0.622 | no | yes | 0.333 |
| tair_constraint_near_mae_proxy | selection | 0.629 | 0.281 | 0.605 | no | no | 0.333 |
| tair_control_horizon_abs_bias | selection | 0.610 | 0.191 | 0.575 | no | yes | 0.333 |
| co2_final_step_mae | selection | 0.088 | 0.126 | 0.550 | no | no | 0.000 |
| tair_final_step_mae | selection | 0.678 | 0.097 | 0.533 | no | no | 0.000 |
| co2_full_horizon_mae | selection | 0.123 | 0.082 | 0.533 | no | yes | 0.333 |
| co2_weighted_horizon_mae | selection | 0.145 | 0.032 | 0.508 | no | yes | 0.333 |
| rhair_full_horizon_mae | selection | 0.720 | 0.021 | 0.508 | no | yes | 0.333 |
| rhair_control_horizon_abs_bias | selection | -0.286 | -0.006 | 0.508 | no | no | 0.333 |
| rhair_final_step_mae | selection | 0.590 | 0.010 | 0.504 | yes | yes | 0.333 |
| rhair_weighted_horizon_mae | selection | 0.744 | -0.004 | 0.479 | no | yes | 0.333 |
| tair_transfer_selection_score | selection | 0.400 | -0.029 | 0.475 | no | no | 0.000 |
| co2_first_step_mae | selection | 0.463 | -0.104 | 0.471 | no | no | 0.000 |
| co2_control_horizon_mae | selection | 0.320 | -0.113 | 0.462 | no | no | 0.000 |
| tair_first_step_mae | selection | 0.823 | -0.112 | 0.458 | no | no | 0.000 |
| multiobjective_transfer_selection_score | selection | 0.445 | -0.074 | 0.458 | yes | yes | 0.333 |
| tair_full_horizon_mae | selection | 0.720 | -0.124 | 0.450 | no | no | 0.000 |
| rhair_constraint_near_mae_proxy | selection | 0.470 | -0.112 | 0.450 | no | no | 0.000 |
| forecast_only_transfer_rank | selection | 0.461 | -0.099 | 0.449 | no | yes | 0.333 |
| tair_control_horizon_mae | selection | 0.818 | -0.122 | 0.445 | no | no | 0.000 |
| co2_transfer_selection_score | selection | 0.072 | -0.244 | 0.433 | no | no | 0.000 |
| tair_weighted_horizon_mae | selection | 0.732 | -0.132 | 0.433 | no | no | 0.000 |
| co2_constraint_near_mae_proxy | selection | -0.214 | -0.271 | 0.392 | no | yes | 0.333 |
| co2_control_horizon_abs_bias | selection | -0.184 | -0.435 | 0.350 | no | no | 0.000 |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.343 | -0.649 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.342 | 0.626 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | 0.001 | 0.584 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | -0.072 | 0.493 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | -0.304 | -0.481 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | 0.307 | 0.478 |  |  |  |  |
| assim_sp_first_grad | diagnostic | 0.237 | 0.467 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.369 | -0.449 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | -0.104 | 0.428 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | -0.069 | 0.402 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | 0.224 | 0.331 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | -0.230 | 0.266 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | 0.203 | 0.221 |  |  |  |  |
| co2_sp_first_grad | diagnostic | 0.250 | 0.122 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | 0.328 | 0.069 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | 0.652 | -0.022 |  |  |  |  |

## Robustness Summary

### Target: `mpc_tair_mae`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | offline_or_diagnostic_only | -0.056 | -0.200 .. 0.089 | -0.200 .. 0.089 | 0.438 |
| tair_control_horizon_mae | offline_or_diagnostic_only | -0.146 | -0.288 .. 0.034 | -0.290 .. 0.034 | 0.404 |
| tair_weighted_horizon_mae | offline_or_diagnostic_only | -0.324 | -0.504 .. -0.182 | -0.504 .. -0.182 | 0.314 |
| tair_full_horizon_mae | offline_or_diagnostic_only | -0.318 | -0.496 .. -0.175 | -0.496 .. -0.175 | 0.333 |
| tair_final_step_mae | offline_or_diagnostic_only | -0.085 | -0.239 .. 0.050 | -0.239 .. 0.050 | 0.448 |
| tair_control_horizon_abs_bias | offline_or_diagnostic_only | 0.068 | -0.032 .. 0.296 | -0.060 .. 0.280 | 0.514 |
| tair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.057 | -0.088 .. 0.198 | -0.088 .. 0.198 | 0.471 |
| rhair_first_step_mae | weak_selection | 0.291 | 0.182 .. 0.429 | 0.182 .. 0.505 | 0.562 |
| rhair_control_horizon_mae | offline_or_diagnostic_only | 0.079 | -0.082 .. 0.161 | -0.082 .. 0.220 | 0.476 |
| rhair_weighted_horizon_mae | offline_or_diagnostic_only | -0.205 | -0.402 .. -0.088 | -0.402 .. -0.036 | 0.365 |
| rhair_full_horizon_mae | offline_or_diagnostic_only | -0.209 | -0.404 .. -0.093 | -0.404 .. -0.033 | 0.381 |
| rhair_final_step_mae | offline_or_diagnostic_only | -0.152 | -0.320 .. -0.023 | -0.320 .. -0.023 | 0.394 |
| rhair_control_horizon_abs_bias | offline_or_diagnostic_only | -0.085 | -0.225 .. 0.111 | -0.225 .. 0.111 | 0.438 |
| rhair_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.362 | -0.568 .. -0.246 | -0.568 .. -0.246 | 0.295 |
| co2_first_step_mae | offline_or_diagnostic_only | 0.072 | -0.063 .. 0.175 | -0.063 .. 0.175 | 0.481 |
| co2_control_horizon_mae | offline_or_diagnostic_only | 0.081 | -0.041 .. 0.189 | -0.041 .. 0.189 | 0.490 |
| co2_weighted_horizon_mae | offline_or_diagnostic_only | 0.012 | -0.164 .. 0.104 | -0.164 .. 0.143 | 0.448 |
| co2_full_horizon_mae | offline_or_diagnostic_only | -0.012 | -0.196 .. 0.093 | -0.196 .. 0.143 | 0.419 |
| co2_final_step_mae | offline_or_diagnostic_only | -0.006 | -0.196 .. 0.136 | -0.196 .. 0.136 | 0.457 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | -0.262 | -0.425 .. -0.186 | -0.425 .. -0.186 | 0.381 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.150 | -0.354 .. -0.032 | -0.354 .. -0.032 | 0.390 |
| forecast_only_transfer_rank | offline_or_diagnostic_only | -0.256 | -0.443 .. -0.107 | -0.443 .. -0.182 | 0.333 |
| tair_transfer_selection_score | offline_or_diagnostic_only | -0.044 | -0.168 .. 0.154 | -0.204 .. 0.154 | 0.438 |
| rhair_transfer_selection_score | offline_or_diagnostic_only | 0.075 | -0.075 .. 0.193 | -0.075 .. 0.269 | 0.471 |
| co2_transfer_selection_score | offline_or_diagnostic_only | -0.021 | -0.143 .. 0.079 | -0.143 .. 0.079 | 0.457 |
| multiobjective_transfer_selection_score | offline_or_diagnostic_only | -0.079 | -0.239 .. 0.107 | -0.239 .. 0.055 | 0.429 |

### Target: `mpc_rhair_mae`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | secondary_selection | 0.571 | 0.479 .. 0.682 | 0.473 .. 0.654 | 0.676 |
| tair_control_horizon_mae | weak_selection | 0.347 | 0.207 .. 0.554 | 0.187 .. 0.561 | 0.567 |
| tair_weighted_horizon_mae | offline_or_diagnostic_only | 0.224 | 0.057 .. 0.404 | 0.051 .. 0.582 | 0.533 |
| tair_full_horizon_mae | offline_or_diagnostic_only | 0.235 | 0.071 .. 0.418 | 0.064 .. 0.593 | 0.552 |
| tair_final_step_mae | offline_or_diagnostic_only | 0.171 | -0.007 .. 0.264 | -0.007 .. 0.434 | 0.495 |
| tair_control_horizon_abs_bias | offline_or_diagnostic_only | 0.241 | 0.079 .. 0.454 | 0.079 .. 0.385 | 0.543 |
| tair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.165 | -0.014 .. 0.257 | -0.014 .. 0.297 | 0.500 |
| rhair_first_step_mae | offline_or_diagnostic_only | -0.068 | -0.296 .. 0.046 | -0.296 .. 0.046 | 0.381 |
| rhair_control_horizon_mae | offline_or_diagnostic_only | 0.032 | -0.175 .. 0.146 | -0.175 .. 0.178 | 0.448 |
| rhair_weighted_horizon_mae | secondary_selection | 0.368 | 0.232 .. 0.497 | 0.232 .. 0.506 | 0.567 |
| rhair_full_horizon_mae | secondary_selection | 0.397 | 0.268 .. 0.493 | 0.268 .. 0.511 | 0.581 |
| rhair_final_step_mae | secondary_selection | 0.456 | 0.340 .. 0.561 | 0.340 .. 0.572 | 0.615 |
| rhair_control_horizon_abs_bias | offline_or_diagnostic_only | -0.371 | -0.568 .. -0.246 | -0.549 .. -0.231 | 0.314 |
| rhair_constraint_near_mae_proxy | secondary_selection | 0.388 | 0.257 .. 0.471 | 0.257 .. 0.566 | 0.590 |
| co2_first_step_mae | secondary_selection | 0.539 | 0.458 .. 0.676 | 0.458 .. 0.655 | 0.663 |
| co2_control_horizon_mae | secondary_selection | 0.421 | 0.315 .. 0.593 | 0.315 .. 0.600 | 0.635 |
| co2_weighted_horizon_mae | offline_or_diagnostic_only | 0.191 | 0.036 .. 0.382 | 0.036 .. 0.429 | 0.533 |
| co2_full_horizon_mae | offline_or_diagnostic_only | 0.121 | -0.050 .. 0.300 | -0.050 .. 0.374 | 0.505 |
| co2_final_step_mae | offline_or_diagnostic_only | 0.015 | -0.082 .. 0.182 | -0.082 .. 0.182 | 0.486 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | 0.191 | 0.061 .. 0.354 | 0.061 .. 0.354 | 0.571 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.056 | -0.164 .. 0.111 | -0.164 .. 0.170 | 0.438 |
| forecast_only_transfer_rank | weak_selection | 0.342 | 0.218 .. 0.593 | 0.218 .. 0.654 | 0.552 |
| tair_transfer_selection_score | secondary_selection | 0.382 | 0.250 .. 0.550 | 0.218 .. 0.505 | 0.581 |
| rhair_transfer_selection_score | offline_or_diagnostic_only | -0.004 | -0.220 .. 0.093 | -0.220 .. 0.130 | 0.423 |
| co2_transfer_selection_score | secondary_selection | 0.365 | 0.282 .. 0.539 | 0.282 .. 0.529 | 0.610 |
| multiobjective_transfer_selection_score | secondary_selection | 0.424 | 0.311 .. 0.629 | 0.311 .. 0.604 | 0.629 |

### Target: `mpc_co2_mae`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | offline_or_diagnostic_only | -0.282 | -0.439 .. -0.129 | -0.439 .. -0.129 | 0.333 |
| tair_control_horizon_mae | offline_or_diagnostic_only | -0.137 | -0.266 .. 0.041 | -0.294 .. 0.041 | 0.442 |
| tair_weighted_horizon_mae | offline_or_diagnostic_only | 0.059 | -0.071 .. 0.257 | -0.231 .. 0.257 | 0.476 |
| tair_full_horizon_mae | offline_or_diagnostic_only | 0.041 | -0.075 .. 0.236 | -0.236 .. 0.236 | 0.457 |
| tair_final_step_mae | offline_or_diagnostic_only | 0.021 | -0.079 .. 0.175 | -0.154 .. 0.175 | 0.495 |
| tair_control_horizon_abs_bias | offline_or_diagnostic_only | 0.053 | -0.050 .. 0.200 | -0.104 .. 0.200 | 0.467 |
| tair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.237 | 0.080 .. 0.399 | -0.096 .. 0.399 | 0.529 |
| rhair_first_step_mae | offline_or_diagnostic_only | 0.206 | 0.068 .. 0.368 | 0.068 .. 0.577 | 0.543 |
| rhair_control_horizon_mae | weak_selection | 0.253 | 0.121 .. 0.386 | 0.115 .. 0.533 | 0.552 |
| rhair_weighted_horizon_mae | offline_or_diagnostic_only | 0.010 | -0.116 .. 0.113 | -0.173 .. 0.256 | 0.481 |
| rhair_full_horizon_mae | offline_or_diagnostic_only | 0.038 | -0.086 .. 0.139 | -0.132 .. 0.253 | 0.505 |
| rhair_final_step_mae | offline_or_diagnostic_only | -0.140 | -0.281 .. 0.002 | -0.281 .. 0.002 | 0.394 |
| rhair_control_horizon_abs_bias | secondary_selection | 0.515 | 0.457 .. 0.689 | 0.429 .. 0.689 | 0.648 |
| rhair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.038 | -0.075 .. 0.225 | -0.075 .. 0.225 | 0.438 |
| co2_first_step_mae | offline_or_diagnostic_only | -0.263 | -0.402 .. -0.123 | -0.402 .. 0.080 | 0.346 |
| co2_control_horizon_mae | offline_or_diagnostic_only | -0.290 | -0.427 .. -0.148 | -0.427 .. 0.102 | 0.337 |
| co2_weighted_horizon_mae | offline_or_diagnostic_only | -0.259 | -0.393 .. -0.111 | -0.393 .. -0.027 | 0.371 |
| co2_full_horizon_mae | offline_or_diagnostic_only | -0.174 | -0.293 .. -0.007 | -0.293 .. 0.016 | 0.400 |
| co2_final_step_mae | offline_or_diagnostic_only | -0.062 | -0.218 .. 0.114 | -0.171 .. 0.039 | 0.429 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | -0.194 | -0.314 .. -0.057 | -0.302 .. -0.059 | 0.381 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.429 | -0.586 .. -0.307 | -0.586 .. -0.214 | 0.314 |
| forecast_only_transfer_rank | offline_or_diagnostic_only | -0.108 | -0.236 .. -0.032 | -0.396 .. 0.019 | 0.457 |
| tair_transfer_selection_score | offline_or_diagnostic_only | -0.129 | -0.250 .. 0.054 | -0.264 .. 0.054 | 0.448 |
| rhair_transfer_selection_score | secondary_selection | 0.353 | 0.250 .. 0.486 | 0.236 .. 0.621 | 0.567 |
| co2_transfer_selection_score | offline_or_diagnostic_only | -0.321 | -0.436 .. -0.186 | -0.436 .. 0.049 | 0.324 |
| multiobjective_transfer_selection_score | offline_or_diagnostic_only | -0.124 | -0.286 .. -0.057 | -0.390 .. 0.143 | 0.400 |

### Target: `mpc_objective`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | offline_or_diagnostic_only | -0.112 | -0.350 .. 0.004 | -0.350 .. 0.004 | 0.381 |
| tair_control_horizon_mae | offline_or_diagnostic_only | -0.122 | -0.363 .. 0.005 | -0.363 .. 0.002 | 0.365 |
| tair_weighted_horizon_mae | offline_or_diagnostic_only | -0.132 | -0.375 .. -0.007 | -0.375 .. -0.032 | 0.352 |
| tair_full_horizon_mae | offline_or_diagnostic_only | -0.124 | -0.364 .. 0.004 | -0.364 .. -0.021 | 0.371 |
| tair_final_step_mae | offline_or_diagnostic_only | 0.097 | -0.096 .. 0.264 | -0.096 .. 0.264 | 0.467 |
| tair_control_horizon_abs_bias | offline_or_diagnostic_only | 0.191 | 0.018 .. 0.393 | 0.018 .. 0.401 | 0.514 |
| tair_constraint_near_mae_proxy | weak_selection | 0.281 | 0.127 .. 0.488 | 0.127 .. 0.488 | 0.548 |
| rhair_first_step_mae | objective_secondary_selection | 0.379 | 0.246 .. 0.507 | 0.246 .. 0.588 | 0.581 |
| rhair_control_horizon_mae | weak_selection | 0.268 | 0.111 .. 0.375 | 0.111 .. 0.445 | 0.571 |
| rhair_weighted_horizon_mae | offline_or_diagnostic_only | -0.004 | -0.220 .. 0.084 | -0.220 .. 0.168 | 0.404 |
| rhair_full_horizon_mae | offline_or_diagnostic_only | 0.021 | -0.189 .. 0.114 | -0.189 .. 0.170 | 0.438 |
| rhair_final_step_mae | offline_or_diagnostic_only | 0.010 | -0.202 .. 0.091 | -0.202 .. 0.129 | 0.433 |
| rhair_control_horizon_abs_bias | offline_or_diagnostic_only | -0.006 | -0.086 .. 0.179 | -0.121 .. 0.179 | 0.476 |
| rhair_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.112 | -0.350 .. -0.011 | -0.350 .. -0.011 | 0.371 |
| co2_first_step_mae | offline_or_diagnostic_only | -0.104 | -0.306 .. -0.002 | -0.306 .. 0.085 | 0.404 |
| co2_control_horizon_mae | offline_or_diagnostic_only | -0.113 | -0.316 .. -0.039 | -0.316 .. 0.080 | 0.394 |
| co2_weighted_horizon_mae | offline_or_diagnostic_only | 0.032 | -0.161 .. 0.111 | -0.161 .. 0.165 | 0.448 |
| co2_full_horizon_mae | offline_or_diagnostic_only | 0.082 | -0.100 .. 0.168 | -0.100 .. 0.220 | 0.476 |
| co2_final_step_mae | offline_or_diagnostic_only | 0.126 | 0.025 .. 0.271 | 0.025 .. 0.247 | 0.524 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | -0.435 | -0.568 .. -0.386 | -0.568 .. -0.385 | 0.314 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.271 | -0.493 .. -0.182 | -0.493 .. -0.154 | 0.324 |
| forecast_only_transfer_rank | offline_or_diagnostic_only | -0.099 | -0.336 .. 0.039 | -0.336 .. -0.025 | 0.371 |
| tair_transfer_selection_score | offline_or_diagnostic_only | -0.029 | -0.250 .. 0.114 | -0.250 .. 0.114 | 0.400 |
| rhair_transfer_selection_score | weak_selection | 0.271 | 0.114 .. 0.386 | 0.114 .. 0.467 | 0.567 |
| co2_transfer_selection_score | offline_or_diagnostic_only | -0.244 | -0.339 .. -0.139 | -0.339 .. -0.082 | 0.400 |
| multiobjective_transfer_selection_score | offline_or_diagnostic_only | -0.074 | -0.311 .. 0.018 | -0.311 .. 0.066 | 0.371 |
