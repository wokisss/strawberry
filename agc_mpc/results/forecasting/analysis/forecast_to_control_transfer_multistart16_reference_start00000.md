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
| mpc_rhair_mae | tair_first_step_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | tair_control_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | tair_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | tair_full_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | tair_final_step_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | tair_control_horizon_abs_bias | weak_selection |
| mpc_rhair_mae | tair_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_first_step_mae | weak_selection |
| mpc_rhair_mae | rhair_control_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_full_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_final_step_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_first_step_mae | weak_selection |
| mpc_rhair_mae | co2_control_horizon_mae | weak_selection |
| mpc_rhair_mae | co2_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_full_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_final_step_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_rhair_mae | forecast_only_transfer_rank | offline_or_diagnostic_only |
| mpc_rhair_mae | tair_transfer_selection_score | weak_selection |
| mpc_rhair_mae | rhair_transfer_selection_score | offline_or_diagnostic_only |
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
| mpc_objective | tair_final_step_mae | offline_or_diagnostic_only |
| mpc_objective | tair_control_horizon_abs_bias | weak_selection |
| mpc_objective | tair_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_objective | rhair_first_step_mae | objective_secondary_selection |
| mpc_objective | rhair_control_horizon_mae | weak_selection |
| mpc_objective | rhair_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_objective | rhair_full_horizon_mae | offline_or_diagnostic_only |
| mpc_objective | rhair_final_step_mae | offline_or_diagnostic_only |
| mpc_objective | rhair_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_objective | rhair_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_objective | co2_first_step_mae | weak_selection |
| mpc_objective | co2_control_horizon_mae | weak_selection |
| mpc_objective | co2_weighted_horizon_mae | weak_selection |
| mpc_objective | co2_full_horizon_mae | weak_selection |
| mpc_objective | co2_final_step_mae | offline_or_diagnostic_only |
| mpc_objective | co2_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_objective | co2_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_objective | forecast_only_transfer_rank | offline_or_diagnostic_only |
| mpc_objective | tair_transfer_selection_score | offline_or_diagnostic_only |
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
| 1 | current_hybrid_transformer | 5.521 | 6.750 | 4.438 | 5.375 | 6.722 | 0.362 | 1.206 | 18.818 | 0.0442 |
| 2 | itransformer_residual | 5.646 | 4.625 | 5.688 | 6.625 | 9.167 | 2.216 | 5.675 | 11.532 | 0.1924 |
| 3 | itransformer_co2_control_aware_fusion | 6.135 | 8.531 | 8.375 | 1.500 | 8.556 | 2.217 | 4.261 | 6.623 | 0.1505 |
| 4 | itransformer_co2_late_frozen_expert | 6.469 | 7.969 | 9.375 | 2.062 | 9.944 | 2.202 | 4.302 | 6.442 | 0.1538 |
| 5 | itransformer_co2_protected_expert | 6.479 | 4.312 | 9.812 | 5.312 | 9.278 | 0.880 | 1.441 | 14.206 | 0.0606 |
| 6 | transformer_hybrid_residual | 6.667 | 5.750 | 3.062 | 11.188 | 9.167 | 1.672 | 4.584 | 18.168 | 0.1062 |
| 7 | itransformer_co2_late_residual | 7.312 | 9.375 | 6.562 | 6.000 | 9.000 | 1.153 | 1.618 | 10.125 | 0.0705 |
| 8 | segrnn_forecaster | 8.104 | 14.188 | 6.375 | 3.750 | 9.389 | 0.391 | 2.195 | 14.425 | 0.0486 |
| 9 | itransformer_co2_horizon_mixture | 8.729 | 3.250 | 12.375 | 10.562 | 13.722 | 3.329 | 5.668 | 29.380 | 0.3734 |
| 10 | dlinear_forecaster | 9.042 | 11.500 | 5.688 | 9.938 | 15.056 | 3.436 | 6.459 | 37.824 | 0.3962 |
| 11 | itransformer_co2_residual | 9.708 | 7.688 | 10.125 | 11.312 | 10.778 | 0.938 | 1.500 | 6.331 | 0.0558 |
| 12 | transformer_forecaster | 9.729 | 7.562 | 11.562 | 10.062 | 13.056 | 1.037 | 4.063 | 16.466 | 0.0858 |
| 13 | itransformer_co2_wavelet_residual | 10.625 | 4.938 | 10.938 | 16.000 | 13.611 | 1.061 | 2.148 | 7.711 | 0.0636 |
| 14 | nlinear_forecaster | 10.708 | 13.312 | 5.188 | 13.625 | 15.500 | 1.867 | 4.182 | 25.236 | 0.1526 |
| 15 | patchtst_residual | 11.646 | 10.250 | 13.062 | 11.625 | 14.833 | 3.089 | 7.961 | 36.014 | 0.2628 |
| 16 | frequency_forecaster | 13.479 | 16.000 | 13.375 | 11.062 | 18.722 | 1.725 | 8.759 | 15.530 | 0.4338 |

## Metric Transfer Quality

### Target: `mpc_tair_mae`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rhair_first_step_mae | selection | 0.220 | 0.324 | 0.617 | no | yes | 0.667 |
| rhair_control_horizon_abs_bias | selection | 0.297 | 0.238 | 0.575 | no | no | 0.000 |
| rhair_transfer_selection_score | selection | 0.204 | 0.144 | 0.563 | no | yes | 0.333 |
| tair_control_horizon_abs_bias | selection | -0.014 | 0.171 | 0.558 | no | yes | 0.333 |
| multiobjective_transfer_selection_score | selection | 0.181 | 0.150 | 0.558 | yes | yes | 0.333 |
| rhair_control_horizon_mae | selection | 0.162 | 0.109 | 0.558 | no | no | 0.000 |
| co2_first_step_mae | selection | -0.058 | 0.144 | 0.546 | no | no | 0.333 |
| tair_transfer_selection_score | selection | -0.013 | 0.100 | 0.542 | no | no | 0.333 |
| co2_control_horizon_mae | selection | -0.038 | 0.124 | 0.538 | no | no | 0.333 |
| co2_transfer_selection_score | selection | 0.156 | 0.118 | 0.533 | no | no | 0.333 |
| co2_full_horizon_mae | selection | -0.012 | 0.082 | 0.533 | no | yes | 0.667 |
| co2_weighted_horizon_mae | selection | -0.021 | 0.088 | 0.525 | no | yes | 0.667 |
| co2_constraint_near_mae_proxy | selection | 0.014 | 0.088 | 0.525 | no | yes | 0.667 |
| tair_constraint_near_mae_proxy | selection | -0.200 | 0.009 | 0.513 | no | no | 0.000 |
| tair_control_horizon_mae | selection | -0.071 | -0.012 | 0.504 | no | no | 0.333 |
| co2_control_horizon_abs_bias | selection | -0.098 | -0.015 | 0.500 | no | no | 0.333 |
| co2_final_step_mae | selection | -0.082 | 0.003 | 0.500 | no | no | 0.333 |
| tair_first_step_mae | selection | -0.016 | -0.009 | 0.492 | no | no | 0.333 |
| forecast_only_transfer_rank | selection | -0.066 | -0.090 | 0.475 | no | yes | 0.333 |
| rhair_final_step_mae | selection | -0.189 | -0.215 | 0.412 | yes | yes | 0.333 |
| rhair_full_horizon_mae | selection | -0.128 | -0.232 | 0.408 | no | yes | 0.333 |
| rhair_weighted_horizon_mae | selection | -0.098 | -0.224 | 0.403 | no | yes | 0.333 |
| tair_final_step_mae | selection | -0.300 | -0.291 | 0.400 | no | no | 0.000 |
| tair_weighted_horizon_mae | selection | -0.226 | -0.359 | 0.367 | no | no | 0.000 |
| tair_full_horizon_mae | selection | -0.243 | -0.371 | 0.350 | no | no | 0.000 |
| rhair_constraint_near_mae_proxy | selection | -0.406 | -0.450 | 0.333 | no | no | 0.000 |
| assim_sp_first_grad | diagnostic | 0.606 | 0.650 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | 0.297 | 0.521 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | 0.478 | 0.518 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | 0.346 | 0.450 |  |  |  |  |
| co2_sp_first_grad | diagnostic | 0.340 | 0.412 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | 0.366 | 0.353 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | 0.392 | 0.312 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.342 | -0.300 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | 0.201 | 0.268 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | 0.184 | 0.244 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.115 | -0.241 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.112 | 0.188 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | 0.053 | 0.185 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | -0.205 | -0.141 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | 0.202 | 0.074 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | 0.106 | -0.044 |  |  |  |  |

### Target: `mpc_rhair_mae`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| multiobjective_transfer_selection_score | selection | 0.497 | 0.332 | 0.642 | yes | yes | 0.333 |
| co2_first_step_mae | selection | 0.240 | 0.331 | 0.622 | no | no | 0.333 |
| rhair_first_step_mae | selection | 0.549 | 0.282 | 0.617 | no | yes | 0.333 |
| co2_control_horizon_mae | selection | 0.163 | 0.272 | 0.597 | no | no | 0.000 |
| rhair_transfer_selection_score | selection | 0.351 | 0.240 | 0.597 | no | yes | 0.333 |
| tair_control_horizon_abs_bias | selection | 0.372 | 0.324 | 0.592 | no | yes | 0.333 |
| tair_transfer_selection_score | selection | 0.346 | 0.285 | 0.592 | no | no | 0.333 |
| rhair_control_horizon_mae | selection | 0.521 | 0.226 | 0.575 | no | no | 0.000 |
| tair_first_step_mae | selection | 0.498 | 0.221 | 0.575 | no | no | 0.333 |
| co2_weighted_horizon_mae | selection | 0.035 | 0.174 | 0.575 | no | yes | 0.333 |
| tair_constraint_near_mae_proxy | selection | 0.266 | 0.222 | 0.571 | no | no | 0.333 |
| co2_transfer_selection_score | selection | 0.254 | 0.209 | 0.567 | no | no | 0.000 |
| co2_full_horizon_mae | selection | 0.015 | 0.138 | 0.567 | no | yes | 0.333 |
| tair_control_horizon_mae | selection | 0.463 | 0.187 | 0.563 | no | no | 0.333 |
| forecast_only_transfer_rank | selection | 0.335 | 0.108 | 0.551 | no | yes | 0.333 |
| rhair_control_horizon_abs_bias | selection | 0.086 | 0.059 | 0.525 | no | no | 0.333 |
| rhair_full_horizon_mae | selection | 0.409 | 0.053 | 0.525 | no | yes | 0.333 |
| rhair_final_step_mae | selection | 0.343 | 0.081 | 0.521 | yes | yes | 0.333 |
| co2_final_step_mae | selection | -0.115 | 0.009 | 0.517 | no | no | 0.000 |
| rhair_weighted_horizon_mae | selection | 0.429 | 0.057 | 0.513 | no | yes | 0.333 |
| co2_constraint_near_mae_proxy | selection | -0.150 | -0.012 | 0.492 | no | yes | 0.333 |
| rhair_constraint_near_mae_proxy | selection | 0.137 | -0.065 | 0.483 | no | no | 0.000 |
| co2_control_horizon_abs_bias | selection | -0.135 | -0.065 | 0.483 | no | no | 0.000 |
| tair_final_step_mae | selection | 0.229 | -0.100 | 0.467 | no | no | 0.000 |
| tair_full_horizon_mae | selection | 0.309 | -0.147 | 0.433 | no | no | 0.000 |
| tair_weighted_horizon_mae | selection | 0.326 | -0.132 | 0.433 | no | no | 0.000 |
| assim_sp_first_grad | diagnostic | 0.423 | 0.514 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | -0.289 | -0.364 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | 0.474 | 0.358 |  |  |  |  |
| co2_sp_first_grad | diagnostic | 0.367 | 0.355 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | 0.528 | 0.305 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | 0.305 | 0.287 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | 0.305 | 0.266 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | -0.264 | -0.184 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.097 | -0.140 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | -0.092 | -0.137 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | -0.030 | 0.134 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | 0.040 | -0.102 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | 0.080 | -0.050 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | -0.105 | -0.047 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | -0.093 | -0.004 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | -0.083 | 0.000 |  |  |  |  |

### Target: `mpc_co2_mae`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| co2_control_horizon_abs_bias | selection | 0.097 | 0.491 | 0.667 | no | no | 0.667 |
| co2_first_step_mae | selection | -0.042 | 0.366 | 0.630 | no | no | 0.667 |
| co2_transfer_selection_score | selection | 0.375 | 0.312 | 0.617 | no | no | 0.667 |
| multiobjective_transfer_selection_score | selection | 0.337 | 0.309 | 0.608 | no | no | 0.333 |
| co2_control_horizon_mae | selection | -0.015 | 0.278 | 0.605 | no | no | 0.667 |
| tair_weighted_horizon_mae | selection | -0.037 | 0.206 | 0.583 | no | no | 0.000 |
| tair_full_horizon_mae | selection | -0.048 | 0.182 | 0.567 | no | no | 0.000 |
| tair_constraint_near_mae_proxy | selection | -0.058 | 0.149 | 0.563 | no | yes | 0.333 |
| forecast_only_transfer_rank | selection | 0.190 | 0.156 | 0.559 | no | no | 0.000 |
| tair_first_step_mae | selection | 0.077 | 0.185 | 0.558 | no | no | 0.000 |
| co2_constraint_near_mae_proxy | selection | 0.026 | 0.147 | 0.558 | no | no | 0.333 |
| tair_control_horizon_mae | selection | 0.046 | 0.190 | 0.555 | no | no | 0.000 |
| tair_transfer_selection_score | selection | 0.199 | 0.141 | 0.542 | no | no | 0.000 |
| rhair_control_horizon_abs_bias | selection | 0.258 | 0.150 | 0.525 | no | yes | 0.333 |
| co2_weighted_horizon_mae | selection | -0.057 | 0.088 | 0.525 | no | no | 0.333 |
| tair_control_horizon_abs_bias | selection | 0.031 | 0.035 | 0.525 | no | no | 0.000 |
| rhair_constraint_near_mae_proxy | selection | -0.074 | 0.000 | 0.517 | no | no | 0.333 |
| rhair_full_horizon_mae | selection | -0.018 | 0.006 | 0.508 | no | no | 0.000 |
| rhair_weighted_horizon_mae | selection | -0.013 | -0.034 | 0.504 | no | no | 0.000 |
| co2_full_horizon_mae | selection | -0.070 | 0.050 | 0.500 | no | no | 0.000 |
| tair_final_step_mae | selection | -0.122 | -0.138 | 0.467 | no | no | 0.000 |
| co2_final_step_mae | selection | -0.208 | -0.074 | 0.467 | no | no | 0.333 |
| rhair_final_step_mae | selection | -0.029 | -0.122 | 0.462 | no | no | 0.000 |
| rhair_transfer_selection_score | selection | 0.018 | -0.100 | 0.462 | no | no | 0.000 |
| rhair_control_horizon_mae | selection | 0.023 | -0.118 | 0.442 | no | no | 0.000 |
| rhair_first_step_mae | selection | 0.046 | -0.162 | 0.433 | no | no | 0.000 |
| rhair_dx_sp_first_grad | diagnostic | -0.315 | -0.540 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | 0.305 | 0.467 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | -0.401 | -0.443 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | -0.306 | -0.438 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | 0.303 | 0.412 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | -0.243 | -0.378 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | -0.158 | -0.369 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | 0.106 | 0.269 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | 0.280 | 0.237 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | 0.229 | 0.190 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | -0.193 | -0.187 |  |  |  |  |
| assim_sp_first_grad | diagnostic | -0.036 | -0.175 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | -0.233 | -0.168 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | -0.036 | -0.107 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | 0.009 | -0.102 |  |  |  |  |
| co2_sp_first_grad | diagnostic | 0.042 | -0.010 |  |  |  |  |

### Target: `mpc_objective`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rhair_first_step_mae | selection | 0.673 | 0.418 | 0.658 | no | yes | 0.667 |
| rhair_transfer_selection_score | selection | 0.356 | 0.299 | 0.622 | no | yes | 0.333 |
| multiobjective_transfer_selection_score | selection | 0.465 | 0.338 | 0.617 | yes | yes | 0.333 |
| tair_control_horizon_abs_bias | selection | 0.360 | 0.276 | 0.617 | no | yes | 0.333 |
| co2_first_step_mae | selection | 0.250 | 0.309 | 0.613 | no | no | 0.000 |
| co2_control_horizon_mae | selection | 0.178 | 0.280 | 0.605 | no | no | 0.333 |
| co2_weighted_horizon_mae | selection | 0.061 | 0.279 | 0.600 | no | yes | 0.667 |
| rhair_control_horizon_mae | selection | 0.601 | 0.259 | 0.600 | no | no | 0.000 |
| co2_full_horizon_mae | selection | 0.044 | 0.253 | 0.592 | no | yes | 0.667 |
| tair_transfer_selection_score | selection | 0.300 | 0.215 | 0.583 | no | no | 0.000 |
| tair_constraint_near_mae_proxy | selection | 0.306 | 0.153 | 0.580 | no | no | 0.333 |
| co2_transfer_selection_score | selection | 0.237 | 0.200 | 0.575 | no | no | 0.333 |
| co2_final_step_mae | selection | -0.117 | 0.153 | 0.558 | no | no | 0.333 |
| tair_control_horizon_mae | selection | 0.459 | 0.150 | 0.555 | no | no | 0.000 |
| forecast_only_transfer_rank | selection | 0.290 | 0.133 | 0.551 | no | yes | 0.333 |
| tair_first_step_mae | selection | 0.490 | 0.153 | 0.550 | no | no | 0.000 |
| rhair_control_horizon_abs_bias | selection | 0.123 | 0.100 | 0.550 | no | no | 0.333 |
| co2_constraint_near_mae_proxy | selection | -0.087 | 0.068 | 0.517 | no | yes | 0.667 |
| co2_control_horizon_abs_bias | selection | -0.148 | -0.038 | 0.508 | no | no | 0.333 |
| rhair_full_horizon_mae | selection | 0.344 | 0.000 | 0.500 | no | yes | 0.333 |
| rhair_final_step_mae | selection | 0.208 | 0.006 | 0.496 | yes | yes | 0.333 |
| rhair_weighted_horizon_mae | selection | 0.379 | -0.009 | 0.487 | no | yes | 0.333 |
| tair_final_step_mae | selection | 0.255 | -0.097 | 0.475 | no | no | 0.000 |
| rhair_constraint_near_mae_proxy | selection | -0.032 | -0.162 | 0.442 | no | no | 0.000 |
| tair_full_horizon_mae | selection | 0.311 | -0.135 | 0.442 | no | no | 0.000 |
| tair_weighted_horizon_mae | selection | 0.327 | -0.132 | 0.442 | no | no | 0.000 |
| assim_sp_first_grad | diagnostic | 0.443 | 0.571 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | 0.372 | 0.465 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | 0.386 | 0.453 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | 0.423 | 0.447 |  |  |  |  |
| co2_sp_first_grad | diagnostic | 0.373 | 0.436 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | 0.407 | 0.341 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | -0.045 | 0.303 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.355 | -0.297 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | 0.059 | 0.256 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.187 | -0.200 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | -0.176 | -0.162 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.185 | 0.157 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | -0.109 | -0.103 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | 0.035 | 0.100 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | -0.093 | 0.032 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | -0.049 | -0.029 |  |  |  |  |

## Robustness Summary

### Target: `mpc_tair_mae`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | offline_or_diagnostic_only | -0.009 | -0.161 .. 0.125 | -0.203 .. 0.125 | 0.438 |
| tair_control_horizon_mae | offline_or_diagnostic_only | -0.012 | -0.132 .. 0.157 | -0.187 .. 0.157 | 0.462 |
| tair_weighted_horizon_mae | offline_or_diagnostic_only | -0.359 | -0.461 .. -0.257 | -0.549 .. -0.257 | 0.324 |
| tair_full_horizon_mae | offline_or_diagnostic_only | -0.371 | -0.475 .. -0.271 | -0.560 .. -0.271 | 0.305 |
| tair_final_step_mae | offline_or_diagnostic_only | -0.291 | -0.364 .. -0.171 | -0.467 .. -0.171 | 0.371 |
| tair_control_horizon_abs_bias | offline_or_diagnostic_only | 0.171 | 0.032 .. 0.325 | 0.016 .. 0.325 | 0.505 |
| tair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.009 | -0.057 .. 0.193 | -0.105 .. 0.193 | 0.490 |
| rhair_first_step_mae | weak_selection | 0.324 | 0.207 .. 0.421 | 0.207 .. 0.582 | 0.581 |
| rhair_control_horizon_mae | offline_or_diagnostic_only | 0.109 | -0.018 .. 0.264 | -0.018 .. 0.358 | 0.514 |
| rhair_weighted_horizon_mae | offline_or_diagnostic_only | -0.224 | -0.408 .. -0.086 | -0.408 .. -0.059 | 0.337 |
| rhair_full_horizon_mae | offline_or_diagnostic_only | -0.232 | -0.418 .. -0.096 | -0.418 .. -0.112 | 0.343 |
| rhair_final_step_mae | offline_or_diagnostic_only | -0.215 | -0.475 .. -0.107 | -0.475 .. -0.107 | 0.327 |
| rhair_control_horizon_abs_bias | offline_or_diagnostic_only | 0.238 | 0.136 .. 0.371 | 0.136 .. 0.371 | 0.533 |
| rhair_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.450 | -0.579 .. -0.336 | -0.579 .. -0.336 | 0.276 |
| co2_first_step_mae | offline_or_diagnostic_only | 0.144 | 0.057 .. 0.279 | 0.047 .. 0.279 | 0.519 |
| co2_control_horizon_mae | offline_or_diagnostic_only | 0.124 | 0.021 .. 0.261 | 0.021 .. 0.261 | 0.500 |
| co2_weighted_horizon_mae | offline_or_diagnostic_only | 0.088 | -0.104 .. 0.168 | -0.104 .. 0.319 | 0.467 |
| co2_full_horizon_mae | offline_or_diagnostic_only | 0.082 | -0.111 .. 0.164 | -0.111 .. 0.390 | 0.476 |
| co2_final_step_mae | offline_or_diagnostic_only | 0.003 | -0.196 .. 0.129 | -0.196 .. 0.154 | 0.438 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | -0.015 | -0.139 .. 0.093 | -0.132 .. 0.093 | 0.457 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.088 | -0.100 .. 0.189 | -0.100 .. 0.346 | 0.467 |
| forecast_only_transfer_rank | offline_or_diagnostic_only | -0.090 | -0.261 .. -0.007 | -0.261 .. -0.011 | 0.419 |
| tair_transfer_selection_score | offline_or_diagnostic_only | 0.100 | -0.021 .. 0.293 | -0.082 .. 0.293 | 0.495 |
| rhair_transfer_selection_score | offline_or_diagnostic_only | 0.144 | -0.007 .. 0.279 | -0.007 .. 0.345 | 0.514 |
| co2_transfer_selection_score | offline_or_diagnostic_only | 0.118 | 0.014 .. 0.254 | 0.014 .. 0.254 | 0.495 |
| multiobjective_transfer_selection_score | offline_or_diagnostic_only | 0.150 | -0.036 .. 0.264 | -0.036 .. 0.243 | 0.486 |

### Target: `mpc_rhair_mae`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | offline_or_diagnostic_only | 0.221 | 0.054 .. 0.293 | 0.054 .. 0.293 | 0.514 |
| tair_control_horizon_mae | offline_or_diagnostic_only | 0.187 | 0.013 .. 0.306 | 0.013 .. 0.360 | 0.500 |
| tair_weighted_horizon_mae | offline_or_diagnostic_only | -0.132 | -0.375 .. -0.043 | -0.375 .. 0.088 | 0.352 |
| tair_full_horizon_mae | offline_or_diagnostic_only | -0.147 | -0.393 .. -0.061 | -0.393 .. 0.093 | 0.352 |
| tair_final_step_mae | offline_or_diagnostic_only | -0.100 | -0.336 .. 0.039 | -0.336 .. 0.209 | 0.390 |
| tair_control_horizon_abs_bias | weak_selection | 0.324 | 0.179 .. 0.489 | 0.159 .. 0.500 | 0.533 |
| tair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.222 | 0.055 .. 0.413 | 0.055 .. 0.413 | 0.510 |
| rhair_first_step_mae | weak_selection | 0.282 | 0.129 .. 0.400 | 0.129 .. 0.604 | 0.562 |
| rhair_control_horizon_mae | offline_or_diagnostic_only | 0.226 | 0.061 .. 0.321 | 0.061 .. 0.462 | 0.514 |
| rhair_weighted_horizon_mae | offline_or_diagnostic_only | 0.057 | -0.145 .. 0.173 | -0.145 .. 0.229 | 0.442 |
| rhair_full_horizon_mae | offline_or_diagnostic_only | 0.053 | -0.150 .. 0.168 | -0.150 .. 0.191 | 0.457 |
| rhair_final_step_mae | offline_or_diagnostic_only | 0.081 | -0.116 .. 0.155 | -0.116 .. 0.198 | 0.452 |
| rhair_control_horizon_abs_bias | offline_or_diagnostic_only | 0.059 | -0.054 .. 0.257 | -0.077 .. 0.257 | 0.486 |
| rhair_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.065 | -0.293 .. 0.032 | -0.293 .. 0.042 | 0.410 |
| co2_first_step_mae | weak_selection | 0.331 | 0.227 .. 0.470 | 0.223 .. 0.531 | 0.577 |
| co2_control_horizon_mae | weak_selection | 0.272 | 0.155 .. 0.416 | 0.155 .. 0.569 | 0.548 |
| co2_weighted_horizon_mae | offline_or_diagnostic_only | 0.174 | 0.014 .. 0.300 | 0.014 .. 0.588 | 0.524 |
| co2_full_horizon_mae | offline_or_diagnostic_only | 0.138 | -0.029 .. 0.261 | -0.029 .. 0.610 | 0.514 |
| co2_final_step_mae | offline_or_diagnostic_only | 0.009 | -0.075 .. 0.146 | -0.075 .. 0.341 | 0.486 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | -0.065 | -0.189 .. 0.054 | -0.225 .. 0.054 | 0.438 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.012 | -0.182 .. 0.121 | -0.182 .. 0.341 | 0.438 |
| forecast_only_transfer_rank | offline_or_diagnostic_only | 0.108 | -0.096 .. 0.207 | -0.096 .. 0.234 | 0.486 |
| tair_transfer_selection_score | weak_selection | 0.285 | 0.132 .. 0.418 | 0.099 .. 0.418 | 0.533 |
| rhair_transfer_selection_score | offline_or_diagnostic_only | 0.240 | 0.073 .. 0.331 | 0.077 .. 0.434 | 0.529 |
| co2_transfer_selection_score | offline_or_diagnostic_only | 0.209 | 0.139 .. 0.339 | 0.088 .. 0.456 | 0.543 |
| multiobjective_transfer_selection_score | weak_selection | 0.332 | 0.175 .. 0.454 | 0.175 .. 0.451 | 0.581 |

### Target: `mpc_co2_mae`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | offline_or_diagnostic_only | 0.185 | 0.082 .. 0.336 | 0.002 .. 0.336 | 0.533 |
| tair_control_horizon_mae | offline_or_diagnostic_only | 0.190 | 0.080 .. 0.366 | -0.035 .. 0.366 | 0.519 |
| tair_weighted_horizon_mae | offline_or_diagnostic_only | 0.206 | 0.136 .. 0.411 | -0.016 .. 0.411 | 0.562 |
| tair_full_horizon_mae | offline_or_diagnostic_only | 0.182 | 0.107 .. 0.382 | -0.022 .. 0.401 | 0.543 |
| tair_final_step_mae | offline_or_diagnostic_only | -0.138 | -0.250 .. 0.018 | -0.412 .. 0.060 | 0.419 |
| tair_control_horizon_abs_bias | offline_or_diagnostic_only | 0.035 | -0.068 .. 0.157 | -0.147 .. 0.157 | 0.495 |
| tair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.149 | -0.009 .. 0.264 | -0.162 .. 0.264 | 0.519 |
| rhair_first_step_mae | offline_or_diagnostic_only | -0.162 | -0.346 .. -0.068 | -0.346 .. 0.093 | 0.381 |
| rhair_control_horizon_mae | offline_or_diagnostic_only | -0.118 | -0.289 .. 0.004 | -0.289 .. 0.116 | 0.390 |
| rhair_weighted_horizon_mae | offline_or_diagnostic_only | -0.034 | -0.159 .. 0.155 | -0.113 .. 0.141 | 0.462 |
| rhair_full_horizon_mae | offline_or_diagnostic_only | 0.006 | -0.111 .. 0.204 | -0.055 .. 0.114 | 0.467 |
| rhair_final_step_mae | offline_or_diagnostic_only | -0.122 | -0.248 .. 0.027 | -0.195 .. 0.009 | 0.433 |
| rhair_control_horizon_abs_bias | offline_or_diagnostic_only | 0.150 | 0.014 .. 0.271 | 0.014 .. 0.327 | 0.476 |
| rhair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.000 | -0.100 .. 0.207 | -0.099 .. 0.168 | 0.486 |
| co2_first_step_mae | secondary_selection | 0.366 | 0.268 .. 0.538 | 0.268 .. 0.657 | 0.590 |
| co2_control_horizon_mae | weak_selection | 0.278 | 0.168 .. 0.470 | 0.168 .. 0.657 | 0.562 |
| co2_weighted_horizon_mae | offline_or_diagnostic_only | 0.088 | 0.004 .. 0.239 | -0.077 .. 0.473 | 0.495 |
| co2_full_horizon_mae | offline_or_diagnostic_only | 0.050 | -0.032 .. 0.193 | -0.099 .. 0.478 | 0.476 |
| co2_final_step_mae | offline_or_diagnostic_only | -0.074 | -0.207 .. 0.093 | -0.218 .. 0.181 | 0.419 |
| co2_control_horizon_abs_bias | secondary_selection | 0.491 | 0.404 .. 0.671 | 0.404 .. 0.703 | 0.638 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.147 | 0.043 .. 0.382 | 0.011 .. 0.632 | 0.514 |
| forecast_only_transfer_rank | offline_or_diagnostic_only | 0.156 | 0.036 .. 0.286 | 0.099 .. 0.281 | 0.529 |
| tair_transfer_selection_score | offline_or_diagnostic_only | 0.141 | 0.018 .. 0.304 | -0.059 .. 0.304 | 0.495 |
| rhair_transfer_selection_score | offline_or_diagnostic_only | -0.100 | -0.256 .. 0.032 | -0.256 .. 0.147 | 0.413 |
| co2_transfer_selection_score | weak_selection | 0.312 | 0.211 .. 0.489 | 0.211 .. 0.687 | 0.581 |
| multiobjective_transfer_selection_score | weak_selection | 0.309 | 0.171 .. 0.407 | 0.165 .. 0.495 | 0.562 |

### Target: `mpc_objective`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | offline_or_diagnostic_only | 0.153 | -0.029 .. 0.286 | -0.029 .. 0.286 | 0.486 |
| tair_control_horizon_mae | offline_or_diagnostic_only | 0.150 | -0.032 .. 0.300 | -0.040 .. 0.300 | 0.490 |
| tair_weighted_horizon_mae | offline_or_diagnostic_only | -0.132 | -0.375 .. -0.007 | -0.375 .. 0.000 | 0.362 |
| tair_full_horizon_mae | offline_or_diagnostic_only | -0.135 | -0.379 .. -0.011 | -0.379 .. 0.011 | 0.362 |
| tair_final_step_mae | offline_or_diagnostic_only | -0.097 | -0.332 .. 0.039 | -0.332 .. 0.039 | 0.400 |
| tair_control_horizon_abs_bias | weak_selection | 0.276 | 0.121 .. 0.425 | 0.121 .. 0.421 | 0.562 |
| tair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.153 | -0.029 .. 0.343 | -0.029 .. 0.343 | 0.519 |
| rhair_first_step_mae | objective_secondary_selection | 0.418 | 0.293 .. 0.532 | 0.293 .. 0.742 | 0.610 |
| rhair_control_horizon_mae | weak_selection | 0.259 | 0.100 .. 0.400 | 0.100 .. 0.495 | 0.543 |
| rhair_weighted_horizon_mae | offline_or_diagnostic_only | -0.009 | -0.225 .. 0.125 | -0.225 .. 0.149 | 0.413 |
| rhair_full_horizon_mae | offline_or_diagnostic_only | 0.000 | -0.214 .. 0.136 | -0.214 .. 0.154 | 0.429 |
| rhair_final_step_mae | offline_or_diagnostic_only | 0.006 | -0.207 .. 0.118 | -0.207 .. 0.118 | 0.423 |
| rhair_control_horizon_abs_bias | offline_or_diagnostic_only | 0.100 | 0.007 .. 0.314 | -0.044 .. 0.314 | 0.514 |
| rhair_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.162 | -0.411 .. -0.039 | -0.411 .. -0.039 | 0.362 |
| co2_first_step_mae | weak_selection | 0.309 | 0.200 .. 0.450 | 0.200 .. 0.528 | 0.567 |
| co2_control_horizon_mae | weak_selection | 0.280 | 0.164 .. 0.415 | 0.164 .. 0.545 | 0.558 |
| co2_weighted_horizon_mae | weak_selection | 0.279 | 0.129 .. 0.418 | 0.129 .. 0.643 | 0.552 |
| co2_full_horizon_mae | weak_selection | 0.253 | 0.096 .. 0.393 | 0.096 .. 0.670 | 0.543 |
| co2_final_step_mae | offline_or_diagnostic_only | 0.153 | -0.014 .. 0.275 | -0.014 .. 0.423 | 0.505 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | -0.038 | -0.175 .. 0.075 | -0.175 .. 0.075 | 0.467 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.068 | -0.125 .. 0.189 | -0.125 .. 0.363 | 0.457 |
| forecast_only_transfer_rank | offline_or_diagnostic_only | 0.133 | -0.068 .. 0.221 | -0.068 .. 0.308 | 0.476 |
| tair_transfer_selection_score | offline_or_diagnostic_only | 0.215 | 0.046 .. 0.389 | 0.033 .. 0.389 | 0.524 |
| rhair_transfer_selection_score | weak_selection | 0.299 | 0.148 .. 0.425 | 0.148 .. 0.490 | 0.567 |
| co2_transfer_selection_score | offline_or_diagnostic_only | 0.200 | 0.100 .. 0.318 | 0.100 .. 0.412 | 0.543 |
| multiobjective_transfer_selection_score | weak_selection | 0.338 | 0.182 .. 0.439 | 0.182 .. 0.500 | 0.552 |
