# Forecast-To-Control Transfer Analysis

Model count: `24`.

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
| mpc_rhair_mae | tair_control_horizon_abs_bias | secondary_selection |
| mpc_rhair_mae | tair_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_first_step_mae | secondary_selection |
| mpc_rhair_mae | rhair_control_horizon_mae | secondary_selection |
| mpc_rhair_mae | rhair_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_full_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_final_step_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_rhair_mae | rhair_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_first_step_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_control_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_weighted_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_full_horizon_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_final_step_mae | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_rhair_mae | co2_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_rhair_mae | forecast_only_transfer_rank | offline_or_diagnostic_only |
| mpc_rhair_mae | tair_transfer_selection_score | weak_selection |
| mpc_rhair_mae | rhair_transfer_selection_score | secondary_selection |
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
| mpc_co2_mae | tair_control_horizon_abs_bias | weak_selection |
| mpc_co2_mae | tair_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_co2_mae | rhair_first_step_mae | weak_selection |
| mpc_co2_mae | rhair_control_horizon_mae | weak_selection |
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
| mpc_co2_mae | co2_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_co2_mae | co2_constraint_near_mae_proxy | offline_or_diagnostic_only |
| mpc_co2_mae | forecast_only_transfer_rank | offline_or_diagnostic_only |
| mpc_co2_mae | tair_transfer_selection_score | offline_or_diagnostic_only |
| mpc_co2_mae | rhair_transfer_selection_score | weak_selection |
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
| mpc_objective | tair_final_step_mae | offline_or_diagnostic_only |
| mpc_objective | tair_control_horizon_abs_bias | offline_or_diagnostic_only |
| mpc_objective | tair_constraint_near_mae_proxy | offline_or_diagnostic_only |
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
| 1 | current_hybrid_transformer | 7.292 | 8.875 | 5.062 | 7.938 | 6.722 | 0.286 | 0.921 | 17.831 | 0.0461 |
| 2 | itransformer_residual | 7.375 | 6.188 | 6.688 | 9.250 | 9.167 | 2.216 | 5.675 | 11.532 | 0.1924 |
| 3 | itransformer_co2_control_aware_fusion | 7.552 | 11.219 | 9.750 | 1.688 | 8.556 | 2.202 | 4.267 | 6.415 | 0.1491 |
| 4 | itransformer_co2_late_frozen_expert | 7.885 | 10.656 | 10.750 | 2.250 | 9.944 | 2.192 | 4.316 | 6.298 | 0.1533 |
| 5 | itransformer_co2_protected_expert | 8.104 | 5.688 | 11.625 | 7.000 | 9.278 | 0.880 | 1.441 | 14.206 | 0.0606 |
| 6 | transformer_hybrid_residual | 8.979 | 7.312 | 3.438 | 16.188 | 9.167 | 0.472 | 1.628 | 16.128 | 0.0436 |
| 7 | itransformer_co2_late_residual | 9.406 | 12.688 | 7.531 | 8.000 | 9.000 | 1.153 | 1.618 | 10.125 | 0.0705 |
| 8 | itransformer_co2_frozen_backbone_horizon_mixture | 9.531 | 12.812 | 7.344 | 8.438 | 9.556 | 1.158 | 1.615 | 10.000 | 0.0718 |
| 9 | itransformer_co2_protected_terminal | 11.208 | 6.500 | 18.188 | 8.938 | 13.778 | 3.380 | 6.179 | 27.089 | 0.3837 |
| 10 | segrnn_forecaster | 11.542 | 22.000 | 7.375 | 5.250 | 9.389 | 0.391 | 2.195 | 14.425 | 0.0486 |
| 11 | dlinear_forecaster | 12.000 | 15.000 | 7.062 | 13.938 | 15.056 | 3.436 | 6.459 | 37.824 | 0.3962 |
| 12 | itransformer_co2_teacher_distill | 12.521 | 14.125 | 16.312 | 7.125 | 12.500 | 2.789 | 6.877 | 27.338 | 0.3502 |
| 13 | itransformer_co2_horizon_mixture | 12.521 | 4.188 | 18.438 | 14.938 | 13.722 | 3.313 | 5.696 | 28.696 | 0.3713 |
| 14 | itransformer_co2_residual | 12.750 | 9.500 | 12.625 | 16.125 | 10.778 | 0.936 | 1.503 | 6.421 | 0.0557 |
| 15 | transformer_forecaster | 12.875 | 9.875 | 14.688 | 14.062 | 13.056 | 1.039 | 4.072 | 16.455 | 0.0861 |
| 16 | itransformer_co2_wavelet_residual | 14.625 | 6.125 | 13.938 | 23.812 | 13.611 | 1.075 | 2.142 | 7.776 | 0.0639 |
| 17 | nlinear_forecaster | 15.104 | 20.188 | 6.500 | 18.625 | 15.500 | 1.867 | 4.182 | 25.236 | 0.1526 |
| 18 | gru_forecaster | 15.271 | 16.125 | 21.500 | 8.188 | 14.278 | 0.409 | 4.957 | 49.973 | 0.1108 |
| 19 | itransformer_co2_frozen_expert | 15.562 | 10.938 | 17.812 | 17.938 | 15.444 | 0.917 | 2.263 | 20.140 | 0.0649 |
| 20 | patchtst_residual | 15.646 | 14.250 | 16.438 | 16.250 | 14.833 | 1.047 | 4.412 | 17.127 | 0.1384 |
| 21 | itransformer_co2_wavelet_blend | 15.854 | 10.750 | 13.625 | 23.188 | 15.000 | 1.023 | 1.928 | 8.020 | 0.0771 |
| 22 | lstm_forecaster | 16.479 | 20.250 | 15.688 | 13.500 | 15.111 | 1.491 | 4.497 | 23.014 | 0.1780 |
| 23 | frequency_forecaster | 19.896 | 24.000 | 19.688 | 16.000 | 18.722 | 1.725 | 8.759 | 15.530 | 0.4338 |
| 24 | itransformer_co2_recoupled_expert | 20.021 | 20.750 | 17.938 | 21.375 | 17.833 | 0.826 | 2.692 | 16.749 | 0.0651 |

## Metric Transfer Quality

### Target: `mpc_tair_mae`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rhair_first_step_mae | selection | 0.353 | 0.325 | 0.591 | no | yes | 0.667 |
| rhair_control_horizon_mae | selection | 0.199 | 0.094 | 0.531 | no | no | 0.000 |
| rhair_control_horizon_abs_bias | selection | 0.230 | 0.098 | 0.529 | no | no | 0.000 |
| rhair_transfer_selection_score | selection | 0.132 | 0.080 | 0.522 | no | yes | 0.333 |
| tair_control_horizon_mae | selection | -0.033 | -0.058 | 0.502 | no | no | 0.000 |
| tair_control_horizon_abs_bias | selection | -0.038 | -0.010 | 0.489 | no | yes | 0.333 |
| co2_first_step_mae | selection | -0.105 | -0.060 | 0.487 | no | no | 0.000 |
| co2_control_horizon_mae | selection | -0.128 | -0.051 | 0.487 | no | no | 0.333 |
| tair_transfer_selection_score | selection | -0.163 | -0.079 | 0.486 | no | no | 0.000 |
| co2_final_step_mae | selection | -0.156 | -0.118 | 0.467 | no | no | 0.333 |
| co2_transfer_selection_score | selection | -0.168 | -0.110 | 0.467 | no | no | 0.333 |
| co2_weighted_horizon_mae | selection | -0.163 | -0.070 | 0.467 | no | yes | 0.667 |
| tair_first_step_mae | selection | -0.030 | -0.123 | 0.464 | no | no | 0.000 |
| co2_full_horizon_mae | selection | -0.161 | -0.091 | 0.464 | no | yes | 0.667 |
| multiobjective_transfer_selection_score | selection | -0.112 | -0.120 | 0.458 | yes | yes | 0.333 |
| tair_constraint_near_mae_proxy | selection | -0.117 | -0.134 | 0.455 | no | no | 0.000 |
| co2_constraint_near_mae_proxy | selection | -0.146 | -0.168 | 0.431 | no | no | 0.333 |
| forecast_only_transfer_rank | selection | -0.302 | -0.277 | 0.413 | no | yes | 0.333 |
| tair_full_horizon_mae | selection | -0.154 | -0.262 | 0.413 | no | no | 0.000 |
| tair_weighted_horizon_mae | selection | -0.143 | -0.282 | 0.409 | no | no | 0.000 |
| co2_control_horizon_abs_bias | selection | -0.301 | -0.348 | 0.399 | no | no | 0.000 |
| tair_final_step_mae | selection | -0.187 | -0.283 | 0.395 | no | no | 0.000 |
| rhair_weighted_horizon_mae | selection | -0.279 | -0.332 | 0.364 | no | yes | 0.333 |
| rhair_full_horizon_mae | selection | -0.320 | -0.364 | 0.355 | no | yes | 0.333 |
| rhair_final_step_mae | selection | -0.422 | -0.397 | 0.345 | yes | yes | 0.333 |
| rhair_constraint_near_mae_proxy | selection | -0.639 | -0.593 | 0.276 | no | no | 0.000 |
| assim_sp_first_grad | diagnostic | 0.558 | 0.658 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | 0.444 | 0.605 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | 0.456 | 0.577 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | 0.342 | 0.516 |  |  |  |  |
| co2_sp_first_grad | diagnostic | 0.345 | 0.449 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.378 | -0.399 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.378 | 0.388 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | 0.388 | 0.379 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.318 | -0.371 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | 0.308 | 0.366 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | 0.186 | 0.291 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | 0.271 | 0.254 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | 0.187 | 0.190 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | -0.088 | -0.184 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | -0.271 | -0.168 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | 0.233 | 0.092 |  |  |  |  |

### Target: `mpc_rhair_mae`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rhair_first_step_mae | selection | 0.702 | 0.592 | 0.732 | no | yes | 0.333 |
| rhair_transfer_selection_score | selection | 0.442 | 0.481 | 0.685 | no | yes | 0.333 |
| rhair_control_horizon_mae | selection | 0.630 | 0.485 | 0.669 | no | no | 0.000 |
| tair_control_horizon_abs_bias | selection | 0.345 | 0.359 | 0.623 | no | yes | 0.333 |
| tair_transfer_selection_score | selection | 0.268 | 0.290 | 0.598 | no | no | 0.333 |
| rhair_control_horizon_abs_bias | selection | 0.160 | 0.231 | 0.598 | no | no | 0.333 |
| multiobjective_transfer_selection_score | selection | 0.280 | 0.265 | 0.596 | yes | yes | 0.333 |
| tair_constraint_near_mae_proxy | selection | 0.273 | 0.187 | 0.571 | no | no | 0.000 |
| tair_first_step_mae | selection | 0.482 | 0.190 | 0.569 | no | no | 0.333 |
| tair_control_horizon_mae | selection | 0.445 | 0.162 | 0.560 | no | no | 0.000 |
| co2_first_step_mae | selection | 0.103 | 0.144 | 0.553 | no | no | 0.333 |
| forecast_only_transfer_rank | selection | 0.117 | 0.082 | 0.533 | no | yes | 0.333 |
| rhair_weighted_horizon_mae | selection | 0.326 | 0.105 | 0.531 | no | yes | 0.333 |
| rhair_full_horizon_mae | selection | 0.287 | 0.077 | 0.525 | no | yes | 0.333 |
| tair_final_step_mae | selection | 0.278 | 0.059 | 0.522 | no | no | 0.000 |
| tair_full_horizon_mae | selection | 0.343 | 0.059 | 0.511 | no | no | 0.000 |
| tair_weighted_horizon_mae | selection | 0.356 | 0.048 | 0.500 | no | no | 0.000 |
| co2_control_horizon_mae | selection | -0.036 | 0.006 | 0.495 | no | no | 0.000 |
| co2_transfer_selection_score | selection | -0.105 | -0.037 | 0.486 | no | no | 0.000 |
| rhair_final_step_mae | selection | 0.162 | -0.032 | 0.484 | yes | yes | 0.333 |
| co2_weighted_horizon_mae | selection | -0.195 | -0.112 | 0.464 | no | yes | 0.333 |
| co2_full_horizon_mae | selection | -0.215 | -0.129 | 0.453 | no | yes | 0.333 |
| rhair_constraint_near_mae_proxy | selection | -0.107 | -0.167 | 0.444 | no | no | 0.000 |
| co2_final_step_mae | selection | -0.247 | -0.186 | 0.435 | no | no | 0.000 |
| co2_constraint_near_mae_proxy | selection | -0.318 | -0.243 | 0.420 | no | no | 0.000 |
| co2_control_horizon_abs_bias | selection | -0.380 | -0.388 | 0.388 | no | no | 0.000 |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.273 | -0.360 |  |  |  |  |
| assim_sp_first_grad | diagnostic | 0.289 | 0.294 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | -0.245 | -0.287 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | -0.245 | -0.275 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.218 | -0.261 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.217 | 0.242 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | -0.221 | -0.204 |  |  |  |  |
| co2_sp_first_grad | diagnostic | 0.251 | 0.197 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | 0.203 | 0.173 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | -0.139 | -0.142 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | 0.041 | 0.084 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | -0.001 | 0.069 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | -0.062 | -0.047 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | -0.047 | -0.031 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | -0.007 | -0.011 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | 0.270 | 0.010 |  |  |  |  |

### Target: `mpc_co2_mae`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rhair_control_horizon_abs_bias | selection | 0.399 | 0.348 | 0.620 | no | no | 0.333 |
| rhair_transfer_selection_score | selection | 0.364 | 0.339 | 0.620 | no | no | 0.000 |
| rhair_first_step_mae | selection | 0.234 | 0.344 | 0.609 | no | no | 0.000 |
| rhair_control_horizon_mae | selection | 0.264 | 0.276 | 0.604 | no | no | 0.000 |
| multiobjective_transfer_selection_score | selection | 0.264 | 0.301 | 0.593 | no | no | 0.333 |
| tair_control_horizon_abs_bias | selection | 0.138 | 0.257 | 0.580 | no | no | 0.000 |
| tair_weighted_horizon_mae | selection | 0.036 | 0.217 | 0.572 | no | no | 0.000 |
| tair_transfer_selection_score | selection | 0.206 | 0.229 | 0.569 | no | no | 0.000 |
| forecast_only_transfer_rank | selection | 0.167 | 0.190 | 0.569 | no | no | 0.333 |
| tair_full_horizon_mae | selection | 0.038 | 0.206 | 0.562 | no | no | 0.000 |
| tair_control_horizon_mae | selection | 0.007 | 0.149 | 0.556 | no | no | 0.000 |
| tair_first_step_mae | selection | 0.090 | 0.137 | 0.554 | no | no | 0.000 |
| tair_constraint_near_mae_proxy | selection | -0.023 | 0.138 | 0.553 | no | no | 0.000 |
| co2_first_step_mae | selection | -0.168 | 0.168 | 0.549 | no | yes | 0.667 |
| co2_transfer_selection_score | selection | -0.013 | 0.102 | 0.529 | no | yes | 0.667 |
| tair_final_step_mae | selection | 0.030 | 0.051 | 0.522 | no | no | 0.000 |
| rhair_weighted_horizon_mae | selection | 0.308 | 0.065 | 0.520 | no | no | 0.000 |
| co2_control_horizon_abs_bias | selection | -0.164 | 0.078 | 0.518 | no | yes | 0.667 |
| rhair_full_horizon_mae | selection | 0.303 | 0.077 | 0.518 | no | no | 0.000 |
| rhair_constraint_near_mae_proxy | selection | 0.160 | -0.027 | 0.509 | no | no | 0.000 |
| co2_constraint_near_mae_proxy | selection | -0.147 | 0.015 | 0.507 | no | no | 0.333 |
| co2_control_horizon_mae | selection | -0.204 | 0.061 | 0.498 | no | yes | 0.667 |
| rhair_final_step_mae | selection | 0.211 | -0.082 | 0.487 | no | no | 0.000 |
| co2_weighted_horizon_mae | selection | -0.245 | -0.086 | 0.464 | no | no | 0.000 |
| co2_full_horizon_mae | selection | -0.255 | -0.110 | 0.453 | no | no | 0.000 |
| co2_final_step_mae | selection | -0.295 | -0.243 | 0.406 | no | no | 0.000 |
| rhair_t_vent_sp_first_grad | diagnostic | -0.472 | -0.421 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | -0.445 | -0.411 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | -0.384 | -0.405 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | -0.412 | -0.349 |  |  |  |  |
| assim_sp_first_grad | diagnostic | -0.319 | -0.342 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | -0.295 | -0.302 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | -0.343 | -0.300 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | -0.329 | -0.277 |  |  |  |  |
| co2_sp_first_grad | diagnostic | -0.219 | -0.244 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.273 | -0.184 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | -0.344 | -0.123 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.052 | -0.097 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.052 | 0.094 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | 0.099 | -0.082 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | -0.328 | -0.045 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | -0.417 | -0.022 |  |  |  |  |

### Target: `mpc_objective`

| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rhair_first_step_mae | selection | 0.683 | 0.507 | 0.703 | yes | yes | 1.000 |
| rhair_transfer_selection_score | selection | 0.348 | 0.368 | 0.649 | yes | yes | 0.667 |
| rhair_control_horizon_mae | selection | 0.557 | 0.349 | 0.629 | yes | yes | 0.333 |
| rhair_control_horizon_abs_bias | selection | 0.183 | 0.234 | 0.591 | no | no | 0.000 |
| tair_transfer_selection_score | selection | 0.111 | 0.199 | 0.583 | no | no | 0.000 |
| tair_control_horizon_abs_bias | selection | 0.228 | 0.205 | 0.572 | no | no | 0.333 |
| multiobjective_transfer_selection_score | selection | 0.173 | 0.167 | 0.564 | no | no | 0.333 |
| tair_constraint_near_mae_proxy | selection | 0.225 | 0.090 | 0.564 | no | no | 0.000 |
| tair_first_step_mae | selection | 0.373 | 0.140 | 0.547 | no | yes | 0.333 |
| tair_control_horizon_mae | selection | 0.362 | 0.139 | 0.545 | no | no | 0.000 |
| tair_full_horizon_mae | selection | 0.243 | 0.023 | 0.525 | no | no | 0.000 |
| forecast_only_transfer_rank | selection | -0.026 | 0.026 | 0.518 | no | no | 0.333 |
| co2_first_step_mae | selection | 0.093 | 0.053 | 0.516 | no | no | 0.000 |
| tair_weighted_horizon_mae | selection | 0.257 | 0.013 | 0.514 | no | no | 0.000 |
| tair_final_step_mae | selection | 0.183 | 0.008 | 0.507 | no | yes | 0.333 |
| co2_control_horizon_mae | selection | -0.009 | -0.009 | 0.502 | no | no | 0.333 |
| rhair_weighted_horizon_mae | selection | 0.108 | 0.006 | 0.495 | yes | yes | 0.667 |
| rhair_full_horizon_mae | selection | 0.058 | -0.017 | 0.489 | yes | yes | 0.667 |
| co2_transfer_selection_score | selection | -0.078 | -0.079 | 0.486 | no | no | 0.333 |
| co2_weighted_horizon_mae | selection | -0.157 | -0.065 | 0.486 | no | no | 0.667 |
| co2_full_horizon_mae | selection | -0.176 | -0.080 | 0.475 | no | no | 0.667 |
| co2_final_step_mae | selection | -0.234 | -0.117 | 0.464 | no | no | 0.333 |
| rhair_final_step_mae | selection | -0.099 | -0.103 | 0.462 | no | no | 0.333 |
| co2_constraint_near_mae_proxy | selection | -0.243 | -0.230 | 0.420 | no | no | 0.333 |
| rhair_constraint_near_mae_proxy | selection | -0.351 | -0.257 | 0.411 | no | no | 0.000 |
| co2_control_horizon_abs_bias | selection | -0.341 | -0.382 | 0.388 | no | no | 0.000 |
| assim_sp_first_grad | diagnostic | 0.399 | 0.437 |  |  |  |  |
| tair_window_pos_lee_sp_first_grad | diagnostic | -0.338 | -0.397 |  |  |  |  |
| co2_first_grad_mean_abs | diagnostic | 0.294 | 0.375 |  |  |  |  |
| co2_sp_first_grad_positive_fraction | diagnostic | -0.279 | -0.356 |  |  |  |  |
| co2_sp_first_grad_flat_fraction | diagnostic | 0.278 | 0.339 |  |  |  |  |
| rhair_window_pos_lee_sp_first_grad | diagnostic | -0.191 | -0.321 |  |  |  |  |
| co2_sp_first_grad | diagnostic | 0.283 | 0.321 |  |  |  |  |
| t_vent_sp_first_grad | diagnostic | 0.116 | 0.288 |  |  |  |  |
| tair_t_vent_sp_first_grad | diagnostic | 0.151 | 0.232 |  |  |  |  |
| tair_first_grad_mean_abs | diagnostic | 0.126 | 0.228 |  |  |  |  |
| rhair_water_sup_intervals_sp_min_first_grad | diagnostic | 0.075 | 0.200 |  |  |  |  |
| tair_t_heat_sp_first_grad | diagnostic | 0.274 | 0.172 |  |  |  |  |
| rhair_dx_sp_first_grad | diagnostic | -0.097 | -0.159 |  |  |  |  |
| rhair_first_grad_mean_abs | diagnostic | 0.046 | 0.052 |  |  |  |  |
| cost_grad_mean_abs | diagnostic | -0.160 | -0.034 |  |  |  |  |
| rhair_t_vent_sp_first_grad | diagnostic | -0.026 | 0.008 |  |  |  |  |

## Robustness Summary

### Target: `mpc_tair_mae`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | offline_or_diagnostic_only | -0.123 | -0.209 .. -0.035 | -0.270 .. 0.036 | 0.435 |
| tair_control_horizon_mae | offline_or_diagnostic_only | -0.058 | -0.135 .. 0.053 | -0.186 .. 0.038 | 0.480 |
| tair_weighted_horizon_mae | offline_or_diagnostic_only | -0.282 | -0.343 .. -0.199 | -0.391 .. -0.183 | 0.383 |
| tair_full_horizon_mae | offline_or_diagnostic_only | -0.262 | -0.325 .. -0.176 | -0.375 .. -0.156 | 0.383 |
| tair_final_step_mae | offline_or_diagnostic_only | -0.283 | -0.393 .. -0.215 | -0.426 .. -0.187 | 0.356 |
| tair_control_horizon_abs_bias | offline_or_diagnostic_only | -0.010 | -0.129 .. 0.082 | -0.129 .. 0.112 | 0.447 |
| tair_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.134 | -0.198 .. -0.034 | -0.198 .. -0.050 | 0.437 |
| rhair_first_step_mae | weak_selection | 0.325 | 0.237 .. 0.395 | 0.237 .. 0.433 | 0.561 |
| rhair_control_horizon_mae | offline_or_diagnostic_only | 0.094 | 0.003 .. 0.196 | 0.033 .. 0.267 | 0.500 |
| rhair_weighted_horizon_mae | offline_or_diagnostic_only | -0.332 | -0.488 .. -0.260 | -0.488 .. -0.228 | 0.313 |
| rhair_full_horizon_mae | offline_or_diagnostic_only | -0.364 | -0.525 .. -0.297 | -0.525 .. -0.253 | 0.304 |
| rhair_final_step_mae | offline_or_diagnostic_only | -0.397 | -0.588 .. -0.340 | -0.588 .. -0.276 | 0.286 |
| rhair_control_horizon_abs_bias | offline_or_diagnostic_only | 0.098 | -0.010 .. 0.163 | -0.100 .. 0.211 | 0.490 |
| rhair_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.593 | -0.690 .. -0.540 | -0.690 .. -0.538 | 0.234 |
| co2_first_step_mae | offline_or_diagnostic_only | -0.060 | -0.126 .. 0.015 | -0.197 .. 0.071 | 0.468 |
| co2_control_horizon_mae | offline_or_diagnostic_only | -0.051 | -0.156 .. 0.030 | -0.258 .. 0.085 | 0.452 |
| co2_weighted_horizon_mae | offline_or_diagnostic_only | -0.070 | -0.214 .. 0.002 | -0.334 .. 0.245 | 0.423 |
| co2_full_horizon_mae | offline_or_diagnostic_only | -0.091 | -0.238 .. -0.021 | -0.347 .. 0.243 | 0.419 |
| co2_final_step_mae | offline_or_diagnostic_only | -0.118 | -0.233 .. -0.031 | -0.282 .. 0.061 | 0.431 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | -0.348 | -0.468 .. -0.294 | -0.500 .. -0.250 | 0.360 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.168 | -0.305 .. -0.098 | -0.345 .. 0.083 | 0.387 |
| forecast_only_transfer_rank | offline_or_diagnostic_only | -0.277 | -0.407 .. -0.200 | -0.407 .. -0.159 | 0.369 |
| tair_transfer_selection_score | offline_or_diagnostic_only | -0.079 | -0.139 .. 0.029 | -0.185 .. 0.091 | 0.466 |
| rhair_transfer_selection_score | offline_or_diagnostic_only | 0.080 | -0.036 .. 0.193 | -0.036 .. 0.248 | 0.482 |
| co2_transfer_selection_score | offline_or_diagnostic_only | -0.110 | -0.220 .. -0.037 | -0.303 .. 0.007 | 0.435 |
| multiobjective_transfer_selection_score | offline_or_diagnostic_only | -0.120 | -0.271 .. -0.044 | -0.271 .. 0.024 | 0.406 |

### Target: `mpc_rhair_mae`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | offline_or_diagnostic_only | 0.190 | 0.079 .. 0.283 | 0.079 .. 0.320 | 0.530 |
| tair_control_horizon_mae | offline_or_diagnostic_only | 0.162 | 0.048 .. 0.263 | 0.038 .. 0.286 | 0.520 |
| tair_weighted_horizon_mae | offline_or_diagnostic_only | 0.048 | -0.082 .. 0.139 | -0.082 .. 0.148 | 0.455 |
| tair_full_horizon_mae | offline_or_diagnostic_only | 0.059 | -0.069 .. 0.152 | -0.069 .. 0.164 | 0.466 |
| tair_final_step_mae | offline_or_diagnostic_only | 0.059 | -0.069 .. 0.157 | -0.069 .. 0.157 | 0.478 |
| tair_control_horizon_abs_bias | secondary_selection | 0.359 | 0.272 .. 0.479 | 0.194 .. 0.482 | 0.589 |
| tair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.187 | 0.077 .. 0.327 | 0.077 .. 0.293 | 0.532 |
| rhair_first_step_mae | secondary_selection | 0.592 | 0.537 .. 0.666 | 0.526 .. 0.711 | 0.708 |
| rhair_control_horizon_mae | secondary_selection | 0.485 | 0.415 .. 0.589 | 0.409 .. 0.639 | 0.639 |
| rhair_weighted_horizon_mae | offline_or_diagnostic_only | 0.105 | -0.017 .. 0.214 | -0.017 .. 0.283 | 0.488 |
| rhair_full_horizon_mae | offline_or_diagnostic_only | 0.077 | -0.043 .. 0.183 | -0.043 .. 0.262 | 0.486 |
| rhair_final_step_mae | offline_or_diagnostic_only | -0.032 | -0.172 .. 0.055 | -0.172 .. 0.156 | 0.437 |
| rhair_control_horizon_abs_bias | offline_or_diagnostic_only | 0.231 | 0.169 .. 0.388 | 0.059 .. 0.388 | 0.573 |
| rhair_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.167 | -0.321 .. -0.081 | -0.321 .. -0.009 | 0.397 |
| co2_first_step_mae | offline_or_diagnostic_only | 0.144 | 0.059 .. 0.207 | 0.034 .. 0.367 | 0.520 |
| co2_control_horizon_mae | offline_or_diagnostic_only | 0.006 | -0.097 .. 0.066 | -0.097 .. 0.271 | 0.456 |
| co2_weighted_horizon_mae | offline_or_diagnostic_only | -0.112 | -0.234 .. -0.024 | -0.234 .. 0.319 | 0.423 |
| co2_full_horizon_mae | offline_or_diagnostic_only | -0.129 | -0.253 .. -0.042 | -0.253 .. 0.331 | 0.411 |
| co2_final_step_mae | offline_or_diagnostic_only | -0.186 | -0.253 .. -0.091 | -0.253 .. 0.105 | 0.407 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | -0.388 | -0.447 .. -0.326 | -0.456 .. -0.275 | 0.372 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.243 | -0.371 .. -0.145 | -0.371 .. 0.130 | 0.379 |
| forecast_only_transfer_rank | offline_or_diagnostic_only | 0.082 | -0.035 .. 0.197 | -0.035 .. 0.272 | 0.490 |
| tair_transfer_selection_score | weak_selection | 0.290 | 0.193 .. 0.402 | 0.128 .. 0.408 | 0.561 |
| rhair_transfer_selection_score | secondary_selection | 0.481 | 0.413 .. 0.592 | 0.413 .. 0.640 | 0.659 |
| co2_transfer_selection_score | offline_or_diagnostic_only | -0.037 | -0.133 .. 0.035 | -0.133 .. 0.194 | 0.456 |
| multiobjective_transfer_selection_score | weak_selection | 0.265 | 0.159 .. 0.371 | 0.159 .. 0.390 | 0.554 |

### Target: `mpc_co2_mae`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | offline_or_diagnostic_only | 0.137 | 0.051 .. 0.248 | 0.005 .. 0.269 | 0.526 |
| tair_control_horizon_mae | offline_or_diagnostic_only | 0.149 | 0.089 .. 0.278 | 0.035 .. 0.294 | 0.540 |
| tair_weighted_horizon_mae | offline_or_diagnostic_only | 0.217 | 0.157 .. 0.351 | 0.139 .. 0.334 | 0.553 |
| tair_full_horizon_mae | offline_or_diagnostic_only | 0.206 | 0.145 .. 0.339 | 0.125 .. 0.320 | 0.542 |
| tair_final_step_mae | offline_or_diagnostic_only | 0.051 | -0.028 .. 0.148 | -0.082 .. 0.125 | 0.498 |
| tair_control_horizon_abs_bias | weak_selection | 0.257 | 0.213 .. 0.380 | 0.086 .. 0.356 | 0.565 |
| tair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.138 | 0.076 .. 0.255 | 0.055 .. 0.238 | 0.528 |
| rhair_first_step_mae | weak_selection | 0.344 | 0.277 .. 0.409 | 0.247 .. 0.444 | 0.585 |
| rhair_control_horizon_mae | weak_selection | 0.276 | 0.190 .. 0.381 | 0.181 .. 0.485 | 0.571 |
| rhair_weighted_horizon_mae | offline_or_diagnostic_only | 0.065 | -0.050 .. 0.177 | -0.081 .. 0.224 | 0.480 |
| rhair_full_horizon_mae | offline_or_diagnostic_only | 0.077 | -0.048 .. 0.191 | -0.073 .. 0.238 | 0.474 |
| rhair_final_step_mae | offline_or_diagnostic_only | -0.082 | -0.230 .. 0.006 | -0.263 .. 0.097 | 0.440 |
| rhair_control_horizon_abs_bias | weak_selection | 0.348 | 0.287 .. 0.453 | 0.228 .. 0.530 | 0.601 |
| rhair_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.027 | -0.167 .. 0.079 | -0.205 .. 0.129 | 0.464 |
| co2_first_step_mae | offline_or_diagnostic_only | 0.168 | 0.055 .. 0.275 | 0.055 .. 0.516 | 0.510 |
| co2_control_horizon_mae | offline_or_diagnostic_only | 0.061 | -0.066 .. 0.156 | -0.066 .. 0.411 | 0.455 |
| co2_weighted_horizon_mae | offline_or_diagnostic_only | -0.086 | -0.187 .. -0.003 | -0.187 .. 0.314 | 0.431 |
| co2_full_horizon_mae | offline_or_diagnostic_only | -0.110 | -0.202 .. -0.027 | -0.202 .. 0.309 | 0.427 |
| co2_final_step_mae | offline_or_diagnostic_only | -0.243 | -0.348 .. -0.142 | -0.348 .. 0.049 | 0.368 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | 0.078 | -0.046 .. 0.166 | -0.046 .. 0.402 | 0.478 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.015 | -0.089 .. 0.111 | -0.089 .. 0.522 | 0.474 |
| forecast_only_transfer_rank | offline_or_diagnostic_only | 0.190 | 0.124 .. 0.281 | 0.030 .. 0.441 | 0.545 |
| tair_transfer_selection_score | offline_or_diagnostic_only | 0.229 | 0.171 .. 0.357 | 0.126 .. 0.333 | 0.549 |
| rhair_transfer_selection_score | weak_selection | 0.339 | 0.232 .. 0.475 | 0.162 .. 0.577 | 0.573 |
| co2_transfer_selection_score | offline_or_diagnostic_only | 0.102 | -0.020 .. 0.201 | -0.020 .. 0.461 | 0.490 |
| multiobjective_transfer_selection_score | weak_selection | 0.301 | 0.223 .. 0.399 | 0.205 .. 0.502 | 0.565 |

### Target: `mpc_objective`

| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |
| --- | --- | --- | --- | --- | --- |
| tair_first_step_mae | offline_or_diagnostic_only | 0.140 | 0.023 .. 0.242 | 0.023 .. 0.256 | 0.506 |
| tair_control_horizon_mae | offline_or_diagnostic_only | 0.139 | 0.022 .. 0.254 | 0.022 .. 0.257 | 0.504 |
| tair_weighted_horizon_mae | offline_or_diagnostic_only | 0.013 | -0.122 .. 0.114 | -0.122 .. 0.106 | 0.470 |
| tair_full_horizon_mae | offline_or_diagnostic_only | 0.023 | -0.111 .. 0.125 | -0.111 .. 0.119 | 0.482 |
| tair_final_step_mae | offline_or_diagnostic_only | 0.008 | -0.127 .. 0.085 | -0.127 .. 0.085 | 0.462 |
| tair_control_horizon_abs_bias | offline_or_diagnostic_only | 0.205 | 0.097 .. 0.304 | 0.097 .. 0.310 | 0.534 |
| tair_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.090 | -0.035 .. 0.202 | -0.035 .. 0.182 | 0.524 |
| rhair_first_step_mae | objective_secondary_selection | 0.507 | 0.440 .. 0.578 | 0.440 .. 0.615 | 0.676 |
| rhair_control_horizon_mae | weak_selection | 0.349 | 0.260 .. 0.455 | 0.260 .. 0.530 | 0.595 |
| rhair_weighted_horizon_mae | offline_or_diagnostic_only | 0.006 | -0.129 .. 0.099 | -0.129 .. 0.142 | 0.448 |
| rhair_full_horizon_mae | offline_or_diagnostic_only | -0.017 | -0.156 .. 0.073 | -0.145 .. 0.108 | 0.443 |
| rhair_final_step_mae | offline_or_diagnostic_only | -0.103 | -0.249 .. -0.027 | -0.249 .. 0.037 | 0.417 |
| rhair_control_horizon_abs_bias | offline_or_diagnostic_only | 0.234 | 0.166 .. 0.394 | -0.042 .. 0.394 | 0.561 |
| rhair_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.257 | -0.418 .. -0.180 | -0.418 .. -0.192 | 0.361 |
| co2_first_step_mae | offline_or_diagnostic_only | 0.053 | -0.045 .. 0.128 | -0.045 .. 0.267 | 0.480 |
| co2_control_horizon_mae | offline_or_diagnostic_only | -0.009 | -0.116 .. 0.055 | -0.116 .. 0.226 | 0.464 |
| co2_weighted_horizon_mae | offline_or_diagnostic_only | -0.065 | -0.196 .. -0.006 | -0.179 .. 0.311 | 0.447 |
| co2_full_horizon_mae | offline_or_diagnostic_only | -0.080 | -0.213 .. -0.022 | -0.197 .. 0.316 | 0.435 |
| co2_final_step_mae | offline_or_diagnostic_only | -0.117 | -0.222 .. -0.045 | -0.185 .. 0.105 | 0.431 |
| co2_control_horizon_abs_bias | offline_or_diagnostic_only | -0.382 | -0.499 .. -0.341 | -0.479 .. -0.322 | 0.352 |
| co2_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.230 | -0.362 .. -0.173 | -0.350 .. 0.020 | 0.379 |
| forecast_only_transfer_rank | offline_or_diagnostic_only | 0.026 | -0.099 .. 0.125 | -0.099 .. 0.191 | 0.478 |
| tair_transfer_selection_score | offline_or_diagnostic_only | 0.199 | 0.090 .. 0.319 | 0.082 .. 0.316 | 0.545 |
| rhair_transfer_selection_score | objective_secondary_selection | 0.368 | 0.285 .. 0.489 | 0.285 .. 0.564 | 0.617 |
| co2_transfer_selection_score | offline_or_diagnostic_only | -0.079 | -0.176 .. 0.011 | -0.175 .. 0.100 | 0.458 |
| multiobjective_transfer_selection_score | offline_or_diagnostic_only | 0.167 | 0.056 .. 0.262 | 0.056 .. 0.298 | 0.526 |
