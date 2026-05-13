# FCTV Multi-Start Transfer Robustness

This report reuses the same forecast-side FCTV metrics and replaces closed-loop outcomes with repeated `GradientMPC` 96-step rollouts from multiple test-set start indices.

| start_idx | model_count | control_target | metric | role | spearman | pairwise | leave-model spearman min |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 16 | mpc_tair_mae | tair_first_step_mae | offline_or_diagnostic_only | -0.274 | 0.408 | -0.393 |
| 0 | 16 | mpc_rhair_mae | rhair_first_step_mae | secondary_selection | 0.435 | 0.683 | 0.314 |
| 0 | 16 | mpc_co2_mae | co2_first_step_mae | secondary_selection | 0.364 | 0.613 | 0.263 |
| 0 | 16 | mpc_co2_mae | co2_constraint_near_mae_proxy | weak_selection | 0.312 | 0.608 | 0.236 |
| 0 | 16 | mpc_objective | multiobjective_transfer_selection_score | objective_secondary_selection | 0.406 | 0.642 | 0.268 |
| 96 | 16 | mpc_tair_mae | tair_first_step_mae | offline_or_diagnostic_only | -0.091 | 0.475 | -0.221 |
| 96 | 16 | mpc_rhair_mae | rhair_first_step_mae | offline_or_diagnostic_only | 0.153 | 0.533 | -0.014 |
| 96 | 16 | mpc_co2_mae | co2_first_step_mae | offline_or_diagnostic_only | 0.037 | 0.504 | -0.116 |
| 96 | 16 | mpc_co2_mae | co2_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.309 | 0.417 | -0.486 |
| 96 | 16 | mpc_objective | multiobjective_transfer_selection_score | offline_or_diagnostic_only | 0.235 | 0.583 | 0.043 |
| 192 | 16 | mpc_tair_mae | tair_first_step_mae | secondary_selection | 0.659 | 0.742 | 0.614 |
| 192 | 16 | mpc_rhair_mae | rhair_first_step_mae | weak_selection | 0.250 | 0.625 | 0.089 |
| 192 | 16 | mpc_co2_mae | co2_first_step_mae | offline_or_diagnostic_only | -0.149 | 0.445 | -0.257 |
| 192 | 16 | mpc_co2_mae | co2_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.341 | 0.367 | -0.450 |
| 192 | 16 | mpc_objective | multiobjective_transfer_selection_score | offline_or_diagnostic_only | 0.174 | 0.567 | 0.039 |
| 288 | 16 | mpc_tair_mae | tair_first_step_mae | offline_or_diagnostic_only | -0.315 | 0.350 | -0.486 |
| 288 | 16 | mpc_rhair_mae | rhair_first_step_mae | offline_or_diagnostic_only | 0.159 | 0.550 | -0.021 |
| 288 | 16 | mpc_co2_mae | co2_first_step_mae | offline_or_diagnostic_only | 0.243 | 0.588 | 0.139 |
| 288 | 16 | mpc_co2_mae | co2_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.185 | 0.567 | 0.111 |
| 288 | 16 | mpc_objective | multiobjective_transfer_selection_score | objective_secondary_selection | 0.362 | 0.625 | 0.225 |
| 384 | 16 | mpc_tair_mae | tair_first_step_mae | offline_or_diagnostic_only | 0.097 | 0.533 | -0.096 |
| 384 | 16 | mpc_rhair_mae | rhair_first_step_mae | offline_or_diagnostic_only | 0.050 | 0.508 | -0.139 |
| 384 | 16 | mpc_co2_mae | co2_first_step_mae | offline_or_diagnostic_only | -0.319 | 0.387 | -0.559 |
| 384 | 16 | mpc_co2_mae | co2_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.429 | 0.350 | -0.514 |
| 384 | 16 | mpc_objective | multiobjective_transfer_selection_score | offline_or_diagnostic_only | 0.141 | 0.542 | -0.054 |

Interpretation rule:

- A metric is reusable only if its role and rank/pairwise statistics remain stable across start indices.
- If a metric changes role across start indices, report it as segment-dependent rather than as a universal selector.
- Whole-objective screening still requires final closed-loop validation even when per-target metrics are stable.
