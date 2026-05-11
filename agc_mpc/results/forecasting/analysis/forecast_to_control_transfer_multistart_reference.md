# FCTV Multi-Start Transfer Robustness

This report reuses the same forecast-side FCTV metrics and replaces closed-loop outcomes with repeated `GradientMPC` 96-step rollouts from multiple test-set start indices.

| start_idx | model_count | control_target | metric | role | spearman | pairwise | leave-model spearman min |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 10 | mpc_tair_mae | tair_first_step_mae | offline_or_diagnostic_only | -0.273 | 0.422 | -0.533 |
| 0 | 10 | mpc_rhair_mae | rhair_first_step_mae | secondary_selection | 0.418 | 0.667 | 0.200 |
| 0 | 10 | mpc_co2_mae | co2_first_step_mae | secondary_selection | 0.498 | 0.705 | 0.360 |
| 0 | 10 | mpc_co2_mae | co2_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.006 | 0.511 | -0.183 |
| 0 | 10 | mpc_objective | multiobjective_transfer_selection_score | weak_selection | 0.285 | 0.600 | 0.017 |
| 96 | 10 | mpc_tair_mae | tair_first_step_mae | offline_or_diagnostic_only | -0.115 | 0.467 | -0.400 |
| 96 | 10 | mpc_rhair_mae | rhair_first_step_mae | offline_or_diagnostic_only | -0.103 | 0.444 | -0.517 |
| 96 | 10 | mpc_co2_mae | co2_first_step_mae | offline_or_diagnostic_only | -0.146 | 0.409 | -0.427 |
| 96 | 10 | mpc_co2_mae | co2_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.067 | 0.578 | -0.133 |
| 96 | 10 | mpc_objective | multiobjective_transfer_selection_score | offline_or_diagnostic_only | 0.188 | 0.556 | -0.117 |
| 192 | 10 | mpc_tair_mae | tair_first_step_mae | weak_selection | 0.358 | 0.600 | 0.183 |
| 192 | 10 | mpc_rhair_mae | rhair_first_step_mae | offline_or_diagnostic_only | 0.091 | 0.578 | -0.117 |
| 192 | 10 | mpc_co2_mae | co2_first_step_mae | offline_or_diagnostic_only | -0.243 | 0.432 | -0.433 |
| 192 | 10 | mpc_co2_mae | co2_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.345 | 0.356 | -0.517 |
| 192 | 10 | mpc_objective | multiobjective_transfer_selection_score | weak_selection | 0.285 | 0.600 | 0.117 |

Interpretation rule:

- A metric is reusable only if its role and rank/pairwise statistics remain stable across start indices.
- If a metric changes role across start indices, report it as segment-dependent rather than as a universal selector.
- Whole-objective screening still requires final closed-loop validation even when per-target metrics are stable.
