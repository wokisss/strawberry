# FCTV Multi-Start Transfer Robustness

This report reuses the same forecast-side FCTV metrics and replaces closed-loop outcomes with repeated `GradientMPC` 96-step rollouts from multiple test-set start indices.

| start_idx | model_count | control_target | metric | role | spearman | pairwise | leave-model spearman min |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 16 | mpc_tair_mae | tair_first_step_mae | offline_or_diagnostic_only | -0.009 | 0.492 | -0.161 |
| 0 | 16 | mpc_rhair_mae | rhair_first_step_mae | weak_selection | 0.282 | 0.617 | 0.129 |
| 0 | 16 | mpc_co2_mae | co2_first_step_mae | secondary_selection | 0.366 | 0.630 | 0.268 |
| 0 | 16 | mpc_co2_mae | co2_constraint_near_mae_proxy | offline_or_diagnostic_only | 0.147 | 0.558 | 0.043 |
| 0 | 16 | mpc_objective | multiobjective_transfer_selection_score | weak_selection | 0.338 | 0.617 | 0.182 |
| 96 | 16 | mpc_tair_mae | tair_first_step_mae | offline_or_diagnostic_only | -0.056 | 0.492 | -0.200 |
| 96 | 16 | mpc_rhair_mae | rhair_first_step_mae | offline_or_diagnostic_only | -0.068 | 0.458 | -0.296 |
| 96 | 16 | mpc_co2_mae | co2_first_step_mae | offline_or_diagnostic_only | -0.263 | 0.395 | -0.402 |
| 96 | 16 | mpc_co2_mae | co2_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.429 | 0.375 | -0.586 |
| 96 | 16 | mpc_objective | multiobjective_transfer_selection_score | offline_or_diagnostic_only | -0.074 | 0.458 | -0.311 |
| 192 | 16 | mpc_tair_mae | tair_first_step_mae | weak_selection | 0.332 | 0.608 | 0.221 |
| 192 | 16 | mpc_rhair_mae | rhair_first_step_mae | offline_or_diagnostic_only | 0.174 | 0.583 | 0.046 |
| 192 | 16 | mpc_co2_mae | co2_first_step_mae | offline_or_diagnostic_only | -0.243 | 0.412 | -0.359 |
| 192 | 16 | mpc_co2_mae | co2_constraint_near_mae_proxy | offline_or_diagnostic_only | -0.415 | 0.333 | -0.507 |
| 192 | 16 | mpc_objective | multiobjective_transfer_selection_score | offline_or_diagnostic_only | 0.144 | 0.567 | -0.011 |

Interpretation rule:

- A metric is reusable only if its role and rank/pairwise statistics remain stable across start indices.
- If a metric changes role across start indices, report it as segment-dependent rather than as a universal selector.
- Whole-objective screening still requires final closed-loop validation even when per-target metrics are stable.
