# Forecast-To-Control Attribution Notes

Scope: `Reference` compartment, strict three-target protocol (`Tair`, `Rhair`, `CO2air`), 17-model FCTV pool, 96-step closed-loop `GradientMPC`.

## What Can Be Claimed

FCTV currently supports metric-mediated screening claims, not standalone causal claims about one architecture.

- A model family or module is useful for control only if it improves a forecast-side metric that is validated against closed-loop outcomes, or if it directly improves closed-loop outcomes under a fixed controller.
- Ordinary offline forecast gains are not enough for attribution. A module that improves final-step or full-horizon CO2 MAE can still fail to improve closed-loop CO2 tracking.
- Framework and module effects must be separated by family-level ablations: same backbone with different modules, and ideally the same module concept across different backbones.

## Current Metric Roles

| Metric | Closed-loop target | Current role | Interpretation |
| --- | --- | --- | --- |
| `rhair_first_step_mae` | `mpc_rhair_mae` | `primary_selection` | Strongest current target-specific signal after standard baseline expansion. |
| `co2_first_step_mae` | `mpc_co2_mae` | `secondary_selection` | Still the strongest CO2 screening signal, but weaker after adding standard baselines. |
| `co2_constraint_near_mae_proxy` | `mpc_co2_mae` | `secondary_selection` | Useful supporting CO2 signal. |
| `co2_final_step_mae` | `mpc_co2_mae` | `offline_or_diagnostic_only` | Terminal CO2 accuracy does not explain receding-horizon CO2 control. |
| `tair_first_step_mae` | `mpc_tair_mae` | `offline_or_diagnostic_only` | Current Tair control benefit is not explained by target-matched first-step forecast error. |
| `multiobjective_transfer_selection_score` | `mpc_objective` | `weak_selection` | Directionally useful but not strong enough as an objective selector. |

## Framework-Level Evidence

The standard baseline expansion weakens any claim that the PHF / CO2 family alone proves the FCTV method.

- `segrnn_forecaster` has weak offline CO2 forecasting (`CO2air` final MAE `84.046`) but good closed-loop CO2 tracking (`CO2air MAE=14.425`).
- `frequency_forecaster` has weak offline CO2 forecasting (`CO2air` final MAE `91.544`) but moderate closed-loop CO2 tracking (`CO2air MAE=15.530`) and poor whole-objective behavior (`objective=0.4338`).
- These cases show that framework choice can change control behavior through sensitivity and short-horizon behavior, not just through ordinary forecast MAE.

Conclusion: framework effects exist, but they cannot be summarized by offline forecasting rank.

## Module-Level Evidence

Within the iTransformer / PHF family, module changes alter different FCTV metrics:

| Predictor | Relevant interpretation |
| --- | --- |
| `itransformer_residual` | Generic residual baseline. |
| `itransformer_co2_late_residual` | CO2 late adapter improves short-horizon CO2 screening metrics and improves closed-loop behavior relative to some generic residual settings. |
| `itransformer_co2_late_frozen_expert` | Strong short-horizon CO2 signal, but poor terminal CO2 metric; useful evidence that terminal accuracy is not the control mechanism. |
| `itransformer_co2_horizon_mixture` | Better terminal-style CO2 behavior but weaker short-horizon/control transfer. |
| `itransformer_co2_control_aware_fusion` | Improves report-facing balance, but should be described as one application of FCTV rather than the method itself. |

Conclusion: module effects are target- and horizon-specific. The same module can help one metric and hurt another, so attribution must be stated through metric changes.

## Metric-Mediated Attribution Rule

Use the following wording pattern:

1. A framework/module changes forecast-side behavior.
2. The changed behavior is measured by a specific FCTV metric.
3. That FCTV metric has a validated relationship to a closed-loop target.
4. Therefore the framework/module has evidence of control relevance for that target.

Avoid the unsupported shortcut:

- "This framework is better, therefore control improves."

Preferred statement:

- "`CO2air` control transfer is most consistently associated with first-step and constraint-near CO2 errors, while terminal CO2 accuracy is diagnostic only in the current pool."
- "`Rhair` has the strongest current target-specific selection signal through `rhair_first_step_mae`."
- "Whole-objective selection remains weak, so model choice should still be validated in closed-loop MPC."

## Remaining Attribution Limits

- The current pool is broader than before, but it is still one dataset, one compartment, one reference mode, and one 96-step rollout setup.
- Family-level ablations are available mainly inside the iTransformer / PHF branch; standard baselines are not yet ablated module-by-module.
- Gradient diagnostics are currently best used for explanation and failure diagnosis, not as direct ranking metrics.
- A stronger causal claim would require repeated rollouts across start indices, leave-family robustness, and controlled module swaps across more than one backbone.
