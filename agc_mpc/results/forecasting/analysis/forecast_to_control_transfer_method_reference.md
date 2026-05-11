# Forecast-To-Control Transfer Validation Method

Scope: `Reference` compartment, strict three-target protocol (`Tair`, `Rhair`, `CO2air`), current 24-model extended FCTV pool, 96-step `GradientMPC` closed-loop linkage.

## Method Position

Forecast-to-Control Transfer Validation (FCTV) is a screening and diagnosis protocol between offline forecasting evaluation and closed-loop MPC validation. It is not a new predictor architecture, not a stability proof, and not a replacement for final closed-loop rollout tests.

The method asks a narrower question than ordinary forecasting evaluation: which forecast-side errors are control-relevant for the current greenhouse MPC setup?

## Metric Origins

Ordinary metrics such as MAE, RMSE, and R2 come from forecasting and regression evaluation. They mainly measure global fit. R2 is the regression goodness-of-fit metric `R2 = 1 - SSE / SST`, which measures the fraction of target variance explained by a model.

These metrics are not enough for MPC predictor selection because receding-horizon MPC optimizes a plan but executes only the front part of that plan. The value of a forecast error depends on where it occurs, which target it affects, whether it is biased, whether it appears near an operational boundary or reference band, and whether the predictor preserves usable sensitivity to future control inputs.

FCTV therefore derives candidate metrics from MPC mechanics before validating them against closed-loop outcomes:

- `first_step_mae`: immediate prediction error for the step most directly coupled to the executed control move.
- `control_horizon_mae`: short-horizon error over the `control_horizon=6` steps optimized by GradientMPC.
- `control_horizon_abs_bias`: systematic short-horizon overprediction or underprediction risk.
- `constraint_near_mae_proxy`: error when the state is near the empirical low/high operating bands.
- Gradient diagnostics: whether forecast outputs and objective retain useful sensitivity to future requested controls.
- Transfer selection scores: normalized rank composites used for reporting, not universal guarantees.

## Validation Procedure

1. Compute the FCTV metrics offline using the logged future controls and held-out Reference samples.
2. Link every predictor to the corresponding 96-step `GradientMPC` closed-loop outcomes: `mpc_tair_mae`, `mpc_rhair_mae`, `mpc_co2_mae`, and `mpc_objective`.
3. For each candidate forecast-side metric and each closed-loop target, compute Pearson correlation, Spearman rank correlation, pairwise consistency, top-k hit rate, leave-one-model robustness, and leave-one-family robustness.
4. Classify metric roles from the validated transfer statistics rather than from intuition alone.

## Role Classification

- `primary_selection`: strong enough for target-specific screening in the current pool.
- `secondary_selection`: useful supporting signal, but not sufficient alone.
- `weak_selection`: directionally useful, mainly for triage.
- `offline_or_diagnostic_only`: useful for forecasting evaluation or interpretation, not for direct control selection.
- `diagnostic_only`: gradient/sensitivity evidence for explanation and failure diagnosis.

## Current 24-Model Conclusion

The current evidence supports variable-specific metric roles, not one universal selector.

- `Rhair`: `rhair_first_step_mae` remains the strongest target-specific signal for `mpc_rhair_mae`, but its role is now `secondary_selection` with Spearman `0.592` and pairwise consistency `0.732`.
- `CO2air`: `co2_first_step_mae` and `co2_constraint_near_mae_proxy` are no longer stable selectors in the 24-model extended pool. Their Spearman values are `0.168` and `0.015`, so they should be reported as diagnostic / pool-dependent rather than selection metrics.
- `Tair`: `tair_first_step_mae` is still not a reliable selector for `mpc_tair_mae`; its Spearman is `-0.123`.
- Whole objective: `multiobjective_transfer_selection_score` is not a reliable whole-objective selector in the extended pool with Spearman `0.167` and pairwise consistency `0.564`.
- Whole objective has one useful secondary signal: `rhair_first_step_mae` against `mpc_objective`, with Spearman `0.507` and pairwise consistency `0.703`.

This supports the paper-facing claim that forecast quality must be evaluated through control-relevant timing, bias, near-boundary behavior, and target-specific transfer roles. Final closed-loop MPC validation is still required for whole-objective model claims.

## Attribution Rule

Use metric-mediated attribution:

1. A framework or module changes forecast-side behavior.
2. The changed behavior is measured by a specific FCTV metric.
3. That metric has a validated relationship to a closed-loop target.
4. Therefore the framework or module has evidence of control relevance for that target.

Avoid the unsupported shortcut: "this framework is better, therefore control improves."

## Remaining Robustness Requirement

The current 24-model result is a stronger stress test than the earlier 17-model pool and shows that some previously useful CO2 screening signals are pool-dependent.

An initial multi-start robustness run was first completed on a representative 10-model subset across start indices `0`, `96`, and `192`. It was then expanded to a stronger 16-model subset by adding generic residual, PatchTST, Transformer, NLinear, DLinear, and wavelet-residual controls.

The 16-model multi-start result shows that the main metric roles are segment-dependent:

- `co2_first_step_mae -> mpc_co2_mae` is `secondary_selection` at start `0`, but becomes `offline_or_diagnostic_only` at starts `96` and `192`.
- `rhair_first_step_mae -> mpc_rhair_mae` is only `weak_selection` at start `0`, and becomes `offline_or_diagnostic_only` at starts `96` and `192`.
- `multiobjective_transfer_selection_score -> mpc_objective` is only weak at start `0` and diagnostic-only at starts `96` and `192`.
- `tair_first_step_mae -> mpc_tair_mae` remains unreliable.

The strongest model-side observation from the expanded multi-start subset is that `itransformer_co2_residual` is the most stable CO2 closed-loop tracker among the tested predictors:

- start `0`: best CO2, `CO2air MAE=6.331`, objective `0.0558`
- start `96`: best CO2, `CO2air MAE=11.074`, objective `0.0654`
- start `192`: best CO2, `CO2air MAE=10.701`, objective `0.0465`

The expanded subset also shows that simpler baselines remain competitive in some segments:

- start `192`: `dlinear_forecaster` reaches `CO2air MAE=11.316` and objective `0.0449`.
- start `192`: `itransformer_residual` reaches `CO2air MAE=11.644` and objective `0.0360`.

This strengthens the method limitation statement: FCTV metrics should be reported with explicit model-pool and rollout-segment scope. A full 24-model multi-start run is still needed before making a reusable-method claim.
