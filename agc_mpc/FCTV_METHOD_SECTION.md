# Forecast-To-Control Transfer Validation Method

Canonical English version.
Mapped Chinese mirror: [FCTV_METHOD_SECTION.zh-CN.md](FCTV_METHOD_SECTION.zh-CN.md)

## Method Position

This section defines `Forecast-to-Control Transfer Validation` (FCTV), the validation layer used to test whether forecast-side model quality transfers to closed-loop MPC quality.

The method does not assume that a lower prediction error automatically implies better control. Instead, it treats that assumption as the hypothesis to be tested.

In this project, FCTV has two roles:

- Screening test: determine whether a forecast-side metric can rank predictors before running MPC.
- Diagnostic test: explain why a predictor with good offline forecasting may still fail to improve closed-loop control.

The current evidence supports the diagnostic role more strongly than the screening role.

## Validation Object

Each candidate predictor receives the same historical state window, future weather features, and future requested control inputs, then outputs a multi-step forecast for the controlled greenhouse variables:

- `Tair`
- `Rhair`
- `CO2air`

The MPC controller uses the predictor inside a closed-loop rollout. The final comparison therefore has two levels:

- Forecast side: errors computed from offline forecast trajectories.
- Control side: tracking objective and target errors computed from closed-loop MPC rollouts.

The central question is whether forecast-side ranking and control-side ranking agree.

## Forecast-Side Metrics

FCTV keeps metrics that can be computed without running MPC:

- First-step MAE: prediction error at the immediate next step.
- Control-horizon MAE: average prediction error over the short horizon used most directly by the controller.
- Control-horizon absolute bias: absolute mean signed error over the control horizon.
- Constraint-near MAE proxy: forecast error when the state is near an operational boundary or reference band.
- Normalized composite score: a scale-normalized combination of target-level forecast metrics.
- Gradient diagnostics: sensitivity of predicted states to future control inputs.

First-step and control-horizon metrics are included because MPC is repeatedly re-optimized and therefore uses short-horizon forecast quality heavily. Constraint-near metrics are included because mistakes near references or operating limits are more likely to change control actions than mistakes far away from active decision boundaries. Gradient diagnostics are included because MPC needs a predictor whose output reacts meaningfully to candidate control inputs.

## Control-Side Metrics

Closed-loop validation uses the same controller and rollout protocol for every predictor.

Primary metrics:

- `mpc_objective`: the closed-loop tracking/control objective.
- `mpc_tair_mae`: closed-loop `Tair` tracking error.
- `mpc_rhair_mae`: closed-loop `Rhair` tracking error.
- `mpc_co2_mae`: closed-loop `CO2air` tracking error.

Secondary metrics:

- control delta MAE
- action total variation

The primary paper conclusion should be based on closed-loop metrics, not only on offline forecasting metrics.

## Transfer Evidence

For each forecast-side metric and each control-side metric, FCTV computes ranking agreement.

### Spearman Rank Correlation

Spearman correlation measures whether two rankings move in the same direction. In this setting, it asks:

If models are ranked by a forecast metric, do they receive a similar rank under closed-loop MPC performance?

Interpretation:

- Near `1`: strong monotonic agreement; the metric may be useful for screening.
- Near `0`: little ranking relationship; the metric is mostly diagnostic.
- Below `0`: the forecast metric tends to rank models opposite to the control metric.

A practical screening threshold is not a mathematical law. For this project, values around `0.2` to `0.4` are treated as a weak-useful reference band because the sample size is small, model families are heterogeneous, and greenhouse MPC introduces target conflicts. Values below this band are too unstable for model selection; values inside it may support only cautious auxiliary screening; values clearly above it across starts and model families are needed before claiming reliable selection.

### Pairwise Ordering Consistency

Pairwise ordering consistency is the two-model ranking agreement rate.

For any two models, if model A has a better forecast-side metric than model B, FCTV checks whether A also has better closed-loop control performance than B. The ratio of agreeing model pairs is the pairwise consistency.

Interpretation:

- `0.5`: close to random ordering.
- Above `0.6`: weak but potentially useful ordering signal.
- Above `0.7`: stronger ranking evidence, if stable across starts and target metrics.

This statistic is useful because it describes the practical selection question directly: whether choosing the better forecast model would also choose the better control model.

### Top-K Overlap

Top-k overlap compares whether the best forecast-side models are also among the best closed-loop models.

This matters because model selection usually cares more about selecting a short list of promising candidates than ranking every weak model exactly.

### Robustness Checks

FCTV should test whether conclusions survive:

- different rollout start indices
- leave-one-model analysis
- leave-one-family analysis, when family labels are available
- target-specific comparisons for `Tair`, `Rhair`, and `CO2air`

A forecast metric should not be called a reliable selector if it only works for one start, one target, or one narrow local model family.

## Current Empirical Pattern

The exploratory stage produced a clear degradation pattern:

- In the early CO2-focused pool, short-horizon CO2 metrics looked useful.
- After expanding to a broader model pool, CO2 forecast metrics lost stable screening power.
- After moving to multi-target and multi-start analysis, transfer became model-pool dependent and start dependent.

Representative evidence:

- In the expanded 24-model analysis, `Rhair` first-step error retained a moderate relationship with `Rhair` closed-loop error (`Spearman = 0.592`, pairwise consistency `0.732`).
- In the same 24-model analysis, CO2 first-step transfer was weak (`Spearman = 0.168`, pairwise consistency `0.549`), and CO2 constraint-near transfer was nearly random (`Spearman = 0.015`, pairwise consistency `0.507`).
- In the final 16-model, 5-start analysis, CO2 first-step transfer remained start dependent (`0.364`, `0.037`, `-0.149`, `0.243`, `-0.319` for starts `0`, `96`, `192`, `288`, and `384`).
- The multi-objective transfer score also remained unstable across starts (`0.406`, `0.235`, `0.174`, `0.362`, `0.141` for starts `0`, `96`, `192`, `288`, and `384`).

Final closed-loop model evidence:

- `current_hybrid_transformer` had the best mean objective across the 5 starts (`0.0662 +/- 0.0269`).
- `itransformer_co2_residual` had the best mean CO2 closed-loop tracking error (`CO2air MAE = 10.215 +/- 2.043`) and the second-best mean objective (`0.0701 +/- 0.0234`).
- This separates the model conclusion from the metric conclusion: robust closed-loop winners can be identified by direct MPC validation, but the tested offline forecast metrics still cannot reliably select them alone.

This pattern means the current forecast-side metrics should not be used as deterministic closed-loop model selectors.

## Paper-Facing Claim

The defensible claim is:

Standard offline forecast metrics, even when targeted to short horizons and constraint-near regions, are not reliable substitutes for direct closed-loop MPC validation in the tested cross-family greenhouse control benchmark.

The method still has value because it shows where the forecast-control assumption breaks:

- target dependence
- model-family dependence
- rollout-segment dependence
- mismatch between prediction error and control-action sensitivity

The paper should therefore present FCTV as a validation and diagnostic framework, not as a finished universal forecast-derived control score.
