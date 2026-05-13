# FCTV Stage Report

Canonical English version.
Mapped Chinese mirror: [FCTV_STAGE_REPORT.zh-CN.md](FCTV_STAGE_REPORT.zh-CN.md)

## Reporting Position

This report summarizes the closed FCTV stage for supervisor reporting. The stage should be presented as a controlled negative/diagnostic result, not as a failed project.

The core message is:

Offline forecast metrics can describe predictor behavior, but the tested metrics do not reliably select closed-loop MPC winners after the benchmark is expanded across model families, targets, and rollout starts. Direct closed-loop validation remains necessary.

## Research Question

Main question:

Can forecast-side metrics computed before MPC reliably screen predictors for greenhouse closed-loop MPC?

The tested hypothesis was reasonable:

- MPC uses forecasts inside optimization.
- Short-horizon forecast errors should matter because MPC re-optimizes repeatedly.
- Errors near references or operational boundaries should matter more because they can change control actions.
- Therefore, forecast-side metrics might predict closed-loop control benefit.

The final evidence shows that this hypothesis is only partially true. It can work locally, but it is not stable enough to replace closed-loop validation.

## Experiment Chain

Stage 1: CO2-focused metric discovery.

- The early model pool focused on CO2 and local PHF/CO2-aware variants.
- Short-horizon CO2 metrics appeared useful.
- This justified expanding the experiment instead of immediately declaring a selector.

Stage 2: Expanded model-pool validation.

- The pool was expanded to standard and modern model families, including linear, recurrent, Transformer-style, PatchTST/iTransformer residual, and CO2/PHF variants.
- After this expansion, the CO2 first-step and constraint-near metrics lost stable screening power.
- This showed that the early signal was partly model-pool dependent.

Stage 3: Multi-target validation.

- The analysis moved from CO2-only to `Tair`, `Rhair`, `CO2air`, and whole-objective comparisons.
- `Rhair` first-step error retained moderate transfer in one expanded setting, but CO2 and whole-objective transfer were weak.
- This showed that transfer is target dependent.

Stage 4: Multi-start closed-loop validation.

- The benchmark was repeated across rollout starts.
- The final run used `16` predictors, starts `0`, `96`, `192`, `288`, `384`, and `96` closed-loop steps.
- This produced `80` closed-loop records.
- The final result confirmed start dependence.

## Final Benchmark Outputs

Main final suite:

- `results/control/summaries/fctv_multistart_gradient_mpc_reference_96steps_16predictors_25890932c3_starts_0_96_192_288_384.json`

Transfer analysis:

- `results/forecasting/analysis/forecast_to_control_transfer_final_reference.{json,csv,md}`
- `results/forecasting/figures/comparisons/forecast_to_control_transfer_final_reference.png`

Closed-loop ranking:

- `results/forecasting/analysis/fctv_final_multistart_model_rankings_reference.{csv,md}`
- `results/forecasting/figures/comparisons/fctv_final_multistart_model_rankings_reference.png`

Weekly reporting figure:

- `results/forecasting/figures/comparisons/fctv_weekly_metric_degradation_summary.png`

## Key Quantitative Results

Transfer metrics:

- CO2 first-step transfer across starts was unstable:
  - start `0`: Spearman `0.364`, pairwise `0.613`
  - start `96`: Spearman `0.037`, pairwise `0.504`
  - start `192`: Spearman `-0.149`, pairwise `0.445`
  - start `288`: Spearman `0.243`, pairwise `0.588`
  - start `384`: Spearman `-0.319`, pairwise `0.387`
- Multi-objective transfer score was also unstable:
  - start `0`: Spearman `0.406`, pairwise `0.642`
  - start `96`: Spearman `0.235`, pairwise `0.583`
  - start `192`: Spearman `0.174`, pairwise `0.567`
  - start `288`: Spearman `0.362`, pairwise `0.625`
  - start `384`: Spearman `0.141`, pairwise `0.542`

Closed-loop winners:

- Best mean objective: `current_hybrid_transformer`, objective `0.0662 +/- 0.0269`.
- Best mean CO2 tracking: `itransformer_co2_residual`, `CO2air MAE = 10.215 +/- 2.043`.
- `itransformer_co2_residual` was also second-best by mean objective: `0.0701 +/- 0.0234`.

## Interpretation

The result is not that forecasting is irrelevant. The result is that ordinary forecast-side metrics are not sufficient as universal selectors.

Reasons:

- MPC converts forecasts into actions, so action sensitivity matters.
- A predictor can reduce offline error in regions that do not affect the active control decision.
- Multi-target control creates conflicts: improving CO2 can worsen `Tair` or `Rhair` behavior.
- Rankings change across rollout starts because greenhouse dynamics and reference difficulty change by segment.
- Local model-family comparisons can make a metric look useful, but the relationship can disappear after adding broader model families.

## Defensible Conclusion

The defensible paper-facing conclusion is:

Under the tested cross-family AGC greenhouse benchmark, standard offline forecast metrics, including short-horizon and constraint-near variants, are not reliable substitutes for direct closed-loop MPC validation.

FCTV remains useful because it diagnoses where the forecast-control assumption breaks:

- target dependence
- model-family dependence
- rollout-start dependence
- mismatch between prediction error and control-action sensitivity

## Recommended Next Step

The next research step should move from tracking-only MPC to economic/resource-aware MPC.

Reason:

- Tracking-only MPC answers whether a predictor can follow recorded or reference trajectories.
- Greenhouse control is ultimately an economic trade-off between tracking, heating, CO2 dosing, lighting, ventilation, irrigation, and actuator movement.
- The FCTV stage already showed that closed-loop validation is necessary; the next stage should make the closed-loop objective more realistic.

