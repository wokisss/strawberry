# FCTV Experiment Design

Canonical English version.
Mapped Chinese mirror: [FCTV_EXPERIMENT_DESIGN.zh-CN.md](FCTV_EXPERIMENT_DESIGN.zh-CN.md)

## Position

This document freezes the paper-facing experiment design for the next phase. The exploratory FCTV stage is now closed: ordinary forecast-side metrics were useful as diagnostic signals, but they did not remain stable universal selectors after expanding the model pool and repeated closed-loop starts.

The next phase should not add experiments opportunistically. It should answer a fixed paper question under a fixed protocol.

## Paper Question

Working title:

**Do Better Forecasts Lead to Better Control? Forecast-to-Control Validation for Greenhouse MPC**

Main question:

Can offline forecast-side metrics reliably select predictors that improve closed-loop greenhouse MPC?

Research questions:

- `RQ1`: Do standard forecast-side metrics predict closed-loop MPC performance across model families?
- `RQ2`: Are forecast-to-control relationships stable across rollout segments?
- `RQ3`: If ordinary forecast metrics are not stable selectors, which predictors are robust under direct closed-loop validation?

Allowed claim:

- Standard offline forecast metrics are not reliable substitutes for closed-loop MPC validation under the tested cross-family greenhouse benchmark.

Disallowed claim:

- No possible forecast-derived metric can ever predict control performance.

## Final Model Pool

The final closed-loop benchmark uses `16` predictors. The pool is deliberately cross-family and avoids filling the experiment with only local PHF variants.

| family | predictors |
| --- | --- |
| Linear | `dlinear_forecaster`, `nlinear_forecaster` |
| Recurrent | `gru_forecaster`, `lstm_forecaster` |
| Segmented recurrent | `segrnn_forecaster` |
| Frequency/decomposition-style | `frequency_forecaster` |
| Transformer | `transformer_forecaster`, `current_hybrid_transformer`, `transformer_hybrid_residual` |
| Patch / iTransformer residual | `patchtst_residual`, `itransformer_residual` |
| CO2-aware / PHF | `itransformer_co2_residual`, `itransformer_co2_late_residual`, `itransformer_co2_late_frozen_expert`, `itransformer_co2_horizon_mixture`, `itransformer_co2_control_aware_fusion` |

Rationale:

- The pool includes standard baselines, modern sequence models, residual variants, and representative CO2/PHF models.
- It contains enough model-family diversity to test transfer stability.
- It is small enough for repeated closed-loop validation.
- It intentionally excludes `diffmpc_style_transformer` until its protocol is aligned with the 288-step AGC history setup.

## Closed-Loop Benchmark Protocol

Fixed protocol:

- Dataset: AGC 2019 Reference compartment.
- Targets: `Tair`, `Rhair`, `CO2air`.
- Forecast history: current AGC three-target protocol.
- Controller: `GradientMPC`.
- Rollout mode: recorded weather / current surrogate closed-loop setup.
- Rollout length: `96` steps.
- Start indices: `0`, `96`, `192`, `288`, `384`.
- Primary closed-loop metrics: `mpc_objective`, `mpc_tair_mae`, `mpc_rhair_mae`, `mpc_co2_mae`.
- Secondary closed-loop metrics: control delta MAE and action total variation.

The final benchmark should report mean and standard deviation across starts, not a single-start leaderboard.

## Forecast-To-Control Metrics

The paper-facing FCTV analysis should retain:

- first-step MAE per target
- control-horizon MAE per target
- control-horizon absolute bias per target
- constraint-near MAE proxy per target
- normalized transfer selection scores
- gradient diagnostics as diagnostic-only evidence

Selection evidence should be evaluated with:

- Spearman rank correlation
- pairwise ordering consistency
- top-k overlap
- leave-one-model robustness
- leave-one-family robustness where family labels are available

## Formal Experiment Matrix

Experiment 1: Forecasting benchmark.

- Goal: establish offline forecasting behavior for the final model pool.
- Output: per-target first-step, control-horizon, full-horizon, and final-step metrics.

Experiment 2: Closed-loop MPC benchmark.

- Goal: determine robust closed-loop winners under the fixed model pool and five starts.
- Output: objective leaderboard and per-target leaderboards with mean/std across starts.

Experiment 3: FCTV transfer analysis.

- Goal: test whether forecast-side metrics explain the closed-loop benchmark.
- Output: Spearman, pairwise consistency, top-k, robustness, and metric-role tables.

Experiment 4: Diagnostic discussion.

- Goal: explain where transfer fails.
- Output: model-family dependence, start dependence, and target-conflict discussion.

## Runnable Entry Points

Print the fixed benchmark plan:

```powershell
C:\Users\wokis\.conda\envs\strawberry_env\python.exe agc_mpc\run_fctv_final_closed_loop_benchmark.py --print-plan
```

Run the formal closed-loop benchmark:

```powershell
C:\Users\wokis\.conda\envs\strawberry_env\python.exe agc_mpc\run_fctv_final_closed_loop_benchmark.py
```

Analyze the generated suite:

```powershell
C:\Users\wokis\.conda\envs\strawberry_env\python.exe agc_mpc\analyze_fctv_multistart_transfer.py --suite-json <generated_suite_json> --prefix forecast_to_control_transfer_final_reference
```

## Executed Final Benchmark

The formal 16-model, 5-start closed-loop benchmark was executed on `2026-05-12`.

Generated suite:

- `results/control/summaries/fctv_multistart_gradient_mpc_reference_96steps_16predictors_25890932c3_starts_0_96_192_288_384.json`

Generated FCTV analysis:

- `results/forecasting/analysis/forecast_to_control_transfer_final_reference.{json,csv,md}`
- per-start `forecast_to_control_transfer_final_reference_start*.{json,csv,md}` and robustness CSVs
- `results/forecasting/figures/comparisons/forecast_to_control_transfer_final_reference.png`
- per-start summary and robustness figures under `results/forecasting/figures/comparisons`

Generated closed-loop ranking output:

- `results/forecasting/analysis/fctv_final_multistart_model_rankings_reference.{csv,md}`
- `results/forecasting/figures/comparisons/fctv_final_multistart_model_rankings_reference.png`

Final benchmark conclusion:

- Forecast-side transfer metrics remain start dependent and are not stable universal selectors.
- `current_hybrid_transformer` is the best mean-objective predictor across the 5 starts.
- `itransformer_co2_residual` is the best mean-CO2 closed-loop tracker across the 5 starts.

## Current-Week Scope

This week should complete:

- A: freeze this paper-style experiment design.
- B: prepare and, when compute time is available, run the final closed-loop benchmark.
- C: write the FCTV method section in paper language.

Next step, not this scope:

- F: prepare the supervisor-facing staged report.
- E: start economic/resource-aware MPC only after the tracking-control benchmark is stable.
