# CONTEXT.md

English canonical version.
Mapped Chinese mirror: [CONTEXT.zh-CN.md](c:/repositories/strawberry/CONTEXT.zh-CN.md)
Last synchronized: `2026-04-07`

## 0. Purpose And Maintenance Policy

This file is the long-lived project context for the `strawberry` workspace.

From `2026-04-07` onward, the documentation policy is:

- `*.md` is the English canonical version for long-lived project docs whenever practical.
- `*.zh-CN.md` is the synchronized Chinese mirror.
- When a maintained bilingual document changes, both versions must be updated in the same work turn.
- If any maintained document shows mojibake, encoding corruption, or suspicious characters, report it immediately before continuing.
- Do not silently overwrite a corrupted document without stating what happened.

This policy currently applies to:

- [CONTEXT.md](c:/repositories/strawberry/CONTEXT.md) and [CONTEXT.zh-CN.md](c:/repositories/strawberry/CONTEXT.zh-CN.md)
- [CO2_PAPERS_AND_DIRECTION.md](c:/repositories/strawberry/agc_mpc/CO2_PAPERS_AND_DIRECTION.md) and [CO2_PAPERS_AND_DIRECTION.zh-CN.md](c:/repositories/strawberry/agc_mpc/CO2_PAPERS_AND_DIRECTION.zh-CN.md)
- [CO2_SPECIALIST_REPORT.md](c:/repositories/strawberry/agc_mpc/CO2_SPECIALIST_REPORT.md) and [CO2_SPECIALIST_REPORT.zh-CN.md](c:/repositories/strawberry/agc_mpc/CO2_SPECIALIST_REPORT.zh-CN.md)

## 1. Project Mainline

The main objective is not to reproduce the old strawberry thesis pipeline.

The active mainline is:

**control-oriented greenhouse multi-step forecasting + closed-loop MPC**

Current project split:

- Legacy reference project: [diffmpc](c:/repositories/strawberry/diffmpc)
- Active mainline project: [agc_mpc](c:/repositories/strawberry/agc_mpc)

Rules:

- New implementation work should go to [agc_mpc](c:/repositories/strawberry/agc_mpc) by default.
- Do not move the main development stream back to `diffmpc` unless there is a clear reason.
- Default runtime environment is `strawberry_env`.

## 2. Core Data And Interface

Primary dataset:

- [AutonomousGreenhouseChallenge_edition2](c:/repositories/strawberry/AutonomousGreenhouseChallenge_edition2)

Secondary / historical dataset:

- [Strawberry Greenhouse Environmental Control Dataset(version2).csv](c:/repositories/strawberry/Strawberry%20Greenhouse%20Environmental%20Control%20Dataset(version2).csv)

Key AGC interpretation:

- `Weather.csv` provides future exogenous weather.
- `GreenhouseClimate.csv` provides indoor climate, actuator states, and setpoints.
- `*_sp` means requested setpoints.
- `*_vip` means realized setpoints / realized commands.

Current forecasting interface:

- `x_past`: historical indoor states and actuator feedback
- `w_future`: future weather and time features
- `u_future`: future requested control inputs
- `y_future`: future targets

Default four-target configuration in code:

- `Tair`
- `Rhair`
- `CO2air`
- `Tot_PAR`

Current fair-budget benchmarking often focuses on the three-target subset:

- `Tair`
- `Rhair`
- `CO2air`

## 3. Current Codebase Status

Stable implemented components:

- AGC data loading, cleaning, alignment, and leak-free splitting
- multi-compartment joint training support
- global scaling under joint training
- forecasting baselines:
  - `GRU`
  - `DLinear`
  - `SegRNN`
  - `Transformer`
  - `Transformer-hybrid`
- residual variants:
  - `transformer_hybrid_residual`
  - `itransformer_residual`
  - `itransformer_co2_residual`
  - `itransformer_co2_late_residual`
  - `patchtst_residual`
- closed-loop surrogate control benchmark:
  - `GradientMPC`
  - `CEMMPC`

Recent CO2-specific additions:

- standalone CO2 specialist models:
  - `co2_env_lstm`
  - `co2_vmd_lstm_fusion`
  - `co2_wavelet_gru_attn`

## 4. Default Experimental Protocol

Default forecasting benchmark:

- regime: `joint_all`
- evaluation compartment: `Reference`
- sequence length: `288` steps = `24 h`
- forecast horizon: `24` steps = `2 h`

Fair-budget protocol used for formal comparisons:

- `batch_size = 256`
- `num_epochs = 200`
- `learning_rate = 1e-4`
- `lambda_trend = 0.3`
- `early_stop_patience = 15`

Default control benchmark:

- `trajectory reference`
- `surrogate rollout`
- compare `GradientMPC` vs `CEMMPC`

## 5. Established Findings

### 5.1 Dataset And Regime

- Switching the mainline from the old strawberry dataset to `AGC 2019` was the right move.
- `joint_all + Reference eval` remains the default benchmark protocol.
- Early one-epoch smoke-test results must not be used as formal conclusions.

### 5.2 Forecasting Mainline

- `current_hybrid_transformer` is still the most stable overall multi-target predictor.
- `itransformer_residual` is the strongest established residual baseline worth tracking.
- `itransformer_co2_late_residual` improves `CO2air` relative to the original `itransformer_residual`, but gives back some `Rhair`.

Formal recent `itransformer` residual results under fair budget:

- `itransformer_residual`
  - `Tair`: Full `R2=0.9494`, MAE `0.618`
  - `Rhair`: Full `R2=0.9030`, MAE `3.802`
  - `CO2air`: Full `R2=0.7078`, MAE `51.161`
- `itransformer_co2_residual`
  - `Tair`: Full `R2=0.9435`, MAE `0.639`
  - `Rhair`: Full `R2=0.8787`, MAE `4.244`
  - `CO2air`: Full `R2=0.6991`, MAE `54.001`
- `itransformer_co2_late_residual`
  - `Tair`: Full `R2=0.9503`, MAE `0.595`
  - `Rhair`: Full `R2=0.8849`, MAE `4.172`
  - `CO2air`: Full `R2=0.7553`, MAE `47.797`

Interpretation:

- The first heavy CO2 branch was not good enough.
- A lighter late-horizon CO2 adapter is more promising.
- `CO2air` benefits from specialized correction, but the specialization must not destabilize the whole multi-target model.

### 5.3 Closed-Loop Control

Current control-side conclusion:

- `GradientMPC` is more reliable than `CEMMPC` on the current surrogate benchmark.
- `current_hybrid_transformer + GradientMPC` is the strongest overall closed-loop combination.
- `itransformer_residual + GradientMPC` is especially strong on `CO2air`.

Known summary from the latest predictor suite:

- `itransformer_residual + GradientMPC` reached `CO2air MAE = 5.950` in the recorded control suite comparison.

## 6. CO2 Mainline Status

There are now two active CO2 directions.

### 6.1 Multi-Target CO2 Specialist Branch

Status:

- `DLinear main path + iTransformer residual + dynamic gate` already existed.
- CO2-specialized variants were added and benchmarked.
- The current best multi-target CO2-specific variant is `itransformer_co2_late_residual`.

### 6.2 Standalone CO2 Forecasting Line

Motivation:

- Literature does not support the idea that simply swapping to a larger generic backbone will solve `CO2air`.
- Stronger directions are:
  - decomposition / denoising / multi-scale modeling
  - adaptive fusion
  - eventually carbon-balance gray-box modeling

Current standalone CO2 specialist ranking:

1. `co2_wavelet_gru_attn`
   - Full `R2=0.7519`, MAE `45.209`
   - Final `R2=0.6159`, MAE `58.292`
2. `co2_vmd_lstm_fusion`
   - Full `R2=0.6863`, MAE `52.298`
   - Final `R2=0.6003`, MAE `59.697`
3. `co2_env_lstm`
   - Full `R2=0.3065`, MAE `74.157`
   - Final `R2=-0.4852`, MAE `118.800`

Current interpretation:

- Pure environmental-factor `LSTM` is too weak as a final solution.
- `CO2air` needs an autoregressive anchor plus multi-scale modeling.
- The strongest standalone direction is currently `wavelet-inspired + GRU + adaptive attention`.

Reference documents:

- [CO2_PAPERS_AND_DIRECTION.md](c:/repositories/strawberry/agc_mpc/CO2_PAPERS_AND_DIRECTION.md)
- [CO2_SPECIALIST_REPORT.md](c:/repositories/strawberry/agc_mpc/CO2_SPECIALIST_REPORT.md)

## 7. Weekly Task Board

Maintenance rules:

- Keep the weekly task board permanently.
- Keep historical weeks with explicit date ranges.
- Always maintain `last week`, `this week`, and `next week`.
- This week's tasks have the highest priority.
- Every Wednesday, update the `next week` block explicitly.

### Historical Weekly Tasks

#### 2026-03-30 ~ 2026-04-05

- Completed the formal fair-budget `DLinear` benchmark.
- Completed the latest predictor suite control comparison.
- Consolidated the CO2 literature direction.

### Last Week: 2026-03-30 ~ 2026-04-05

- Formal fair-budget `DLinear` benchmark.
- Latest predictor suite control comparison.
- CO2 literature and direction consolidation.

### This Week: 2026-04-06 ~ 2026-04-12

- Complete the `iTransformer` hybrid line and formalize the benchmarkable implementation.
  - Status: largely done through residual variants and CO2-specialized variants.
- Land usable CO2 specialist branches.
  - Status: standalone CO2 line is implemented and benchmarked.
  - Remaining subtask: integrate the best standalone CO2 idea back into the multi-target mainline and test control impact.

### Next Week: 2026-04-13 ~ 2026-04-19

- Integrate `co2_wavelet_gru_attn` ideas into the multi-target CO2 residual line.
- Run fair-budget comparison against `itransformer_residual` and `itransformer_co2_late_residual`.
- Run closed-loop control comparison for the upgraded CO2-specialized predictor.
- Extend the bilingual document policy to additional maintained project markdown files if needed.

## 8. Current Priorities

Priority 1:

- strengthen `CO2air` in a control-relevant way
- prefer targeted CO2 branches over generic backbone swapping

Priority 2:

- preserve control-side validation
- keep `GradientMPC vs CEMMPC` comparisons
- verify whether offline forecasting gains transfer to closed-loop gains

Priority 3:

- move toward a more realistic economic / resource-aware greenhouse control setup
- eventually include:
  - `Heat_cons`
  - `ElecHigh`
  - `ElecLow`
  - `CO2_cons`
  - `Irr`

## 9. Working Rules

1. Update this document after meaningful code changes, benchmark updates, or direction changes.
2. Keep conclusions aligned with files under:
   - `results/forecasting/analysis/*.json`
   - `results/control/summaries/*.json`
3. Do not mix smoke-test conclusions with formal fair-budget conclusions.
4. Any new model should be evaluated against four questions:
   - Does offline forecasting improve?
   - Does closed-loop control improve?
   - Is the forecast error robust?
   - Can the architecture be explained as control-oriented design?
5. For CO2 work, prefer specialized modeling over blind generic backbone expansion.
6. If a maintained bilingual document is changed, update both the English canonical file and the Chinese mirror in the same turn.
## 10. 2026-04-07 CO2 Wavelet Integration Update

Two multi-target integration attempts were completed for the standalone `co2_wavelet_gru_attn` idea.

Results:

- `itransformer_co2_wavelet_residual`
  - `Tair`: Full `R2=0.9433`, MAE `0.636`
  - `Rhair`: Full `R2=0.8702`, MAE `4.409`
  - `CO2air`: Full `R2=0.5182`, MAE `65.984`
- `itransformer_co2_wavelet_blend`
  - `Tair`: Full `R2=0.9423`, MAE `0.641`
  - `Rhair`: Full `R2=0.8483`, MAE `4.725`
  - `CO2air`: Full `R2=0.5813`, MAE `64.666`

Interpretation:

- The standalone wavelet CO2 expert is strong by itself, but it did not transfer cleanly into end-to-end multi-target training.
- Both the direct residual-integration route and the direct blend-expert route degraded `CO2air` relative to `itransformer_residual` and `itransformer_co2_late_residual`.
- The current evidence suggests that the standalone CO2 specialist should probably be integrated through a more decoupled mechanism such as freezing, distillation, or offline teacher guidance rather than naive end-to-end joint training.