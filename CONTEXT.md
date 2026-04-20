# CONTEXT.md

English canonical version.
Mapped Chinese mirror: [CONTEXT.zh-CN.md](c:/repositories/strawberry/CONTEXT.zh-CN.md)
Last synchronized: `2026-04-20`

## 0. Purpose And Maintenance Policy

This file is the long-lived project context for the `strawberry` workspace.

From `2026-04-07` onward, the documentation policy is:

- `*.md` is the English canonical version for long-lived project docs whenever practical.
- `*.zh-CN.md` is the synchronized Chinese mirror.
- When a maintained bilingual document changes, both versions must be updated in the same work turn.
- If any maintained document shows mojibake, encoding corruption, or suspicious characters, report it immediately before continuing.
- On Windows PowerShell, Chinese markdown can appear corrupted if read with the default `Get-Content` encoding. Before declaring a Chinese mirror damaged, re-read it explicitly with `Get-Content -Raw -Encoding UTF8 <path>` and distinguish terminal decoding issues from real file corruption.
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

#### 2026-04-06 ~ 2026-04-12

- Completed the `iTransformer` hybrid line through residual and CO2-specialized variants.
- Implemented and benchmarked standalone CO2 specialist models.
- Completed first multi-target wavelet CO2 integration attempts and recorded the failed-transfer conclusion.

#### 2026-04-13 ~ 2026-04-19

- Implemented and formally benchmarked the frozen, late-frozen, distillation, recoupled, protected, protected-terminal, horizon-mixture, and frozen-backbone horizon-mixture CO2 expert variants.
- Established `itransformer_co2_horizon_mixture` as the current offline `CO2air` forecasting leader.
- Ran the first `96-step` closed-loop control checks for the new CO2 variants.
- Added control-sensitivity diagnostics and trace-based pair comparison plots.
- Recorded the key conclusion that generic offline forecasting metrics do not automatically transfer to MPC control performance.

### Last Week: 2026-04-13 ~ 2026-04-19

- Implemented and formally benchmarked the latest CO2 expert fusion variants.
- Completed the `itransformer_co2_horizon_mixture` forecasting push.
- Diagnosed the poor MPC transfer of the offline forecasting leader.
- Implemented the `itransformer_co2_frozen_backbone_horizon_mixture` control-safe diagnostic variant.
- Generated trace-based control comparison figures against `late_frozen_expert` and `recoupled_expert`.

### This Week: 2026-04-20 ~ 2026-04-26

- Do not keep adding unrelated new predictors by default.
- Primary task candidate 1: build a standardized control-relevant validation suite.
- Primary task candidate 2: converge the model story around `Protected Horizon Fusion` / `PHF-iTransformer`.
- High-risk/high-reward task candidate: build a control-aware CO2 fusion model that combines `late_frozen_expert` short-horizon controllability with `horizon_mixture` offline terminal gains.
- Supporting task candidate: consolidate the PHF ablation table and figures from existing variants.
- Supporting task candidate: prepare a literature benchmark table across `Tair`, `Rhair`, and `CO2air`.
- Recommended weekly pair unless redirected by the user:
  - control-relevant validation
  - PHF mainline/story convergence

### Next Week: 2026-04-27 ~ 2026-05-03

- If this week finishes validation and story convergence, implement only one control-aware CO2 fusion candidate.
- If the user chooses performance over writing/story, prioritize the control-aware mixture and rerun formal forecasting + `96-step` control.
- If the user chooses paper preparation, prioritize PHF ablation, method diagram, and literature comparison.

## 8. Current Priorities

Priority 1:

- strengthen offline `CO2air` forecasting first
- resolve the split between full-horizon and final-step CO2 leaders
- prefer targeted CO2 branches over generic backbone swapping

Priority 2:

- return to control-side validation after the forecasting leader is stronger
- keep `GradientMPC vs CEMMPC` comparisons when control is rerun
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

## 11. 2026-04-14 Handoff Update: Forecasting-Only CO2 Push

The short-term project focus has changed:

- Do not prioritize closed-loop control yet.
- First make offline forecasting clearly stronger.
- Only after the predictor is consistently stronger should control be used as the next story step.

New multi-target CO2 variants implemented after the previous push:

- `itransformer_co2_frozen_expert`
- `itransformer_co2_late_frozen_expert`
- `itransformer_co2_teacher_distill`
- `itransformer_co2_recoupled_expert`
- `itransformer_co2_protected_expert`
- `itransformer_co2_protected_terminal`
- `itransformer_co2_horizon_mixture`
- `itransformer_co2_frozen_backbone_horizon_mixture`

Implementation notes:

- `training/trainer.py` now supports optional model-provided `compute_auxiliary_loss`.
- `config.py` now has `lambda_auxiliary`.
- Frozen-expert variants load the standalone `co2_wavelet_gru_attn` checkpoint and keep that expert frozen.

Latest fair-budget forecasting results:

- `itransformer_co2_frozen_expert`
  - `Tair`: Full `R2=0.9463`, MAE `0.601`
  - `Rhair`: Full `R2=0.7949`, MAE `5.471`
  - `CO2air`: Full `R2=0.7427`, MAE `46.966`
  - `CO2air`: Final `R2=0.6105`, MAE `59.247`
- `itransformer_co2_late_frozen_expert`
  - `Tair`: Full `R2=0.9460`, MAE `0.632`
  - `Rhair`: Full `R2=0.8908`, MAE `4.117`
  - `CO2air`: Full `R2=0.7757`, MAE `44.727`
  - `CO2air`: Final `R2=0.6292`, MAE `57.193`
- `itransformer_co2_teacher_distill`
  - `Tair`: Full `R2=0.9464`, MAE `0.611`
  - `Rhair`: Full `R2=0.8730`, MAE `4.179`
  - `CO2air`: Full `R2=0.6551`, MAE `56.018`
  - `CO2air`: Final `R2=0.6407`, MAE `57.294`
- `itransformer_co2_recoupled_expert`
  - `Tair`: Full `R2=0.9339`, MAE `0.687`
  - `Rhair`: Full `R2=0.8591`, MAE `4.522`
  - `CO2air`: Full `R2=0.7533`, MAE `47.585`
  - `CO2air`: Final `R2=0.6416`, MAE `58.054`
- `itransformer_co2_protected_expert`
  - `Tair`: Full `R2=0.9431`, MAE `0.660`
  - `Rhair`: Full `R2=0.8829`, MAE `4.197`
  - `CO2air`: Full `R2=0.7765`, MAE `45.190`
  - `CO2air`: Final `R2=0.6410`, MAE `55.984`
- `itransformer_co2_protected_terminal`
  - `Tair`: Full `R2=0.9489`, MAE `0.614`
  - `Rhair`: Full `R2=0.8620`, MAE `4.324`
  - `CO2air`: Full `R2=0.7404`, MAE `48.055`
  - `CO2air`: Final `R2=0.7069`, MAE `52.056`
- `itransformer_co2_horizon_mixture`
  - `Tair`: Full `R2=0.9508`, MAE `0.604`
  - `Rhair`: Full `R2=0.8958`, MAE `3.882`
  - `CO2air`: Full `R2=0.7868`, MAE `43.910`
  - `CO2air`: Final `R2=0.7468`, MAE `47.661`
- `itransformer_co2_frozen_backbone_horizon_mixture`
  - `Tair`: Full `R2=0.9503`, MAE `0.595`
  - `Rhair`: Full `R2=0.8849`, MAE `4.172`
  - `CO2air`: Full `R2=0.7727`, MAE `46.334`
  - `CO2air`: Final `R2=0.7312`, MAE `50.139`

Current forecasting frontier:

- Best `CO2air` Full MAE:
  - `itransformer_co2_horizon_mixture`: `43.910`
- Best `CO2air` Final MAE:
  - `itransformer_co2_horizon_mixture`: `47.661`
- Best practical CO2-focused compromise:
  - `itransformer_co2_horizon_mixture`: `Tair` Full MAE `0.604`, `Rhair` Full MAE `3.882`, `CO2air` Full MAE `43.910`, `CO2air` Final MAE `47.661`
- Best non-CO2 balance:
  - `itransformer_residual` remains strongest on `Rhair`
  - `itransformer_co2_late_residual` remains a strong broad multi-target balance

Important conclusion:

- `itransformer_co2_horizon_mixture` is the first current fair-budget model to unify the previous split between full-horizon and final-step CO2 leaders.
- It does not strictly dominate every non-CO2 metric; `itransformer_residual` is still stronger on `Rhair`.
- The forecasting bottleneck has moved from "can CO2 be improved?" to "can the new CO2 leader preserve or recover the last bit of humidity balance?"

Recommended next forecasting-only direction:

- Treat `itransformer_co2_horizon_mixture` as the new forecasting leader.
- Inspect horizon-wise error and forecast examples to confirm the terminal pullback behaves as intended.
- If the figures look stable, rerun closed-loop control only for `itransformer_co2_horizon_mixture` before spending more time on control-side tuning.
- If humidity balance becomes the limiting issue, tune the horizon gate or auxiliary loss without adding a heavier backbone.

## 12. 2026-04-14 Closed-Loop Check, Now Deprioritized

A `96-step` closed-loop control suite was run for context, but control is no longer the immediate priority.

`GradientMPC` results:

- `itransformer_residual`
  - objective `0.1924`
  - `Tair MAE=2.216`
  - `Rhair MAE=5.675`
  - `CO2air MAE=11.532`
- `itransformer_co2_late_residual`
  - objective `0.0705`
  - `Tair MAE=1.153`
  - `Rhair MAE=1.618`
  - `CO2air MAE=10.125`
- `itransformer_co2_late_frozen_expert`
  - objective `0.1533`
  - `Tair MAE=2.192`
  - `Rhair MAE=4.316`
  - `CO2air MAE=6.298`
- `itransformer_co2_recoupled_expert`
  - objective `0.0651`
  - `Tair MAE=0.826`
  - `Rhair MAE=2.692`
  - `CO2air MAE=16.749`
- `itransformer_co2_horizon_mixture`
  - objective `0.3713`
  - `Tair MAE=3.313`
  - `Rhair MAE=5.696`
  - `CO2air MAE=28.696`

Interpretation:

- `late_frozen_expert` converts CO2 forecasting strength into the best closed-loop `CO2air` control among the compared models.
- `late_residual` and `recoupled_expert` are better on overall objective.
- `horizon_mixture` is the new offline CO2 leader, but its first `96-step` control transfer is poor and should not be treated as the control leader.
- The immediate control-side question is why the terminal-pullback forecast improves offline metrics but destabilizes MPC rollout.
- A follow-up frozen-backbone mixture restores the `late_residual` first-step behavior and control gradients, but remains a control-safe compromise rather than a new control leader.

## 13. Current Weekly Task Update

Current week: `2026-04-13 ~ 2026-04-19`

This week's priority:

- Forecasting-only priority: make `CO2air` prediction clearly stronger before returning to control.
- Completed: implemented and formally benchmarked `itransformer_co2_horizon_mixture`.
- Completed: diagnosed the failed control transfer and implemented `itransformer_co2_frozen_backbone_horizon_mixture`.
- Current best full-horizon CO2 model: `itransformer_co2_horizon_mixture`.
- Current best final-step CO2 model: `itransformer_co2_horizon_mixture`.
- Current best CO2-focused compromise model: `itransformer_co2_horizon_mixture`.
- Completed first closed-loop check for `itransformer_co2_horizon_mixture`; offline gains did not transfer to MPC.
- Current control-safe mixture candidate: `itransformer_co2_frozen_backbone_horizon_mixture`.
- Immediate next subtask: build a control-aware mixture or validation metric that favors first-step and short-horizon sensitivity, not only full/final offline MAE.

Next week: `2026-04-20 ~ 2026-04-26`

- If forecasting frontier improves, rerun closed-loop control only for the new forecasting leader.
- If forecasting remains split between full-horizon and final-step leaders, analyze horizon-wise error and build a more explicit horizon-conditioned gate.
- Update CO2 specialist report with the successful and failed integration patterns.

## 14. Current Repository Change Status

As of `2026-04-20`, the recent code and result changes were pushed to `origin/main` in segmented commits:

- `f5aa3f6` - CO2 specialist fusion models and control diagnostic tools
- `ac98b66` - CO2 specialist forecasting result artifacts
- `86dc2e7` - CO2 control diagnostics, comparison figures, and trace JSON results

Remaining documentation maintenance is tracked in the current context/report updates.

Before switching branches, still check `git status` because documentation may have been updated after the latest result pushes.

## 15. 2026-04-14 Horizon Mixture Forecasting Result

Implemented `itransformer_co2_horizon_mixture`.

Design:

- base predictor: `itransformer_co2_late_residual`
- protected correction: frozen standalone `co2_wavelet_gru_attn` expert
- horizon behavior:
  - early/mid horizons keep protected expert correction
  - terminal horizons are pulled back toward the late-residual predictor
- training: fair budget with `lambda_auxiliary = 0.05`

Formal `joint_all + Reference` result:

- `Tair`: Full `R2=0.9508`, MAE `0.604`; Final `R2=0.9374`, MAE `0.689`
- `Rhair`: Full `R2=0.8958`, MAE `3.882`; Final `R2=0.8615`, MAE `4.568`
- `CO2air`: Full `R2=0.7868`, MAE `43.910`; Final `R2=0.7468`, MAE `47.661`

Interpretation:

- This is now the strongest offline CO2 model in the current fair-budget suite on both Full MAE and Final MAE.
- It improves the previous best `CO2air` Full MAE from `44.727` to `43.910`.
- It improves the previous best `CO2air` Final MAE from `50.139` to `47.661`.
- `Rhair` remains slightly behind the strongest `itransformer_residual` balance, so the result is a clear CO2-frontier improvement rather than a strict all-target domination.

Generated artifacts:

- summary: `results/forecasting/analysis/itransformer_co2_horizon_mixture_joint_all_reference_summary.json`
- checkpoint: `results/forecasting/checkpoints/itransformer_co2_horizon_mixture_joint_all_reference.pt`
- figures under `results/forecasting/figures/residual_variants/`
- updated comparison figure: `results/forecasting/figures/comparisons/itransformer_co2_branch_comparison_metrics.png`

Closed-loop transfer check:

- `96-step` `GradientMPC` with `itransformer_co2_horizon_mixture`:
  - objective `0.3713`
  - `Tair MAE=3.313`
  - `Rhair MAE=5.696`
  - `CO2air MAE=28.696`
- `CEMMPC` with `itransformer_co2_horizon_mixture`:
  - objective `0.4903`
  - `Tair MAE=4.426`
  - `Rhair MAE=7.355`
  - `CO2air MAE=31.294`

Control interpretation:

- Offline `CO2air` improvement did not transfer to the current MPC loop.
- Keep `itransformer_co2_horizon_mixture` as the offline forecasting leader only.
- Do not replace the current control-side leaders with it until the rollout mismatch is understood.
- Next likely diagnostic: compare action sensitivity and horizon-wise forecast gradients against `late_frozen_expert` and `late_residual`.

## 16. 2026-04-14 Control Sensitivity Diagnosis And Frozen-Backbone Mixture

A control-sensitivity diagnostic was added after the poor `horizon_mixture` closed-loop transfer.

Diagnostic file:

- [diagnose_control_sensitivity.py](c:/repositories/strawberry/agc_mpc/diagnose_control_sensitivity.py)

Comparison plotting file:

- [plot_control_pair_comparison.py](c:/repositories/strawberry/agc_mpc/plot_control_pair_comparison.py)
- [plot_control_pair_trace_comparison.py](c:/repositories/strawberry/agc_mpc/plot_control_pair_trace_comparison.py)
- Primary generated figure: `results/control/figures/comparison_itransformer_co2_horizon_mixture_vs_itransformer_co2_late_frozen_expert_gradient_mpc.png`
- Secondary generated figure for the overall-objective leader: `results/control/figures/comparison_itransformer_co2_horizon_mixture_vs_itransformer_co2_recoupled_expert_gradient_mpc.png`
- Trace JSONs are saved under `results/control/summaries/trace_comparison_*_gradient_mpc.json`.

Main diagnosis:

- The current simulator advances state with the first-step prediction.
- `itransformer_co2_horizon_mixture` improved full/final offline `CO2air` metrics, but worsened the control-aligned first-step `CO2air` error.
- This explains why the offline leader failed to transfer into MPC rollout.

Follow-up model:

- `itransformer_co2_frozen_backbone_horizon_mixture`

Design:

- freeze the proven `itransformer_co2_late_residual` main backbone
- freeze the standalone `co2_wavelet_gru_attn` expert
- train only the small horizon gate
- keep gradients through the frozen backbone and expert inputs for MPC

Important implementation detail:

- Do not wrap the frozen backbone or expert forward pass in `torch.no_grad()` during MPC-facing inference.
- Parameters stay frozen via `requires_grad_(False)`, but input gradients must remain available for `GradientMPC`.
- An earlier `no_grad()` version preserved prediction values but cut off control gradients and made `GradientMPC` nearly inactive.

Formal `joint_all + Reference` forecasting result:

- `Tair`: Full `R2=0.9503`, MAE `0.595`; Final `R2=0.9375`, MAE `0.674`
- `Rhair`: Full `R2=0.8849`, MAE `4.172`; Final `R2=0.8531`, MAE `4.774`
- `CO2air`: Full `R2=0.7727`, MAE `46.334`; Final `R2=0.7312`, MAE `50.139`

Control-aligned diagnostic after the gradient fix:

- first-step `CO2air MAE = 27.351`
- full-horizon `CO2air MAE = 36.356`
- final-step `CO2air MAE = 30.574`
- mean absolute control-cost gradient `0.01915`
- strongest cost-gradient controls: `t_vent_sp`, `co2_sp`, `assim_sp`

`96-step` closed-loop control result:

- `GradientMPC`
  - objective `0.0718`
  - `Tair MAE=1.158`
  - `Rhair MAE=1.615`
  - `CO2air MAE=10.000`
- `CEMMPC`
  - objective `0.1632`
  - `Tair MAE=2.631`
  - `Rhair MAE=4.351`
  - `CO2air MAE=25.263`

Interpretation:

- `itransformer_co2_frozen_backbone_horizon_mixture` is not the offline CO2 leader; `itransformer_co2_horizon_mixture` remains stronger offline.
- It is a more control-safe mixture because it preserves short-step behavior and usable control gradients.
- It roughly matches `itransformer_co2_late_residual + GradientMPC` and slightly improves its `CO2air MAE` from `10.125` to `10.000`.
- It still does not beat `itransformer_co2_late_frozen_expert + GradientMPC` on `CO2air`, which previously reached `6.298`.
- The next mainline should be control-aware CO2 fusion: preserve `late_frozen_expert` short-horizon CO2 controllability while keeping the offline terminal gains of the horizon-mixture family.

## 17. 2026-04-20 Story Convergence And Current Week Task Candidates

The latest discussion identified a narrative risk:

- The project has too many recent predictor variants to present all of them as independent main contributions.
- If written as a model-by-model chronology, the story will look like trial-and-error architecture stacking.
- The paper/mainline should now converge around one method family and use the other models as baselines, ablations, or diagnostics.

Recommended method framing:

- Main method name: `Protected Horizon Fusion` / `PHF-iTransformer`.
- Main technical chain:
  - `CO2-WGA` standalone expert
  - protected expert correction
  - horizon-dependent trust
  - terminal pullback
  - MPC-relevant validation
- Do not present every variant as a standalone contribution.

Recommended model roles:

- `itransformer_co2_horizon_mixture`: main offline forecasting method / PHF representative.
- `itransformer_co2_late_frozen_expert`: strongest current CO2 control baseline.
- `itransformer_co2_recoupled_expert`: strongest current overall objective baseline.
- `itransformer_co2_frozen_backbone_horizon_mixture`: control-safety diagnostic variant.
- `frozen_expert`, `teacher_distill`, `protected_terminal`: ablation or appendix material.

Current week task candidates, ranked by value:

1. Control-relevant validation suite
   - first-step MAE
   - first `6`-step control-horizon MAE
   - horizon-weighted MAE
   - control-input sensitivity
   - `GradientMPC` activity metrics
   - standard JSON/table/figure outputs
2. PHF story and method convergence
   - rename and frame the main method
   - write a clean method diagram
   - define which models are main method, baselines, ablations, and diagnostics
3. Control-aware CO2 fusion
   - combine `late_frozen_expert` short-horizon controllability with `horizon_mixture` terminal forecasting gains
   - keep input gradients available for `GradientMPC`
4. PHF ablation consolidation
   - organize existing variant results into one controlled table
   - avoid further architecture sprawl
5. Literature benchmark table
   - compare `Tair`, `Rhair`, and `CO2air`
   - distinguish pure forecasting papers from control-oriented validation

Recommended two-task pair unless the user chooses otherwise:

- control-relevant validation suite
- PHF story and method convergence

Reason:

- The bottleneck is no longer just model capacity.
- The current bottleneck is model-selection logic: the project must explain why an offline forecasting leader is not automatically the control leader, and then use that explanation to justify the next model.
