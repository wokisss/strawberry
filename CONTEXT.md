# CONTEXT.md

English canonical version.
Mapped Chinese mirror: [CONTEXT.zh-CN.md](c:/repositories/strawberry/CONTEXT.zh-CN.md)
Last synchronized: `2026-04-28`

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
- [PHF_MAINLINE.md](c:/repositories/strawberry/agc_mpc/PHF_MAINLINE.md) and [PHF_MAINLINE.zh-CN.md](c:/repositories/strawberry/agc_mpc/PHF_MAINLINE.zh-CN.md)
- [THESIS_LITERATURE_LIBRARY.md](c:/repositories/strawberry/agc_mpc/THESIS_LITERATURE_LIBRARY.md) and [THESIS_LITERATURE_LIBRARY.zh-CN.md](c:/repositories/strawberry/agc_mpc/THESIS_LITERATURE_LIBRARY.zh-CN.md)

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
- [PHF_MAINLINE.md](c:/repositories/strawberry/agc_mpc/PHF_MAINLINE.md)
- [THESIS_LITERATURE_LIBRARY.md](c:/repositories/strawberry/agc_mpc/THESIS_LITERATURE_LIBRARY.md)

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

#### 2026-04-20 ~ 2026-04-26

- Built the standardized `control_relevant_validation.py` suite and upgraded it to v2.
- Consolidated the paper-facing story around `Protected Horizon Fusion` / `PHF-iTransformer`.
- Generated the formal PHF ablation table and figure.
- Implemented, benchmarked, and promoted `itransformer_co2_control_aware_fusion` as the current balanced report model.
- Generated the triplet summary figure comparing `control-aware fusion`, `late_frozen_expert`, and `horizon_mixture`.

### Last Week: 2026-04-20 ~ 2026-04-26

- Established that ordinary offline forecasting metrics alone are not sufficient for MPC predictor selection.
- Implemented `control-aware fusion` to combine the short-horizon controllability of `late_frozen_expert` with most of the terminal forecasting gain of `horizon_mixture`.
- Confirmed `control-aware fusion` as the current best aggregate control-relevant validation model with mean rank `1.750`.
- Confirmed `GradientMPC 96-step` transfer for the promoted revision: objective `0.1491`, `CO2air MAE=6.415`.
- Updated PHF mainline docs, thesis literature library, PHF ablation outputs, and reporting figures.

### This Week: 2026-04-27 ~ 2026-05-03

- Current-week focus is method validation, not declaring another final model.
- The target method is now a multi-objective `Forecast-to-Control Transfer Validation` workflow rather than a CO2-only selector.
- CO2 remains the current stress-test target because it is the hardest and most visibly non-transferable variable, but the method must also quantify `Tair` and `Rhair` forecast-to-control transfer.
- `diffmpc_style_transformer` is excluded from the current strict pool because its 48-step history protocol is not aligned with the current 288-step AGC control-validation protocol.
- Task A: expand the strict validation model pool beyond the local PHF/fusion family.
- Task A target model groups:
  - compatible standard baselines: `DLinear`, then retrained three-target `GRU`, `LSTM`, `SegRNN`, `NLinear`, and pure `Transformer`
  - representative recent time-series baselines where practical: `PatchTST`, `iTransformer`, and at least one decomposition / frequency-style model such as `Autoformer`, `FEDformer`, or `TimesNet`
  - existing residual, CO2-aware, PHF / expert / fusion variants
- Task B: generalize the validation metrics from CO2-only to multi-target control-transfer metrics.
- Task B metric groups:
  - per-target first-step MAE for `Tair`, `Rhair`, and `CO2air`
  - per-target first `control_horizon=6` MAE
  - per-target short-horizon signed bias / absolute bias
  - constraint-near or setpoint-near MAE when the state is close to an operational boundary or reference band
  - control-sensitivity diagnostics: whether forecast outputs retain usable gradients with respect to relevant future control inputs
  - normalized multi-objective composite scores that align with the closed-loop tracking objective
- Task C: quantify whether these metrics predict actual closed-loop benefit.
- Task C analyses:
  - correlation between each forecast-side metric and closed-loop `Tair`, `Rhair`, and `CO2air` MAE
  - correlation between each forecast-side metric and closed-loop objective
  - rank correlation, top-k hit rate, and pairwise consistency
  - leave-one-model and leave-one-family robustness
  - separate per-target selection metrics from whole-objective selection metrics and diagnostic-only metrics
- Expected output: a cross-model method report showing which metric groups can quantify forecast-to-control transfer, where they fail, and how they should be used for multi-objective greenhouse MPC.

### Next Week: 2026-05-04 ~ 2026-05-10

- Convert the multi-objective transfer analysis into a paper-facing method section and figure set if the robustness checks hold.
- If metric transfer is variable-specific or family-dependent, explicitly report that limitation and define variable-specific metric roles instead of forcing one universal score.
- Only after the validation method is stable should model tuning continue; new architectures should be added only if they fill a missing baseline family in the method validation, not just to chase a leaderboard.

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
7. Do not avoid model runs by default. Forecasting training, closed-loop rollout, and FCTV recomputation are necessary experimental work, not optional polish. If a checkpoint, dataset, and runnable command are available, run the experiment instead of only adding tooling or documentation. Defer a run only when the environment blocks it, required artifacts are missing, or the computational cost is clearly too high for the current turn; in that case, state the blocker and the exact command to run next.

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

Work started for the selected pair:

- Added [control_relevant_validation.py](c:/repositories/strawberry/agc_mpc/control_relevant_validation.py).
- Generated:
  - `results/forecasting/analysis/control_relevant_validation_reference.json`
  - `results/forecasting/analysis/control_relevant_validation_reference.csv`
  - `results/forecasting/analysis/control_relevant_validation_reference.md`
  - `results/forecasting/figures/comparisons/control_relevant_validation_reference.png`
- Added [PHF_MAINLINE.md](c:/repositories/strawberry/agc_mpc/PHF_MAINLINE.md) and [PHF_MAINLINE.zh-CN.md](c:/repositories/strawberry/agc_mpc/PHF_MAINLINE.zh-CN.md).
- Added [THESIS_LITERATURE_LIBRARY.md](c:/repositories/strawberry/agc_mpc/THESIS_LITERATURE_LIBRARY.md) and [THESIS_LITERATURE_LIBRARY.zh-CN.md](c:/repositories/strawberry/agc_mpc/THESIS_LITERATURE_LIBRARY.zh-CN.md) as the broad paper-facing literature library. It now consolidates the former control-relevant MPC note with the content from [RECENT_PAPERS_SURVEY.md](c:/repositories/strawberry/agc_mpc/RECENT_PAPERS_SURVEY.md) and [LITERATURE_COMPARISON.md](c:/repositories/strawberry/agc_mpc/LITERATURE_COMPARISON.md), covering greenhouse forecasting, greenhouse control, CO2-specific modeling, general time-series architectures, AGC-vs-literature positioning, prediction-control correlation, and citation-ready thesis paragraphs.
- Upgraded [control_relevant_validation.py](c:/repositories/strawberry/agc_mpc/control_relevant_validation.py) to v2 with signed CO2 bias, constraint-near proxy MAE, signed/flat gradient diagnostics, recorded-policy CO2 improvement, and action-activity diagnostics.
- Added [summarize_phf_ablation.py](c:/repositories/strawberry/agc_mpc/summarize_phf_ablation.py) and generated the PHF ablation JSON/CSV/Markdown/figure outputs.

Initial validation conclusion:

- `itransformer_co2_late_residual`, `itransformer_co2_late_frozen_expert`, and `itransformer_co2_frozen_backbone_horizon_mixture` rank best on the initial control-relevant validation aggregate.
- `itransformer_co2_horizon_mixture` remains the offline full/final CO2 forecasting leader, but ranks poorly on first-step, first-6-step, and closed-loop CO2 validation.
- This supports the current PHF story: `horizon_mixture` is the offline PHF representative, while MPC selection needs separate control-relevant validation.

## 18. 2026-04-21 Control-Relevant Validation v2 And PHF Ablation

New generated validation outputs:

- `results/forecasting/analysis/control_relevant_validation_reference.json`
- `results/forecasting/analysis/control_relevant_validation_reference.csv`
- `results/forecasting/analysis/control_relevant_validation_reference.md`
- `results/forecasting/figures/comparisons/control_relevant_validation_reference.png`

Validation v2 adds:

- signed CO2 bias
- constraint-near proxy MAE
- signed and flat gradient diagnostics
- recorded-policy CO2 improvement
- action-activity diagnostics

Current control-relevant mean rank:

1. `itransformer_co2_late_frozen_expert`: `2.250`
2. `itransformer_co2_late_residual`: `2.500`
3. `itransformer_residual`: `3.250`
4. `itransformer_co2_frozen_backbone_horizon_mixture`: `3.375`
5. `itransformer_co2_horizon_mixture`: `4.500`
6. `itransformer_co2_recoupled_expert`: `5.125`

New PHF ablation outputs:

- `results/forecasting/analysis/phf_ablation_reference.json`
- `results/forecasting/analysis/phf_ablation_reference.csv`
- `results/forecasting/analysis/phf_ablation_reference.md`
- `results/forecasting/figures/comparisons/phf_ablation_reference.png`

PHF ablation conclusion:

- `itransformer_co2_horizon_mixture` remains the offline PHF representative and CO2 forecasting leader.
- `itransformer_co2_late_frozen_expert` remains the strongest CO2 closed-loop control baseline.
- `itransformer_co2_recoupled_expert` remains the strongest overall closed-loop objective baseline.
- `itransformer_co2_frozen_backbone_horizon_mixture` remains a control-safety diagnostic, not the main offline method.

Next recommended technical step:

- Add only one control-aware fusion candidate after this validation/story layer is committed.
- It should preserve the short-horizon controllability of `late_frozen_expert` while trying to recover the terminal offline gains of `horizon_mixture`.

## 19. 2026-04-21 Control-Aware Fusion Candidate

Implemented `itransformer_co2_control_aware_fusion`.

Design:

- freeze `itransformer_co2_late_frozen_expert` as the short-horizon anchor
- freeze `itransformer_co2_horizon_mixture` as the terminal-gain reference
- train only a CO2 fusion gate that stays near the late-frozen anchor in the first `6` control steps and opens mainly in the later half of the horizon
- current promoted revision smooths the imported terminal delta after the control horizon instead of increasing tail trust directly
- add auxiliary protection for:
  - first-step `CO2air`
  - first `6`-step `CO2air`
  - `co2_sp` first-step gradient matching against the late-frozen anchor

Formal `joint_all + Reference` forecasting result:

- `Tair`: Full `R2=0.9460`, MAE `0.632`; Final `R2=0.9326`, MAE `0.713`
- `Rhair`: Full `R2=0.8908`, MAE `4.117`; Final `R2=0.8580`, MAE `4.762`
- `CO2air`: Full `R2=0.7858`, MAE `43.983`; Final `R2=0.7393`, MAE `49.069`

Control-relevant validation result:

- new best mean rank: `1.750`
- first-step `CO2air MAE = 24.468`
- first `6`-step `CO2air MAE = 26.742`
- final-step `CO2air MAE = 26.601`
- constraint-near proxy `CO2air MAE = 29.392`
- first-step `co2_sp` gradient magnitude `0.3040`

Closed-loop `96-step` result:

- `GradientMPC`
  - objective `0.1491`
  - `Tair MAE=2.202`
  - `Rhair MAE=4.267`
  - `CO2air MAE=6.415`
- `CEMMPC`
  - objective `0.2475`
  - `CO2air MAE=16.045`

Interpretation:

- This candidate preserves the short-horizon control behavior of `late_frozen_expert` almost exactly on the validation suite.
- It recovers most of the offline CO2 gains of `horizon_mixture`:
  - compared with `late_frozen_expert`, Full `CO2air MAE` improves from `44.727` to `43.983`
  - compared with `late_frozen_expert`, Final `CO2air MAE` improves from `57.193` to `49.069`
- The promoted delta-smoothing revision improves closed-loop transfer relative to the previous control-aware fusion checkpoint:
  - `GradientMPC CO2air` improves from `6.521` to `6.415`
  - objective improves from `0.1504` to `0.1491`
- It still does not beat `late_frozen_expert` on closed-loop CO2 (`6.415` vs `6.298`), but it remains close and still ranks first in the current control-relevant validation aggregate.
- It is worth keeping as the main control-aware follow-up, not deleting.

Next step after this candidate:

- do not add another new architecture family
- only tune the existing fusion gate conservatism / late-start schedule / auxiliary weight
- target: keep the current first-step and first `6`-step behavior while trying to close the remaining gap to `late_frozen_expert` on `GradientMPC CO2air`

Additional tuning note:

- A more conservative tail-trust pilot was tested and archived under:
  - `results/forecasting/analysis/itransformer_co2_control_aware_fusion_conservative_tune_holdout_reference_summary.json`
  - `results/control/summaries/itransformer_co2_control_aware_fusion_conservative_tune_holdout_gradient_mpc_summary.json`
- That pilot improved offline CO2 to Full `43.817` / Final `46.784`, but did not improve control transfer enough to justify replacing the current main candidate.
- Current conclusion: pushing terminal trust further upward is not the best next move; the next tuning should instead focus on keeping the current tail gains while shaving the remaining closed-loop gap.

- A gate-shape pilot with extra monotonic/smoothness regularization was also tested and archived under:
  - `results/forecasting/analysis/itransformer_co2_control_aware_fusion_gate_shape_tune_holdout_reference_summary.json`
  - `results/control/summaries/itransformer_co2_control_aware_fusion_gate_shape_tune_holdout_gradient_mpc_summary.json`
- That pilot improved offline CO2 to Full `43.779` / Final `46.916`, but worsened `GradientMPC CO2air` to `6.885`.
- Current conclusion: simply smoothing or monotonizing the late gate is also not the right next move.

- A delta-smoothing selector revision was then tested and promoted into the current main candidate.
- Its key behavior is to smooth the imported `late_frozen_expert -> horizon_mixture` terminal delta after the control horizon, instead of only changing gate timing.
- Current conclusion: selecting a smoother terminal delta is more promising than further changing the gate schedule alone.

## 20. 2026-04-27 Current Week Direction: Cross-Model Forecast-To-Control Validation

The user corrected the scope on `2026-04-27`:

- The research question is not "prove one CO2 model is the final report model."
- The research question is to build a quantifiable methodology for converting forecast-side validation into multi-objective control-benefit evidence.
- CO2 is still the current emphasis because it exposes the forecast-to-control mismatch most clearly, but `Tair` and `Rhair` must be included in the method.
- `diffmpc_style_transformer` should be ignored for now because its protocol is not aligned with the current strict AGC control-validation setup.

Updated rationale:

- Last week established that offline CO2 forecasting gains do not automatically transfer to MPC.
- The next step is to test whether this observation generalizes into a reusable multi-objective validation method.
- The method should quantify which forecast-side metrics predict closed-loop `Tair`, `Rhair`, `CO2air`, and whole-objective gains, and which metrics are only offline diagnostics.

Updated primary tasks:

1. Define a multi-objective FCTV metric set.
   - Per-target first-step MAE: `Tair`, `Rhair`, `CO2air`.
   - Per-target first `control_horizon=6` MAE.
   - Per-target short-horizon bias / absolute bias.
   - Per-target constraint-near or setpoint-near MAE.
   - Per-target and whole-objective weighted forecast ranks.
   - Gradient / controllability diagnostics for relevant control channels.

2. Validate metric-to-control transfer against closed-loop outcomes.
   - Match forecast metrics against `GradientMPC` closed-loop `Tair`, `Rhair`, and `CO2air` MAE.
   - Match forecast metrics against closed-loop objective and action-activity diagnostics.
   - Use Pearson / Spearman correlation, top-k hit rate, pairwise consistency, leave-one-model robustness, and leave-one-family robustness.
   - Report metric roles separately for each target rather than forcing one score to explain every control outcome.

3. Expand model breadth with strict comparability.
   - Keep the current 11 compatible models as the initial pool.
   - Add retrained three-target standard baselines: `GRU`, `LSTM`, `SegRNN`, `NLinear`, and pure `Transformer`.
   - Add representative recent time-series families where feasible: `PatchTST`, `iTransformer`, and at least one of `Autoformer`, `FEDformer`, or `TimesNet`.
   - Treat PHF / expert / fusion variants as depth and ablation coverage, not as the only evidence base.

4. Formalize the methodology.
   - The paper-facing object should be a metric group and validation protocol, not only a model ranking.
   - CO2-specific conclusions are allowed as a case study, but the method section must show how the same logic applies to temperature and humidity.
   - A final model can be mentioned only as an application of the method, not as the method itself.

Expected current-week deliverables:

- Multi-objective FCTV JSON / CSV / Markdown outputs.
- A compact figure showing per-target metric-to-control correlations and robustness.
- A baseline coverage table separating strict comparable models from protocol-mismatched or appendix-only models.
- A short method narrative explaining why first-step / control-horizon / bias / constraint-near / gradient metrics are candidates, and which of them are empirically validated.

Initial implementation and results:

- Added [analyze_forecast_to_control_transfer.py](c:/repositories/strawberry/agc_mpc/analyze_forecast_to_control_transfer.py).
- Expanded the default `control_relevant_validation.py` pool from the PHF-local set to `11` compatible models:
  - `dlinear_forecaster`
  - `current_hybrid_transformer`
  - `transformer_hybrid_residual`
  - `itransformer_residual`
  - `patchtst_residual`
  - `itransformer_co2_late_residual`
  - `itransformer_co2_late_frozen_expert`
  - `itransformer_co2_recoupled_expert`
  - `itransformer_co2_horizon_mixture`
  - `itransformer_co2_frozen_backbone_horizon_mixture`
  - `itransformer_co2_control_aware_fusion`
- Added `dlinear_forecaster` as the compatible three-target DLinear baseline and ran its `96-step` closed-loop control suite:
  - `GradientMPC` objective `0.3962`
  - `GradientMPC CO2air MAE = 37.824`
  - `CEMMPC CO2air MAE = 26.864`
- Excluded older `dlinear_baseline`, `transformer_baseline`, `gru_baseline`, and `segrnn_baseline` checkpoints from the fine-grained validation run because their saved heads are four-target checkpoints and do not load under the current three-target control protocol.
- Excluded `diffmpc_style_transformer` from the pooled validation for now because it uses a 48-step history protocol rather than the current 288-step control-validation protocol.
- Added baseline coverage notes:
  - `results/forecasting/analysis/forecast_to_control_baseline_coverage.md`
- Regenerated:
  - `results/forecasting/analysis/control_relevant_validation_reference.json`
  - `results/forecasting/analysis/control_relevant_validation_reference.csv`
  - `results/forecasting/analysis/control_relevant_validation_reference.md`
  - `results/forecasting/figures/comparisons/control_relevant_validation_reference.png`
- Generated new forecast-to-control transfer outputs:
  - `results/forecasting/analysis/forecast_to_control_transfer_reference.json`
  - `results/forecasting/analysis/forecast_to_control_transfer_reference.csv`
  - `results/forecasting/analysis/forecast_to_control_transfer_reference.md`
  - `results/forecasting/figures/comparisons/forecast_to_control_transfer_reference.png`
- Added robustness outputs:
  - `results/forecasting/analysis/forecast_to_control_transfer_robustness_reference.csv`
  - `results/forecasting/figures/comparisons/forecast_to_control_transfer_robustness_reference.png`
- Added a report-facing summary figure:
  - `results/forecasting/figures/comparisons/forecast_to_control_transfer_summary_reference.png`

Initial CO2-focused transfer conclusions on the `11`-model compatible pool:

- For closed-loop `CO2air MAE`, `co2_first_step_mae` is the strongest current selection metric:
  - Pearson `0.572`
  - Spearman `0.752`
  - pairwise consistency `0.815`
  - top-3 closed-loop winner hit: yes, with top-3 overlap `1.000`
- `co2_control_horizon_mae` is the next strongest CO2-control selection metric:
  - Spearman `0.588`
  - pairwise consistency `0.722`
- `co2_constraint_near_mae_proxy` and `co2_control_horizon_abs_bias` are useful secondary selection metrics, but weaker than first-step / first-6-step MAE.
- `co2_final_step_mae` does not predict closed-loop `CO2air MAE` in this pool:
  - Spearman `0.009`
  - pairwise consistency `0.509`
- Selection metrics for CO2 tracking do not explain overall `mpc_objective` well. This supports keeping `CO2air` tracking selection separate from whole-objective controller quality.
- `control-aware fusion` remains the best forecast-only transfer-ranked model and the best aggregate control-relevant validation model, while `late_frozen_expert` remains the best raw closed-loop `CO2air MAE` model.

Robustness update:

- Added leave-one-model and leave-one-family robustness to the transfer analysis.
- Added `co2_transfer_selection_score` as a weighted composite score using only validated control-transfer metrics:
  - `co2_first_step_mae`: weight `3.0`
  - `co2_control_horizon_mae`: weight `2.0`
  - `co2_constraint_near_mae_proxy`: weight `1.5`
  - `co2_control_horizon_abs_bias`: weight `1.5`
- Current metric roles for closed-loop `CO2air MAE`:
  - `co2_first_step_mae`: `primary_selection`
  - `co2_control_horizon_mae`: `secondary_selection`
  - `co2_constraint_near_mae_proxy`: `secondary_selection`
  - `co2_control_horizon_abs_bias`: `secondary_selection`
  - `forecast_only_transfer_rank`: `secondary_selection`
  - `co2_transfer_selection_score`: `secondary_selection`
  - `co2_weighted_horizon_mae`: `weak_selection`
  - `co2_full_horizon_mae`: `offline_or_diagnostic_only`
  - `co2_final_step_mae`: `offline_or_diagnostic_only`
  - gradient metrics: `diagnostic_only`
- `co2_first_step_mae` is the only current primary selection metric:
  - full Spearman vs closed-loop `CO2air MAE`: `0.752`
  - leave-one-model Spearman range: `0.669 .. 0.839`
  - leave-one-family Spearman range: `0.661 .. 0.839`
  - leave-one-model pairwise minimum: `0.773`
- This strengthens the method claim: first-step CO2 accuracy is not just a local observation inside PHF models; in the current compatible cross-model pool it is the most stable predictor of closed-loop CO2 tracking.
- `co2_transfer_selection_score` is useful as a report-facing composite score, but should not be described as stronger than `co2_first_step_mae`:
  - full Spearman vs closed-loop `CO2air MAE`: `0.582`
  - leave-one-model Spearman range: `0.455 .. 0.770`
  - leave-one-model pairwise minimum: `0.667`
  - top ranked models: `control-aware fusion`, then `late_frozen_expert`
- Current wording should be: first-step CO2 MAE is the strongest current CO2 primary selection signal, while the weighted CO2 score is a secondary composite for ranking and reporting.
- This is not yet a complete multi-objective methodology because `Tair` and `Rhair` transfer roles still need to be computed and stress-tested.

Execution checklist completed in this round:

- Extended the transfer analyzer into a concrete score-and-robustness tool.
- Regenerated the transfer JSON / CSV / Markdown outputs.
- Regenerated the transfer correlation figure.
- Regenerated the leave-one-model robustness figure.
- Added the compact summary figure for reporting.
- Verified that the new scripts parse successfully with AST compilation.

Immediate next technical step:

- Generalize [control_relevant_validation.py](c:/repositories/strawberry/agc_mpc/control_relevant_validation.py) and [analyze_forecast_to_control_transfer.py](c:/repositories/strawberry/agc_mpc/analyze_forecast_to_control_transfer.py) from CO2-only selection metrics to multi-objective FCTV metrics.
- Rerun the current 11-model pool first, then add retrained strict baselines.
- Update the report language so it presents `control-aware fusion` only as one model selected by the current CO2-weighted composite, not as the central contribution of the methodology.

## 21. 2026-04-27 FCTV Paper Story And Scope

The current paper-facing methodological direction is feasible if framed as a screening and diagnosis protocol, not as a theoretical guarantee.

Recommended method name:

- `Forecast-to-Control Transfer Validation (FCTV)`

Core claim:

- Standard offline forecasting metrics such as full-horizon MAE, final-step MAE, RMSE, and R2 are not sufficient for selecting MPC predictors.
- In receding-horizon MPC, forecast errors near the executed control horizon, systematic short-horizon bias, constraint-near errors, and control-input sensitivity can be more predictive of closed-loop benefit than average long-horizon accuracy.
- FCTV provides a low-cost intermediate validation layer between pure offline forecasting evaluation and expensive closed-loop MPC rollout.

What the method is:

- A multi-objective forecast-side metric group for `Tair`, `Rhair`, and `CO2air`.
- A transfer-analysis protocol that relates forecast metrics to closed-loop `GradientMPC` outcomes.
- A model-screening and failure-diagnosis tool.

What the method is not:

- It is not a stability proof.
- It is not a replacement for final closed-loop MPC validation.
- It is not a universal one-number score that must work identically for temperature, humidity, CO2, and every controller.
- It is not a claim that one current PHF / fusion model is the final contribution.

Candidate FCTV metric groups:

- first-step MAE per target
- first `control_horizon=6` MAE per target
- short-horizon signed bias and absolute bias per target
- constraint-near or setpoint-near MAE per target
- gradient / control-sensitivity diagnostics with respect to relevant future control inputs
- target-specific and whole-objective composite ranks

Validation evidence required for a short paper:

- strict comparable model breadth: at least DLinear / NLinear, GRU / LSTM / SegRNN, pure Transformer, PatchTST / iTransformer, residual variants, and PHF / fusion variants
- multi-target analysis for `Tair`, `Rhair`, `CO2air`, and closed-loop objective
- Pearson / Spearman correlation
- pairwise consistency
- top-k winner hit rate
- leave-one-model and leave-one-family robustness
- explicit separation between per-target selection metrics, whole-objective selection metrics, and diagnostic-only metrics

Current strongest partial evidence:

- On the current 11-model compatible pool, CO2 first-step MAE is the strongest observed CO2 selection signal for closed-loop CO2 tracking.
- `co2_final_step_mae` is not predictive of closed-loop CO2 tracking in the current pool.
- This supports the larger story that terminal offline forecasting gains do not necessarily transfer to receding-horizon MPC benefit.
- The result is still incomplete because `Tair` and `Rhair` metric roles must be computed and stress-tested.

Paper positioning:

- Do not claim that no previous work links forecasting and control.
- Existing control-oriented identification, decision-focused learning, and MPC forecast-value studies already recognize that prediction quality affects control.
- The defensible gap is narrower: there is limited systematic work on offline, multi-objective, forecast-side validation metrics that can screen and diagnose deep forecasting predictors before greenhouse MPC rollout.
- The contribution should be stated as a practical bridge between forecasting evaluation and MPC validation for multi-objective greenhouse climate control.

## 21. 2026-04-27 FCTV Paper Positioning

The current method direction is feasible as a small paper only if it is framed correctly.

Do not claim:

- FCTV is a theoretical guarantee of closed-loop optimality.
- A single forecast metric universally determines MPC performance.
- The main contribution is that one specific PHF / fusion model is best.

Claim instead:

- FCTV is a low-cost screening and diagnosis layer between offline forecasting validation and expensive closed-loop MPC evaluation.
- It quantifies which forecast errors are control-relevant for a given MPC setup.
- It helps identify why an offline forecasting improvement does or does not transfer into closed-loop benefit.
- It provides per-target metric roles for `Tair`, `Rhair`, and `CO2air`, rather than forcing one universal score.

Paper-facing story:

1. Greenhouse MPC depends on multi-step predictors, but ordinary forecasting metrics such as full-horizon MAE, RMSE, and final-step MAE are not sufficient for selecting MPC predictors.
2. The mismatch is empirical and practical: a model can improve terminal or full-horizon forecast accuracy while worsening the short-horizon behavior that the receding-horizon controller actually uses.
3. Existing control-oriented identification and decision-focused learning literature recognizes this issue, but there is a missing practical validation layer for modern data-driven greenhouse forecasting models.
4. FCTV fills this layer by measuring first-step accuracy, control-horizon accuracy, short-horizon bias, constraint-near / setpoint-near error, and control-input sensitivity.
5. These metrics are then validated against closed-loop `GradientMPC` outcomes with correlation, rank correlation, top-k hit rate, pairwise consistency, and leave-one-model / leave-one-family robustness.

Candidate FCTV metric groups:

- First-step MAE: directly relevant because receding-horizon MPC applies only the first optimized control move.
- Control-horizon MAE: relevant because near-term predictions shape the imminent control sequence.
- Short-horizon signed / absolute bias: relevant because systematic overprediction or underprediction causes persistent overcompensation or undercompensation.
- Constraint-near / setpoint-near MAE: relevant because the same absolute error has higher control cost near reference bands, constraints, or operational boundaries.
- Gradient / control-sensitivity diagnostics: relevant because gradient-based MPC needs the predictor to respond sensibly to future control inputs.
- Multi-objective composite ranks: useful only after per-target metrics are normalized and validated against closed-loop `Tair`, `Rhair`, `CO2air`, and whole-objective outcomes.

Validation requirements for a credible small paper:

- Strict comparable model breadth: at least standard linear, RNN, Transformer-style, PatchTST / iTransformer-style, residual, and PHF / fusion families.
- Multi-objective scope: report `Tair`, `Rhair`, `CO2air`, and overall objective separately.
- Closed-loop linkage: every proposed selection metric must be compared against `GradientMPC 96-step` outcomes.
- Robustness: include Pearson / Spearman correlation, top-k hit, pairwise consistency, leave-one-model, and leave-one-family checks.
- Limitation statement: FCTV does not replace final closed-loop validation and is not a stability proof.

Current evidence status:

- CO2 already has an initial positive result: `co2_first_step_mae` is the strongest current CO2 selection signal, while `co2_final_step_mae` is not predictive of closed-loop CO2 MAE.
- This is a useful case study but not yet the full method.
- The next evidence gap is computing equivalent roles for `Tair` and `Rhair`, then checking whether a multi-objective composite can explain whole-objective MPC performance.

Recommended thesis / paper wording:

- "Forecast accuracy is necessary for MPC, but not every forecast-accuracy improvement has equal control value."
- "The value of a prediction error depends on when it occurs, which state it affects, whether it is biased, whether it happens near constraints, and whether the model preserves actionable control sensitivity."
- "FCTV is designed to quantify this control relevance before committing to expensive closed-loop MPC trials."

## 22. 2026-04-27 Multi-Objective FCTV Implementation Update

The immediate next technical step from Section 20 has now been completed for the current `11`-model compatible pool.

Implementation update:

- [control_relevant_validation.py](c:/repositories/strawberry/agc_mpc/control_relevant_validation.py) now exports unified target-prefixed forecast metrics for `Tair`, `Rhair`, and `CO2air`.
- [analyze_forecast_to_control_transfer.py](c:/repositories/strawberry/agc_mpc/analyze_forecast_to_control_transfer.py) now computes multi-objective FCTV transfer analysis instead of CO2-only analysis.
- The analyzer reports per-target selection roles against:
  - `mpc_tair_mae`
  - `mpc_rhair_mae`
  - `mpc_co2_mae`
  - `mpc_objective`
- The analyzer now writes target-specific scores:
  - `tair_transfer_selection_score`
  - `rhair_transfer_selection_score`
  - `co2_transfer_selection_score`
  - `multiobjective_transfer_selection_score`

Regenerated outputs:

- `results/forecasting/analysis/control_relevant_validation_reference.{json,csv,md}`
- `results/forecasting/figures/comparisons/control_relevant_validation_reference.png`
- `results/forecasting/analysis/forecast_to_control_transfer_reference.{json,csv,md}`
- `results/forecasting/analysis/forecast_to_control_transfer_robustness_reference.csv`
- `results/forecasting/figures/comparisons/forecast_to_control_transfer_reference.png`
- `results/forecasting/figures/comparisons/forecast_to_control_transfer_robustness_reference.png`
- `results/forecasting/figures/comparisons/forecast_to_control_transfer_summary_reference.png`

Current multi-objective evidence:

- `CO2air`: `co2_first_step_mae` remains the strongest validated selection metric for closed-loop `CO2air MAE`.
  - Spearman `0.752`
  - pairwise consistency `0.815`
  - leave-one-model Spearman range `0.669 .. 0.839`
  - role: `primary_selection`
- `Rhair`: `rhair_first_step_mae` is a useful but weaker selection signal for closed-loop `Rhair MAE`.
  - Spearman `0.627`
  - pairwise consistency `0.727`
  - leave-one-model Spearman range `0.539 .. 0.733`
  - role: `secondary_selection`
- `Tair`: `tair_first_step_mae` is not currently a reliable selector for closed-loop `Tair MAE`.
  - Spearman `-0.236`
  - pairwise consistency `0.400`
  - role: `offline_or_diagnostic_only`
- Whole objective: the current `multiobjective_transfer_selection_score` does not explain `mpc_objective` well in this pool.
  - Spearman `0.136`
  - pairwise consistency `0.564`
  - role: `offline_or_diagnostic_only`

Interpretation:

- FCTV should be presented as a per-target screening and diagnosis protocol, not as one universal score.
- The current strong positive case is still CO2 because it has the clearest receding-horizon transfer signal.
- Humidity has usable secondary transfer evidence.
- Temperature currently exposes a limitation: ordinary target-matched first-step error is not enough to select the best closed-loop Tair controller in this model pool.
- This limitation is useful for the method story because it supports variable-specific metric roles instead of a forced all-in-one score.

Immediate next technical step:

- Expand strict comparable model breadth with three-target retrained `GRU`, `LSTM`, `SegRNN`, `NLinear`, pure `Transformer`, and where practical `iTransformer` / `PatchTST` / decomposition-style baselines.
- Add or improve Tair/Rhair-specific control-sensitivity diagnostics rather than relying only on CO2 gradient diagnostics.
- Re-check whether objective-level screening improves after the baseline pool becomes less PHF-heavy.

## 23. 2026-04-28 Standard Baseline Expansion And FCTV Recheck

The next FCTV baseline-completion step has now been partially executed.

Implementation update:

- [compare_training_regimes.py](c:/repositories/strawberry/agc_mpc/compare_training_regimes.py) now supports:
  - `--control-protocol` for strict three-target `Tair` / `Rhair` / `CO2air` training
  - `--fair-budget` for the formal budget: `batch_size=256`, `num_epochs=200`, `learning_rate=1e-4`, `lambda_trend=0.3`, `early_stop_patience=15`
- [control_main.py](c:/repositories/strawberry/agc_mpc/control_main.py) now exposes strict control predictors:
  - `gru_forecaster`
  - `lstm_forecaster`
  - `nlinear_forecaster`
  - `segrnn_forecaster`
  - `transformer_forecaster`
- [control_relevant_validation.py](c:/repositories/strawberry/agc_mpc/control_relevant_validation.py) now includes those three standard baselines in the default FCTV pool.
- [control/controller.py](c:/repositories/strawberry/agc_mpc/control/controller.py) disables CuDNN RNN kernels during gradient-based MPC optimization so recurrent predictors can be differentiated in eval-mode control rollout.
- FCTV gradient diagnostics are no longer CO2-only; they now include first-step and mean forecast gradients for `Tair`, `Rhair`, and `CO2air`, plus target-relevant control channels.

New strict baseline training results:

- `gru_forecaster`
  - Full MAE: `Tair=0.866`, `Rhair=4.753`, `CO2air=48.396`
  - Final MAE: `Tair=0.986`, `Rhair=6.281`, `CO2air=54.721`
- `segrnn_forecaster`
  - Full MAE: `Tair=0.960`, `Rhair=5.109`, `CO2air=69.209`
  - Final MAE: `Tair=1.186`, `Rhair=6.406`, `CO2air=84.046`
- `lstm_forecaster`
  - Full MAE: `Tair=0.874`, `Rhair=4.832`, `CO2air=69.352`
  - Final MAE: `Tair=1.105`, `Rhair=6.483`, `CO2air=81.987`
- `nlinear_forecaster`
  - Full MAE: `Tair=0.727`, `Rhair=4.236`, `CO2air=61.003`
  - Final MAE: `Tair=0.774`, `Rhair=4.710`, `CO2air=63.283`
- `transformer_forecaster`
  - Full MAE: `Tair=0.597`, `Rhair=4.256`, `CO2air=42.789`
  - Final MAE: `Tair=0.691`, `Rhair=5.175`, `CO2air=48.983`

New 96-step closed-loop `GradientMPC` results:

- `gru_forecaster`: objective `0.1108`, `Tair MAE=0.409`, `Rhair MAE=4.957`, `CO2air MAE=49.973`
- `segrnn_forecaster`: objective `0.0486`, `Tair MAE=0.391`, `Rhair MAE=2.195`, `CO2air MAE=14.425`
- `lstm_forecaster`: objective `0.1780`, `Tair MAE=1.491`, `Rhair MAE=4.497`, `CO2air MAE=23.014`
- `nlinear_forecaster`: objective `0.1526`, `Tair MAE=1.867`, `Rhair MAE=4.182`, `CO2air MAE=25.236`
- `transformer_forecaster`: objective `0.0861`, `Tair MAE=1.039`, `Rhair MAE=4.072`, `CO2air MAE=16.455`

Updated FCTV pool:

- The default strict pool is now `16` models instead of `11`.
- It now covers DLinear, NLinear, GRU, LSTM, SegRNN, pure Transformer, Transformer-hybrid, PatchTST-style residual, iTransformer-style residual, CO2-aware residuals, and PHF / control-aware fusion variants.

Updated transfer conclusions after adding standard baselines:

- `CO2air`: `co2_first_step_mae` remains the strongest CO2 screening signal, but its role is now more conservative:
  - Spearman `0.593`
  - pairwise consistency `0.723`
  - role: `secondary_selection`
- `Rhair`: `rhair_first_step_mae` is now the strongest validated per-target signal in the expanded pool:
  - Spearman `0.653`
  - pairwise consistency `0.758`
  - role: `primary_selection`
- `Tair`: `tair_first_step_mae` remains unreliable for closed-loop Tair selection:
  - Spearman `-0.335`
  - pairwise consistency `0.383`
  - role: `offline_or_diagnostic_only`
- Whole objective: the current `multiobjective_transfer_selection_score` still does not explain `mpc_objective`:
  - Spearman `0.153`
  - pairwise consistency `0.567`
  - role: `offline_or_diagnostic_only`

Interpretation update:

- Adding non-PHF standard baselines weakened the earlier CO2 primary-selection claim. This is a useful correction, not a failure.
- The defensible claim is now: first-step CO2 error is the best current CO2 screening metric, but it should be treated as a secondary selection signal until the pool includes more families and more closed-loop runs.
- The standard baselines show why FCTV is needed: `segrnn_forecaster` has weak offline CO2 forecasting but much better closed-loop CO2 tracking than its offline final-step CO2 MAE would suggest.
- This supports the paper story that forecast quality must be evaluated through control-relevant timing, bias, sensitivity, and target-specific transfer roles rather than through ordinary offline forecasting rank alone.

Remaining baseline gaps:

- Add at least one decomposition / frequency-style baseline if feasible.
- Use family-level ablations to separate framework effects from module effects.

## 24. 2026-04-28 Frequency Baseline And Attribution Report

The remaining baseline and attribution tasks have been completed for the current work turn.

Implementation update:

- Added [frequency_forecaster.py](c:/repositories/strawberry/agc_mpc/models/frequency_forecaster.py), a lightweight frequency-style conditional baseline.
  - It encodes low-frequency FFT modes from the historical state sequence.
  - It fuses that frequency context with future weather and future requested controls.
  - It is an in-repository frequency-style baseline, not a formal reproduction of Autoformer / FEDformer / TimesNet.
- Added `frequency_baseline` / `frequency_forecaster` to:
  - [compare_training_regimes.py](c:/repositories/strawberry/agc_mpc/compare_training_regimes.py)
  - [control_main.py](c:/repositories/strawberry/agc_mpc/control_main.py)
  - [control_relevant_validation.py](c:/repositories/strawberry/agc_mpc/control_relevant_validation.py)
  - [analyze_forecast_to_control_transfer.py](c:/repositories/strawberry/agc_mpc/analyze_forecast_to_control_transfer.py)
- Generated attribution notes:
  - `results/forecasting/analysis/forecast_to_control_attribution_reference.md`

Frequency baseline results:

- Offline:
  - Full MAE: `Tair=1.253`, `Rhair=4.624`, `CO2air=90.101`
  - Final MAE: `Tair=1.383`, `Rhair=5.284`, `CO2air=91.544`
- 96-step `GradientMPC`:
  - objective `0.4338`
  - `Tair MAE=1.725`
  - `Rhair MAE=8.759`
  - `CO2air MAE=15.530`

Updated FCTV pool:

- The default strict pool is now `17` models.
- It covers DLinear, NLinear, frequency-style MLP, GRU, LSTM, SegRNN, pure Transformer, Transformer-hybrid, PatchTST-style residual, iTransformer-style residual, CO2-aware residuals, and PHF / control-aware fusion variants.

Updated FCTV metric roles in the `17`-model pool:

- `rhair_first_step_mae -> mpc_rhair_mae`
  - role: `primary_selection`
  - Spearman `0.711`
  - pairwise consistency `0.787`
- `co2_first_step_mae -> mpc_co2_mae`
  - role: `secondary_selection`
  - Spearman `0.516`
  - pairwise consistency `0.681`
- `co2_constraint_near_mae_proxy -> mpc_co2_mae`
  - role: `secondary_selection`
  - Spearman `0.522`
  - pairwise consistency `0.676`
- `tair_first_step_mae -> mpc_tair_mae`
  - role: `offline_or_diagnostic_only`
  - Spearman `-0.270`
  - pairwise consistency `0.412`
- `multiobjective_transfer_selection_score -> mpc_objective`
  - role: `weak_selection`
  - Spearman `0.267`
  - pairwise consistency `0.618`

Attribution conclusion:

- Current evidence supports metric-mediated attribution, not a simple "framework X is better" claim.
- Framework effects are visible: for example, `segrnn_forecaster` and `frequency_forecaster` have weak offline CO2 forecasts but much better closed-loop CO2 tracking than their final-step CO2 MAE would suggest.
- Module effects inside the PHF / iTransformer family are horizon-specific: late expert, horizon mixture, frozen-backbone mixture, and control-aware fusion change different FCTV metrics in different directions.
- The defensible wording is:
  - model structures and modules affect control through specific forecast-side behaviors;
  - FCTV identifies which of those behaviors are control-relevant for each target;
  - final closed-loop MPC validation is still required for whole-objective claims.

Remaining optional future work:

- If the paper requires exact external baselines, add a formal Autoformer / FEDformer / TimesNet implementation.
- Run repeated closed-loop rollouts across multiple start indices to strengthen causal robustness.
- Add controlled module swaps across more than one backbone if making stronger module-causality claims.

## 25. 2026-04-28 FCTV Follow-Up Checklist And Metric-Origin Rationale

The next step should not be blindly adding more models. The priority is to complete the logic chain explaining why forecast-side metrics can explain control-side benefit.

P0: write the method-level FCTV story.

- State FCTV as a screening / diagnosis protocol between forecasting evaluation and closed-loop MPC validation, not as a new predictor model.
- Convert `results/forecasting/analysis/forecast_to_control_transfer_reference.md` into a method section covering candidate metric origins, validation procedure, and role classification.
- State the current `17`-model conclusion clearly: `Rhair first-step MAE` is the strongest signal, `CO2 first-step / constraint-near` metrics are supporting screening signals, `Tair` is not currently explained by target-matched forecast error, and whole-objective claims still require closed-loop validation.

P1: add robustness experiments.

- Repeat 96-step closed-loop rollouts across multiple start indices to verify that the current FCTV relationship is not an artifact of one segment.
- Recompute `mpc_tair_mae`, `mpc_rhair_mae`, `mpc_co2_mae`, and `mpc_objective` for each start index.
- Recompute Spearman, pairwise consistency, top-k hit, leave-one-model robustness, and leave-one-family robustness.
- If the relationships remain stable, FCTV can be presented as a reusable validation method rather than only a current-pool observation.

P1: add attribution experiments.

- Fix the backbone and swap modules: for example iTransformer residual, CO2 late adapter, frozen expert, horizon mixture, and control-aware fusion.
- Fix the module idea and change the backbone: if feasible, port a CO2 late / fusion-style module to more than one backbone.
- The goal is to separate framework effects, module effects, and metric-mediated effects where a module improves an FCTV metric that then transfers to control.

P2: add external baselines if needed.

- If the paper requires stronger external comparison, add formal Autoformer / FEDformer / TimesNet implementations.
- This is not the immediate bottleneck because the strict pool already contains `17` models; robustness and attribution are more urgent.

P2: add presentation assets.

- Add one logic-chain figure: model / module -> forecast-side behavior -> FCTV metric -> closed-loop target.
- Add one metric-role table: selection metric, secondary metric, diagnostic-only metric.
- Add one counterexample figure: better final-step CO2 MAE does not necessarily imply better control, showing why ordinary forecast rank is insufficient.

Metric-origin rationale:

- Ordinary metrics such as MAE, RMSE, and R2 come from the forecasting / regression tradition and mainly answer whether predictions fit observations overall.
- R2 is a regression goodness-of-fit metric, defined as `R2 = 1 - SSE / SST`, and measures how much target variance is explained by the model.
- These ordinary metrics do not directly answer whether a predictor improves MPC control, because receding-horizon MPC only executes the front part of each optimized plan.
- FCTV metrics come from MPC execution mechanics, control-target structure, and optimizer sensitivity requirements.
- `first-step MAE` and `control_horizon MAE` come from receding-horizon execution, because the controller depends most directly on short-horizon predictions that are actually acted on.
- `bias` comes from control-direction risk, because systematic overprediction or underprediction can push MPC toward wrong actions.
- `constraint-near MAE` comes from constraint / setpoint risk, because errors near operational limits are more likely to change the optimal control decision.
- `gradient diagnostics` come from GradientMPC requirements and test whether the predictor has reasonable sensitivity to future control inputs.
- The paper wording should be: derive candidate metrics from MPC mechanics first, then validate which metrics actually predict closed-loop outcomes across models, rather than inventing metrics after seeing results.

## 26. 2026-04-28 FCTV Method Report, Multi-Start Tooling, And Presentation Assets

The Section 25 follow-up checklist has been advanced from an open task list into concrete method/report assets and repeatable experiment tooling.

Completed P0 method work:

- Added `results/forecasting/analysis/forecast_to_control_transfer_method_reference.md`.
- The method report states FCTV as a screening / diagnosis protocol between offline forecasting and closed-loop MPC validation.
- It explains candidate metric origins from receding-horizon execution, short-horizon bias risk, constraint-near risk, and GradientMPC sensitivity requirements.
- It records the current 17-model conclusion:
  - `rhair_first_step_mae` is the strongest target-specific signal for `mpc_rhair_mae`.
  - `co2_first_step_mae` and `co2_constraint_near_mae_proxy` are supporting CO2 screening metrics.
  - `tair_first_step_mae` is not a reliable selector for `mpc_tair_mae`.
  - `multiobjective_transfer_selection_score` remains only weakly useful for whole-objective screening.

Completed P1 robustness tooling:

- Added `run_fctv_multistart_control.py` to repeat 96-step `GradientMPC` rollouts across multiple closed-loop start indices.
- Added `analyze_fctv_multistart_transfer.py` to recompute FCTV transfer statistics after replacing the closed-loop target metrics with each start-index rollout result.
- `AGCConfig` now has optional `control_output_tag`.
- `AGCClosedLoopSimulator` now includes `start_idx` in summaries and uses `control_output_tag` to avoid overwriting repeated-rollout figures and summaries.
- `control_main.py` suite summaries now record `start_idx` and `output_tag`.

Execution note:

- The full multi-start robustness benchmark has not been run in this work turn because it requires many expensive 96-step GradientMPC rollouts across the 17-model pool.
- The reproducible command path is now available:
  - `python agc_mpc/run_fctv_multistart_control.py --start-indices 0 96 192 --steps 96`
  - `python agc_mpc/analyze_fctv_multistart_transfer.py --suite-json <generated_suite_json>`

Completed P2 presentation assets:

- Added `plot_fctv_presentation_assets.py`.
- Generated:
  - `results/forecasting/figures/comparisons/fctv_presentation_reference_logic_chain.png`
  - `results/forecasting/figures/comparisons/fctv_presentation_reference_metric_roles.png`
  - `results/forecasting/figures/comparisons/fctv_presentation_reference_co2_counterexample.png`

Attribution status:

- The current supported claim remains metric-mediated attribution, not a blanket framework-causality claim.
- Existing PHF / iTransformer variants provide same-family module evidence, while standard baselines provide framework contrast.
- Stronger causal attribution still requires the multi-start rollouts and controlled module swaps across more than one backbone.

Validation performed:

- Generated the new presentation assets from the current transfer JSON.
- Validated syntax of the changed Python files with AST parsing because the environment blocks `__pycache__` bytecode replacement.

## 27. 2026-04-28 Remaining Runnable Models And Experiment-Run Policy

Current runnable gaps after the 17-model FCTV pool:

- Already supported by `control_main.py` and already checkpointed, but not in the current strict 17-model FCTV pool:
  - `itransformer_co2_residual`
  - `itransformer_co2_frozen_expert`
  - `itransformer_co2_teacher_distill`
  - `itransformer_co2_protected_expert`
  - `itransformer_co2_protected_terminal`
  - `itransformer_co2_wavelet_residual`
  - `itransformer_co2_wavelet_blend`
- Also runnable but lower priority for the strict current pool:
  - `dlinear_baseline`
  - `transformer_hybrid_baseline`
  - `transformer_baseline`
- Excluded unless protocol is aligned:
  - `diffmpc_style_transformer`, because its historical protocol does not match the current strict 288-step AGC control-validation protocol.
- Not yet implemented as formal external baselines:
  - Autoformer / FEDformer / TimesNet.

Immediate runnable experiment priority:

1. Run 96-step `GradientMPC` closed-loop checks for the checkpointed CO2 / PHF variants missing from the current FCTV pool.
2. Recompute `control_relevant_validation.py` and `analyze_forecast_to_control_transfer.py` with an extended pool if those runs complete.
3. Run multi-start robustness for the most important predictors once the single-start extended pool is complete.

Policy clarification:

- For future experimental turns, do not treat model execution as something to avoid by default.
- If the user asks to advance an experiment and the required model/checkpoint/script exists, run it.
- If a full run is expensive, choose a defensible subset first, report what was run, and leave the exact remaining command.

## 28. 2026-04-28 Extended 24-Model FCTV Run

The missing checkpointed CO2 / PHF variants from Section 27 were actually run instead of being left as planned work.

New 96-step `GradientMPC` closed-loop results:

- `itransformer_co2_residual`: objective `0.0557`, `Tair MAE=0.936`, `Rhair MAE=1.503`, `CO2air MAE=6.421`
- `itransformer_co2_frozen_expert`: objective `0.0649`, `Tair MAE=0.917`, `Rhair MAE=2.263`, `CO2air MAE=20.140`
- `itransformer_co2_teacher_distill`: objective `0.3502`, `Tair MAE=2.789`, `Rhair MAE=6.877`, `CO2air MAE=27.338`
- `itransformer_co2_protected_expert`: objective `0.0606`, `Tair MAE=0.880`, `Rhair MAE=1.441`, `CO2air MAE=14.206`
- `itransformer_co2_protected_terminal`: objective `0.3837`, `Tair MAE=3.380`, `Rhair MAE=6.179`, `CO2air MAE=27.089`
- `itransformer_co2_wavelet_residual`: objective `0.0639`, `Tair MAE=1.075`, `Rhair MAE=2.142`, `CO2air MAE=7.776`
- `itransformer_co2_wavelet_blend`: objective `0.0771`, `Tair MAE=1.023`, `Rhair MAE=1.928`, `CO2air MAE=8.020`

Generated / updated outputs:

- `results/control/summaries/predictor_suite_missing_co2_phf_reference_96steps.json`
- `results/forecasting/analysis/control_relevant_validation_reference.{json,csv,md}`
- `results/forecasting/analysis/forecast_to_control_transfer_reference.{json,csv,md}`
- `results/forecasting/analysis/forecast_to_control_transfer_robustness_reference.csv`
- FCTV comparison, robustness, summary, and presentation figures under `results/forecasting/figures/comparisons/`

Updated 24-model FCTV conclusions:

- The pool now contains `24` models.
- `rhair_first_step_mae -> mpc_rhair_mae` remains the strongest target-specific signal, but its role is now `secondary_selection`:
  - Spearman `0.592`
  - pairwise consistency `0.732`
  - leave-one-model Spearman minimum `0.537`
- `co2_first_step_mae -> mpc_co2_mae` is no longer a stable selector in the extended pool:
  - role: `offline_or_diagnostic_only`
  - Spearman `0.168`
  - pairwise consistency `0.549`
- `co2_constraint_near_mae_proxy -> mpc_co2_mae` is also no longer stable:
  - role: `offline_or_diagnostic_only`
  - Spearman `0.015`
  - pairwise consistency `0.507`
- `tair_first_step_mae -> mpc_tair_mae` remains unreliable:
  - Spearman `-0.123`
  - pairwise consistency `0.464`
- `multiobjective_transfer_selection_score -> mpc_objective` remains unsuitable as a whole-objective selector:
  - Spearman `0.167`
  - pairwise consistency `0.564`
- `rhair_first_step_mae -> mpc_objective` is the strongest current whole-objective supporting signal:
  - role: `objective_secondary_selection`
  - Spearman `0.507`
  - pairwise consistency `0.703`

Interpretation update:

- The earlier 17-model CO2 screening claim was pool-dependent; adding the missing CO2 / PHF variants weakened CO2 first-step and constraint-near transfer.
- This is useful evidence for the paper story: FCTV should report metric roles under explicit model-pool scope and should not overclaim universal transfer.
- The best new closed-loop CO2 result among the added models is `itransformer_co2_residual` with `CO2air MAE=6.421`, close to `control-aware fusion` (`6.415`) and `late_frozen_expert` (`6.298`), but with a much better whole objective than those two.
- `itransformer_co2_protected_expert` has the best objective among the newly added variants (`0.0606`) and strong `Rhair MAE=1.441`, making it important for the control-side discussion.
- Immediate next experiment after this extended single-start pool is multi-start robustness, not another architecture addition.

## 29. 2026-04-28 Initial Multi-Start FCTV Robustness Run

A representative multi-start robustness run has been completed instead of leaving robustness as a future-only plan.

Executed command scope:

- `10` predictors
- start indices: `0`, `96`, `192`
- rollout length: `96` steps
- controller: `GradientMPC`

Predictor subset:

- `current_hybrid_transformer`
- `transformer_hybrid_residual`
- `segrnn_forecaster`
- `frequency_forecaster`
- `itransformer_co2_residual`
- `itransformer_co2_protected_expert`
- `itransformer_co2_late_residual`
- `itransformer_co2_late_frozen_expert`
- `itransformer_co2_control_aware_fusion`
- `itransformer_co2_horizon_mixture`

Generated outputs:

- `results/control/summaries/fctv_multistart_gradient_mpc_reference_96steps_starts_0_96_192.json`
- `results/forecasting/analysis/forecast_to_control_transfer_multistart_reference.{json,csv,md}`
- per-start transfer reports:
  - `forecast_to_control_transfer_multistart_reference_start00000.*`
  - `forecast_to_control_transfer_multistart_reference_start00096.*`
  - `forecast_to_control_transfer_multistart_reference_start00192.*`
- `results/forecasting/figures/comparisons/forecast_to_control_transfer_multistart_reference.png`

Important execution note:

- The long-running command exceeded the tool timeout after completing and saving the suite JSON, so the shell returned timeout status `124`.
- The output file was present and complete, and the analyzer successfully processed start indices `[0, 96, 192]`.

Multi-start metric conclusion:

- `co2_first_step_mae -> mpc_co2_mae` is not stable across starts:
  - start `0`: `secondary_selection`, Spearman `0.498`, pairwise `0.705`
  - start `96`: `offline_or_diagnostic_only`, Spearman `-0.146`, pairwise `0.409`
  - start `192`: `offline_or_diagnostic_only`, Spearman `-0.243`, pairwise `0.432`
- `rhair_first_step_mae -> mpc_rhair_mae` is also not stable:
  - start `0`: `secondary_selection`, Spearman `0.418`, pairwise `0.667`
  - start `96`: `offline_or_diagnostic_only`, Spearman `-0.103`, pairwise `0.444`
  - start `192`: `offline_or_diagnostic_only`, Spearman `0.091`, pairwise `0.578`
- `multiobjective_transfer_selection_score -> mpc_objective` remains weak or diagnostic only:
  - start `0`: `weak_selection`, Spearman `0.285`, pairwise `0.600`
  - start `96`: `offline_or_diagnostic_only`, Spearman `0.188`, pairwise `0.556`
  - start `192`: `weak_selection`, Spearman `0.285`, pairwise `0.600`
- `tair_first_step_mae -> mpc_tair_mae` remains unreliable.

Multi-start model-side finding:

- `itransformer_co2_residual` is the most stable CO2 closed-loop tracker in the tested subset:
  - start `0`: `CO2air MAE=6.331`, objective `0.0558`
  - start `96`: `CO2air MAE=11.074`, objective `0.0654`
  - start `192`: `CO2air MAE=10.701`, objective `0.0465`
- Best whole-objective model by start:
  - start `0`: `current_hybrid_transformer`, objective `0.0442`
  - start `96`: `current_hybrid_transformer`, objective `0.0517`
  - start `192`: `transformer_hybrid_residual`, objective `0.0235`

Interpretation update:

- The multi-start result strengthens the limitation claim: FCTV metric roles are not only model-pool dependent, but also rollout-segment dependent.
- FCTV should be presented as a diagnostic protocol with explicit scope, not as a universal offline selector.
- The strongest near-term model conclusion is not that one FCTV metric universally selects winners, but that `itransformer_co2_residual` deserves renewed attention as a robust CO2 closed-loop tracker.
- Next experiment priority is either full 24-model multi-start robustness or a smaller repeated-start suite centered on `itransformer_co2_residual`, `current_hybrid_transformer`, `transformer_hybrid_residual`, and the main PHF/fusion variants.

## 30. 2026-04-28 Expanded 16-Model Multi-Start FCTV Robustness Run

The initial 10-model multi-start subset was expanded to `16` predictors by adding:

- `itransformer_residual`
- `patchtst_residual`
- `transformer_forecaster`
- `nlinear_forecaster`
- `dlinear_forecaster`
- `itransformer_co2_wavelet_residual`

Execution notes:

- The second long-running command also exceeded the tool timeout after finishing and saving its suite JSON, returning status `124`.
- The saved suite was complete and was merged with the earlier 10-model suite.

Generated outputs:

- `results/control/summaries/fctv_multistart_gradient_mpc_reference_96steps_6predictors_8e102971d9_starts_0_96_192.json`
- `results/control/summaries/fctv_multistart_gradient_mpc_reference_96steps_16predictors_starts_0_96_192.json`
- `results/forecasting/analysis/forecast_to_control_transfer_multistart16_reference.{json,csv,md}`
- per-start `forecast_to_control_transfer_multistart16_reference_start*.{json,csv,md}` and robustness CSVs
- `results/forecasting/figures/comparisons/forecast_to_control_transfer_multistart16_reference.png`
- `results/forecasting/analysis/fctv_multistart_model_rankings_reference.{csv,md}`
- `results/forecasting/figures/comparisons/fctv_multistart_model_rankings_reference.png`

16-model multi-start metric conclusion:

- `co2_first_step_mae -> mpc_co2_mae` remains segment-dependent:
  - start `0`: `secondary_selection`, Spearman `0.366`, pairwise `0.630`
  - start `96`: `offline_or_diagnostic_only`, Spearman `-0.263`, pairwise `0.395`
  - start `192`: `offline_or_diagnostic_only`, Spearman `-0.243`, pairwise `0.412`
- `rhair_first_step_mae -> mpc_rhair_mae` weakens further:
  - start `0`: `weak_selection`, Spearman `0.282`, pairwise `0.617`
  - start `96`: `offline_or_diagnostic_only`, Spearman `-0.068`, pairwise `0.458`
  - start `192`: `offline_or_diagnostic_only`, Spearman `0.174`, pairwise `0.583`
- `multiobjective_transfer_selection_score -> mpc_objective` is not stable:
  - start `0`: `weak_selection`, Spearman `0.338`, pairwise `0.617`
  - start `96`: `offline_or_diagnostic_only`, Spearman `-0.074`, pairwise `0.458`
  - start `192`: `offline_or_diagnostic_only`, Spearman `0.144`, pairwise `0.567`
- `tair_first_step_mae -> mpc_tair_mae` remains unreliable.

16-model multi-start model-side conclusion:

- `itransformer_co2_residual` remains the most stable CO2 closed-loop tracker:
  - start `0`: best CO2, `CO2air MAE=6.331`, objective `0.0558`
  - start `96`: best CO2, `CO2air MAE=11.074`, objective `0.0654`
  - start `192`: best CO2, `CO2air MAE=10.701`, objective `0.0465`
- Best whole-objective model by start:
  - start `0`: `current_hybrid_transformer`, objective `0.0442`
  - start `96`: `current_hybrid_transformer`, objective `0.0517`
  - start `192`: `transformer_hybrid_residual`, objective `0.0235`
- Additional important segment-specific findings:
  - start `192`: `dlinear_forecaster` reached `CO2air MAE=11.316`, objective `0.0449`
  - start `192`: `itransformer_residual` reached `CO2air MAE=11.644`, objective `0.0360`

Interpretation update:

- The 16-model multi-start result confirms that no current FCTV forecast-side metric is a stable universal selector.
- FCTV remains useful as a diagnostic protocol for identifying mismatch and segment dependence.
- The model story should now emphasize robust closed-loop evidence:
  - `current_hybrid_transformer` remains the strongest objective-oriented baseline across starts `0` and `96`.
  - `transformer_hybrid_residual` is strongest objective-wise at start `192`.
  - `itransformer_co2_residual` is consistently the strongest CO2 tracker in the expanded multi-start subset.
- The multi-start model ranking figure now directly compares objective and CO2 MAE across starts.

## 31. 2026-04-29 FCTV Weekly Reporting Figure

Generated a supervisor-facing weekly summary figure for the FCTV result chain.

New script and output:

- `agc_mpc/plot_fctv_weekly_metric_degradation.py`
- `results/forecasting/figures/comparisons/fctv_weekly_metric_degradation_summary.png`

Figure message:

- The early `17`-model CO2-focused FCTV stage showed useful screening signals.
- After expanding to the `24`-model pool, the CO2 first-step and constraint-near metrics degraded to diagnostic-only roles.
- After expanding to `16` models across starts `0`, `96`, and `192`, the main forecast-side metrics became model-pool and segment dependent.
- The figure should be used to communicate the current report conclusion: offline forecast metrics cannot reliably screen closed-loop control benefit; FCTV is useful as a diagnostic framework, and closed-loop MPC validation remains necessary.
