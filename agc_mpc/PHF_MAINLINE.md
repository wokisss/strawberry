# Protected Horizon Fusion Mainline

English canonical version.
Mapped Chinese mirror: [PHF_MAINLINE.zh-CN.md](c:/repositories/strawberry/agc_mpc/PHF_MAINLINE.zh-CN.md)
Last synchronized: `2026-04-21`

## 1. Purpose

This document defines the current paper-facing mainline for the CO2 specialist fusion work.

The project should not present every recent model variant as an independent contribution. The story should converge around:

**Protected Horizon Fusion for control-oriented greenhouse multi-step forecasting**

Short name:

- `PHF`
- `PHF-iTransformer`
- `CO2-PHF` when the focus is the CO2 branch

The main paper story is:

1. `CO2air` is harder than `Tair` and `Rhair` because it mixes slow cycles, sharp dosing/ventilation disturbances, and control-dependent dynamics.
2. A standalone multi-scale CO2 specialist is useful, but direct end-to-end integration into a multi-target predictor is unstable.
3. A frozen specialist should be trusted selectively, not blindly.
4. Trust should depend on agreement with the main predictor and on forecast horizon.
5. Offline forecast strength is not enough for MPC; validation must include control-relevant metrics.

## 2. Main Method

The proposed method should be described as `Protected Horizon Fusion`.

The method has three components:

1. **Multi-target residual backbone**
   - predicts `Tair`, `Rhair`, and `CO2air`
   - current implementation family: `iTransformer residual`
   - stable baseline: `itransformer_co2_late_residual`

2. **Frozen CO2 specialist**
   - current expert: `co2_wavelet_gru_attn`
   - trained as a standalone `CO2air` specialist
   - encodes the literature idea that greenhouse CO2 needs multi-scale and horizon-aware modeling

3. **Protected horizon fusion gate**
   - applies expert correction only to the `CO2air` channel
   - reduces trust when the expert and main predictor disagree strongly
   - changes trust by forecast horizon
   - pulls terminal horizons back toward the more stable late-residual backbone

Core formula:

```text
main_co2   = multi_target_backbone(x_past, w_future, u_future)[CO2air]
expert_co2 = frozen_co2_specialist(x_past, w_future, u_future)
delta      = expert_co2 - main_co2

agreement      = exp(-abs(delta) / temperature)
horizon_trust  = horizon_ratio ^ late_power
terminal_back  = terminal_pullback(horizon_ratio)
gate           = learned_gate(context) * agreement * horizon_trust

final_co2 = main_co2 + gate * (1 - terminal_back) * delta
```

The strongest current offline implementation of this idea is:

- `itransformer_co2_horizon_mixture`

## 3. Model Roles

Use these roles consistently in reports and paper drafts.

| Model | Paper role | How to describe it |
|---|---|---|
| `itransformer_residual` | residual backbone baseline | generic multi-target residual predictor |
| `itransformer_co2_late_residual` | strong CO2-aware backbone | late-horizon CO2 adapter without external expert |
| `co2_wavelet_gru_attn` | standalone CO2 expert | multi-scale GRU-attention CO2 specialist |
| `itransformer_co2_frozen_expert` | naive fusion baseline | directly blends frozen expert with main predictor |
| `itransformer_co2_late_frozen_expert` | late-trust control baseline | trusts the frozen expert more at later horizons; currently best closed-loop CO2 control |
| `itransformer_co2_teacher_distill` | distillation ablation | uses the expert only as an auxiliary teacher |
| `itransformer_co2_recoupled_expert` | cross-target recoupling baseline | adds target interaction after expert correction; currently best overall control objective |
| `itransformer_co2_protected_expert` | protection ablation | adds agreement-protected expert correction |
| `itransformer_co2_protected_terminal` | terminal-loss ablation | tests whether terminal loss alone is enough |
| `itransformer_co2_horizon_mixture` | proposed offline PHF model | protected correction plus terminal pullback; current offline CO2 leader |
| `itransformer_co2_frozen_backbone_horizon_mixture` | control-safety diagnostic | freezes the late-residual backbone and trains only the gate; preserves MPC gradients |

## 4. What Not To Claim

Do not claim:

- every CO2 variant is a separate contribution
- `horizon_mixture` is the control leader
- ordinary offline MAE/R2 alone is sufficient to select an MPC predictor
- the current model is CO2-only SOTA against specialist greenhouse CO2 papers

The correct claim is narrower and stronger:

- `PHF-iTransformer` is the current best offline multi-target CO2 specialist fusion model in this repository.
- Closed-loop experiments show that offline forecasting gains do not automatically transfer to MPC.
- Control-relevant validation is therefore a necessary part of predictor selection.

## 5. Required Ablation Logic

The ablation table should answer one question per row.

| Question | Model comparison |
|---|---|
| Does a CO2-aware backbone help? | `itransformer_residual` vs `itransformer_co2_late_residual` |
| Does a frozen standalone expert help? | `late_residual` vs `frozen_expert` / `late_frozen_expert` |
| Is horizon-dependent trust useful? | `frozen_expert` vs `late_frozen_expert` |
| Is agreement protection useful? | `late_frozen_expert` vs `protected_expert` |
| Is terminal loss alone enough? | `protected_expert` vs `protected_terminal` |
| Is explicit terminal pullback useful? | `protected_terminal` vs `horizon_mixture` |
| Does freezing the backbone improve MPC safety? | `horizon_mixture` vs `frozen_backbone_horizon_mixture` |

## 6. Current Evidence

Current offline forecasting leader:

- `itransformer_co2_horizon_mixture`
- `CO2air` Full MAE `43.910`
- `CO2air` Final MAE `47.661`

Current best CO2 closed-loop control baseline:

- `itransformer_co2_late_frozen_expert + GradientMPC`
- `CO2air` MAE `6.298`

Current best overall closed-loop objective baseline:

- `itransformer_co2_recoupled_expert + GradientMPC`
- objective `0.0651`

Current control-safe diagnostic:

- `itransformer_co2_frozen_backbone_horizon_mixture + GradientMPC`
- objective `0.0718`
- `CO2air` MAE `10.000`

Control-relevant validation result:

- `horizon_mixture` is strong at offline full/final forecasting but weak in first-step and first-6-step CO2 validation.
- `late_frozen_expert` is stronger in short-horizon CO2 behavior and closed-loop CO2 control.
- `late_residual` and frozen-backbone horizon mixture are strong control-safe compromises.

Validation v2 result:

- Generated [control_relevant_validation_reference.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/control_relevant_validation_reference.json), [control_relevant_validation_reference.csv](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/control_relevant_validation_reference.csv), [control_relevant_validation_reference.md](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/control_relevant_validation_reference.md), and [control_relevant_validation_reference.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/comparisons/control_relevant_validation_reference.png).
- Added signed CO2 bias, constraint-near proxy MAE, signed/flat gradient diagnostics, recorded-policy CO2 improvement, and action-activity diagnostics.
- Current control-relevant mean ranks:
  - `itransformer_co2_late_frozen_expert`: `2.250`
  - `itransformer_co2_late_residual`: `2.500`
  - `itransformer_residual`: `3.250`
  - `itransformer_co2_frozen_backbone_horizon_mixture`: `3.375`
  - `itransformer_co2_horizon_mixture`: `4.500`
  - `itransformer_co2_recoupled_expert`: `5.125`

PHF ablation result:

- Generated [phf_ablation_reference.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/phf_ablation_reference.json), [phf_ablation_reference.csv](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/phf_ablation_reference.csv), [phf_ablation_reference.md](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/phf_ablation_reference.md), and [phf_ablation_reference.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/comparisons/phf_ablation_reference.png).
- The ablation table supports the current role split:
  - `horizon_mixture`: offline PHF representative and CO2 forecasting leader
  - `late_frozen_expert`: strongest CO2 closed-loop control baseline
  - `recoupled_expert`: strongest overall closed-loop objective baseline
  - `frozen_backbone_horizon_mixture`: control-safety diagnostic

## 7. Current Week Tasks

The current week should prioritize:

1. Control-relevant validation suite
   - first-step MAE
   - first `6`-step control-horizon MAE
   - horizon-weighted MAE
   - control-input sensitivity
   - closed-loop metrics
   - status: v2 implemented with signed bias, constraint-near proxy, gradient sign/flatness, recorded-policy improvement, and PHF-linked outputs

2. PHF story convergence
   - keep `horizon_mixture` as the offline PHF representative
   - keep `late_frozen_expert` and `recoupled_expert` as control baselines
   - keep `frozen_backbone_horizon_mixture` as a diagnostic, not the main method
   - status: PHF ablation table and figure generated

Only after this story is stable should the project add one control-aware fusion model.
