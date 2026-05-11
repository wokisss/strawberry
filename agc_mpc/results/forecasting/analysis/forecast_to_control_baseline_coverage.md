# Multi-Objective Forecast-To-Control Baseline Coverage

This file records which representative forecasting baselines are currently included in the fine-grained forecast-to-control transfer validation.

Scope correction on 2026-04-27:

- The methodology target is multi-objective transfer validation for `Tair`, `Rhair`, and `CO2air`.
- CO2 remains the current stress-test variable, but it should not be treated as the only research target.
- `diffmpc_style_transformer` is intentionally excluded from the current strict pool because its 48-step-history protocol is not aligned with the current 288-step AGC control-validation protocol.

## Included In Fine-Grained Validation

| Category | Model | Status | Notes |
| --- | --- | --- | --- |
| Linear forecasting baseline | `dlinear_forecaster` | included | Three-target fair-budget checkpoint; newly added to control and transfer validation. |
| Recurrent baseline | `gru_forecaster` | included | Three-target fair-budget `gru_baseline_joint_all_reference.pt`; added to 96-step closed-loop and FCTV validation. |
| Recurrent baseline | `lstm_forecaster` | included | Three-target fair-budget `lstm_baseline_joint_all_reference.pt`; added to 96-step closed-loop and FCTV validation. |
| Recurrent segment baseline | `segrnn_forecaster` | included | Three-target fair-budget `segrnn_baseline_joint_all_reference.pt`; added to 96-step closed-loop and FCTV validation. |
| Linear forecasting baseline | `nlinear_forecaster` | included | Three-target fair-budget `nlinear_baseline_joint_all_reference.pt`; added to 96-step closed-loop and FCTV validation. |
| Frequency-style baseline | `frequency_forecaster` | included | Three-target fair-budget `frequency_baseline_joint_all_reference.pt`; low-frequency FFT context plus future conditioning. |
| Pure Transformer baseline | `transformer_forecaster` | included | Three-target fair-budget `transformer_baseline_joint_all_reference.pt`; added to 96-step closed-loop and FCTV validation. |
| Hybrid Transformer baseline | `current_hybrid_transformer` | included | Strong repository baseline with compatible three-target checkpoint. |
| Transformer-hybrid residual | `transformer_hybrid_residual` | included | DLinear main path plus Transformer-hybrid residual correction. |
| iTransformer-style residual | `itransformer_residual` | included | DLinear main path plus iTransformer residual correction. |
| PatchTST-style residual | `patchtst_residual` | included | DLinear main path plus PatchTST residual correction. |
| CO2-aware residual | `itransformer_co2_late_residual` | included | CO2-aware late-horizon residual baseline. |
| PHF / expert variants | `late_frozen_expert`, `recoupled_expert`, `horizon_mixture`, `frozen_backbone_horizon_mixture`, `control_aware_fusion` | included | Main local PHF / control-aware comparison family. |

## Not Yet Included In Fine-Grained Validation

| Category | Model | Status | Reason |
| --- | --- | --- | --- |
| Old DLinear baseline alias | `dlinear_baseline` | superseded | Existing checkpoint is four-target; `dlinear_forecaster` is the compatible three-target DLinear baseline. |
| Legacy GRU / SegRNN / Transformer aliases | `gru_baseline`, `segrnn_baseline`, `transformer_baseline` | superseded for control | The strict control predictors are now exposed as `gru_forecaster`, `segrnn_forecaster`, and `transformer_forecaster` to avoid loading old four-target checkpoints. |
| Legacy LSTM / NLinear aliases | `lstm_baseline`, `nlinear_baseline` | superseded for control | The strict control predictors are exposed as `lstm_forecaster` and `nlinear_forecaster`. |
| DiffMPC-style Transformer | `diffmpc_style_transformer` | excluded for now | Uses a 48-step history protocol, while current validation bundle uses 288 steps. Do not include it in the strict pool until a protocol-aligned run exists. |
| Standalone CO2 specialists | `co2_wavelet_gru_attn`, `co2_vmd_lstm_fusion`, `co2_env_lstm` | pending | CO2-only outputs cannot directly drive multi-target MPC without an adapter. |

## Current Interpretation

The expanded validation now covers the most important in-repository representative families that are immediately compatible with the three-target MPC protocol: DLinear, NLinear, GRU, LSTM, SegRNN, pure Transformer, Transformer-hybrid, PatchTST-style, iTransformer-style, CO2-aware residuals, and PHF / control-aware fusion models.

It does not yet constitute a full external-paper baseline suite. The current `frequency_forecaster` is a lightweight in-repository frequency-style baseline, not a formal reproduction of Autoformer, FEDformer, TimesNet, or FreTS. A future external-paper baseline suite can still add one of those exact architectures if needed.

The transfer analysis has been expanded to multi-objective FCTV. After adding the standard and frequency-style baselines, the current interpretation is more conservative: `co2_first_step_mae` remains the strongest CO2 screening signal but is now a `secondary_selection` metric rather than a primary universal rule; `rhair_first_step_mae` is now the strongest validated per-target signal and is classified as `primary_selection` for humidity; `tair_first_step_mae` remains diagnostic-only for closed-loop Tair; the current multi-objective score is only `weak_selection` for the whole objective.
