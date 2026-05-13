# Economic And Resource-Aware MPC

Canonical English version.
Mapped Chinese mirror: [ECONOMIC_RESOURCE_MPC.zh-CN.md](ECONOMIC_RESOURCE_MPC.zh-CN.md)

## Position

This document defines the E-stage direction after the tracking-only FCTV stage.

The tracking-control benchmark should remain as the baseline. Economic/resource-aware MPC is an extension that asks a more realistic greenhouse question:

How much resource use can be reduced while keeping `Tair`, `Rhair`, and `CO2air` tracking degradation acceptable?

## Objective

The current tracking MPC objective is:

`tracking error + effort + deviation from logged action + action smoothness`

The E-stage objective adds:

`resource proxy cost`

The implemented objective is:

`J = J_tracking + w_effort J_effort + w_deviation J_deviation + w_smooth J_smooth + w_resource J_resource`

Where:

- `J_tracking` is the normalized target tracking loss.
- `J_effort` penalizes high normalized control values.
- `J_deviation` penalizes deviation from recorded AGC actions.
- `J_smooth` penalizes abrupt actuator changes.
- `J_resource` is a weighted normalized resource proxy over future actions.

Default `w_resource = 0`, so historical FCTV and tracking-control benchmarks remain unchanged unless the economic profile is explicitly enabled.

## Resource Proxy

The first implemented resource proxy uses action-level weights:

| action | interpretation | default weight |
| --- | --- | --- |
| `t_heat_sp` | heating demand proxy | `1.0` |
| `co2_sp` | CO2 dosing proxy | `1.0` |
| `assim_sp` | artificial lighting proxy | `1.0` |
| `window_pos_lee_sp` | ventilation proxy | `0.35` |
| `t_vent_sp` | ventilation temperature proxy | `0.25` |
| `water_sup_intervals_sp_min` | irrigation proxy | `0.20` |
| `scr_enrg_sp` | energy screen movement / state proxy | `0.15` |
| `scr_blck_sp` | blackout screen movement / state proxy | `0.10` |

This proxy is intentionally conservative. It is not a physical energy model and should not be described as true cost. It is a first benchmarkable control penalty for resource-aware behavior.

## Implementation

Code changes:

- `AGCConfig.economic_resource_weight`
- `AGCConfig.economic_action_weights`
- `PredictiveControlAdapter.control_cost()`
- `RolloutSummary.resource_proxy_mean`
- `run_economic_resource_mpc_probe.py`

The tracking benchmark remains unchanged when `economic_resource_weight = 0`.

Run a plan-only check:

```powershell
C:\Users\wokis\.conda\envs\strawberry_env\python.exe agc_mpc\run_economic_resource_mpc_probe.py --print-plan
```

Run the first small probe:

```powershell
C:\Users\wokis\.conda\envs\strawberry_env\python.exe agc_mpc\run_economic_resource_mpc_probe.py
```

Suggested first formal comparison:

```powershell
C:\Users\wokis\.conda\envs\strawberry_env\python.exe agc_mpc\run_economic_resource_mpc_probe.py --steps 96 --start-indices 0 96 192 --resource-weight 0.05 --profile-name economic_w005
C:\Users\wokis\.conda\envs\strawberry_env\python.exe agc_mpc\run_economic_resource_mpc_probe.py --steps 96 --start-indices 0 96 192 --resource-weight 0.15 --profile-name economic_w015
C:\Users\wokis\.conda\envs\strawberry_env\python.exe agc_mpc\run_economic_resource_mpc_probe.py --steps 96 --start-indices 0 96 192 --resource-weight 0.30 --profile-name economic_w030
```

## Evaluation

The E-stage should report:

- tracking objective
- `Tair`, `Rhair`, `CO2air` MAE
- `resource_proxy_mean`
- control delta MAE
- action total variation
- resource reduction versus tracking-only MPC
- tracking degradation versus tracking-only MPC

A result is useful only if it provides a trade-off curve, not just a lower-resource single run.

## First Executed Probe

Executed on `2026-05-12`:

- tracking-only comparison profile: `tracking_probe_w000`
- economic/resource profile: `economic_probe_w015`
- predictors: `current_hybrid_transformer`, `itransformer_co2_residual`
- start: `0`
- rollout length: `24` steps

Generated outputs:

- `results/control/summaries/fctv_multistart_gradient_mpc_reference_24steps_2predictors_c5d60ca7a5_tracking_probe_w000_starts_0.json`
- `results/control/summaries/fctv_multistart_gradient_mpc_reference_24steps_2predictors_c5d60ca7a5_economic_probe_w015_starts_0.json`
- `results/control/summaries/economic_resource_probe_comparison.{csv,md}`
- `results/control/figures/economic_resource_probe_comparison.png`

Probe result:

- `current_hybrid_transformer`: resource proxy decreased from `0.354` to `0.332` (`-6.0%`), while CO2 MAE increased from `10.964` to `12.380`.
- `itransformer_co2_residual`: resource proxy decreased from `0.377` to `0.357` (`-5.3%`), while CO2 MAE increased from `2.938` to `4.899`.

Interpretation:

- The code path works: the resource term changes the optimized actions and creates a measurable resource-tracking trade-off.
- The first weight `0.15` is already strong enough to reduce the proxy by about `5%` to `6%`, but it increases CO2 error in the short probe.
- The next formal E-stage experiment should sweep resource weights and use 96-step, multi-start rollouts before making any control claim.

## Top-5 Control Model Probe

A larger short probe was executed for five strong tracking-control models:

- `current_hybrid_transformer`
- `itransformer_co2_residual`
- `segrnn_forecaster`
- `transformer_forecaster`
- `transformer_hybrid_residual`

Setup:

- start `0`
- `24` rollout steps
- tracking-only profile `tracking_top5_w000`
- economic/resource profile `economic_top5_w015`

Generated outputs:

- `results/control/summaries/fctv_multistart_gradient_mpc_reference_24steps_5predictors_e9cead51af_tracking_top5_w000_starts_0.json`
- `results/control/summaries/fctv_multistart_gradient_mpc_reference_24steps_5predictors_e9cead51af_economic_top5_w015_starts_0.json`
- `results/control/summaries/economic_resource_top5_start0_24steps_comparison.{csv,md}`
- `results/control/figures/economic_resource_top5_start0_24steps_comparison.png`

Result summary:

| predictor | resource change | CO2 MAE change |
| --- | --- | --- |
| `current_hybrid_transformer` | `-5.9%` | `10.964 -> 12.357` |
| `itransformer_co2_residual` | `-5.3%` | `2.938 -> 4.899` |
| `segrnn_forecaster` | `-3.0%` | `12.891 -> 14.519` |
| `transformer_forecaster` | `-8.6%` | `8.051 -> 8.486` |
| `transformer_hybrid_residual` | `+2.3%` | `7.913 -> 9.886` |

Interpretation:

- `transformer_forecaster` is the most promising model in this short economic probe: it reduced the resource proxy the most while only slightly increasing CO2 MAE.
- `itransformer_co2_residual` still has the best absolute CO2 tracking after adding the economic term, but its CO2 degradation is larger.
- `transformer_hybrid_residual` increased the resource proxy under the current weight, so the economic objective does not produce uniform behavior across predictors.
- The result is suitable for selecting candidates for a formal weight sweep, not for claiming final economic superiority.

## Top-3 96-Step Multi-Start Weight Sweep

The first multi-start E-stage sweep was executed for three representative strong closed-loop predictors:

- `current_hybrid_transformer`
- `itransformer_co2_residual`
- `transformer_forecaster`

Setup:

- starts `0`, `96`, `192`
- `96` rollout steps
- resource weights `0.00`, `0.05`, `0.15`, `0.30`

Generated suites:

- `results/control/summaries/fctv_multistart_gradient_mpc_reference_96steps_3predictors_e7d9317832_economic_sweep_top3_w000_starts_0_96_192.json`
- `results/control/summaries/fctv_multistart_gradient_mpc_reference_96steps_3predictors_e7d9317832_economic_sweep_top3_w005_starts_0_96_192.json`
- `results/control/summaries/fctv_multistart_gradient_mpc_reference_96steps_3predictors_e7d9317832_economic_sweep_top3_w015_starts_0_96_192.json`
- `results/control/summaries/fctv_multistart_gradient_mpc_reference_96steps_3predictors_e7d9317832_economic_sweep_top3_w030_starts_0_96_192.json`

Generated sweep summary:

- `results/control/summaries/economic_resource_sweep_top3_reference.{csv,md}`
- `results/control/figures/economic_resource_sweep_top3_reference.png`

Mean result across starts:

| predictor | weight | resource change | CO2 change |
| --- | --- | --- | --- |
| `current_hybrid_transformer` | `0.05` | `-9.8%` | `+2.1%` |
| `current_hybrid_transformer` | `0.15` | `-14.9%` | `+19.0%` |
| `current_hybrid_transformer` | `0.30` | `-27.0%` | `+16.9%` |
| `itransformer_co2_residual` | `0.05` | `-7.3%` | `+4.3%` |
| `itransformer_co2_residual` | `0.15` | `-22.5%` | `+24.9%` |
| `itransformer_co2_residual` | `0.30` | `-23.2%` | `+69.6%` |
| `transformer_forecaster` | `0.05` | `-5.9%` | `+3.3%` |
| `transformer_forecaster` | `0.15` | `-16.3%` | `+19.3%` |
| `transformer_forecaster` | `0.30` | `-22.7%` | `+39.7%` |

Interpretation:

- `w=0.05` is the current useful trade-off region. It reduces the resource proxy by about `6%` to `10%` while keeping mean CO2 degradation to about `2%` to `4%`.
- `w=0.15` and `w=0.30` reduce the resource proxy more strongly but cause much larger CO2 degradation.
- `current_hybrid_transformer` has the best low-weight trade-off in this sweep: `-9.8%` resource proxy with only `+2.1%` CO2 MAE.
- `itransformer_co2_residual` remains the best absolute CO2 tracker, but high resource weights damage its CO2 advantage.
- The next E-stage step should refine the low-weight region, for example `0.02`, `0.05`, `0.08`, `0.10`, before trying larger model pools.

## Research Claim Boundary

Allowed claim:

- The extended MPC can explore tracking-resource trade-offs under an action-level resource proxy.

Disallowed claim:

- The proxy is real greenhouse profit, real energy cost, or real CO2 consumption.

Next rigorous step:

- Replace the proxy with calibrated greenhouse cost terms if reliable price, energy, CO2 dosing, and actuator data are available.
