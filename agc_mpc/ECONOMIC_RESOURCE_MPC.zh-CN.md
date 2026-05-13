# Economic And Resource-Aware MPC

中文镜像版本。
对应英文主版：[ECONOMIC_RESOURCE_MPC.md](ECONOMIC_RESOURCE_MPC.md)

## 定位

本文档定义 tracking-only FCTV 阶段之后的 E 阶段方向。

Tracking-control benchmark 应继续作为 baseline。Economic/resource-aware MPC 是一个扩展，用来回答更真实的温室问题：

在 `Tair`、`Rhair`、`CO2air` tracking 退化可接受的前提下，能减少多少资源使用？

## 目标函数

当前 tracking MPC 目标是：

`tracking error + effort + deviation from logged action + action smoothness`

E 阶段新增：

`resource proxy cost`

实现后的目标函数是：

`J = J_tracking + w_effort J_effort + w_deviation J_deviation + w_smooth J_smooth + w_resource J_resource`

其中：

- `J_tracking` 是归一化目标 tracking loss。
- `J_effort` 惩罚较高的归一化控制值。
- `J_deviation` 惩罚偏离记录 AGC 动作。
- `J_smooth` 惩罚执行器剧烈变化。
- `J_resource` 是未来控制动作上的加权归一化资源代理项。

默认 `w_resource = 0`，因此除非显式启用 economic profile，历史 FCTV 和 tracking-control benchmark 不会改变。

## 资源代理项

第一版实现使用 action-level weights：

| action | interpretation | default weight |
| --- | --- | --- |
| `t_heat_sp` | 加热需求代理 | `1.0` |
| `co2_sp` | CO2 施肥代理 | `1.0` |
| `assim_sp` | 人工补光代理 | `1.0` |
| `window_pos_lee_sp` | 通风代理 | `0.35` |
| `t_vent_sp` | 通风温度代理 | `0.25` |
| `water_sup_intervals_sp_min` | 灌溉代理 | `0.20` |
| `scr_enrg_sp` | energy screen 状态 / 动作代理 | `0.15` |
| `scr_blck_sp` | blackout screen 状态 / 动作代理 | `0.10` |

这个 proxy 是刻意保守的。它不是物理能耗模型，不能表述为真实经济成本。它只是第一版可 benchmark 的 resource-aware control penalty。

## 实现

代码改动：

- `AGCConfig.economic_resource_weight`
- `AGCConfig.economic_action_weights`
- `PredictiveControlAdapter.control_cost()`
- `RolloutSummary.resource_proxy_mean`
- `run_economic_resource_mpc_probe.py`

当 `economic_resource_weight = 0` 时，tracking benchmark 保持不变。

只打印计划：

```powershell
C:\Users\wokis\.conda\envs\strawberry_env\python.exe agc_mpc\run_economic_resource_mpc_probe.py --print-plan
```

运行第一轮小规模 probe：

```powershell
C:\Users\wokis\.conda\envs\strawberry_env\python.exe agc_mpc\run_economic_resource_mpc_probe.py
```

建议的第一轮正式对比：

```powershell
C:\Users\wokis\.conda\envs\strawberry_env\python.exe agc_mpc\run_economic_resource_mpc_probe.py --steps 96 --start-indices 0 96 192 --resource-weight 0.05 --profile-name economic_w005
C:\Users\wokis\.conda\envs\strawberry_env\python.exe agc_mpc\run_economic_resource_mpc_probe.py --steps 96 --start-indices 0 96 192 --resource-weight 0.15 --profile-name economic_w015
C:\Users\wokis\.conda\envs\strawberry_env\python.exe agc_mpc\run_economic_resource_mpc_probe.py --steps 96 --start-indices 0 96 192 --resource-weight 0.30 --profile-name economic_w030
```

## 评价方式

E 阶段应报告：

- tracking objective
- `Tair`、`Rhair`、`CO2air` MAE
- `resource_proxy_mean`
- control delta MAE
- action total variation
- 相对 tracking-only MPC 的 resource reduction
- 相对 tracking-only MPC 的 tracking degradation

有价值的结果应该是一条 trade-off curve，而不是单次低资源运行。

## 第一轮已执行 Probe

已在 `2026-05-12` 执行：

- tracking-only 对照 profile：`tracking_probe_w000`
- economic/resource profile：`economic_probe_w015`
- predictors：`current_hybrid_transformer`、`itransformer_co2_residual`
- start：`0`
- rollout 长度：`24` steps

生成输出：

- `results/control/summaries/fctv_multistart_gradient_mpc_reference_24steps_2predictors_c5d60ca7a5_tracking_probe_w000_starts_0.json`
- `results/control/summaries/fctv_multistart_gradient_mpc_reference_24steps_2predictors_c5d60ca7a5_economic_probe_w015_starts_0.json`
- `results/control/summaries/economic_resource_probe_comparison.{csv,md}`
- `results/control/figures/economic_resource_probe_comparison.png`

Probe 结果：

- `current_hybrid_transformer`：resource proxy 从 `0.354` 降到 `0.332`（`-6.0%`），CO2 MAE 从 `10.964` 升到 `12.380`。
- `itransformer_co2_residual`：resource proxy 从 `0.377` 降到 `0.357`（`-5.3%`），CO2 MAE 从 `2.938` 升到 `4.899`。

解释：

- 代码路径有效：resource term 会改变优化动作，并产生可量化的 resource-tracking trade-off。
- 第一版权重 `0.15` 已经足以让 resource proxy 降低约 `5%` 到 `6%`，但在短 probe 中会增加 CO2 error。
- 下一轮正式 E 阶段实验应扫描 resource weights，并使用 96-step、multi-start rollout，之后才能做控制结论。

## Top-5 控制模型 Probe

已对五个 tracking-control 表现较好的模型执行更大的短 probe：

- `current_hybrid_transformer`
- `itransformer_co2_residual`
- `segrnn_forecaster`
- `transformer_forecaster`
- `transformer_hybrid_residual`

设置：

- start `0`
- `24` rollout steps
- tracking-only profile `tracking_top5_w000`
- economic/resource profile `economic_top5_w015`

生成输出：

- `results/control/summaries/fctv_multistart_gradient_mpc_reference_24steps_5predictors_e9cead51af_tracking_top5_w000_starts_0.json`
- `results/control/summaries/fctv_multistart_gradient_mpc_reference_24steps_5predictors_e9cead51af_economic_top5_w015_starts_0.json`
- `results/control/summaries/economic_resource_top5_start0_24steps_comparison.{csv,md}`
- `results/control/figures/economic_resource_top5_start0_24steps_comparison.png`

结果汇总：

| predictor | resource change | CO2 MAE change |
| --- | --- | --- |
| `current_hybrid_transformer` | `-5.9%` | `10.964 -> 12.357` |
| `itransformer_co2_residual` | `-5.3%` | `2.938 -> 4.899` |
| `segrnn_forecaster` | `-3.0%` | `12.891 -> 14.519` |
| `transformer_forecaster` | `-8.6%` | `8.051 -> 8.486` |
| `transformer_hybrid_residual` | `+2.3%` | `7.913 -> 9.886` |

解释：

- `transformer_forecaster` 是这轮短 economic probe 中最值得继续看的模型：resource proxy 降幅最大，同时 CO2 MAE 只小幅上升。
- `itransformer_co2_residual` 加入 economic term 后仍有最好的绝对 CO2 tracking，但 CO2 退化更明显。
- `transformer_hybrid_residual` 在当前权重下 resource proxy 反而上升，说明 economic objective 对不同 predictor 不会产生完全同向的行为。
- 该结果适合用于选择正式 weight sweep 的候选模型，不能直接声称最终经济最优。

## Top-3 96-Step Multi-Start 权重扫描

第一轮 multi-start E 阶段 sweep 已对三个代表性强闭环 predictor 执行：

- `current_hybrid_transformer`
- `itransformer_co2_residual`
- `transformer_forecaster`

设置：

- starts `0`、`96`、`192`
- `96` rollout steps
- resource weights `0.00`、`0.05`、`0.15`、`0.30`

生成 suite：

- `results/control/summaries/fctv_multistart_gradient_mpc_reference_96steps_3predictors_e7d9317832_economic_sweep_top3_w000_starts_0_96_192.json`
- `results/control/summaries/fctv_multistart_gradient_mpc_reference_96steps_3predictors_e7d9317832_economic_sweep_top3_w005_starts_0_96_192.json`
- `results/control/summaries/fctv_multistart_gradient_mpc_reference_96steps_3predictors_e7d9317832_economic_sweep_top3_w015_starts_0_96_192.json`
- `results/control/summaries/fctv_multistart_gradient_mpc_reference_96steps_3predictors_e7d9317832_economic_sweep_top3_w030_starts_0_96_192.json`

生成 sweep 汇总：

- `results/control/summaries/economic_resource_sweep_top3_reference.{csv,md}`
- `results/control/figures/economic_resource_sweep_top3_reference.png`

跨 start 平均结果：

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

解释：

- `w=0.05` 是当前最有价值的 trade-off 区间。它能让 resource proxy 下降约 `6%` 到 `10%`，同时平均 CO2 退化保持在约 `2%` 到 `4%`。
- `w=0.15` 和 `w=0.30` 能进一步降低 resource proxy，但 CO2 退化明显变大。
- `current_hybrid_transformer` 在这轮 sweep 中有最好的低权重 trade-off：resource proxy `-9.8%`，CO2 MAE 仅 `+2.1%`。
- `itransformer_co2_residual` 仍是绝对 CO2 tracking 最好的模型，但高 resource weight 会明显削弱它的 CO2 优势。
- 下一步 E 阶段应细化低权重区间，例如 `0.02`、`0.05`、`0.08`、`0.10`，然后再扩大模型池。

## 研究声明边界

允许声明：

- 扩展后的 MPC 可以在 action-level resource proxy 下探索 tracking-resource trade-off。

不允许声明：

- 该 proxy 是真实温室利润、真实能耗或真实 CO2 消耗。

下一步严谨工作：

- 如果能拿到可靠价格、能耗、CO2 施肥和执行器数据，再把 proxy 替换为校准后的温室成本项。
