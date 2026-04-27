# Protected Horizon Fusion 主线

中文对齐翻译版本。
英文主版本：[PHF_MAINLINE.md](c:/repositories/strawberry/agc_mpc/PHF_MAINLINE.md)
最近同步时间：`2026-04-21`

## 1. 目的

本文档定义当前 CO2 specialist fusion 工作的论文主线。

项目不应该把最近每个模型变体都写成独立贡献。故事应该收敛到：

**面向控制的温室多步预测中的 Protected Horizon Fusion**

简称：

- `PHF`
- `PHF-iTransformer`
- 聚焦 CO2 分支时可称为 `CO2-PHF`

论文主线应该是：

1. `CO2air` 比 `Tair` 和 `Rhair` 更难，因为它混合了慢周期、补碳/通风突变和控制相关动力学。
2. 独立多尺度 CO2 specialist 有用，但直接端到端并回多目标 predictor 不稳定。
3. 冻结 specialist 应该被选择性信任，而不是盲目信任。
4. 信任程度应该依赖 expert 与主模型的一致性，也应该依赖 forecast horizon。
5. 离线预测强度不足以服务 MPC；验证必须包含 control-relevant metrics。

## 2. 主方法

推荐把主方法描述为 `Protected Horizon Fusion`。

该方法包含三个组成部分：

1. **多目标 residual backbone**
   - 同时预测 `Tair`、`Rhair` 和 `CO2air`
   - 当前实现族：`iTransformer residual`
   - 稳定 baseline：`itransformer_co2_late_residual`

2. **冻结 CO2 specialist**
   - 当前 expert：`co2_wavelet_gru_attn`
   - 作为独立 `CO2air` specialist 训练
   - 对应文献里的核心思想：温室 CO2 需要多尺度和 horizon-aware 建模

3. **Protected horizon fusion gate**
   - 只对 `CO2air` 通道施加 expert correction
   - 当 expert 和主模型严重分歧时降低信任
   - 随 forecast horizon 改变信任程度
   - 末端 horizon 回拉到更稳定的 late-residual backbone

核心公式：

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

当前最强离线实现是：

- `itransformer_co2_horizon_mixture`

## 3. 模型角色

后续报告和论文草稿中应统一使用这些角色。

| 模型 | 论文角色 | 描述方式 |
|---|---|---|
| `itransformer_residual` | residual backbone baseline | 通用多目标 residual predictor |
| `itransformer_co2_late_residual` | 强 CO2-aware backbone | 没有外部 expert 的 late-horizon CO2 adapter |
| `co2_wavelet_gru_attn` | 独立 CO2 expert | 多尺度 GRU-attention CO2 specialist |
| `itransformer_co2_frozen_expert` | naive fusion baseline | 直接把冻结 expert 与主模型 blend |
| `itransformer_co2_late_frozen_expert` | late-trust control baseline | 在后段 horizon 更信任冻结 expert；当前闭环 CO2 控制最好 |
| `itransformer_co2_teacher_distill` | distillation ablation | 只把 expert 作为辅助 teacher |
| `itransformer_co2_recoupled_expert` | cross-target recoupling baseline | expert correction 后再加入目标间耦合；当前 overall control objective 最好 |
| `itransformer_co2_protected_expert` | protection ablation | 加入 agreement-protected expert correction |
| `itransformer_co2_protected_terminal` | terminal-loss ablation | 测试只加 terminal loss 是否足够 |
| `itransformer_co2_horizon_mixture` | proposed offline PHF model | protected correction + terminal pullback；当前离线 CO2 leader |
| `itransformer_co2_frozen_backbone_horizon_mixture` | control-safety diagnostic | 冻结 late-residual backbone，只训练 gate；保留 MPC 梯度 |

## 4. 不应该宣称什么

不要宣称：

- 每个 CO2 变体都是独立贡献
- `horizon_mixture` 是 control leader
- 普通离线 MAE / R2 足以选择 MPC predictor
- 当前模型已经是 CO2-only greenhouse specialist 论文中的 SOTA

正确的、更稳的说法是：

- `PHF-iTransformer` 是当前仓库里最强的离线多目标 CO2 specialist fusion 模型。
- 闭环实验表明，离线 forecasting 收益不会自动转化为 MPC 收益。
- 因此，control-relevant validation 是 predictor selection 的必要组成部分。

## 5. 必要消融逻辑

消融表应该保证每一行只回答一个问题。

| 问题 | 模型对比 |
|---|---|
| CO2-aware backbone 是否有用？ | `itransformer_residual` vs `itransformer_co2_late_residual` |
| 冻结 standalone expert 是否有用？ | `late_residual` vs `frozen_expert` / `late_frozen_expert` |
| horizon-dependent trust 是否有用？ | `frozen_expert` vs `late_frozen_expert` |
| agreement protection 是否有用？ | `late_frozen_expert` vs `protected_expert` |
| terminal loss alone 是否足够？ | `protected_expert` vs `protected_terminal` |
| 显式 terminal pullback 是否有用？ | `protected_terminal` vs `horizon_mixture` |
| 冻结 backbone 是否提升 MPC safety？ | `horizon_mixture` vs `frozen_backbone_horizon_mixture` |

## 6. 当前证据

当前离线 forecasting leader：

- `itransformer_co2_horizon_mixture`
- `CO2air` Full MAE `43.910`
- `CO2air` Final MAE `47.661`

当前最强 CO2 闭环控制 baseline：

- `itransformer_co2_late_frozen_expert + GradientMPC`
- `CO2air` MAE `6.298`

当前整体闭环 objective 最强 baseline：

- `itransformer_co2_recoupled_expert + GradientMPC`
- objective `0.0651`

当前 control-safe diagnostic：

- `itransformer_co2_frozen_backbone_horizon_mixture + GradientMPC`
- objective `0.0718`
- `CO2air` MAE `10.000`

Control-relevant validation 结论：

- `horizon_mixture` 在离线 full/final forecasting 上强，但 first-step 和 first-6-step CO2 validation 弱。
- `late_frozen_expert` 在短时域 CO2 行为和闭环 CO2 控制上更强。
- `late_residual` 与 frozen-backbone horizon mixture 是更稳的 control-safe 折中。

Validation v2 结果：

- 已生成 [control_relevant_validation_reference.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/control_relevant_validation_reference.json)、[control_relevant_validation_reference.csv](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/control_relevant_validation_reference.csv)、[control_relevant_validation_reference.md](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/control_relevant_validation_reference.md) 和 [control_relevant_validation_reference.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/comparisons/control_relevant_validation_reference.png)。
- 新增 signed CO2 bias、constraint-near proxy MAE、有符号/平坦梯度诊断、recorded-policy CO2 improvement 和 action-activity diagnostics。
- 当前 control-relevant mean rank：
  - `itransformer_co2_late_frozen_expert`: `2.250`
  - `itransformer_co2_late_residual`: `2.500`
  - `itransformer_residual`: `3.250`
  - `itransformer_co2_frozen_backbone_horizon_mixture`: `3.375`
  - `itransformer_co2_horizon_mixture`: `4.500`
  - `itransformer_co2_recoupled_expert`: `5.125`

PHF 消融结果：

- 已生成 [phf_ablation_reference.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/phf_ablation_reference.json)、[phf_ablation_reference.csv](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/phf_ablation_reference.csv)、[phf_ablation_reference.md](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/phf_ablation_reference.md) 和 [phf_ablation_reference.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/comparisons/phf_ablation_reference.png)。
- 消融表支持当前角色划分：
  - `horizon_mixture`：离线 PHF 代表模型和 CO2 forecasting leader
  - `late_frozen_expert`：最强 CO2 闭环控制 baseline
  - `recoupled_expert`：最强整体闭环 objective baseline
  - `frozen_backbone_horizon_mixture`：control-safety 诊断模型

## 7. 本周任务

当前周应该优先：

1. Control-relevant validation suite
   - first-step MAE
   - 前 `6` 步 control-horizon MAE
   - horizon-weighted MAE
   - 控制输入敏感性
   - 闭环控制指标
   - 状态：v2 已实现，包含 signed bias、constraint-near proxy、gradient sign/flatness、recorded-policy improvement 和 PHF 关联输出

2. PHF 故事收敛
   - 保持 `horizon_mixture` 作为离线 PHF 代表
   - 保持 `late_frozen_expert` 和 `recoupled_expert` 作为控制 baseline
   - 保持 `frozen_backbone_horizon_mixture` 作为 diagnostic，而不是主方法
   - 状态：PHF 消融表和图已生成

只有当这条故事稳定之后，项目才应该新增一个 control-aware fusion 模型。
