# FCTV 阶段汇报

中文镜像版本。
对应英文主版：[FCTV_STAGE_REPORT.md](FCTV_STAGE_REPORT.md)

## 汇报定位

本文档用于向导师汇报已经收束的 FCTV 阶段。这个阶段应表述为一个受控的负结果 / 诊断结果，而不是项目失败。

核心信息是：

离线预测指标可以描述 predictor 的行为，但在模型族、目标变量和 rollout 起点扩展后，已测试的指标不能可靠筛选闭环 MPC winner。直接闭环验证仍然必要。

## 研究问题

主问题：

在运行 MPC 之前计算的预测侧指标，能否可靠筛选温室闭环 MPC 的 predictor？

最初假设是合理的：

- MPC 在优化中使用预测。
- 因为 MPC 会反复滚动优化，短时域预测误差应该重要。
- 接近参考值或运行边界的误差更可能改变控制动作。
- 因此，预测侧指标可能预测闭环控制收益。

最终证据说明这个假设只在局部成立。它可以在局部模型池中表现有效，但不足以替代闭环验证。

## 实验链条

阶段 1：CO2-focused 指标归纳。

- 早期模型池聚焦 CO2 和本地 PHF / CO2-aware 变体。
- 短时域 CO2 指标看起来有用。
- 这说明值得继续扩大验证，而不是立刻宣布一个 selector。

阶段 2：扩大模型池验证。

- 模型池扩展到标准和现代模型族，包括 linear、recurrent、Transformer-style、PatchTST/iTransformer residual 和 CO2/PHF 变体。
- 扩展后，CO2 first-step 和 constraint-near 指标失去稳定筛选能力。
- 这说明早期信号部分依赖模型池。

阶段 3：多目标验证。

- 分析从 CO2-only 扩展到 `Tair`、`Rhair`、`CO2air` 和整体 objective。
- `Rhair` first-step error 在某个扩展设置中保留中等 transfer，但 CO2 和整体 objective transfer 较弱。
- 这说明 transfer 具有目标变量依赖。

阶段 4：多起点闭环验证。

- Benchmark 在多个 rollout start 上重复。
- 最终运行使用 `16` 个 predictor，starts `0`、`96`、`192`、`288`、`384`，闭环长度 `96` 步。
- 共生成 `80` 条闭环记录。
- 最终结果确认 start dependence。

## 最终 Benchmark 输出

主 suite：

- `results/control/summaries/fctv_multistart_gradient_mpc_reference_96steps_16predictors_25890932c3_starts_0_96_192_288_384.json`

Transfer analysis：

- `results/forecasting/analysis/forecast_to_control_transfer_final_reference.{json,csv,md}`
- `results/forecasting/figures/comparisons/forecast_to_control_transfer_final_reference.png`

闭环 ranking：

- `results/forecasting/analysis/fctv_final_multistart_model_rankings_reference.{csv,md}`
- `results/forecasting/figures/comparisons/fctv_final_multistart_model_rankings_reference.png`

周报汇报图：

- `results/forecasting/figures/comparisons/fctv_weekly_metric_degradation_summary.png`

## 关键定量结果

Transfer metrics：

- CO2 first-step transfer 跨 start 不稳定：
  - start `0`：Spearman `0.364`，pairwise `0.613`
  - start `96`：Spearman `0.037`，pairwise `0.504`
  - start `192`：Spearman `-0.149`，pairwise `0.445`
  - start `288`：Spearman `0.243`，pairwise `0.588`
  - start `384`：Spearman `-0.319`，pairwise `0.387`
- Multi-objective transfer score 同样不稳定：
  - start `0`：Spearman `0.406`，pairwise `0.642`
  - start `96`：Spearman `0.235`，pairwise `0.583`
  - start `192`：Spearman `0.174`，pairwise `0.567`
  - start `288`：Spearman `0.362`，pairwise `0.625`
  - start `384`：Spearman `0.141`，pairwise `0.542`

闭环 winner：

- 平均 objective 最优：`current_hybrid_transformer`，objective `0.0662 +/- 0.0269`。
- 平均 CO2 tracking 最优：`itransformer_co2_residual`，`CO2air MAE = 10.215 +/- 2.043`。
- `itransformer_co2_residual` 的平均 objective 也排名第二：`0.0701 +/- 0.0234`。

## 解释

结论不是 forecasting 不重要。结论是普通预测侧指标不足以作为 universal selector。

原因：

- MPC 会把预测转化为动作，因此 action sensitivity 重要。
- Predictor 可能降低了离线误差，但这些误差位于不会影响控制决策的区域。
- 多目标控制存在冲突：改善 CO2 可能损害 `Tair` 或 `Rhair` 行为。
- Greenhouse dynamics 和 reference 难度随片段变化，所以 ranking 会跨 rollout start 改变。
- 在局部模型族里看起来有效的指标，扩展到更广模型族后可能失效。

## 可严谨表述的结论

面向论文可以表述为：

在当前跨模型族 AGC 温室 benchmark 下，标准离线预测指标，即使包括短时域和 constraint-near 变体，也不能可靠替代直接闭环 MPC 验证。

FCTV 仍有价值，因为它诊断了 forecast-control 假设失效的位置：

- 目标变量依赖
- 模型族依赖
- rollout start 依赖
- 预测误差和控制动作敏感性不匹配

## 推荐下一步

下一步应从 tracking-only MPC 转向 economic/resource-aware MPC。

原因：

- Tracking-only MPC 回答的是 predictor 能否跟踪记录轨迹或参考轨迹。
- 温室控制最终是 tracking、加热、CO2 施肥、补光、通风、灌溉和执行器动作之间的经济权衡。
- FCTV 阶段已经说明闭环验证必要；下一阶段应让闭环目标更接近真实控制问题。

