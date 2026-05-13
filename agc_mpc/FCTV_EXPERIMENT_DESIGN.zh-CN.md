# FCTV 实验设计

中文镜像版本。
对应英文主版：[FCTV_EXPERIMENT_DESIGN.md](FCTV_EXPERIMENT_DESIGN.md)

## 定位

本文档固定下一阶段面向论文的实验设计。探索性的 FCTV 阶段已经收束：普通预测侧指标可以作为诊断信号，但在扩大模型池和重复闭环起点后，不能稳定作为通用 selector。

下一阶段不再机会式追加实验，而是用固定协议回答固定论文问题。

## 论文问题

暂定题目：

**Do Better Forecasts Lead to Better Control? Forecast-to-Control Validation for Greenhouse MPC**

主问题：

离线预测侧指标能否可靠筛选出能改善闭环温室 MPC 的预测器？

研究问题：

- `RQ1`：标准预测侧指标是否能跨模型族预测闭环 MPC 表现？
- `RQ2`：预测到控制的关系是否能跨 rollout segment 保持稳定？
- `RQ3`：如果普通预测指标不是稳定 selector，哪些预测器在直接闭环验证中更稳健？

允许声明：

- 在当前跨模型族温室 benchmark 下，标准离线预测指标不能可靠替代闭环 MPC 验证。

不允许声明：

- 不存在任何从预测结果导出的指标可以预测控制表现。

## 最终模型池

最终闭环 benchmark 使用 `16` 个 predictor。该模型池刻意覆盖多个模型族，而不是只填充本地 PHF 变体。

| family | predictors |
| --- | --- |
| Linear | `dlinear_forecaster`, `nlinear_forecaster` |
| Recurrent | `gru_forecaster`, `lstm_forecaster` |
| Segmented recurrent | `segrnn_forecaster` |
| Frequency/decomposition-style | `frequency_forecaster` |
| Transformer | `transformer_forecaster`, `current_hybrid_transformer`, `transformer_hybrid_residual` |
| Patch / iTransformer residual | `patchtst_residual`, `itransformer_residual` |
| CO2-aware / PHF | `itransformer_co2_residual`, `itransformer_co2_late_residual`, `itransformer_co2_late_frozen_expert`, `itransformer_co2_horizon_mixture`, `itransformer_co2_control_aware_fusion` |

理由：

- 模型池包含标准 baseline、现代序列模型、residual 变体和代表性 CO2/PHF 模型。
- 模型族覆盖足够用于检验 transfer 稳定性。
- 模型数量仍可承受重复闭环验证。
- `diffmpc_style_transformer` 在协议与 288 步 AGC 历史窗口对齐前继续排除。

## 闭环 Benchmark 协议

固定协议：

- 数据集：AGC 2019 Reference compartment。
- 目标变量：`Tair`、`Rhair`、`CO2air`。
- 预测历史窗口：当前 AGC 三目标协议。
- 控制器：`GradientMPC`。
- rollout 模式：recorded weather / 当前 surrogate closed-loop 设置。
- rollout 长度：`96` steps。
- start indices：`0`、`96`、`192`、`288`、`384`。
- 主要闭环指标：`mpc_objective`、`mpc_tair_mae`、`mpc_rhair_mae`、`mpc_co2_mae`。
- 次要闭环指标：control delta MAE 和 action total variation。

最终 benchmark 应报告跨 start 的 mean 和 standard deviation，而不是单起点 leaderboard。

## 预测到控制指标

面向论文的 FCTV 分析保留：

- 每个目标变量的 first-step MAE
- 每个目标变量的 control-horizon MAE
- 每个目标变量的 control-horizon absolute bias
- 每个目标变量的 constraint-near MAE proxy
- 归一化 transfer selection score
- 只作为 diagnostic-only evidence 的 gradient diagnostics

筛选证据用以下统计量验证：

- Spearman rank correlation
- pairwise ordering consistency
- top-k overlap
- leave-one-model robustness
- 如果 family 标签可用，增加 leave-one-family robustness

## 正式实验矩阵

实验 1：Forecasting benchmark。

- 目标：建立最终模型池的离线预测行为。
- 输出：逐目标 first-step、control-horizon、full-horizon 和 final-step 指标。

实验 2：Closed-loop MPC benchmark。

- 目标：在固定模型池和 5 个 start 下确定稳健闭环 winner。
- 输出：跨 start mean/std 的 objective leaderboard 和逐目标 leaderboard。

实验 3：FCTV transfer analysis。

- 目标：检验预测侧指标是否能解释闭环 benchmark。
- 输出：Spearman、pairwise consistency、top-k、robustness 和 metric-role 表。

实验 4：Diagnostic discussion。

- 目标：解释 transfer 失效的位置。
- 输出：模型族依赖、start 依赖和目标冲突讨论。

## 可运行入口

打印固定 benchmark 计划：

```powershell
C:\Users\wokis\.conda\envs\strawberry_env\python.exe agc_mpc\run_fctv_final_closed_loop_benchmark.py --print-plan
```

运行正式闭环 benchmark：

```powershell
C:\Users\wokis\.conda\envs\strawberry_env\python.exe agc_mpc\run_fctv_final_closed_loop_benchmark.py
```

分析生成的 suite：

```powershell
C:\Users\wokis\.conda\envs\strawberry_env\python.exe agc_mpc\analyze_fctv_multistart_transfer.py --suite-json <generated_suite_json> --prefix forecast_to_control_transfer_final_reference
```

## 已执行最终 Benchmark

正式 16 模型、5 起点闭环 benchmark 已在 `2026-05-12` 执行。

生成的 suite：

- `results/control/summaries/fctv_multistart_gradient_mpc_reference_96steps_16predictors_25890932c3_starts_0_96_192_288_384.json`

生成的 FCTV analysis：

- `results/forecasting/analysis/forecast_to_control_transfer_final_reference.{json,csv,md}`
- 每个 start 的 `forecast_to_control_transfer_final_reference_start*.{json,csv,md}` 和 robustness CSV
- `results/forecasting/figures/comparisons/forecast_to_control_transfer_final_reference.png`
- `results/forecasting/figures/comparisons` 下每个 start 的 summary 和 robustness 图

生成的闭环 ranking 输出：

- `results/forecasting/analysis/fctv_final_multistart_model_rankings_reference.{csv,md}`
- `results/forecasting/figures/comparisons/fctv_final_multistart_model_rankings_reference.png`

最终 benchmark 结论：

- 预测侧 transfer 指标仍然具有 start dependence，不能作为稳定 universal selector。
- `current_hybrid_transformer` 是 5 个 start 上平均 objective 最好的 predictor。
- `itransformer_co2_residual` 是 5 个 start 上平均 CO2 闭环 tracking 最好的 predictor。

## 本周范围

本周完成：

- A：固定这份论文式实验设计。
- B：准备 final closed-loop benchmark，并在有计算窗口时运行。
- C：用论文语言写 FCTV 方法章节。

下一步，不放在本周主范围：

- F：准备面向导师的阶段汇报。
- E：等 tracking-control benchmark 稳定后，再启动 economic/resource-aware MPC。
