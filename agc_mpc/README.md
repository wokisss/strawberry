# AGC MPC

## Dataset Background

Current main dataset:

- `AutonomousGreenhouseChallenge_edition2`

This dataset is better described as:

- a multi-compartment greenhouse benchmark
- not a fully independent multi-site or multi-year greenhouse dataset

Default compartments used in this project:

- `AICU`
- `Automatoes`
- `Digilog`
- `IUACAAS`
- `Reference`
- `TheAutomators`

Key properties:

- common outdoor weather from `Weather/Weather.csv`
- compartment-specific greenhouse logs from `GreenhouseClimate.csv`
- 5-minute sampling
- explicit control hierarchy:
  - requested setpoints `*_sp`
  - realized setpoints `*_vip`
  - actuator states
  - indoor climate states
  - resource tables for later economic evaluation

Current modeling interface:

- `x_past`: past indoor states and execution feedback
- `w_future`: future weather and time features
- `u_future`: future requested setpoints
- `y_future`: future `Tair / Rhair / CO2air / Tot_PAR`

Why this dataset is more suitable than the old strawberry dataset:

- it is naturally control-oriented
- it supports explicit future weather conditioning
- it supports explicit future setpoint conditioning
- it can be extended from forecasting to surrogate closed-loop control and later economic evaluation

## Single-Compartment vs Multi-Compartment Training

Short answer:

- single-compartment training can be better for one specific compartment
- multi-compartment joint training is better as the default benchmark setting

When single-compartment training can help:

- the target compartment has a distinct control style
- we only care about the best result on one known compartment
- we want to study specialization or transfer

Why joint training across compartments is valuable:

1. More training data
   - each compartment adds additional rolling windows
   - this especially helps nonlinear models

2. More operational diversity
   - the compartments share weather but differ in strategies and trajectories
   - this exposes the model to more regime variation than one compartment alone

3. Better benchmark quality
   - a model that only works on one compartment is less convincing for a control-oriented paper
   - joint training makes the benchmark less fragile

4. Better support for later control work
   - if the predictor is meant to become an MPC surrogate, robustness across compartments matters more than overfitting one compartment

Current project stance:

- use multi-compartment joint training as the default benchmark setting
- keep single-compartment runs as a secondary analysis tool

Training-regime comparison entry point:

```bash
conda activate strawberry_env
python c:\repositories\strawberry\agc_mpc\compare_training_regimes.py --models dlinear --target-compartment Reference
```

Useful options:

- `--describe-only`
- `--epochs 1`
- `--models dlinear transformer transformer_hybrid`
- `--regimes single joint_all leave_one_out`

Outputs:

- summary JSON in `agc_mpc/results/forecasting/analysis`
- summary figure in `agc_mpc/results/forecasting/analysis`
- per-run forecasting figures for each regime in `agc_mpc/results/forecasting/analysis`

## DiffMPC-Style Transformer Benchmark

Purpose:

- keep the old `diffmpc` Transformer-hybrid family and training budget as close as possible
- move that model family onto `AGC`
- test whether the new dataset is structurally more suitable for conditional Transformer-style forecasting

Why this matters:

- one major reason for switching datasets is not just "better scores"
- it is that `AGC` better supports `history + future weather + future control` style modeling
- this is exactly the type of interface where encoder-decoder / hybrid Transformer designs make more sense

Benchmark entry point:

```bash
conda activate strawberry_env
python c:\repositories\strawberry\agc_mpc\benchmark_diffmpc_style_transformer.py --regime single --target-compartment Reference
```

Default protocol:

- targets: `Tair / Rhair / CO2air`
- `seq_len = 48`
  - equivalent to old `240 min` history under AGC `5 min` sampling
- `horizon = 24`
  - equivalent to old `120 min` forecast horizon under AGC `5 min` sampling
- `d_model = 64`
- `nhead = 4`
- `num_layers = 4`
- `dim_feedforward = 128`
- `dropout = 0.1`
- `batch_size = 256`
- `num_epochs = 200`
- `learning_rate = 1e-4`
- `lambda_trend = 0.3`
- `early_stop_patience = 15`

Design choice:

- this benchmark writes summary JSON by default
- it does not generate a large comparison figure automatically
- the intention is to lock the protocol first, then compare datasets with controlled variables

这是独立于 `diffmpc` 的新项目目录，用于基于 `Autonomous Greenhouse Challenge, Second Edition (2019)` 数据集重建预测控制主线。

当前已经具备：

- AGC 数据读取与字段标准化
- `sp/vip` 缺失回填
- `x_past / w_future / u_future / y_future` 样本构造
- 单隔间与多隔间联合训练
- leak-free 时间切分与全局标准化
- `GRU / DLinear / SegRNN / Transformer / Transformer-hybrid` 五个预测 baseline
- 自动保存 forecasting / control 结果到分层后的 `results` 目录

运行方式：

```bash
conda activate strawberry_env
python c:\repositories\strawberry\agc_mpc\main.py
```

结果输出：

- 模型权重：`agc_mpc/results/forecasting/checkpoints/*.pt`
- 预测示例图：`agc_mpc/results/forecasting/figures/*_forecast_examples.png`
- Horizon MAE 图：`agc_mpc/results/forecasting/figures/*_horizon_mae.png`
- Rolling multi-step forecast windows：`agc_mpc/results/forecasting/figures/*_forecast_rollout.png`
- First-step stitched rollout：`agc_mpc/results/forecasting/figures/*_forecast_first_step_rollout.png`
- Forecast error heatmap：`agc_mpc/results/forecasting/figures/*_forecast_error_heatmap.png`
- 控制闭环图：`agc_mpc/results/control/figures/*_closed_loop.png`
- 控制 summary：`agc_mpc/results/control/summaries/*_summary.json`

当前默认设置：

- 数据集：`AutonomousGreenhouseChallenge_edition2`
- 隔间：6 个 compartment 联合训练
- 历史窗口：`seq_len = 288`（24 小时）
- 预测窗口：`horizon = 24`（2 小时）
- 目标：`Tair / Rhair / CO2air / Tot_PAR`

当前结果概览：

- `DLinear` 当前整体最强，尤其是 `Tair / Rhair`
- `Transformer-hybrid` 在 `Tot_PAR` 最终步最好，也在 `CO2air` 上接近最优
- `GRU` 当前不再是整体最优，但仍然是重要的时序 baseline
- `SegRNN` 当前表现不如前三者，仍保留作结构化 RNN 对照

下一步主线：

1. 分析哪类预测器更适合后续控制
2. 把预测模型接入 AGC 上的 MPC
   当前区分的是两种 MPC 求解器：gradient-based MPC 和 CEM-based MPC
3. 再引入资源成本和经济指标
- forecasting 图现在会直接在图内标注 `R2 / MAE` 等关键指标
- control 图现在会同时展示状态跟踪、控制代价/动作偏移，以及归一化动作轨迹
- 文献与项目对照文档：`agc_mpc/LITERATURE_COMPARISON.md`
