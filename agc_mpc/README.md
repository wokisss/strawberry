# AGC MPC

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
