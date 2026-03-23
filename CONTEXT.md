# CONTEXT.md

## 1. 使用方式

这是本项目的长期上下文文件。

建议规则：

1. 每次开启新对话时，优先读取本文件。
2. 如果使用支持文件上下文的 IDE / AI 助手，直接引用本文件。
3. 每次完成有意义的代码改动、实验结果更新、路线调整后，都要更新本文件。
4. 本文件优先记录稳定事实、当前主线、关键决策、最新结果、TODO 和工作规则。


## 2. 项目当前主线

当前目标不是复现原草莓论文，而是做：

**面向控制的温室多步预测 + 闭环 MPC / DPC**

核心设定：

- 使用多变量温室数据
- 输入历史室内状态
- 输入未来天气 / 外生量
- 输入未来控制设定值
- 预测未来室内温室状态
- 最终服务于 MPC / DPC
- SAC 仅作为 baseline，不是主线方法


## 3. 当前项目分工

### 3.1 旧项目

- [diffmpc](c:/repositories/strawberry/diffmpc)

说明：

- 这是旧草莓主线项目
- 不再作为新的主要落地方向
- 仅保留参考价值

### 3.2 新主项目

- [agc_mpc](c:/repositories/strawberry/agc_mpc)

说明：

- 这是新的 AGC 2019 主线项目
- 后续主要代码工作都应优先放在这里
- 原则上不要再把新开发继续堆回 `diffmpc`


## 4. 当前核心数据集

### 4.1 主数据集

- [AutonomousGreenhouseChallenge_edition2](c:/repositories/strawberry/AutonomousGreenhouseChallenge_edition2)

这是当前主建模数据源。

### 4.2 原始包 / 备份

- [Autonomous Greenhouse Challenge, Second Edition (2019)_1_all](c:/repositories/strawberry/Autonomous%20Greenhouse%20Challenge,%20Second%20Edition%20(2019)_1_all)

说明：

- 这是原始下载包及其解压后的备份结构
- 不作为主建模入口
- 仅在需要回查原始格式时使用

### 4.3 草莓数据

- [Strawberry Greenhouse Environmental Control Dataset(version2).csv](c:/repositories/strawberry/Strawberry%20Greenhouse%20Environmental%20Control%20Dataset(version2).csv)

说明：

- 现在降级为 secondary dataset / stress test
- 不再作为论文主实验数据集


## 5. AGC 数据理解

权威参考：

- [ReadMe.pdf](c:/repositories/strawberry/AutonomousGreenhouseChallenge_edition2/ReadMe.pdf)
- [Economics.pdf](c:/repositories/strawberry/AutonomousGreenhouseChallenge_edition2/Economics.pdf)
- [AGC_DATA_SCHEMA.md](c:/repositories/strawberry/AGC_DATA_SCHEMA.md)

关键结论：

- `Weather.csv` = 未来外生天气
- `GreenhouseClimate.csv` = 室内状态 + 执行器状态 + 设定值
- `*_sp` = 请求设定值
- `*_vip` = realized setpoints
- `Resources.csv` = 日尺度资源消耗
- `Production.csv` = 收获时点产量
- `TomQuality.csv` = 品质
- `GrodanSens.csv` = 根区 / 基质数据

当前第一版建模接口：

- `x_past`: 历史室内状态 + 执行器反馈
- `w_future`: 天气 + 时间特征
- `u_future`: 未来 setpoints
- `y_future`: 未来 `Tair / Rhair / CO2air / Tot_PAR`


## 6. 当前新工程代码状态

项目目录：

- [agc_mpc](c:/repositories/strawberry/agc_mpc)

核心文件：

- [config.py](c:/repositories/strawberry/agc_mpc/config.py)
- [schema.py](c:/repositories/strawberry/agc_mpc/schema.py)
- [processor.py](c:/repositories/strawberry/agc_mpc/data_processing/processor.py)
- [gru_forecaster.py](c:/repositories/strawberry/agc_mpc/models/gru_forecaster.py)
- [dlinear_forecaster.py](c:/repositories/strawberry/agc_mpc/models/dlinear_forecaster.py)
- [seg_rnn_forecaster.py](c:/repositories/strawberry/agc_mpc/models/seg_rnn_forecaster.py)
- [transformer_hybrid_forecaster.py](c:/repositories/strawberry/agc_mpc/models/transformer_hybrid_forecaster.py)
- [trainer.py](c:/repositories/strawberry/agc_mpc/training/trainer.py)
- [evaluator.py](c:/repositories/strawberry/agc_mpc/evaluation/evaluator.py)
- [main.py](c:/repositories/strawberry/agc_mpc/main.py)
- [README.md](c:/repositories/strawberry/agc_mpc/README.md)

当前已完成：

- AGC 数据读取
- 时间字段标准化
- 天气与气候表对齐
- `sp/vip` 缺失回填
- `x_past / w_future / u_future / y_future` 样本切片
- 单隔间和多隔间支持
- 全局 leak-free 时序切分
- 多隔间联合训练下的全局标准化
- 条件 GRU baseline
- 条件 DLinear baseline
- 条件 SegRNN baseline
- 条件 Transformer-hybrid baseline
- 离线评估输出
- 自动保存结果图到 `agc_mpc/results/figures`
- AGC 控制侧初版接入
- `DLinear / Transformer-hybrid` 已接到 AGC 上的 `DPC / MPC`
- 新增基于测试集真实天气/时间推进的 semi-grounded surrogate closed-loop rollout
- 控制结果自动保存到 `agc_mpc/results/control`

当前未完成：

- 完整物理级 / economic 级 AGC 闭环环境
- 更严格的 actuator / VIP / resource-aware AGC 控制建模
- 资源成本 / 经济指标纳入控制目标


## 7. 当前默认实验设置

来自 [config.py](c:/repositories/strawberry/agc_mpc/config.py)：

- 默认隔间：6 个全部联合训练
- `seq_len = 288`  
  含义：24 小时历史窗口
- `horizon = 24`  
  含义：2 小时预测窗口
- `batch_size = 256`
- `num_epochs = 12`
- `early_stop_patience = 4`

当前目标变量：

- `Tair`
- `Rhair`
- `CO2air`
- `Tot_PAR`


## 8. 最新基线结果

最新运行方式：

```bash
conda activate strawberry_env
python c:\repositories\strawberry\agc_mpc\main.py
```

最新数据规模：

- 6 个隔间联合训练
- `train = 199488`
- `val = 40878`
- `test = 40878`

### 8.1 GRU baseline

- `Tair`: Full `R²=0.9293`, MAE `0.886`; Final `R²=0.9136`, MAE `1.026`
- `Rhair`: Full `R²=0.8277`, MAE `3.996`; Final `R²=0.7424`, MAE `5.067`
- `CO2air`: Full `R²=0.7718`, MAE `55.797`; Final `R²=0.7092`, MAE `64.391`
- `Tot_PAR`: Full `R²=0.9688`, MAE `37.947`; Final `R²=0.9660`, MAE `39.784`

结果图：

- [gru_forecast_examples.png](c:/repositories/strawberry/agc_mpc/results/figures/gru_baseline_forecast_examples.png)
- [gru_horizon_mae.png](c:/repositories/strawberry/agc_mpc/results/figures/gru_baseline_horizon_mae.png)

### 8.2 DLinear baseline

- `Tair`: Full `R²=0.9639`, MAE `0.638`; Final `R²=0.9526`, MAE `0.729`
- `Rhair`: Full `R²=0.8607`, MAE `3.684`; Final `R²=0.8184`, MAE `4.209`
- `CO2air`: Full `R²=0.8205`, MAE `48.084`; Final `R²=0.7928`, MAE `51.481`
- `Tot_PAR`: Full `R²=0.9790`, MAE `30.483`; Final `R²=0.9779`, MAE `31.295`

结果图：

- [dlinear_forecast_examples.png](c:/repositories/strawberry/agc_mpc/results/figures/dlinear_baseline_forecast_examples.png)
- [dlinear_horizon_mae.png](c:/repositories/strawberry/agc_mpc/results/figures/dlinear_baseline_horizon_mae.png)

### 8.3 SegRNN baseline

- `Tair`: Full `R²=0.9228`, MAE `0.944`; Final `R²=0.9076`, MAE `1.069`
- `Rhair`: Full `R²=0.7512`, MAE `4.951`; Final `R²=0.6662`, MAE `5.922`
- `CO2air`: Full `R²=0.7856`, MAE `53.093`; Final `R²=0.7176`, MAE `62.168`
- `Tot_PAR`: Full `R²=0.9689`, MAE `38.705`; Final `R²=0.9672`, MAE `40.208`

结果图：

- [segrnn_forecast_examples.png](c:/repositories/strawberry/agc_mpc/results/figures/segrnn_baseline_forecast_examples.png)
- [segrnn_horizon_mae.png](c:/repositories/strawberry/agc_mpc/results/figures/segrnn_baseline_horizon_mae.png)

### 8.4 Transformer-hybrid baseline

- `Tair`: Full `R²=0.9517`, MAE `0.748`; Final `R²=0.9435`, MAE `0.836`
- `Rhair`: Full `R²=0.8070`, MAE `4.191`; Final `R²=0.7445`, MAE `4.899`
- `CO2air`: Full `R²=0.8097`, MAE `47.653`; Final `R²=0.7638`, MAE `53.823`
- `Tot_PAR`: Full `R²=0.9783`, MAE `30.745`; Final `R²=0.9801`, MAE `29.211`

结果图：

- [transformer_forecast_examples.png](c:/repositories/strawberry/agc_mpc/results/figures/transformer_hybrid_baseline_forecast_examples.png)
- [transformer_horizon_mae.png](c:/repositories/strawberry/agc_mpc/results/figures/transformer_hybrid_baseline_horizon_mae.png)

当前结论：

- `DLinear` 当前整体最强，尤其是 `Tair / Rhair`
- `Transformer-hybrid` 在 `Tot_PAR` 最终步最好，在 `CO2air` 上接近 `DLinear`
- `GRU` 当前不再是整体最优，但仍然是重要的时序 baseline
- `SegRNN` 当前未超过前三者
- 这初步支持一个重要论文论点：  
  **最好的离线预测模型可能因目标变量不同而分化，不存在单一绝对最优结构**

### 8.5 初版控制侧 benchmark（2026-03-23）

运行方式：

```bash
conda activate strawberry_env
python c:\repositories\strawberry\agc_mpc\control_main.py --steps 12 --start-idx 0 --reference-mode trajectory
```

协议说明：

- 控制隔间：`Reference`
- 控制器：`recorded` / `DPC` / `MPC(CEM)`
- 预测器：`DLinear` 与 `Transformer-hybrid`
- 参考目标：测试集真实未来 `y_future` trajectory
- 当前闭环协议不是完整物理仿真器，而是：
  - 天气、时间和未建模列继续来自 AGC 测试集真实序列
  - 被控目标状态由预测器一步滚动产生
  - 控制相关历史反馈由真实下一行打底，再按执行 setpoints 做启发式覆盖

结果：

#### DLinear as control surrogate

- `recorded`: `Tair=0.938`, `Rhair=1.394`, `CO2air=43.372`, `Tot_PAR=64.345`
- `DPC`: `Tair=0.317`, `Rhair=0.927`, `CO2air=5.443`, `Tot_PAR=57.345`
- `MPC`: `Tair=3.022`, `Rhair=2.768`, `CO2air=59.089`, `Tot_PAR=69.307`

结论：

- 在当前 12-step surrogate rollout 上，`DLinear + DPC` 明显优于 logged control 和当前 `MPC(CEM)`
- 当前 `MPC(CEM)` 还不稳，需要继续调参或改搜索策略

#### Transformer-hybrid as control surrogate

- `recorded`: `Tair=0.235`, `Rhair=0.823`, `CO2air=5.831`, `Tot_PAR=65.404`
- `DPC`: `Tair=0.247`, `Rhair=1.139`, `CO2air=12.814`, `Tot_PAR=53.317`
- `MPC`: `Tair=0.384`, `Rhair=1.021`, `CO2air=8.580`, `Tot_PAR=57.377`

结论：

- `Transformer-hybrid` 在当前 surrogate rollout 中对 `Tot_PAR` 改善最明显
- 但在 `Tair / CO2air` 上，当前配置下未整体超过 logged control
- 这进一步提示：**最强离线预测器不一定自动变成最强闭环控制 surrogate**


## 9. 当前耗时经验值

在当前机器和当前配置下：

- 单隔间 GRU baseline：约 `20 秒`
- 6 隔间联合 `GRU + DLinear + SegRNN`：约 `136 秒`
- 6 隔间联合 `GRU + DLinear + SegRNN + Transformer-hybrid`：约 `541 秒`

粗略估计：

- 轻量 baseline：`2 分钟内`
- 中等规模 GRU / SegRNN：`2~5 分钟`
- Transformer / hybrid：`6~10 分钟`


## 10. 当前论文定位

当前最优定位不是：

- “改进 Transformer 做温室预测”

更合理的定位是：

- “面向控制的温室多步预测”
- “利用未来天气与未来控制信息的闭环预测控制框架”
- “预测模型与控制性能之间关系的系统 benchmark”


## 11. 当前创新点判断

### 可以成立的创新点

- 面向控制的多步预测，而不是纯离线拟合
- 显式利用未来天气和未来控制
- 严格闭环评估
- 预测模型与控制结果之间差异的系统分析
- 多变量耦合建模

### 不能单独作为强创新点的内容

- “用了 Transformer”
- “多参数耦合”
- “用了 SAC baseline”
- “做了 MPC”

这些只能作为背景或组成部分，不能单独撑起论文主创新。


## 12. 当前优先级

### 第一优先级

先稳住控制 benchmark：

- 调整 `MPC(CEM)` 搜索稳定性
- 验证 `DLinear` 与 `Transformer-hybrid` 在更长 rollout 下的闭环排名
- 逐步把 `sp -> actuator feedback -> climate` 的 surrogate 更新做实

第二层继续补强预测 benchmark：

- 可能的 `hybrid residual model`

### 第二优先级

把 AGC 主线接到控制层：

- 从当前 surrogate closed-loop 继续推进到更严格的 AGC 闭环环境
- 再看 SAC on AGC

### 第三优先级

把资源指标纳入：

- `Heat_cons`
- `ElecHigh`
- `ElecLow`
- `CO2_cons`
- `Irr`

向 economic MPC 延伸。


## 13. 当前工作规则

1. 新开发优先放在 [agc_mpc](c:/repositories/strawberry/agc_mpc)。
2. 除非有明确需要，不要继续把主工作流堆回 `diffmpc`。
3. 代码默认运行环境是 `strawberry_env`。
4. 每次做完关键代码改动、实验结果更新或路线变化后，都要更新本文件。
5. 任何新模型都要同时回答四个问题：
   - 离线预测是否提升
   - 闭环控制是否提升
   - 对 forecast error 是否稳健
   - 是否能解释为面向控制的设计


## 14. 下次对话建议起手内容

建议先说明：

- 当前主项目目录：`agc_mpc`
- 当前主数据集：`AutonomousGreenhouseChallenge_edition2`
- 当前已完成：数据管线 + GRU baseline + DLinear baseline
- 当前已完成：数据管线 + GRU baseline + DLinear baseline + SegRNN baseline + Transformer-hybrid baseline + 自动结果图
- 当前已完成：`DLinear / Transformer-hybrid` 已接入 AGC 上的 `DPC / MPC` 初版 surrogate closed-loop benchmark
- 当前下一步：稳住 `MPC(CEM)`、拉长闭环评估窗口，并把 surrogate rollout 逐步替换为更严格的 AGC 控制环境；或并行开始做一个 `hybrid residual model`
