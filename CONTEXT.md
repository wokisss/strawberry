# CONTEXT.md

## 0. 对话与表达规则

- 默认把回答对象视为“从零开始了解项目的人”。
- 解释概念时优先使用中文。
- 除非必须，不要中英夹杂。
- 如果必须使用英文术语，要立刻在后面补中文释义，例如“平均绝对误差（Mean Absolute Error，MAE）”。
- 讲图、讲指标、讲模型时，先说“它是什么”，再说“它说明了什么”，最后再说“怎么解读”。
- 涉及容易混淆的术语时，优先给出直白解释，不默认假设读者已有机器学习或控制背景。
- forecasting 默认结果图只保留三类：
  - `forecast_examples`：单次预测样例图
  - `forecast_rollout`：滚动多窗预测图
  - `horizon_mae`：预测步长误差图
- `forecast_error_heatmap` 已移除，不再作为默认结果图输出。
- `forecast_first_step_rollout` 也已移除，不再作为默认结果图输出。

## 1. 使用方式

这是本项目的长期上下文文件。

建议规则：

1. 每次开启新对话时，优先读取本文件。
2. 如果使用支持文件上下文的 IDE / AI 助手，直接引用本文件。
3. 每次完成有意义的代码改动、实验结果更新、路线调整后，都要更新本文件。
4. 本文件优先记录稳定事实、当前主线、关键决策、最新结果、TODO 和工作规则。


## 2. 项目当前主线

当前目标不是复现原草莓论文，而是做：

**面向控制的温室多步预测 + 闭环 MPC**

核心设定：

- 使用多变量温室数据
- 输入历史室内状态
- 输入未来天气 / 外生量
- 输入未来控制设定值
- 预测未来室内温室状态
- 最终服务于 MPC
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
- [transformer_forecaster.py](c:/repositories/strawberry/agc_mpc/models/transformer_forecaster.py)
- [transformer_hybrid_forecaster.py](c:/repositories/strawberry/agc_mpc/models/transformer_hybrid_forecaster.py)
- [hybrid_residual_forecaster.py](c:/repositories/strawberry/agc_mpc/models/hybrid_residual_forecaster.py)
- [trainer.py](c:/repositories/strawberry/agc_mpc/training/trainer.py)
- [evaluator.py](c:/repositories/strawberry/agc_mpc/evaluation/evaluator.py)
- [main.py](c:/repositories/strawberry/agc_mpc/main.py)
- [benchmark_hybrid_residual_forecaster.py](c:/repositories/strawberry/agc_mpc/benchmark_hybrid_residual_forecaster.py)
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
- 条件纯 Transformer baseline
- 条件 Transformer-hybrid baseline
- `DLinear main path + Transformer-hybrid residual` 混合残差模型原型
- 离线评估输出
- forecast 图支持“历史上下文 + 未来 horizon”联合展示，不再只盯着纯 future window
- forecast 图新增 rolling multi-step rollout 展示，用更长时间轴显示连续多窗预测，而不只是一段 24-step future window
- forecasting 现在默认保留 3 类预测图：单次预测样例图、滚动多窗预测图、预测步长误差图
- 上述 3 类长时间轴图已经为 `GRU / DLinear / SegRNN / Transformer / Transformer-hybrid` 全部补齐
- `results` 目录开始按 `forecasting / control` 分层整理
- forecasting checkpoint 统一收敛到 `agc_mpc/results/forecasting/checkpoints`
- forecasting 图统一收敛到 `agc_mpc/results/forecasting/figures`
- control summary 统一收敛到 `agc_mpc/results/control/summaries`
- AGC 控制侧初版接入
- `DLinear / Transformer-hybrid` 已接到 AGC 上的两类 MPC 求解器
- `CEMMPC` 已补上固定随机种子、warm start、candidate injection 和更平滑的 CEM 更新
- 闭环 rollout 默认切到更严格的 `surrogate` 模式，不再默认用真实下一行状态打底
- surrogate 状态更新里会重算 `HumDef`，并用 persistence + action proxy 更新非目标状态
- 控制结果自动保存到 `agc_mpc/results/control`
- 已新增 `benchmark_hybrid_residual_forecaster.py`，用于在公平训练预算下单独评估混合残差模型

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
- 这意味着“单个 forecast 窗口图”天然只会显示 24 个未来步；如果想看更长时间轴，需要看 rolling forecast rollout 图，或直接把 `horizon` 改大后重训
- `batch_size = 256`
- `num_epochs = 12`
- `early_stop_patience = 4`
- `control_eval_steps = 96`
- `control_rollout_mode = surrogate`

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

- [gru_forecast_examples.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/gru_baseline_forecast_examples.png)
- [gru_horizon_mae.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/gru_baseline_horizon_mae.png)

### 8.2 DLinear baseline

- `Tair`: Full `R²=0.9639`, MAE `0.638`; Final `R²=0.9526`, MAE `0.729`
- `Rhair`: Full `R²=0.8607`, MAE `3.684`; Final `R²=0.8184`, MAE `4.209`
- `CO2air`: Full `R²=0.8205`, MAE `48.084`; Final `R²=0.7928`, MAE `51.481`
- `Tot_PAR`: Full `R²=0.9790`, MAE `30.483`; Final `R²=0.9779`, MAE `31.295`

结果图：

- [dlinear_forecast_examples.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/dlinear_baseline_forecast_examples.png)
- [dlinear_horizon_mae.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/dlinear_baseline_horizon_mae.png)

### 8.3 SegRNN baseline

- `Tair`: Full `R²=0.9228`, MAE `0.944`; Final `R²=0.9076`, MAE `1.069`
- `Rhair`: Full `R²=0.7512`, MAE `4.951`; Final `R²=0.6662`, MAE `5.922`
- `CO2air`: Full `R²=0.7856`, MAE `53.093`; Final `R²=0.7176`, MAE `62.168`
- `Tot_PAR`: Full `R²=0.9689`, MAE `38.705`; Final `R²=0.9672`, MAE `40.208`

结果图：

- [segrnn_forecast_examples.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/segrnn_baseline_forecast_examples.png)
- [segrnn_horizon_mae.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/segrnn_baseline_horizon_mae.png)

### 8.4 纯 Transformer baseline

- `Tair`: Full `R²=0.9483`, MAE `0.765`; Final `R²=0.9413`, MAE `0.823`
- `Rhair`: Full `R²=0.8038`, MAE `4.249`; Final `R²=0.7454`, MAE `4.919`
- `CO2air`: Full `R²=0.8509`, MAE `43.206`; Final `R²=0.8242`, MAE `47.229`
- `Tot_PAR`: Full `R²=0.9853`, MAE `26.484`; Final `R²=0.9859`, MAE `24.964`

结果图：

- [transformer_forecast_examples.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/transformer_baseline_forecast_examples.png)
- [transformer_horizon_mae.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/transformer_baseline_horizon_mae.png)

### 8.5 Transformer-hybrid baseline

- `Tair`: Full `R²=0.9544`, MAE `0.708`; Final `R²=0.9480`, MAE `0.770`
- `Rhair`: Full `R²=0.7539`, MAE `4.650`; Final `R²=0.6927`, MAE `5.306`
- `CO2air`: Full `R²=0.7870`, MAE `51.905`; Final `R²=0.7434`, MAE `58.318`
- `Tot_PAR`: Full `R²=0.9848`, MAE `28.237`; Final `R²=0.9846`, MAE `28.509`

结果图：

- [transformer_hybrid_forecast_examples.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/transformer_hybrid_baseline_forecast_examples.png)
- [transformer_hybrid_horizon_mae.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/transformer_hybrid_baseline_horizon_mae.png)

当前离线结论：

- `DLinear` 仍然是 `Tair / Rhair` 上最稳的整体 baseline
- 纯 `Transformer` 在当前设置下对 `CO2air / Tot_PAR` 最强，且整体强于当前 `Transformer-hybrid`
- `Transformer-hybrid` 仍保留结构价值，但当前实现没有在所有目标上超过纯 Transformer
- `GRU` 当前不再是整体最优，但仍然是重要的时序 baseline
- `SegRNN` 当前未超过前三者
- 这继续支持一个重要论文论点：  
  **最好的离线预测模型可能因目标变量不同而分化，不存在单一绝对最优结构**

### 8.6 控制侧 benchmark（2026-03-23, stricter surrogate update）

运行方式：

```bash
conda activate strawberry_env
python c:\repositories\strawberry\agc_mpc\control_main.py --steps 48 --start-idx 0 --reference-mode trajectory
```

协议说明：

- 控制隔间：`Reference`
- 控制器：`recorded` / `GradientMPC` / `CEMMPC`
- 预测器：`DLinear`、纯 `Transformer`、`Transformer-hybrid`
- 参考目标：测试集真实未来 `y_future` trajectory
- 当前闭环协议仍不是完整物理仿真器，但比上一版更严格：
  - 天气、时间和参考轨迹继续来自 AGC 测试集真实序列
  - 被控目标状态由预测器一步滚动产生
  - 历史状态默认不再直接拷贝真实下一行，而是从当前状态出发，用 persistence + action proxy + predicted targets 更新
  - `HumDef` 根据预测的 `Tair / Rhair` 重新计算
  - `CEMMPC` 现在使用固定随机种子，因此同一命令重跑时 summary 哈希保持一致

结果：

术语说明：

- 这里原来写作 `DPC` 的方法，现在统一记为 `GradientMPC`
- 它不是独立于 MPC 的另一类控制范式，而是“通过梯度直接求解滚动时域优化问题的 MPC 求解器”
- 原来写作 `MPC(CEM)` 的方法，现在统一记为 `CEMMPC`
- 因此当前控制对比更准确地说是：`GradientMPC vs CEMMPC`

#### DLinear as control surrogate

- `recorded`: `Tair=0.527`, `Rhair=4.533`, `CO2air=66.691`, `Tot_PAR=36.827`
- `GradientMPC`: `Tair=0.167`, `Rhair=0.458`, `CO2air=2.592`, `Tot_PAR=11.558`
- `CEMMPC`: `Tair=0.303`, `Rhair=1.237`, `CO2air=11.970`, `Tot_PAR=12.647`

#### Pure Transformer as control surrogate

- `recorded`: `Tair=1.482`, `Rhair=4.004`, `CO2air=29.998`, `Tot_PAR=20.712`
- `GradientMPC`: `Tair=0.251`, `Rhair=2.818`, `CO2air=15.884`, `Tot_PAR=18.789`
- `CEMMPC`: `Tair=0.423`, `Rhair=3.800`, `CO2air=21.208`, `Tot_PAR=24.774`

#### Transformer-hybrid as control surrogate

- `recorded`: `Tair=2.253`, `Rhair=3.134`, `CO2air=28.444`, `Tot_PAR=26.593`
- `GradientMPC`: `Tair=0.194`, `Rhair=1.861`, `CO2air=18.029`, `Tot_PAR=12.909`
- `CEMMPC`: `Tair=0.642`, `Rhair=4.026`, `CO2air=20.468`, `Tot_PAR=12.491`

当前控制结论：

- 在更严格的 48-step surrogate rollout 上，`GradientMPC` 仍普遍优于 `CEMMPC`
- `DLinear + GradientMPC` 是当前最强的严格 surrogate 控制组合，四个目标都显著优于 recorded
- `CEMMPC` 现在已经可复现，同一命令重复运行时其 summary 哈希保持一致，但性能仍落后于 `GradientMPC`
- surrogate 协议一旦收紧，recorded control 和各 predictor 的误差都会明显变大，这说明上一版 semi-grounded rollout 确实偏乐观
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

- 已完成：`CEMMPC` 的可复现性和基础稳定性
- 正在做：验证 `DLinear / Transformer / Transformer-hybrid` 在更长 rollout 下的闭环排名
- 下一步：逐步把 `sp -> actuator feedback -> climate` 的 surrogate 更新做实

第二层继续补强预测 benchmark：

- 已启动 `hybrid residual model`
- 下一步是给 `hybrid residual model` 跑正式预算，并与 `DLinear / Transformer / current hybrid-transformer` 做统一口径对比

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
5. 当前控制术语约定：
   - `GradientMPC` = 通过梯度直接求解滚动时域优化问题的 MPC 求解器
   - `CEMMPC` = 通过 CEM 采样搜索求解同一 MPC 目标的 MPC 求解器
   - 不再把 `DPC` 和 `MPC` 记成两个平级范式，以免术语混淆
6. 任何新模型都要同时回答四个问题：
   - 离线预测是否提升
   - 闭环控制是否提升
   - 对 forecast error 是否稳健
   - 是否能解释为面向控制的设计
7. Git 提交默认采用“小步分段提交”，不要把结果目录重构、模型新增、控制实验结果、文档更新一次性混成一个大提交。
8. 当前仓库在本机上曾出现 `.git` ACL / `index.lock` 写入受限问题；如果 `git add` / `git commit` 报 `Unable to create .git/index.lock: Permission denied`：
   - 不要反复重试很多次
   - 先检查 `.git` 的 ACL
   - 必要时一次性递归移除 `.git` 下针对当前用户的 `DENY` ACL 后再继续提交
9. 推荐的提交拆分顺序：
   - 先提结果目录结构 / plotting / 基础设施
   - 再提新模型与 forecasting 结果
   - 最后提 control benchmark、结果图 / summary 和 `CONTEXT.md`
10. 如果后续 push 因 pack 过大或二进制结果过多失败，优先考虑继续拆提交，必要时把“代码变更”和“实验产物”分开处理，而不是无限重试 push。
11. 当前环境下，`Remove-Item` 一类删除动作也可能被沙箱拦住并报 `Access is denied`，即使文件 ACL 看起来正常；如果需要清理 legacy 结果文件：
   - 先区分是沙箱/提权限制还是文件自身 ACL 问题，不要默认是文件损坏
   - 优先用“精确过滤 + 提权删除”，不要用会误伤新文件的宽泛通配
   - 例如清理旧控制结果时，只删除旧命名的 `_dpc_` 和旧 `_mpc_` 文件，不要匹配到 `gradient_mpc` / `cem_mpc`


## 14. 下次对话建议起手内容

建议先说明：

- 当前主项目目录：`agc_mpc`
- 当前主数据集：`AutonomousGreenhouseChallenge_edition2`
- 当前已完成：数据管线 + GRU baseline + DLinear baseline
- 当前已完成：数据管线 + GRU baseline + DLinear baseline + SegRNN baseline + Transformer baseline + Transformer-hybrid baseline + hybrid residual 原型 + 自动结果图
- 当前已完成：`DLinear / Transformer / Transformer-hybrid` 已接入 AGC 上的 `GradientMPC / CEMMPC` 初版 surrogate closed-loop benchmark
- 当前已完成：forecast 侧新增 rolling multi-step rollout 图；control 侧默认切到更严格的 `surrogate` rollout，并验证了 `CEMMPC` 的可复现性
- 当前下一步：控制侧继续把 surrogate 从 `state persistence + action proxy` 推到更接近 `sp -> vip -> actuator -> climate` 的层级建模；预测侧把 `hybrid residual model` 跑成正式预算并做统一对比
## 15. Strawberry vs AGC 对比图

- 已新增导师展示用脚本：[compare_dataset_switch.py](c:/repositories/strawberry/agc_mpc/compare_dataset_switch.py)
- 运行方式：
  ```bash
  conda activate strawberry_env
  python c:\repositories\strawberry\agc_mpc\compare_dataset_switch.py
  ```
- 输出文件：
  - [strawberry_vs_agc_dataset_switch.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/strawberry_vs_agc_dataset_switch.png)
  - [strawberry_vs_agc_dataset_switch_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/strawberry_vs_agc_dataset_switch_summary.json)
- 图的比较口径：
  - 只比较共同变量 `Temperature / Humidity / CO2`
  - 只比较 final-step 指标
  - 两边都按“2 小时预测任务”对齐：旧 Strawberry = `120 x 1 min`，AGC = `24 x 5 min`
- 当前结论：
  - 旧 Strawberry Transformer-hybrid 的 final MAE 为 `3.36 / 6.78 / 105.88`
  - AGC `DLinear` 的 final MAE 为 `0.76 / 4.46 / 54.73`
  - AGC `Transformer` 的 final MAE 为 `0.82 / 4.92 / 47.23`
  - AGC `Transformer-hybrid` 的 final MAE 为 `0.77 / 5.31 / 58.32`
  - 旧 Strawberry 在 `CO2` 上 final `R2` 只有 `0.073`；AGC 三个模型对应为 `0.776 / 0.824 / 0.743`
- 对导师的推荐表述：
  - 这张图不证明 “AGC 已经做到理想极限”
  - 它证明的是：在当前 baseline-first 实现下，AGC 已经能提供更稳定、更可控、对闭环更友好的预测基座
- 因此切换数据集的主要理由应表述为“任务匹配度更高 + 结果更稳 + 能自然扩展到闭环控制”，而不只是“旧数据集分数差”
- 已新增代表性预测窗对比图：[strawberry_vs_agc_forecast_windows.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/strawberry_vs_agc_forecast_windows.png)
- 该图只展示 `Strawberry / old Transformer-hybrid`、`AGC / Transformer`、`AGC / Transformer-hybrid`
- 该图使用两边测试集各自的 midpoint sample，不做样本对齐，不用于严格统计比较，只用于给导师做“预测轨迹形态”的直观说明
- forecasting 图已升级为“图内直接显示指标”：
- `forecast_examples / rollout / horizon_mae` 现在都会直接关联当前模型的 `Full R2 / Full MAE / Final R2 / Final MAE`
  - `horizon_mae` 图会在图下方汇总全部目标的指标
- control 图已升级为“状态 + 指标 + 动作”联合展示：
  - 前四行仍是 `Tair / Rhair / CO2air / Tot_PAR`
  - 第五行显示 `objective / |u-u_log| / action_tv`
  - 第六行显示所有控制量的归一化动作轨迹，实线是 executed，虚线是 logged baseline
- 已新增文献对照文档：[LITERATURE_COMPARISON.md](c:/repositories/strawberry/agc_mpc/LITERATURE_COMPARISON.md)
- 文献对照文档的定位：
  - 不做伪 leaderboard
  - 按任务、输入、输出、horizon、模型、控制设定、结果和可比性分开写
  - 当前结论是：AGC 结果还不是 final-paper quality，但已处于可辩护的 literature band 内；真正短板在 `Rhair`、uncertainty、economic objective 和更完整闭环
- 已新增表格式近期论文综述文档：[RECENT_PAPERS_SURVEY.md](c:/repositories/strawberry/agc_mpc/RECENT_PAPERS_SURVEY.md)
- 该文档按“论文 / 任务 / 主模型 / 对比-baseline / 启发 / 链接”组织，分为：
  - 温室预测论文
  - 温室控制论文
  - 通用时序模型参考
- 用途：
  - 快速回答“最近相似论文都用了什么模型、baseline 和对比对象是什么”
  - 为后续新模型路线提供文献锚点，避免反复手工整理
- 已在该文档中补充 `Mao et al., 2024` 的重点详解小节，专门回答：
  - 为什么该文 `PSO-BiGRU-Attention-LightGBM` 的 `R2` 很高
  - 它和当前 `AGC` 数据集到底有多相似
  - 它是否可以被严格复现，哪些部分只能做方法级复现
- `README.md` 已补充数据集背景与训练设定说明：
  - 明确 `AGC` 更准确是 multi-compartment benchmark，而不是 fully independent multi-greenhouse dataset
  - 补充当前 `x_past / w_future / u_future / y_future` 的控制导向接口说明
  - 补充 single-compartment training 与 multi-compartment joint training 的取舍，当前默认仍以 joint training 为主
- 已新增训练策略对照脚本：[compare_training_regimes.py](c:/repositories/strawberry/agc_mpc/compare_training_regimes.py)
- 该脚本支持围绕一个目标隔间比较三种 regime：
  - `single`: 只在目标隔间上训练并在该隔间测试
  - `joint_all`: 在全部隔间上训练，但只在目标隔间测试
  - `leave_one_out`: 在除目标隔间外的其余隔间上训练，再在目标隔间测试
- 数据管线已新增自定义 bundle 组装能力，可按 train/eval compartments 自由拼接并仅用训练集拟合 scaler
- 结果统一落到：`agc_mpc/results/forecasting/analysis`
- 已做 1-epoch smoke test（目标隔间 `Reference`，模型 `DLinear`）：
  - `single`：`Tair/Rhair/CO2air/Tot_PAR` Final MAE = `0.772 / 4.815 / 93.219 / 53.889`
  - `joint_all`：`0.776 / 3.798 / 53.866 / 32.658`
  - `leave_one_out`：`0.671 / 5.469 / 56.336 / 38.663`
- 初步信号：
  - joint training 对 `Rhair / CO2air / Tot_PAR` 明显更有帮助
  - leave-one-out 在 `Reference` 的 `Tair` 上很强，但对湿度和 CO2 不占优
  - 单隔间训练并不天然更好，至少在当前 `Reference + DLinear` 的 smoke test 上不是
- 已新增 `diffmpc` 风格 Transformer 迁移基准脚本：[benchmark_diffmpc_style_transformer.py](c:/repositories/strawberry/agc_mpc/benchmark_diffmpc_style_transformer.py)
- 该脚本的目的不是追当前 `agc_mpc` 最强分数，而是控制变量地回答：
  - 在尽量保留旧 `diffmpc` Transformer-hybrid 架构与训练预算时，`AGC` 是否比旧 Strawberry 更适合作为 Transformer 的数据基座
- 固定协议：
  - targets = `Tair / Rhair / CO2air`
  - `seq_len = 48`（对应旧项目 `240 min` 历史）
  - `horizon = 24`（对应旧项目 `120 min` 预测窗）
  - `d_model = 64`, `nhead = 4`, `num_layers = 4`, `ff_dim = 128`, `dropout = 0.1`
  - `batch_size = 256`, `num_epochs = 200`, `lr = 1e-4`, `lambda_trend = 0.3`, `patience = 15`
- 设计原则：
  - 默认只落 summary JSON，不自动生成大图
  - 先把“模型结构/训练预算/时间口径”对齐，再谈数据集是否更适合 Transformer
- 已做 1-epoch smoke test（`single + Reference`）并成功落盘：
  - [diffmpc_style_transformer_single_reference_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/diffmpc_style_transformer_single_reference_summary.json)
  - 当前仅用于验证入口与协议，不用于正式结论
- 该基准现已完成 `Reference` 上的正式三组运行：
  - [diffmpc_style_transformer_single_reference_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/diffmpc_style_transformer_single_reference_summary.json)
  - [diffmpc_style_transformer_joint_all_reference_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/diffmpc_style_transformer_joint_all_reference_summary.json)
  - [diffmpc_style_transformer_leave_one_out_reference_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/diffmpc_style_transformer_leave_one_out_reference_summary.json)
- `diffmpc` 风格 Transformer 在 AGC / `Reference` 上的最终指标：
  - `single`
    - `Tair`: Final `R2=0.5198`, MAE `2.023`
    - `Rhair`: Final `R2=0.6850`, MAE `6.874`
    - `CO2air`: Final `R2=0.3543`, MAE `75.857`
  - `joint_all`
    - `Tair`: Final `R2=0.8007`, MAE `1.358`
    - `Rhair`: Final `R2=0.6470`, MAE `7.891`
    - `CO2air`: Final `R2=0.3899`, MAE `72.867`
  - `leave_one_out`
    - `Tair`: Final `R2=0.8859`, MAE `0.926`
    - `Rhair`: Final `R2=0.5763`, MAE `8.169`
    - `CO2air`: Final `R2=0.3140`, MAE `78.422`
- 当前读法：
  - 旧 `diffmpc` 风格结构迁到 AGC 后，`Tair / CO2air` 明显好于旧 Strawberry 上的旧 Transformer-hybrid 结果，说明数据集切换确实帮助了这类 conditional Transformer
  - 但 `Rhair` 没有同步变成强项，说明“数据集更适合 Transformer”不等于“旧结构无需改造就会全面变强”
  - 三种 AGC 训练 regime 没有单一绝对最优：
    - `single` 在 `Rhair` 上最好
    - `joint_all` 在 `CO2air` 上最好
    - `leave_one_out` 在 `Tair` 上最好
  - 因此对导师更稳的表述应是：
    - AGC 给旧 Transformer 风格提供了更合理的数据接口和更高的上限空间
    - 但真正把该架构做强，仍然需要进一步面向 AGC/控制任务改造，而不是直接照搬旧结构
- 已新增直观对比图：[diffmpc_style_transformer_dataset_suitability.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/diffmpc_style_transformer_dataset_suitability.png)
- 该图只比较：
  - `Strawberry / old Transformer-hybrid`
  - `AGC / diffmpc-style / single`
  - `AGC / diffmpc-style / joint_all`
  - `AGC / diffmpc-style / leave_one_out`
- 该图的定位：
  - 用于直观展示“尽量相似的 Transformer 风格与训练预算”下，换到 AGC 后 `Temperature / Humidity / CO2` 的 final MAE 与 final R2 如何变化
  - 不混入当前 `agc_mpc` 的 `DLinear / Transformer / Transformer-hybrid` 新 baseline，避免论证口径漂移
- 已新增更适合汇报的两张中文图：
  - [diffmpc_style_transformer_best_vs_old_line_cn.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/diffmpc_style_transformer_best_vs_old_line_cn.png)
  - [diffmpc_style_transformer_old_vs_agc_joint_all_windows_cn.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/diffmpc_style_transformer_old_vs_agc_joint_all_windows_cn.png)
- 两图定位：
  - `best_vs_old_line_cn`：只看 `old Strawberry` vs `AGC joint_all`，用中文折线图展示 `Temperature / Humidity / CO2` 的 final MAE 与 final R2
  - `old_vs_agc_joint_all_windows_cn`：并排展示旧 Strawberry 与 AGC joint_all 的代表性预测窗，让导师直接看轨迹贴合与偏移方式
- 已新增“旧数据集旧 hybrid-transformer vs 新数据集新 hybrid-transformer”的公平预算对照主线：
  - 旧侧：`diffmpc` 原始 `TransformerHybridModel`
  - 新侧：`agc_mpc` 当前 `ConditionalTransformerHybridForecaster`
  - 共同口径：只看 `Tair / Rhair / CO2air`，统一按 `2h` 预测任务讨论
  - 旧侧保留旧项目方法与架构；新侧保留 AGC 当前 `x_past / w_future / u_future -> y_future` 的 control-oriented 接口
- 已新增脚本：[benchmark_current_hybrid_transformer.py](c:/repositories/strawberry/agc_mpc/benchmark_current_hybrid_transformer.py)
  - 目的：给 AGC 当前 hybrid-transformer 一个比 12 epoch baseline 更公平的训练预算，再与旧 Strawberry 的 old hybrid-transformer 做对比
  - 当前正式跑通的配置为：`joint_all + Reference`
  - 训练预算：`batch_size=256`, `num_epochs=200`, `lr=1e-4`, `lambda_trend=0.3`, `patience=15`
  - 模型参数：`hidden_dim=96`, `nhead=4`, `num_layers=2`, `ff_dim=192`, `dropout=0.1`
- 当前正式结果文件：
  - [current_hybrid_transformer_joint_all_reference_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/current_hybrid_transformer_joint_all_reference_summary.json)
  - [current_hybrid_transformer_joint_all_reference.pt](c:/repositories/strawberry/agc_mpc/results/forecasting/checkpoints/current_hybrid_transformer_joint_all_reference.pt)
- `AGC + current hybrid-transformer + joint_all + Reference` 正式结果：
  - `Tair`: Full `R2=0.9344`, MAE `0.630`; Final `R2=0.9318`, MAE `0.651`
  - `Rhair`: Full `R2=0.8951`, MAE `3.698`; Final `R2=0.8553`, MAE `4.403`
  - `CO2air`: Full `R2=0.8184`, MAE `41.201`; Final `R2=0.7860`, MAE `44.567`
- 与 `AGC + diffmpc-style hybrid-transformer + joint_all + Reference` 的直接对比：
  - `Tair`: Final MAE `1.358 -> 0.651`, Final `R2 0.8007 -> 0.9318`
  - `Rhair`: Final MAE `7.891 -> 4.403`, Final `R2 0.6470 -> 0.8553`
  - `CO2air`: Final MAE `72.867 -> 44.567`, Final `R2 0.3899 -> 0.7860`
- 当前更稳的表述应改为：
  - 不是“AGC 自动让旧 Transformer 变强”
  - 而是“AGC 更适合当前这套面向控制的 hybrid-transformer 接口与训练范式”
  - 旧 Strawberry + old hybrid-transformer 与 AGC + current hybrid-transformer 的对比，才更能支持‘换数据集 + 换方法是合理主线’这一结论
- 已新增 `AGC + current hybrid-transformer + joint_all + Reference + horizon=120` 正式实验：
  - [current_hybrid_transformer_h120_joint_all_reference_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/current_hybrid_transformer_h120_joint_all_reference_summary.json)
  - [current_hybrid_transformer_h120_joint_all_reference.pt](c:/repositories/strawberry/agc_mpc/results/forecasting/checkpoints/current_hybrid_transformer_h120_joint_all_reference.pt)
  - 注意：这里的 `120-step` 指 `120 x 5min = 600 min`，不再等价于旧 Strawberry 的 `120 x 1min = 120 min`
- `AGC current hybrid-transformer` 在 `horizon=120` 下的正式结果：
  - `Tair`: Full `R2=0.9204`, MAE `0.764`; Final `R2=0.9153`, MAE `0.820`
  - `Rhair`: Full `R2=0.7302`, MAE `6.705`; Final `R2=0.7149`, MAE `6.875`
  - `CO2air`: Full `R2=0.5754`, MAE `63.666`; Final `R2=0.5573`, MAE `65.198`
- 与当前 `horizon=24` 对比的读法：
  - `Tair`: Final MAE `0.651 -> 0.820`
  - `Rhair`: Final MAE `4.403 -> 6.875`
  - `CO2air`: Final MAE `44.567 -> 65.198`
  - 说明：把 AGC 任务从 `2h` 拉到 `10h` 后，性能明显下降，但 `Tair` 仍保持较强；`Rhair / CO2air` 更容易随 horizon 拉长而退化
- 已新增两张中文 horizon 对比图：
  - [current_hybrid_transformer_h24_vs_h120_metrics_cn.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/current_hybrid_transformer_h24_vs_h120_metrics_cn.png)
  - [current_hybrid_transformer_h24_vs_h120_windows_cn.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/current_hybrid_transformer_h24_vs_h120_windows_cn.png)
  - 用途：把 `24-step (120 min)` 和 `120-step (600 min)` 放在同一页上，看指标和轨迹如何随 horizon 拉长而退化
- 已新增更符合当前主线的两张中文汇报图：
  - [current_hybrid_transformer_best_vs_old_line_cn.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/current_hybrid_transformer_best_vs_old_line_cn.png)
  - [current_hybrid_transformer_old_vs_agc_joint_all_windows_cn.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/current_hybrid_transformer_old_vs_agc_joint_all_windows_cn.png)
- 已新增“分钟对齐展示版”窗口图：
  - [current_hybrid_transformer_old_vs_agc_joint_all_windows_minute_aligned_cn.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/current_hybrid_transformer_old_vs_agc_joint_all_windows_minute_aligned_cn.png)
  - 用途：让左侧 `120 x 1min` 和右侧 `24 x 5min` 在视觉上都展开到 `120 min` 时间轴，便于导师肉眼比较
  - 重要说明：右侧只是把 `24 x 5min` 的真实/预测轨迹插值到 `120` 个分钟点做显示，不代表模型真的做了 `120` 步 AGC 预测
- 这两张图的口径是：
  - 左侧固定为“旧 Strawberry + old hybrid-transformer”
  - 右侧固定为“AGC + current hybrid-transformer + joint_all”
  - 用于向导师说明：真正值得讲的不是“旧结构迁到新数据集”，而是“新数据集让新的 control-oriented hybrid-transformer 变得合理且有效”
- 已新增混合残差模型脚本：[benchmark_hybrid_residual_forecaster.py](c:/repositories/strawberry/agc_mpc/benchmark_hybrid_residual_forecaster.py)
  - 定位：作为当前预测主线的下一步，不再继续堆 plain Transformer，而是把 `DLinear` 的稳健主路径与 `Transformer-hybrid` 的高阶残差建模结合起来
  - 结构：`ConditionalDLinearForecaster` 负责 main path，`ConditionalTransformerHybridForecaster` 负责 residual path，最终输出为 `base + gated residual`
  - 当前已接入 [main.py](c:/repositories/strawberry/agc_mpc/main.py) 的 baseline 入口，也支持独立 fair-budget benchmark
- 已做 1-epoch smoke test（`joint_all + Reference`，targets = `Tair / Rhair / CO2air`）：
  - [hybrid_residual_forecaster_joint_all_reference_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/hybrid_residual_forecaster_joint_all_reference_summary.json)
  - [hybrid_residual_forecaster_joint_all_reference.pt](c:/repositories/strawberry/agc_mpc/results/forecasting/checkpoints/hybrid_residual_forecaster_joint_all_reference.pt)
  - `Tair`: Full `R2=0.8960`, MAE `0.912`; Final `R2=0.8904`, MAE `0.925`
  - `Rhair`: Full `R2=0.8828`, MAE `4.145`; Final `R2=0.8435`, MAE `4.887`
  - `CO2air`: Full `R2=0.6480`, MAE `58.135`; Final `R2=0.5861`, MAE `62.729`
- 已补做同协议 `DLinear` 1-epoch quick benchmark（`joint_all + Reference`，targets = `Tair / Rhair / CO2air`）：
  - [dlinear_forecaster_joint_all_reference_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/dlinear_forecaster_joint_all_reference_summary.json)
  - [dlinear_forecaster_joint_all_reference.pt](c:/repositories/strawberry/agc_mpc/results/forecasting/checkpoints/dlinear_forecaster_joint_all_reference.pt)
  - `Tair`: Full `R2=0.8870`, MAE `1.003`; Final `R2=0.8745`, MAE `1.047`
  - `Rhair`: Full `R2=0.8872`, MAE `3.865`; Final `R2=0.8385`, MAE `4.651`
  - `CO2air`: Full `R2=0.5086`, MAE `71.191`; Final `R2=0.4850`, MAE `72.943`
- 已新增快速对比图：
  - [hybrid_residual_vs_dlinear_joint_all_reference.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/hybrid_residual_vs_dlinear_joint_all_reference.png)
  - 定位：用于明天汇报时快速展示“在同一 1-epoch quick benchmark 下，混合残差模型相对 `DLinear` 是否已经出现早期优势信号”
- 当前读法：
  - 这组结果仅是 1-epoch smoke test，不用于正式结论
  - 但它已经证明：`hybrid residual` 这条线的代码入口、训练、checkpoint、summary 落盘都已打通，可直接继续跑正式预算
  - 在当前 1-epoch quick benchmark 下，`hybrid residual` 已经在 `Tair / CO2air` 上明显优于同协议 `DLinear`，而 `Rhair` 与其接近但略逊
  - 更合理的下一步是与 `current_hybrid_transformer` 使用同一预算（如 `200 epoch, lr=1e-4, lambda_trend=0.3, patience=15`）做正式对比，再决定是否接入控制侧 benchmark
