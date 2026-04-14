# CONTEXT.zh-CN.md

中文对齐翻译版本。
英文主版本： [CONTEXT.md](c:/repositories/strawberry/CONTEXT.md)
最近同步时间：`2026-04-07`

## 0. 目的与维护规则

这是 `strawberry` 工作区的长期项目上下文文档。

从 `2026-04-07` 起，文档规则如下：

- 对长期维护的项目文档，尽量使用 `*.md` 作为英文主版本。
- `*.zh-CN.md` 作为同步维护的中文镜像版本。
- 只要某个双语维护文档发生变化，英文版和中文版必须在同一轮工作中一起更新。
- 只要发现乱码、编码损坏或可疑字符污染，必须先报告再继续。
- 在没有说明问题来源之前，不要默默覆盖损坏文档。

当前已经按这套规则维护的文档：

- [CONTEXT.md](c:/repositories/strawberry/CONTEXT.md) 与 [CONTEXT.zh-CN.md](c:/repositories/strawberry/CONTEXT.zh-CN.md)
- [CO2_PAPERS_AND_DIRECTION.md](c:/repositories/strawberry/agc_mpc/CO2_PAPERS_AND_DIRECTION.md) 与 [CO2_PAPERS_AND_DIRECTION.zh-CN.md](c:/repositories/strawberry/agc_mpc/CO2_PAPERS_AND_DIRECTION.zh-CN.md)
- [CO2_SPECIALIST_REPORT.md](c:/repositories/strawberry/agc_mpc/CO2_SPECIALIST_REPORT.md) 与 [CO2_SPECIALIST_REPORT.zh-CN.md](c:/repositories/strawberry/agc_mpc/CO2_SPECIALIST_REPORT.zh-CN.md)

## 1. 项目主线

当前主要目标不是复现旧的草莓论文流程。

当前主线是：

**面向控制的温室多步预测 + 闭环 MPC**

当前项目划分：

- 旧参考项目：[diffmpc](c:/repositories/strawberry/diffmpc)
- 当前主线项目：[agc_mpc](c:/repositories/strawberry/agc_mpc)

规则：

- 新实现默认放在 [agc_mpc](c:/repositories/strawberry/agc_mpc)。
- 除非有明确理由，不要把主开发流移回 `diffmpc`。
- 默认运行环境是 `strawberry_env`。

## 2. 核心数据与接口

主数据集：

- [AutonomousGreenhouseChallenge_edition2](c:/repositories/strawberry/AutonomousGreenhouseChallenge_edition2)

次级 / 历史数据集：

- [Strawberry Greenhouse Environmental Control Dataset(version2).csv](c:/repositories/strawberry/Strawberry%20Greenhouse%20Environmental%20Control%20Dataset(version2).csv)

当前对 AGC 数据的理解：

- `Weather.csv` 提供未来外生天气。
- `GreenhouseClimate.csv` 提供室内气候、执行器状态和设定值。
- `*_sp` 表示请求设定值。
- `*_vip` 表示实际设定值 / 实际执行值。

当前 forecasting 接口：

- `x_past`：历史室内状态和执行器反馈
- `w_future`：未来天气和时间特征
- `u_future`：未来请求控制输入
- `y_future`：未来目标变量

代码中的默认四目标配置：

- `Tair`
- `Rhair`
- `CO2air`
- `Tot_PAR`

当前正式 fair-budget benchmark 经常聚焦三目标子集：

- `Tair`
- `Rhair`
- `CO2air`

## 3. 当前代码库状态

已经稳定落地的部分：

- AGC 数据读取、清洗、对齐和无泄漏切分
- 多隔间联合训练支持
- 联合训练下的全局标准化
- forecasting baselines：
  - `GRU`
  - `DLinear`
  - `SegRNN`
  - `Transformer`
  - `Transformer-hybrid`
- residual variants：
  - `transformer_hybrid_residual`
  - `itransformer_residual`
  - `itransformer_co2_residual`
  - `itransformer_co2_late_residual`
  - `patchtst_residual`
- surrogate 闭环控制 benchmark：
  - `GradientMPC`
  - `CEMMPC`

最近新增的 CO2 专项部分：

- 独立 CO2 专项模型：
  - `co2_env_lstm`
  - `co2_vmd_lstm_fusion`
  - `co2_wavelet_gru_attn`

## 4. 默认实验协议

默认 forecasting benchmark：

- regime：`joint_all`
- 评估隔间：`Reference`
- 历史窗口长度：`288` 步 = `24 h`
- 预测窗口长度：`24` 步 = `2 h`

正式对比使用的 fair-budget 协议：

- `batch_size = 256`
- `num_epochs = 200`
- `learning_rate = 1e-4`
- `lambda_trend = 0.3`
- `early_stop_patience = 15`

默认控制 benchmark：

- `trajectory reference`
- `surrogate rollout`
- 对比 `GradientMPC` 与 `CEMMPC`

## 5. 已建立结论

### 5.1 数据集与训练协议

- 把主线从旧草莓数据切换到 `AGC 2019` 是正确的。
- `joint_all + Reference eval` 仍然是默认 benchmark 协议。
- 早期 `1 epoch` smoke-test 结果不能作为正式结论。

### 5.2 Forecasting 主线

- `current_hybrid_transformer` 仍然是最稳定的整体多目标 predictor。
- `itransformer_residual` 是当前最强、最值得跟踪的 residual baseline。
- `itransformer_co2_late_residual` 在 `CO2air` 上优于原始 `itransformer_residual`，但会牺牲一部分 `Rhair`。

最近 `itransformer` residual 系列在 fair-budget 下的正式结果：

- `itransformer_residual`
  - `Tair`: Full `R2=0.9494`, MAE `0.618`
  - `Rhair`: Full `R2=0.9030`, MAE `3.802`
  - `CO2air`: Full `R2=0.7078`, MAE `51.161`
- `itransformer_co2_residual`
  - `Tair`: Full `R2=0.9435`, MAE `0.639`
  - `Rhair`: Full `R2=0.8787`, MAE `4.244`
  - `CO2air`: Full `R2=0.6991`, MAE `54.001`
- `itransformer_co2_late_residual`
  - `Tair`: Full `R2=0.9503`, MAE `0.595`
  - `Rhair`: Full `R2=0.8849`, MAE `4.172`
  - `CO2air`: Full `R2=0.7553`, MAE `47.797`

当前解读：

- 第一版重型 CO2 分支效果不够好。
- 更轻的后段 horizon CO2 adapter 更有前景。
- `CO2air` 确实能从专项校正中受益，但专项结构不能把整个多目标模型拖坏。

### 5.3 闭环控制

当前控制侧结论：

- 在当前 surrogate benchmark 上，`GradientMPC` 比 `CEMMPC` 更可靠。
- `current_hybrid_transformer + GradientMPC` 是当前最强的整体闭环组合。
- `itransformer_residual + GradientMPC` 在 `CO2air` 上尤其强。

latest predictor suite 中已知的汇总结论：

- `itransformer_residual + GradientMPC` 在记录式 control suite 对比中达到 `CO2air MAE = 5.950`。

## 6. CO2 主线状态

当前存在两条活跃的 CO2 方向。

### 6.1 多目标 CO2 专项分支

状态：

- `DLinear main path + iTransformer residual + dynamic gate` 早已存在。
- 后续补充了 CO2 专项变体并完成 benchmark。
- 当前最好的多目标 CO2 专项版本是 `itransformer_co2_late_residual`。

### 6.2 独立 CO2 预测线

动机：

- 文献并不支持“换一个更大的通用 backbone 就能解决 `CO2air`”这种想法。
- 更强方向是：
  - 分解 / 去噪 / 多尺度建模
  - 自适应融合
  - 最终再走向 carbon-balance 灰盒建模

当前独立 CO2 专项模型排序：

1. `co2_wavelet_gru_attn`
   - Full `R2=0.7519`, MAE `45.209`
   - Final `R2=0.6159`, MAE `58.292`
2. `co2_vmd_lstm_fusion`
   - Full `R2=0.6863`, MAE `52.298`
   - Final `R2=0.6003`, MAE `59.697`
3. `co2_env_lstm`
   - Full `R2=0.3065`, MAE `74.157`
   - Final `R2=-0.4852`, MAE `118.800`

当前解读：

- 纯环境因子 `LSTM` 作为最终方案太弱。
- `CO2air` 需要自回归锚点加多尺度建模。
- 当前最强的独立方向是 `wavelet-inspired + GRU + adaptive attention`。

参考文档：

- [CO2_PAPERS_AND_DIRECTION.md](c:/repositories/strawberry/agc_mpc/CO2_PAPERS_AND_DIRECTION.md)
- [CO2_SPECIALIST_REPORT.md](c:/repositories/strawberry/agc_mpc/CO2_SPECIALIST_REPORT.md)

## 7. 周任务看板

维护规则：

- 长期保留周任务看板。
- 历史周必须带明确时间区间。
- 始终保留 `上周`、`本周` 和 `下周`。
- 本周任务优先级最高。
- 每到周三，必须显式更新 `下周` 区块。

### 历史周任务

#### 2026-03-30 ~ 2026-04-05

- 完成正式 fair-budget `DLinear` benchmark。
- 完成 latest predictor suite 控制对比。
- 完成 CO2 文献方向整理。

### 上周：2026-03-30 ~ 2026-04-05

- 正式 fair-budget `DLinear` benchmark。
- latest predictor suite 控制对比。
- CO2 文献与方向整理。

### 本周：2026-04-06 ~ 2026-04-12

- 补全 `iTransformer` 混合线并形成正式可 benchmark 实现。
  - 状态：通过 residual 变体和 CO2 专项变体，主体已经基本完成。
- 落地可用的 CO2 专项分支。
  - 状态：独立 CO2 线已经实现并完成 benchmark。
  - 剩余子任务：把最强的独立 CO2 思路并回多目标主线，并验证控制收益。

### 下周：2026-04-13 ~ 2026-04-19

- 把 `co2_wavelet_gru_attn` 的思路并入多目标 CO2 residual 线。
- 与 `itransformer_residual`、`itransformer_co2_late_residual` 做正式 fair-budget 对照。
- 为升级后的 CO2 专项 predictor 跑闭环控制对比。
- 如有需要，把双语文档规则继续扩展到其他长期维护的 markdown 文档。

## 8. 当前优先级

优先级 1：

- 以面向控制的方式强化 `CO2air`
- 优先做定向 CO2 分支，而不是继续更换 generic backbone

优先级 2：

- 保持控制侧验证
- 持续保留 `GradientMPC vs CEMMPC` 对照
- 验证离线 forecasting 提升是否能转化为闭环收益

优先级 3：

- 走向更真实的经济 / 资源约束温室控制设定
- 最终纳入：
  - `Heat_cons`
  - `ElecHigh`
  - `ElecLow`
  - `CO2_cons`
  - `Irr`

## 9. 工作规则

1. 每次有意义的代码改动、benchmark 更新或方向变化后，都更新本文件。
2. 文中结论默认要与以下结果文件对齐：
   - `results/forecasting/analysis/*.json`
   - `results/control/summaries/*.json`
3. 不要把 smoke-test 结论和正式 fair-budget 结论混用。
4. 任何新模型都要回答四个问题：
   - 离线 forecasting 是否提升？
   - 闭环控制是否提升？
   - 误差是否稳健？
   - 结构是否能解释成面向控制的设计？
5. 做 CO2 时，优先专项建模，不要盲目扩展通用 backbone。
6. 只要维护中的双语文档发生变化，英文主版本和中文镜像版本必须在同一轮同步更新。
## 10. 2026-04-07 CO2 Wavelet 并回主线更新

针对独立 `co2_wavelet_gru_attn` 思路，已经完成两种多目标并回尝试。

结果如下：

- `itransformer_co2_wavelet_residual`
  - `Tair`: Full `R2=0.9433`, MAE `0.636`
  - `Rhair`: Full `R2=0.8702`, MAE `4.409`
  - `CO2air`: Full `R2=0.5182`, MAE `65.984`
- `itransformer_co2_wavelet_blend`
  - `Tair`: Full `R2=0.9423`, MAE `0.641`
  - `Rhair`: Full `R2=0.8483`, MAE `4.725`
  - `CO2air`: Full `R2=0.5813`, MAE `64.666`

当前解读：

- 独立 wavelet CO2 expert 单独训练时表现强，但它没有顺利迁移到端到端多目标联合训练中。
- 不管是直接做 residual integration，还是直接做 blend expert，`CO2air` 都比 `itransformer_residual` 和 `itransformer_co2_late_residual` 更差。
- 当前证据说明，独立 CO2 specialist 更可能需要通过更解耦的方式并回主线，例如冻结、蒸馏，或离线 teacher guidance，而不是朴素的端到端联合训练。