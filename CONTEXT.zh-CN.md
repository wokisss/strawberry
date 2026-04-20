# CONTEXT.zh-CN.md

中文对齐翻译版本。
英文主版本： [CONTEXT.md](c:/repositories/strawberry/CONTEXT.md)
最近同步时间：`2026-04-20`

## 0. 目的与维护规则

这是 `strawberry` 工作区的长期项目上下文文档。

从 `2026-04-07` 起，文档规则如下：

- 对长期维护的项目文档，尽量使用 `*.md` 作为英文主版本。
- `*.zh-CN.md` 作为同步维护的中文镜像版本。
- 只要某个双语维护文档发生变化，英文版和中文版必须在同一轮工作中一起更新。
- 只要发现乱码、编码损坏或可疑字符污染，必须先报告再继续。
- 在 Windows PowerShell 中，如果直接用默认 `Get-Content` 读取中文 markdown，可能因为默认编码解读而显示成乱码。判断中文镜像损坏之前，必须先用 `Get-Content -Raw -Encoding UTF8 <path>` 显式重读，并区分终端解码问题和真实文件损坏。
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

#### 2026-04-06 ~ 2026-04-12

- 通过 residual 变体和 CO2 专项变体补全 `iTransformer` 混合线。
- 实现并 benchmark 独立 CO2 专项模型。
- 完成第一轮多目标 wavelet CO2 并回尝试，并记录迁移失败结论。

#### 2026-04-13 ~ 2026-04-19

- 实现并正式 benchmark frozen、late-frozen、distillation、recoupled、protected、protected-terminal、horizon-mixture 和 frozen-backbone horizon-mixture 等 CO2 expert 变体。
- 确立 `itransformer_co2_horizon_mixture` 为当前离线 `CO2air` forecasting leader。
- 完成新 CO2 变体的第一轮 `96-step` 闭环控制检查。
- 新增控制敏感性诊断和 trace-based 成对控制对比图。
- 记录关键结论：普通离线 forecasting 指标不会自动转化成 MPC 控制收益。

### 上周：2026-04-13 ~ 2026-04-19

- 实现并正式 benchmark 最新 CO2 expert fusion 变体。
- 完成 `itransformer_co2_horizon_mixture` forecasting 攻坚。
- 诊断离线 forecasting leader 的 MPC 转化失败原因。
- 实现 `itransformer_co2_frozen_backbone_horizon_mixture` 作为 control-safe 诊断变体。
- 生成与 `late_frozen_expert`、`recoupled_expert` 的 trace-based 控制对比图。

### 本周：2026-04-20 ~ 2026-04-26

- 默认不要继续添加无关的新 predictor。
- 主任务候选 1：建立标准化 control-relevant validation suite。
- 主任务候选 2：把模型故事收敛到 `Protected Horizon Fusion` / `PHF-iTransformer`。
- 高风险高收益候选：构建 control-aware CO2 fusion，把 `late_frozen_expert` 的短时域可控性和 `horizon_mixture` 的离线末端收益结合起来。
- 支撑任务候选：整理 PHF 消融表和已有变体图。
- 支撑任务候选：整理跨 `Tair`、`Rhair`、`CO2air` 的文献 benchmark 表。
- 除非用户另行选择，本周推荐组合：
  - control-relevant validation
  - PHF 主线 / 论文故事收敛

### 下周：2026-04-27 ~ 2026-05-03

- 如果本周完成 validation 和故事收敛，下周只实现一个 control-aware CO2 fusion 候选。
- 如果用户选择性能优先，则优先 control-aware mixture，并重跑正式 forecasting + `96-step` control。
- 如果用户选择论文准备优先，则优先 PHF 消融、方法图和文献对比。

## 8. 当前优先级

优先级 1：

- 先强化离线 `CO2air` forecasting
- 解决全时域 CO2 leader 和末步 CO2 leader 分裂的问题
- 优先做定向 CO2 分支，而不是继续更换 generic backbone

优先级 2：

- 等 forecasting leader 变得更强后，再回到控制侧验证
- 重新跑控制时，持续保留 `GradientMPC vs CEMMPC` 对照
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

## 11. 2026-04-14 接力更新：Forecasting-Only CO2 攻坚

短期项目重点已经改变：

- 暂时不要优先做闭环控制。
- 先把离线 forecasting 明确做强。
- 只有 predictor 持续变强之后，控制才作为下一步故事线。

上一轮 push 后新增的多目标 CO2 变体：

- `itransformer_co2_frozen_expert`
- `itransformer_co2_late_frozen_expert`
- `itransformer_co2_teacher_distill`
- `itransformer_co2_recoupled_expert`
- `itransformer_co2_protected_expert`
- `itransformer_co2_protected_terminal`
- `itransformer_co2_horizon_mixture`
- `itransformer_co2_frozen_backbone_horizon_mixture`

实现备注：

- `training/trainer.py` 现在支持模型自带的可选 `compute_auxiliary_loss`。
- `config.py` 现在有 `lambda_auxiliary`。
- frozen-expert 系列会加载独立 `co2_wavelet_gru_attn` checkpoint，并冻结该 expert。

最新 fair-budget forecasting 结果：

- `itransformer_co2_frozen_expert`
  - `Tair`: Full `R2=0.9463`, MAE `0.601`
  - `Rhair`: Full `R2=0.7949`, MAE `5.471`
  - `CO2air`: Full `R2=0.7427`, MAE `46.966`
  - `CO2air`: Final `R2=0.6105`, MAE `59.247`
- `itransformer_co2_late_frozen_expert`
  - `Tair`: Full `R2=0.9460`, MAE `0.632`
  - `Rhair`: Full `R2=0.8908`, MAE `4.117`
  - `CO2air`: Full `R2=0.7757`, MAE `44.727`
  - `CO2air`: Final `R2=0.6292`, MAE `57.193`
- `itransformer_co2_teacher_distill`
  - `Tair`: Full `R2=0.9464`, MAE `0.611`
  - `Rhair`: Full `R2=0.8730`, MAE `4.179`
  - `CO2air`: Full `R2=0.6551`, MAE `56.018`
  - `CO2air`: Final `R2=0.6407`, MAE `57.294`
- `itransformer_co2_recoupled_expert`
  - `Tair`: Full `R2=0.9339`, MAE `0.687`
  - `Rhair`: Full `R2=0.8591`, MAE `4.522`
  - `CO2air`: Full `R2=0.7533`, MAE `47.585`
  - `CO2air`: Final `R2=0.6416`, MAE `58.054`
- `itransformer_co2_protected_expert`
  - `Tair`: Full `R2=0.9431`, MAE `0.660`
  - `Rhair`: Full `R2=0.8829`, MAE `4.197`
  - `CO2air`: Full `R2=0.7765`, MAE `45.190`
  - `CO2air`: Final `R2=0.6410`, MAE `55.984`
- `itransformer_co2_protected_terminal`
  - `Tair`: Full `R2=0.9489`, MAE `0.614`
  - `Rhair`: Full `R2=0.8620`, MAE `4.324`
  - `CO2air`: Full `R2=0.7404`, MAE `48.055`
  - `CO2air`: Final `R2=0.7069`, MAE `52.056`
- `itransformer_co2_horizon_mixture`
  - `Tair`: Full `R2=0.9508`, MAE `0.604`
  - `Rhair`: Full `R2=0.8958`, MAE `3.882`
  - `CO2air`: Full `R2=0.7868`, MAE `43.910`
  - `CO2air`: Final `R2=0.7468`, MAE `47.661`
- `itransformer_co2_frozen_backbone_horizon_mixture`
  - `Tair`: Full `R2=0.9503`, MAE `0.595`
  - `Rhair`: Full `R2=0.8849`, MAE `4.172`
  - `CO2air`: Full `R2=0.7727`, MAE `46.334`
  - `CO2air`: Final `R2=0.7312`, MAE `50.139`

当前 forecasting frontier：

- 最强 `CO2air` Full MAE：
  - `itransformer_co2_horizon_mixture`: `43.910`
- 最强 `CO2air` Final MAE：
  - `itransformer_co2_horizon_mixture`: `47.661`
- 当前最好 CO2-focused 折中：
  - `itransformer_co2_horizon_mixture`: `Tair` Full MAE `0.604`，`Rhair` Full MAE `3.882`，`CO2air` Full MAE `43.910`，`CO2air` Final MAE `47.661`
- 最强非 CO2 平衡：
  - `itransformer_residual` 仍然在 `Rhair` 上最强
  - `itransformer_co2_late_residual` 仍然是强整体多目标平衡模型

重要结论：

- `itransformer_co2_horizon_mixture` 是当前 fair-budget 下第一个统一全时域 CO2 leader 和末步 CO2 leader 的模型。
- 它还没有严格支配所有非 CO2 指标；`itransformer_residual` 在 `Rhair` 上仍然更强。
- forecasting 瓶颈已经从“CO2 能不能提升”转为“新 CO2 leader 能不能保持或找回最后一点湿度平衡”。

建议的下一步 forecasting-only 方向：

- 把 `itransformer_co2_horizon_mixture` 视为新的 forecasting leader。
- 检查 horizon-wise error 和 forecast examples，确认 terminal pullback 的行为符合预期。
- 如果图形稳定，只为 `itransformer_co2_horizon_mixture` 重新跑闭环控制，再决定是否继续优化控制侧。
- 如果湿度平衡成为限制，再调 horizon gate 或 auxiliary loss，不要急着加重 backbone。

## 12. 2026-04-14 闭环检查，当前已降优先级

已经跑过一组 `96-step` 闭环控制 suite 作为背景，但控制不再是当前立即优先级。

`GradientMPC` 结果：

- `itransformer_residual`
  - objective `0.1924`
  - `Tair MAE=2.216`
  - `Rhair MAE=5.675`
  - `CO2air MAE=11.532`
- `itransformer_co2_late_residual`
  - objective `0.0705`
  - `Tair MAE=1.153`
  - `Rhair MAE=1.618`
  - `CO2air MAE=10.125`
- `itransformer_co2_late_frozen_expert`
  - objective `0.1533`
  - `Tair MAE=2.192`
  - `Rhair MAE=4.316`
  - `CO2air MAE=6.298`
- `itransformer_co2_recoupled_expert`
  - objective `0.0651`
  - `Tair MAE=0.826`
  - `Rhair MAE=2.692`
  - `CO2air MAE=16.749`
- `itransformer_co2_horizon_mixture`
  - objective `0.3713`
  - `Tair MAE=3.313`
  - `Rhair MAE=5.696`
  - `CO2air MAE=28.696`

解读：

- `late_frozen_expert` 把 CO2 forecasting 强度转化成了当前对比中最强的闭环 `CO2air` 控制。
- `late_residual` 和 `recoupled_expert` 在整体 objective 上更好。
- `horizon_mixture` 是新的离线 CO2 leader，但第一次 `96-step` 控制转化很差，不能当作 control leader。
- 当前控制侧最直接的问题是：为什么 terminal-pullback forecast 改善了离线指标，却会破坏 MPC rollout。
- 后续的 frozen-backbone mixture 恢复了 `late_residual` 的第一步行为和控制梯度，但它仍然只是 control-safe 折中，不是新的控制 leader。

## 13. 当前周任务更新

当前周：`2026-04-13 ~ 2026-04-19`

本周优先级：

- Forecasting-only 优先：先把 `CO2air` 预测做强，再回到控制。
- 已完成：实现并正式 benchmark `itransformer_co2_horizon_mixture`。
- 已完成：诊断控制转化失败原因，并实现 `itransformer_co2_frozen_backbone_horizon_mixture`。
- 当前最强全时域 CO2 模型：`itransformer_co2_horizon_mixture`。
- 当前最强末步 CO2 模型：`itransformer_co2_horizon_mixture`。
- 当前最好 CO2-focused 折中模型：`itransformer_co2_horizon_mixture`。
- 已完成 `itransformer_co2_horizon_mixture` 的第一轮闭环检查；离线收益没有转化到 MPC。
- 当前 control-safe mixture 候选：`itransformer_co2_frozen_backbone_horizon_mixture`。
- 立即下一步：构建 control-aware mixture 或验证指标，优先考虑第一步和短时域敏感性，而不是只看 full/final 离线 MAE。

下周：`2026-04-20 ~ 2026-04-26`

- 如果 forecasting frontier 有提升，只为新的 forecasting leader 重新跑闭环控制。
- 如果 forecasting 仍然分裂在全时域 leader 和末步 leader 之间，分析 horizon-wise error 并构建更明确的 horizon-conditioned gate。
- 更新 CO2 specialist report，补充成功和失败的并回模式。

## 14. 当前仓库改动状态

截至 `2026-04-20`，最近代码和结果改动已经分段推送到 `origin/main`：

- `f5aa3f6` - CO2 专项融合模型与控制诊断工具
- `ac98b66` - CO2 专项 forecasting 结果产物
- `86dc2e7` - CO2 控制诊断、对比图和 trace JSON 结果

剩余文档维护会继续体现在当前 context / report 更新中。

切换分支之前仍然要检查 `git status`，因为最近结果推送之后，文档可能又被更新过。

## 15. 2026-04-14 Horizon Mixture Forecasting 结果

已实现 `itransformer_co2_horizon_mixture`。

设计：

- base predictor：`itransformer_co2_late_residual`
- protected correction：冻结的独立 `co2_wavelet_gru_attn` expert
- horizon 行为：
  - 早中段保留 protected expert correction
  - 末端 horizon 回拉到 late-residual predictor
- 训练：fair budget，`lambda_auxiliary = 0.05`

正式 `joint_all + Reference` 结果：

- `Tair`: Full `R2=0.9508`, MAE `0.604`; Final `R2=0.9374`, MAE `0.689`
- `Rhair`: Full `R2=0.8958`, MAE `3.882`; Final `R2=0.8615`, MAE `4.568`
- `CO2air`: Full `R2=0.7868`, MAE `43.910`; Final `R2=0.7468`, MAE `47.661`

解读：

- 这是当前 fair-budget suite 里离线 CO2 最强模型，同时刷新 Full MAE 和 Final MAE。
- 它把之前最强 `CO2air` Full MAE 从 `44.727` 提升到 `43.910`。
- 它把之前最强 `CO2air` Final MAE 从 `50.139` 提升到 `47.661`。
- `Rhair` 仍然略弱于最强的 `itransformer_residual` 平衡，因此这是明确的 CO2 frontier 提升，但还不是所有目标的严格全面支配。

生成文件：

- summary：`results/forecasting/analysis/itransformer_co2_horizon_mixture_joint_all_reference_summary.json`
- checkpoint：`results/forecasting/checkpoints/itransformer_co2_horizon_mixture_joint_all_reference.pt`
- figures：`results/forecasting/figures/residual_variants/`
- 已更新对比图：`results/forecasting/figures/comparisons/itransformer_co2_branch_comparison_metrics.png`

闭环转化检查：

- `96-step` `GradientMPC` + `itransformer_co2_horizon_mixture`：
  - objective `0.3713`
  - `Tair MAE=3.313`
  - `Rhair MAE=5.696`
  - `CO2air MAE=28.696`
- `CEMMPC` + `itransformer_co2_horizon_mixture`：
  - objective `0.4903`
  - `Tair MAE=4.426`
  - `Rhair MAE=7.355`
  - `CO2air MAE=31.294`

控制侧解读：

- 离线 `CO2air` 提升没有转化到当前 MPC loop。
- `itransformer_co2_horizon_mixture` 目前只应作为离线 forecasting leader。
- 在理解 rollout 错配之前，不要用它替换当前控制侧 leader。
- 下一步可能的诊断：对比它和 `late_frozen_expert`、`late_residual` 的 action sensitivity 与 horizon-wise forecast gradients。

## 16. 2026-04-14 控制敏感性诊断与 Frozen-Backbone Mixture

在 `horizon_mixture` 闭环转化失败后，新增了一个控制敏感性诊断脚本。

诊断文件：

- [diagnose_control_sensitivity.py](c:/repositories/strawberry/agc_mpc/diagnose_control_sensitivity.py)

对比绘图文件：

- [plot_control_pair_comparison.py](c:/repositories/strawberry/agc_mpc/plot_control_pair_comparison.py)
- [plot_control_pair_trace_comparison.py](c:/repositories/strawberry/agc_mpc/plot_control_pair_trace_comparison.py)
- 主对比图：`results/control/figures/comparison_itransformer_co2_horizon_mixture_vs_itransformer_co2_late_frozen_expert_gradient_mpc.png`
- 面向整体 objective leader 的补充对比图：`results/control/figures/comparison_itransformer_co2_horizon_mixture_vs_itransformer_co2_recoupled_expert_gradient_mpc.png`
- trace JSON 保存到 `results/control/summaries/trace_comparison_*_gradient_mpc.json`。

主要诊断：

- 当前 simulator 用第一步预测推进状态。
- `itransformer_co2_horizon_mixture` 改善了 full/final 离线 `CO2air` 指标，但控制对齐窗口里的第一步 `CO2air` 误差变差。
- 这解释了为什么离线 leader 没有转化成 MPC rollout 收益。

后续模型：

- `itransformer_co2_frozen_backbone_horizon_mixture`

设计：

- 冻结已经验证过的 `itransformer_co2_late_residual` 主 backbone
- 冻结独立 `co2_wavelet_gru_attn` expert
- 只训练小型 horizon gate
- 为 MPC 保留穿过冻结 backbone 和 expert 输入的梯度

重要实现细节：

- 面向 MPC 的 forward 里，不要用 `torch.no_grad()` 包住冻结 backbone 或 expert。
- 参数通过 `requires_grad_(False)` 保持冻结，但输入梯度必须保留给 `GradientMPC`。
- 早先带 `no_grad()` 的版本虽然预测数值正常，但切断了控制梯度，导致 `GradientMPC` 几乎不动作。

正式 `joint_all + Reference` forecasting 结果：

- `Tair`: Full `R2=0.9503`, MAE `0.595`; Final `R2=0.9375`, MAE `0.674`
- `Rhair`: Full `R2=0.8849`, MAE `4.172`; Final `R2=0.8531`, MAE `4.774`
- `CO2air`: Full `R2=0.7727`, MAE `46.334`; Final `R2=0.7312`, MAE `50.139`

修复梯度后的控制对齐诊断：

- first-step `CO2air MAE = 27.351`
- full-horizon `CO2air MAE = 36.356`
- final-step `CO2air MAE = 30.574`
- mean absolute control-cost gradient `0.01915`
- 最强 cost-gradient 控制量：`t_vent_sp`、`co2_sp`、`assim_sp`

`96-step` 闭环控制结果：

- `GradientMPC`
  - objective `0.0718`
  - `Tair MAE=1.158`
  - `Rhair MAE=1.615`
  - `CO2air MAE=10.000`
- `CEMMPC`
  - objective `0.1632`
  - `Tair MAE=2.631`
  - `Rhair MAE=4.351`
  - `CO2air MAE=25.263`

解读：

- `itransformer_co2_frozen_backbone_horizon_mixture` 不是离线 CO2 leader；`itransformer_co2_horizon_mixture` 离线仍然更强。
- 它是更 control-safe 的 mixture，因为它保留了短步行为和可用控制梯度。
- 它大致追平 `itransformer_co2_late_residual + GradientMPC`，并把后者的 `CO2air MAE` 从 `10.125` 小幅改善到 `10.000`。
- 它仍然没有超过 `itransformer_co2_late_frozen_expert + GradientMPC` 的 `CO2air` 表现，后者之前达到 `6.298`。
- 下一条主线应该是 control-aware CO2 fusion：保留 `late_frozen_expert` 的短时域 CO2 可控性，同时保留 horizon-mixture 家族的离线末端收益。

## 17. 2026-04-20 故事收敛与本周任务候选

最新讨论确认了一个叙事风险：

- 最近 predictor 变体太多，不能把每个都当成独立主贡献来讲。
- 如果按模型流水账写，论文故事会显得像不断试错和堆结构。
- 当前主线应该收敛到一个方法族，其他模型只作为 baseline、ablation 或 diagnostic。

推荐的方法表述：

- 主方法名：`Protected Horizon Fusion` / `PHF-iTransformer`。
- 主技术链条：
  - `CO2-WGA` 独立 expert
  - protected expert correction
  - horizon-dependent trust
  - terminal pullback
  - MPC-relevant validation
- 不要把每个变体都写成独立贡献。

推荐模型角色：

- `itransformer_co2_horizon_mixture`：离线 forecasting 主方法 / PHF 代表模型。
- `itransformer_co2_late_frozen_expert`：当前最强 CO2 控制 baseline。
- `itransformer_co2_recoupled_expert`：当前整体 objective 最强 baseline。
- `itransformer_co2_frozen_backbone_horizon_mixture`：control-safety 诊断变体。
- `frozen_expert`、`teacher_distill`、`protected_terminal`：消融或 appendix 材料。

本周任务候选，按价值排序：

1. Control-relevant validation suite
   - first-step MAE
   - 前 `6` 步 control-horizon MAE
   - horizon-weighted MAE
   - 控制输入敏感性
   - `GradientMPC` 活跃度指标
   - 标准 JSON / 表格 / 图输出
2. PHF 故事与方法收敛
   - 统一主方法命名和叙述
   - 画清晰方法图
   - 明确哪些模型是主方法、baseline、消融和诊断
3. Control-aware CO2 fusion
   - 结合 `late_frozen_expert` 的短时域可控性和 `horizon_mixture` 的离线末端收益
   - 为 `GradientMPC` 保留输入梯度
4. PHF 消融整理
   - 把已有变体结果整理成一个受控表格
   - 避免继续架构发散
5. 文献 benchmark 表
   - 同时比较 `Tair`、`Rhair`、`CO2air`
   - 区分纯 forecasting 论文和 control-oriented validation

除非用户另行选择，推荐本周两个主任务：

- control-relevant validation suite
- PHF 故事与方法收敛

原因：

- 当前瓶颈已经不只是模型容量。
- 当前瓶颈是模型选择逻辑：项目必须解释为什么离线 forecasting leader 不会自动成为 control leader，然后用这个解释指导下一版模型。
