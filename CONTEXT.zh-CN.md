# CONTEXT.zh-CN.md

中文对齐翻译版本。
英文主版本： [CONTEXT.md](c:/repositories/strawberry/CONTEXT.md)
最近同步时间：`2026-05-12`

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
- [PHF_MAINLINE.md](c:/repositories/strawberry/agc_mpc/PHF_MAINLINE.md) 与 [PHF_MAINLINE.zh-CN.md](c:/repositories/strawberry/agc_mpc/PHF_MAINLINE.zh-CN.md)
- [THESIS_LITERATURE_LIBRARY.md](c:/repositories/strawberry/agc_mpc/THESIS_LITERATURE_LIBRARY.md) 与 [THESIS_LITERATURE_LIBRARY.zh-CN.md](c:/repositories/strawberry/agc_mpc/THESIS_LITERATURE_LIBRARY.zh-CN.md)
- [FCTV_EXPERIMENT_DESIGN.md](c:/repositories/strawberry/agc_mpc/FCTV_EXPERIMENT_DESIGN.md) 与 [FCTV_EXPERIMENT_DESIGN.zh-CN.md](c:/repositories/strawberry/agc_mpc/FCTV_EXPERIMENT_DESIGN.zh-CN.md)
- [FCTV_METHOD_SECTION.md](c:/repositories/strawberry/agc_mpc/FCTV_METHOD_SECTION.md) 与 [FCTV_METHOD_SECTION.zh-CN.md](c:/repositories/strawberry/agc_mpc/FCTV_METHOD_SECTION.zh-CN.md)
- [FCTV_STAGE_REPORT.md](c:/repositories/strawberry/agc_mpc/FCTV_STAGE_REPORT.md) 与 [FCTV_STAGE_REPORT.zh-CN.md](c:/repositories/strawberry/agc_mpc/FCTV_STAGE_REPORT.zh-CN.md)
- [ECONOMIC_RESOURCE_MPC.md](c:/repositories/strawberry/agc_mpc/ECONOMIC_RESOURCE_MPC.md) 与 [ECONOMIC_RESOURCE_MPC.zh-CN.md](c:/repositories/strawberry/agc_mpc/ECONOMIC_RESOURCE_MPC.zh-CN.md)

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
- [PHF_MAINLINE.md](c:/repositories/strawberry/agc_mpc/PHF_MAINLINE.md)
- [THESIS_LITERATURE_LIBRARY.md](c:/repositories/strawberry/agc_mpc/THESIS_LITERATURE_LIBRARY.md)
- [FCTV_EXPERIMENT_DESIGN.md](c:/repositories/strawberry/agc_mpc/FCTV_EXPERIMENT_DESIGN.md)
- [FCTV_METHOD_SECTION.md](c:/repositories/strawberry/agc_mpc/FCTV_METHOD_SECTION.md)
- [FCTV_STAGE_REPORT.md](c:/repositories/strawberry/agc_mpc/FCTV_STAGE_REPORT.md)
- [ECONOMIC_RESOURCE_MPC.md](c:/repositories/strawberry/agc_mpc/ECONOMIC_RESOURCE_MPC.md)

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

#### 2026-04-20 ~ 2026-04-26

- 建立标准化 `control_relevant_validation.py` suite，并升级到 v2。
- 将论文主线收敛到 `Protected Horizon Fusion` / `PHF-iTransformer`。
- 生成正式 PHF 消融表和图。
- 实现、benchmark 并晋升 `itransformer_co2_control_aware_fusion` 为当前均衡型汇报模型。
- 生成 `control-aware fusion`、`late_frozen_expert` 和 `horizon_mixture` 的三模型汇报对比图。

#### 2026-04-27 ~ 2026-05-03

- 将 FCTV 从 CO2-only selector 思路扩展为多目标 forecast-to-control validation 协议。
- 建立扩展后的 24 模型 transfer analysis，并生成周报指标退化汇总图。
- 跑完 16 模型 multi-start 子集，start 为 `0`、`96`、`192`。
- 确认当前结论：已测试的预测侧指标都不能稳定作为闭环 MPC 收益的 universal selector。

### 上周：2026-05-04 ~ 2026-05-10

- 收束探索性 FCTV 阶段，并用更小粒度 commit 推送阶段性代码和结果。
- 准备面向导师汇报的解释：随着模型池和 start 扩大，预测侧指标解释性明显退化。
- 确认下一阶段应转向论文式研究设计，而不是机会式追模型。
- 保留核心限制：FCTV 目前更适合作为诊断 / 验证框架，而不是确定性的闭环模型 selector。

### 本周：2026-05-11 ~ 2026-05-17

- 任务 A：在 `agc_mpc/FCTV_EXPERIMENT_DESIGN.md` 中固定论文式 FCTV 实验设计。
- 任务 B：在 `agc_mpc/run_fctv_final_closed_loop_benchmark.py` 中准备并运行最终 16 模型、5 起点闭环 benchmark 入口。
- 任务 B 执行规则：以后不能默认回避跑模型或闭环实验。只要研究问题需要、且计算窗口允许，就应该运行；如果暂缓，必须说明计算成本和准确运行命令。
- 任务 C：在 `agc_mpc/FCTV_METHOD_SECTION.md` 中写面向论文的方法章节。
- 当前 benchmark 范围：16 个 predictor，starts `0`、`96`、`192`、`288`、`384`，`96` rollout steps，`GradientMPC`，三个目标变量 `Tair`、`Rhair`、`CO2air`。
- 预期输出：固定实验设计、已执行最终 benchmark、分析输出，以及清楚区分筛选声明和诊断声明的方法文字。

### 下周：2026-05-18 ~ 2026-05-24

- 任务 F：基于最终 FCTV 设计和已有闭环证据，准备面向导师的阶段汇报。
- 任务 F 预期输出：简洁报告段落，说明研究问题、实验链条、负结果 / 诊断结果，以及为什么直接闭环验证仍然必要。
- 任务 E：等 tracking-control benchmark 稳定后，再启动 economic/resource-aware MPC。
- 任务 E 预期输出：先定义温室 MPC 的能耗、CO2 施肥、执行器动作和 tracking trade-off 目标扩展，再实现新控制器。

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
7. 不要默认回避跑模型。forecasting 训练、闭环 rollout 和 FCTV 复算是必要实验步骤，不是可选润色。如果 checkpoint、数据和可运行命令都存在，应直接跑实验，而不是只补工具或文档。只有在环境阻塞、必要产物缺失，或当前轮计算成本明显过高时才推迟；推迟时必须写清楚阻塞原因和下一步精确命令。

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

已为用户选择的两个任务开始落地：

- 新增 [control_relevant_validation.py](c:/repositories/strawberry/agc_mpc/control_relevant_validation.py)。
- 已生成：
  - `results/forecasting/analysis/control_relevant_validation_reference.json`
  - `results/forecasting/analysis/control_relevant_validation_reference.csv`
  - `results/forecasting/analysis/control_relevant_validation_reference.md`
  - `results/forecasting/figures/comparisons/control_relevant_validation_reference.png`
- 新增 [PHF_MAINLINE.md](c:/repositories/strawberry/agc_mpc/PHF_MAINLINE.md) 与 [PHF_MAINLINE.zh-CN.md](c:/repositories/strawberry/agc_mpc/PHF_MAINLINE.zh-CN.md)。
- 新增 [THESIS_LITERATURE_LIBRARY.md](c:/repositories/strawberry/agc_mpc/THESIS_LITERATURE_LIBRARY.md) 与 [THESIS_LITERATURE_LIBRARY.zh-CN.md](c:/repositories/strawberry/agc_mpc/THESIS_LITERATURE_LIBRARY.zh-CN.md)，作为更宽的论文文献库。它已经合并原 control-relevant MPC 文献笔记、[RECENT_PAPERS_SURVEY.md](c:/repositories/strawberry/agc_mpc/RECENT_PAPERS_SURVEY.md) 和 [LITERATURE_COMPARISON.md](c:/repositories/strawberry/agc_mpc/LITERATURE_COMPARISON.md) 的内容，覆盖温室 forecasting、温室控制、CO2 专项建模、通用时序架构、AGC 与文献定位、预测-控制相关性和可直接写进论文的段落。
- 将 [control_relevant_validation.py](c:/repositories/strawberry/agc_mpc/control_relevant_validation.py) 升级到 v2，新增 signed CO2 bias、constraint-near proxy MAE、有符号/平坦梯度诊断、recorded-policy CO2 improvement 和 action-activity diagnostics。
- 新增 [summarize_phf_ablation.py](c:/repositories/strawberry/agc_mpc/summarize_phf_ablation.py)，并生成 PHF 消融 JSON / CSV / Markdown / figure 输出。

初步 validation 结论：

- `itransformer_co2_late_residual`、`itransformer_co2_late_frozen_expert` 和 `itransformer_co2_frozen_backbone_horizon_mixture` 在初版 control-relevant validation 综合排名中最好。
- `itransformer_co2_horizon_mixture` 仍然是离线 full/final CO2 forecasting leader，但在 first-step、前 6 步和闭环 CO2 validation 中排名靠后。
- 这支持当前 PHF 叙事：`horizon_mixture` 是离线 PHF 代表模型，而 MPC predictor selection 必须单独引入 control-relevant validation。

## 18. 2026-04-21 Control-Relevant Validation v2 与 PHF 消融

新生成的 validation 输出：

- `results/forecasting/analysis/control_relevant_validation_reference.json`
- `results/forecasting/analysis/control_relevant_validation_reference.csv`
- `results/forecasting/analysis/control_relevant_validation_reference.md`
- `results/forecasting/figures/comparisons/control_relevant_validation_reference.png`

Validation v2 新增：

- signed CO2 bias
- constraint-near proxy MAE
- signed and flat gradient diagnostics
- recorded-policy CO2 improvement
- action-activity diagnostics

当前 control-relevant mean rank：

1. `itransformer_co2_late_frozen_expert`: `2.250`
2. `itransformer_co2_late_residual`: `2.500`
3. `itransformer_residual`: `3.250`
4. `itransformer_co2_frozen_backbone_horizon_mixture`: `3.375`
5. `itransformer_co2_horizon_mixture`: `4.500`
6. `itransformer_co2_recoupled_expert`: `5.125`

新生成的 PHF 消融输出：

- `results/forecasting/analysis/phf_ablation_reference.json`
- `results/forecasting/analysis/phf_ablation_reference.csv`
- `results/forecasting/analysis/phf_ablation_reference.md`
- `results/forecasting/figures/comparisons/phf_ablation_reference.png`

PHF 消融结论：

- `itransformer_co2_horizon_mixture` 仍是离线 PHF 代表模型和 CO2 forecasting leader。
- `itransformer_co2_late_frozen_expert` 仍是最强 CO2 闭环控制 baseline。
- `itransformer_co2_recoupled_expert` 仍是最强整体闭环 objective baseline。
- `itransformer_co2_frozen_backbone_horizon_mixture` 仍是 control-safety diagnostic，不是离线主方法。

下一步推荐技术任务：

- 在当前 validation / story 层提交之后，只新增一个 control-aware fusion 候选。
- 该候选应保留 `late_frozen_expert` 的短时域可控性，同时尝试恢复 `horizon_mixture` 的离线末端收益。

## 19. 2026-04-21 Control-Aware Fusion 候选

已实现 `itransformer_co2_control_aware_fusion`。

设计：

- 冻结 `itransformer_co2_late_frozen_expert` 作为短时域 anchor
- 冻结 `itransformer_co2_horizon_mixture` 作为末端离线收益参考
- 只训练一个 CO2 fusion gate：前 `6` 个 control step 尽量贴住 late-frozen anchor，主要在 horizon 后半段逐步打开
- 当前晋升版本不再单纯提高 tail trust，而是在 control horizon 之后对导入的 terminal delta 做平滑选择
- 加入辅助保护项，保护：
  - first-step `CO2air`
  - first `6`-step `CO2air`
  - 相对 late-frozen anchor 的 `co2_sp` first-step 梯度

正式 `joint_all + Reference` forecasting 结果：

- `Tair`：Full `R2=0.9460`，MAE `0.632`；Final `R2=0.9326`，MAE `0.713`
- `Rhair`：Full `R2=0.8908`，MAE `4.117`；Final `R2=0.8580`，MAE `4.762`
- `CO2air`：Full `R2=0.7858`，MAE `43.983`；Final `R2=0.7393`，MAE `49.069`

Control-relevant validation 结果：

- 新的最佳 mean rank：`1.750`
- first-step `CO2air MAE = 24.468`
- first `6`-step `CO2air MAE = 26.742`
- final-step `CO2air MAE = 26.601`
- constraint-near proxy `CO2air MAE = 29.392`
- first-step `co2_sp` gradient magnitude `0.3040`

闭环 `96-step` 结果：

- `GradientMPC`
  - objective `0.1491`
  - `Tair MAE=2.202`
  - `Rhair MAE=4.267`
  - `CO2air MAE=6.415`
- `CEMMPC`
  - objective `0.2475`
  - `CO2air MAE=16.045`

解释：

- 这个候选在 validation suite 上几乎完整保住了 `late_frozen_expert` 的短时域控制行为。
- 它拿回了 `horizon_mixture` 的大部分离线 CO2 收益：
  - 相比 `late_frozen_expert`，Full `CO2air MAE` 从 `44.727` 改进到 `43.983`
  - 相比 `late_frozen_expert`，Final `CO2air MAE` 从 `57.193` 改进到 `49.069`
- 这次晋升的 delta-smoothing 版本，相比上一版 control-aware fusion，又把闭环转化往前推了一小步：
  - `GradientMPC CO2air` 从 `6.521` 改进到 `6.415`
  - objective 从 `0.1504` 改进到 `0.1491`
- 它仍然没有超过 `late_frozen_expert` 的闭环 CO2（`6.415` vs `6.298`），但仍然很接近，而且当前 control-relevant validation aggregate 里仍排第一。
- 因此它值得保留，作为主线中的 control-aware follow-up，而不是删掉。

这个候选之后的下一步：

- 不再新增一个新的架构家族
- 只调现有 fusion gate 的保守度 / late-start schedule / auxiliary weight
- 目标：在保持当前 first-step 和 first `6`-step 行为的前提下，把 `GradientMPC CO2air` 继续往 `late_frozen_expert` 靠近

补充调参记录：

- 还测试了一版更保守的 tail-trust pilot，并单独存档在：
  - `results/forecasting/analysis/itransformer_co2_control_aware_fusion_conservative_tune_holdout_reference_summary.json`
  - `results/control/summaries/itransformer_co2_control_aware_fusion_conservative_tune_holdout_gradient_mpc_summary.json`
- 这版 pilot 把离线 CO2 提到 Full `43.817` / Final `46.784`，但控制转化没有继续改善到足以替换当前主候选。
- 当前结论：继续单纯提高 terminal trust 不是最好的下一步；下一轮更应该在保住当前 tail gain 的同时，继续压缩剩余的闭环差距。

- 还测试了一版额外加入 gate 单调 / 平滑正则的 pilot，并单独存档在：
  - `results/forecasting/analysis/itransformer_co2_control_aware_fusion_gate_shape_tune_holdout_reference_summary.json`
  - `results/control/summaries/itransformer_co2_control_aware_fusion_gate_shape_tune_holdout_gradient_mpc_summary.json`
- 这版 pilot 把离线 CO2 提到 Full `43.779` / Final `46.916`，但 `GradientMPC CO2air` 反而变差到 `6.885`。
- 当前结论：单纯把 late gate 做得更平滑或更单调，也不是正确的下一步。

- 随后又测试了 delta-smoothing selector 版本，并已经晋升为当前主候选。
- 它的关键动作是在 control horizon 之后，对 `late_frozen_expert -> horizon_mixture` 导入的 terminal delta 做平滑选择，而不是只调 gate 的时间曲线。
- 当前结论：相比继续改 gate schedule，选择更平滑的 terminal delta 更有希望。

## 20. 2026-04-27 当前周方向：跨模型 Forecast-To-Control 验证

用户在 `2026-04-27` 进一步修正了任务范围：

- 研究问题不是“证明某个 CO2 模型是最终汇报模型”。
- 研究问题是形成一套可量化的方法论，用预测侧 validation 去解释和预测多目标控制收益。
- CO2 仍然是当前重点，因为它最明显地暴露 forecast-to-control mismatch；但 `Tair` 和 `Rhair` 必须纳入方法。
- `diffmpc_style_transformer` 当前先不管，因为它的协议与当前严格 AGC 控制验证设置不一致。

更新后的理由：

- 上周已经确认离线 CO2 forecasting gain 不会自动转化成 MPC 收益。
- 下一步是判断这个观察能否扩展成可复用的多目标 validation 方法。
- 方法需要量化哪些预测侧指标能预测闭环 `Tair`、`Rhair`、`CO2air` 和整体 objective 收益，哪些指标只能作为离线诊断。

更新后的主要任务：

1. 定义多目标 FCTV 指标组。
   - 逐目标 first-step MAE：`Tair`、`Rhair`、`CO2air`。
   - 逐目标前 `control_horizon=6` 步 MAE。
   - 逐目标短时域 bias / absolute bias。
   - 逐目标 constraint-near 或 setpoint-near MAE。
   - 逐目标和整体 objective 的加权 forecast rank。
   - 相关控制通道的 gradient / controllability diagnostics。

2. 用闭环结果验证 metric-to-control transfer。
   - 将预测指标与 `GradientMPC` 闭环 `Tair`、`Rhair`、`CO2air` MAE 对齐。
   - 将预测指标与闭环 objective 和 action-activity 诊断对齐。
   - 使用 Pearson / Spearman correlation、top-k hit rate、pairwise consistency、leave-one-model robustness 和 leave-one-family robustness。
   - 分目标报告 metric roles，不强行用一个 score 解释所有控制结果。

3. 用严格可比原则扩展模型广度。
   - 当前 11 个兼容模型作为初始池。
   - 补训三目标标准 baseline：`GRU`、`LSTM`、`SegRNN`、`NLinear` 和纯 `Transformer`。
   - 可行时纳入代表性近年时序模型：`PatchTST`、`iTransformer`，以及 `Autoformer`、`FEDformer` 或 `TimesNet` 中至少一个。
   - PHF / expert / fusion 变体作为模型深度和消融覆盖，而不是唯一证据来源。

4. 正式化方法论。
   - 论文面向的对象应是指标组和验证协议，而不仅是模型排名。
   - CO2 专项结论可以作为 case study，但方法章节必须说明同一逻辑如何用于温度和湿度。
   - 最终模型只能作为方法应用结果来讲，不能替代方法本身。

当前周预期产出：

- 多目标 FCTV JSON / CSV / Markdown 输出。
- 一张简洁图，展示逐目标 metric-to-control 相关性和稳健性。
- 一个 baseline coverage 表，区分严格可比模型与协议不匹配 / appendix-only 模型。
- 一段方法叙述，解释为什么 first-step / control-horizon / bias / constraint-near / gradient 是候选指标，以及哪些已经被实验证实。

初步实现和结果：

- 新增 [analyze_forecast_to_control_transfer.py](c:/repositories/strawberry/agc_mpc/analyze_forecast_to_control_transfer.py)。
- 将 `control_relevant_validation.py` 默认模型池从 PHF 本地集合扩展到 `11` 个兼容模型：
  - `dlinear_forecaster`
  - `current_hybrid_transformer`
  - `transformer_hybrid_residual`
  - `itransformer_residual`
  - `patchtst_residual`
  - `itransformer_co2_late_residual`
  - `itransformer_co2_late_frozen_expert`
  - `itransformer_co2_recoupled_expert`
  - `itransformer_co2_horizon_mixture`
  - `itransformer_co2_frozen_backbone_horizon_mixture`
  - `itransformer_co2_control_aware_fusion`
- 新增 `dlinear_forecaster` 作为兼容的三目标 DLinear baseline，并运行其 `96-step` 闭环控制 suite：
  - `GradientMPC` objective `0.3962`
  - `GradientMPC CO2air MAE = 37.824`
  - `CEMMPC CO2air MAE = 26.864`
- 旧的 `dlinear_baseline`、`transformer_baseline`、`gru_baseline` 和 `segrnn_baseline` 没有纳入细粒度 validation run，因为它们保存的是四目标 checkpoint，不能直接加载到当前三目标 control protocol。
- `diffmpc_style_transformer` 暂未纳入 pooled validation，因为它使用 48 步历史窗口协议，而当前 control-validation protocol 使用 288 步历史窗口。
- 新增 baseline coverage 说明：
  - `results/forecasting/analysis/forecast_to_control_baseline_coverage.md`
- 已重新生成：
  - `results/forecasting/analysis/control_relevant_validation_reference.json`
  - `results/forecasting/analysis/control_relevant_validation_reference.csv`
  - `results/forecasting/analysis/control_relevant_validation_reference.md`
  - `results/forecasting/figures/comparisons/control_relevant_validation_reference.png`
- 已生成新的 forecast-to-control transfer 输出：
  - `results/forecasting/analysis/forecast_to_control_transfer_reference.json`
  - `results/forecasting/analysis/forecast_to_control_transfer_reference.csv`
  - `results/forecasting/analysis/forecast_to_control_transfer_reference.md`
  - `results/forecasting/figures/comparisons/forecast_to_control_transfer_reference.png`
- 已新增 robustness 输出：
  - `results/forecasting/analysis/forecast_to_control_transfer_robustness_reference.csv`
  - `results/forecasting/figures/comparisons/forecast_to_control_transfer_robustness_reference.png`
- 已新增汇报型 summary 图：
  - `results/forecasting/figures/comparisons/forecast_to_control_transfer_summary_reference.png`

基于 `11` 个兼容模型池的初步 CO2-focused transfer 结论：

- 对闭环 `CO2air MAE` 来说，`co2_first_step_mae` 是当前最强 selection metric：
  - Pearson `0.572`
  - Spearman `0.752`
  - pairwise consistency `0.815`
  - top-3 闭环优胜模型命中：yes，top-3 overlap `1.000`
- `co2_control_horizon_mae` 是第二强的 CO2 控制 selection metric：
  - Spearman `0.588`
  - pairwise consistency `0.722`
- `co2_constraint_near_mae_proxy` 和 `co2_control_horizon_abs_bias` 是有用的辅助 selection metric，但弱于 first-step / first-6-step MAE。
- `co2_final_step_mae` 在当前模型池中不能预测闭环 `CO2air MAE`：
  - Spearman `0.009`
  - pairwise consistency `0.509`
- 面向 CO2 tracking 的 selection metrics 不能很好解释整体 `mpc_objective`。这支持把 `CO2air` tracking selection 和整体 controller objective quality 分开处理。
- `control-aware fusion` 仍是 forecast-only transfer rank 和 aggregate control-relevant validation 下最好的模型，而 `late_frozen_expert` 仍是 raw closed-loop `CO2air MAE` 最好的模型。

稳健性更新：

- transfer 分析已加入 leave-one-model 和 leave-one-family robustness。
- 新增 `co2_transfer_selection_score`，作为只使用已验证 control-transfer 指标的加权 composite score：
  - `co2_first_step_mae`：权重 `3.0`
  - `co2_control_horizon_mae`：权重 `2.0`
  - `co2_constraint_near_mae_proxy`：权重 `1.5`
  - `co2_control_horizon_abs_bias`：权重 `1.5`
- 当前针对闭环 `CO2air MAE` 的指标角色：
  - `co2_first_step_mae`：`primary_selection`
  - `co2_control_horizon_mae`：`secondary_selection`
  - `co2_constraint_near_mae_proxy`：`secondary_selection`
  - `co2_control_horizon_abs_bias`：`secondary_selection`
  - `forecast_only_transfer_rank`：`secondary_selection`
  - `co2_transfer_selection_score`：`secondary_selection`
  - `co2_weighted_horizon_mae`：`weak_selection`
  - `co2_full_horizon_mae`：`offline_or_diagnostic_only`
  - `co2_final_step_mae`：`offline_or_diagnostic_only`
  - gradient metrics：`diagnostic_only`
- `co2_first_step_mae` 是当前唯一的 primary selection metric：
  - 相对闭环 `CO2air MAE` 的 full Spearman：`0.752`
  - leave-one-model Spearman 范围：`0.669 .. 0.839`
  - leave-one-family Spearman 范围：`0.661 .. 0.839`
  - leave-one-model pairwise minimum：`0.773`
- 这加强了方法论主张：first-step CO2 accuracy 不只是 PHF 本地模型族内的观察，在当前兼容的跨模型池中，它也是预测闭环 CO2 tracking 表现最稳定的指标。
- `co2_transfer_selection_score` 适合作为汇报用的 composite score，但不应表述为强于 `co2_first_step_mae`：
  - 相对闭环 `CO2air MAE` 的 full Spearman：`0.582`
  - leave-one-model Spearman 范围：`0.455 .. 0.770`
  - leave-one-model pairwise minimum：`0.667`
  - 排名前两位：`control-aware fusion`，然后是 `late_frozen_expert`
- 当前建议表述：first-step CO2 MAE 是当前最强 CO2 primary selection signal，加权 CO2 score 是用于排序和汇报的 secondary composite。
- 这还不是完整的多目标方法论，因为 `Tair` 和 `Rhair` 的 transfer roles 仍需计算和压力测试。

本轮执行清单已完成：

- 将 transfer analyzer 扩展成明确的 score-and-robustness 工具。
- 重新生成 transfer JSON / CSV / Markdown 输出。
- 重新生成 transfer correlation 图。
- 重新生成 leave-one-model robustness 图。
- 新增一张紧凑的汇报型 summary 图。
- 用 AST compilation 验证新脚本语法正常。

立即下一步技术任务：

- 将 [control_relevant_validation.py](c:/repositories/strawberry/agc_mpc/control_relevant_validation.py) 和 [analyze_forecast_to_control_transfer.py](c:/repositories/strawberry/agc_mpc/analyze_forecast_to_control_transfer.py) 从 CO2-only selection metrics 推广为多目标 FCTV metrics。
- 先重跑当前 11 模型池，再补严格可比的标准 baseline。
- 更新汇报语言：`control-aware fusion` 只能表述为当前 CO2-weighted composite 选出的一个模型，而不是方法论的中心贡献。

## 21. 2026-04-27 FCTV 论文故事与边界

当前面向论文的方法论方向是可行的，但必须表述为 screening 和 diagnosis protocol，而不是理论保证。

推荐方法名：

- `Forecast-to-Control Transfer Validation (FCTV)`

核心主张：

- full-horizon MAE、final-step MAE、RMSE、R2 等普通离线 forecasting 指标不足以单独用于 MPC predictor selection。
- 在 receding-horizon MPC 中，靠近真实执行控制时域的预测误差、短时域系统性 bias、约束附近误差和控制输入敏感性，可能比长时域平均精度更能预测闭环收益。
- FCTV 是纯离线 forecasting evaluation 和昂贵闭环 MPC rollout 之间的低成本中间验证层。

这个方法是什么：

- 面向 `Tair`、`Rhair`、`CO2air` 的多目标预测侧指标组。
- 将预测指标与闭环 `GradientMPC` 结果关联起来的 transfer-analysis protocol。
- 模型筛选和失败诊断工具。

这个方法不是什么：

- 不是稳定性证明。
- 不能替代最终闭环 MPC 验证。
- 不是一个必须同时解释温度、湿度、CO2 和所有控制器的万能单一分数。
- 不是把某个当前 PHF / fusion 模型包装成最终贡献。

候选 FCTV 指标组：

- 逐目标 first-step MAE
- 逐目标前 `control_horizon=6` 步 MAE
- 逐目标短时域 signed bias 和 absolute bias
- 逐目标 constraint-near 或 setpoint-near MAE
- 相对相关未来控制输入的 gradient / control-sensitivity diagnostics
- 逐目标和整体 objective 的 composite ranks

支撑小论文所需的验证证据：

- 严格可比的模型广度：至少覆盖 DLinear / NLinear、GRU / LSTM / SegRNN、纯 Transformer、PatchTST / iTransformer、residual 变体和 PHF / fusion 变体
- 同时分析 `Tair`、`Rhair`、`CO2air` 和闭环 objective
- Pearson / Spearman correlation
- pairwise consistency
- top-k winner hit rate
- leave-one-model 和 leave-one-family robustness
- 明确区分逐目标 selection metrics、整体 objective selection metrics 和 diagnostic-only metrics

当前最强的部分证据：

- 在当前 11 个兼容模型池中，CO2 first-step MAE 是目前观察到的最强闭环 CO2 tracking selection signal。
- `co2_final_step_mae` 在当前模型池中不能预测闭环 CO2 tracking。
- 这支持更大的故事：terminal offline forecasting gain 不一定能转化成 receding-horizon MPC 收益。
- 这个结果仍不完整，因为 `Tair` 和 `Rhair` 的 metric roles 还需要计算和压力测试。

论文定位：

- 不要声称过去没有人研究 forecast 和 control 的关系。
- 已有 control-oriented identification、decision-focused learning 和 MPC forecast-value 研究已经承认预测质量会影响控制。
- 更稳妥的研究空缺是：缺少面向温室多目标 MPC 的、离线可计算的、多目标 forecast-side validation 指标组，用来在进入闭环 rollout 前筛选和诊断深度预测模型。
- 贡献应表述为在多目标温室气候控制中，搭建 forecasting evaluation 和 MPC validation 之间的实用桥梁。

## 22. 2026-04-27 多目标 FCTV 实现更新

第 20 节里写的立即下一步技术任务，已经在当前 `11` 个兼容模型池上完成第一版。

实现更新：

- [control_relevant_validation.py](c:/repositories/strawberry/agc_mpc/control_relevant_validation.py) 现在会导出 `Tair`、`Rhair`、`CO2air` 三个目标统一前缀格式的 forecast metrics。
- [analyze_forecast_to_control_transfer.py](c:/repositories/strawberry/agc_mpc/analyze_forecast_to_control_transfer.py) 现在执行多目标 FCTV transfer analysis，而不是 CO2-only analysis。
- analyzer 现在分别针对以下闭环结果报告逐目标 selection role：
  - `mpc_tair_mae`
  - `mpc_rhair_mae`
  - `mpc_co2_mae`
  - `mpc_objective`
- analyzer 现在输出以下目标特定 score：
  - `tair_transfer_selection_score`
  - `rhair_transfer_selection_score`
  - `co2_transfer_selection_score`
  - `multiobjective_transfer_selection_score`

已重新生成的输出：

- `results/forecasting/analysis/control_relevant_validation_reference.{json,csv,md}`
- `results/forecasting/figures/comparisons/control_relevant_validation_reference.png`
- `results/forecasting/analysis/forecast_to_control_transfer_reference.{json,csv,md}`
- `results/forecasting/analysis/forecast_to_control_transfer_robustness_reference.csv`
- `results/forecasting/figures/comparisons/forecast_to_control_transfer_reference.png`
- `results/forecasting/figures/comparisons/forecast_to_control_transfer_robustness_reference.png`
- `results/forecasting/figures/comparisons/forecast_to_control_transfer_summary_reference.png`

当前多目标证据：

- `CO2air`：`co2_first_step_mae` 仍然是预测闭环 `CO2air MAE` 最强的已验证 selection metric。
  - Spearman `0.752`
  - pairwise consistency `0.815`
  - leave-one-model Spearman 范围 `0.669 .. 0.839`
  - role：`primary_selection`
- `Rhair`：`rhair_first_step_mae` 对闭环 `Rhair MAE` 有用，但强度低于 CO2。
  - Spearman `0.627`
  - pairwise consistency `0.727`
  - leave-one-model Spearman 范围 `0.539 .. 0.733`
  - role：`secondary_selection`
- `Tair`：`tair_first_step_mae` 目前不能稳定选择闭环 `Tair MAE` 更好的模型。
  - Spearman `-0.236`
  - pairwise consistency `0.400`
  - role：`offline_or_diagnostic_only`
- 整体 objective：当前 `multiobjective_transfer_selection_score` 不能很好解释 `mpc_objective`。
  - Spearman `0.136`
  - pairwise consistency `0.564`
  - role：`offline_or_diagnostic_only`

当前解释：

- FCTV 应该表述为逐目标 screening 和 diagnosis protocol，而不是一个万能单一分数。
- 当前最强的正面案例仍然是 CO2，因为它的 receding-horizon transfer signal 最清楚。
- 湿度有可用的 secondary transfer evidence。
- 温度目前暴露了一个方法边界：只看 target-matched first-step error，还不足以选择闭环 Tair controller。
- 这个限制本身对方法论叙事有价值，因为它支持 variable-specific metric roles，而不是强行给出 all-in-one score。

立即下一步技术任务：

- 扩展严格可比模型广度，补三目标重训的 `GRU`、`LSTM`、`SegRNN`、`NLinear`、纯 `Transformer`，以及可行时的 `iTransformer` / `PatchTST` / decomposition-style baseline。
- 增加或改进 Tair/Rhair-specific control-sensitivity diagnostics，不要只依赖 CO2 gradient diagnostics。
- 在 baseline pool 不再过度偏向 PHF 家族之后，重新检查 objective-level screening 是否会改善。

## 23. 2026-04-28 标准 Baseline 扩展与 FCTV 复查

FCTV 的下一步 baseline 补全任务已经完成一部分。

实现更新：

- [compare_training_regimes.py](c:/repositories/strawberry/agc_mpc/compare_training_regimes.py) 现在支持：
  - `--control-protocol`：严格三目标 `Tair` / `Rhair` / `CO2air` 训练
  - `--fair-budget`：正式预算，`batch_size=256`、`num_epochs=200`、`learning_rate=1e-4`、`lambda_trend=0.3`、`early_stop_patience=15`
- [control_main.py](c:/repositories/strawberry/agc_mpc/control_main.py) 现在暴露严格控制 predictor：
  - `gru_forecaster`
  - `lstm_forecaster`
  - `nlinear_forecaster`
  - `segrnn_forecaster`
  - `transformer_forecaster`
- [control_relevant_validation.py](c:/repositories/strawberry/agc_mpc/control_relevant_validation.py) 已把这三个标准 baseline 加入默认 FCTV 模型池。
- [control/controller.py](c:/repositories/strawberry/agc_mpc/control/controller.py) 在 gradient-based MPC 优化时禁用 CuDNN RNN kernel，使 recurrent predictor 可以在 eval-mode 控制 rollout 中被求梯度。
- FCTV gradient diagnostics 不再是 CO2-only；现在包含 `Tair`、`Rhair`、`CO2air` 的 first-step 和 mean forecast gradients，以及逐目标相关控制通道。

新严格 baseline 训练结果：

- `gru_forecaster`
  - Full MAE：`Tair=0.866`、`Rhair=4.753`、`CO2air=48.396`
  - Final MAE：`Tair=0.986`、`Rhair=6.281`、`CO2air=54.721`
- `segrnn_forecaster`
  - Full MAE：`Tair=0.960`、`Rhair=5.109`、`CO2air=69.209`
  - Final MAE：`Tair=1.186`、`Rhair=6.406`、`CO2air=84.046`
- `lstm_forecaster`
  - Full MAE：`Tair=0.874`、`Rhair=4.832`、`CO2air=69.352`
  - Final MAE：`Tair=1.105`、`Rhair=6.483`、`CO2air=81.987`
- `nlinear_forecaster`
  - Full MAE：`Tair=0.727`、`Rhair=4.236`、`CO2air=61.003`
  - Final MAE：`Tair=0.774`、`Rhair=4.710`、`CO2air=63.283`
- `transformer_forecaster`
  - Full MAE：`Tair=0.597`、`Rhair=4.256`、`CO2air=42.789`
  - Final MAE：`Tair=0.691`、`Rhair=5.175`、`CO2air=48.983`

新的 96-step 闭环 `GradientMPC` 结果：

- `gru_forecaster`：objective `0.1108`，`Tair MAE=0.409`，`Rhair MAE=4.957`，`CO2air MAE=49.973`
- `segrnn_forecaster`：objective `0.0486`，`Tair MAE=0.391`，`Rhair MAE=2.195`，`CO2air MAE=14.425`
- `lstm_forecaster`：objective `0.1780`，`Tair MAE=1.491`，`Rhair MAE=4.497`，`CO2air MAE=23.014`
- `nlinear_forecaster`：objective `0.1526`，`Tair MAE=1.867`，`Rhair MAE=4.182`，`CO2air MAE=25.236`
- `transformer_forecaster`：objective `0.0861`，`Tair MAE=1.039`，`Rhair MAE=4.072`，`CO2air MAE=16.455`

更新后的 FCTV 模型池：

- 默认严格模型池已经从 `11` 个扩展到 `16` 个。
- 当前覆盖 DLinear、NLinear、GRU、LSTM、SegRNN、纯 Transformer、Transformer-hybrid、PatchTST-style residual、iTransformer-style residual、CO2-aware residual，以及 PHF / control-aware fusion 变体。

加入标准 baseline 后的 transfer 结论：

- `CO2air`：`co2_first_step_mae` 仍是当前最强 CO2 screening signal，但角色需要更保守：
  - Spearman `0.593`
  - pairwise consistency `0.723`
  - role：`secondary_selection`
- `Rhair`：`rhair_first_step_mae` 现在是扩展模型池里最强的逐目标已验证 signal：
  - Spearman `0.653`
  - pairwise consistency `0.758`
  - role：`primary_selection`
- `Tair`：`tair_first_step_mae` 仍不能稳定选择闭环 Tair 更好的模型：
  - Spearman `-0.335`
  - pairwise consistency `0.383`
  - role：`offline_or_diagnostic_only`
- 整体 objective：当前 `multiobjective_transfer_selection_score` 仍不能解释 `mpc_objective`：
  - Spearman `0.153`
  - pairwise consistency `0.567`
  - role：`offline_or_diagnostic_only`

解释更新：

- 加入非 PHF 标准 baseline 后，早先的 CO2 primary-selection 结论变弱。这是有价值的修正，不是失败。
- 当前稳妥表述应该是：first-step CO2 error 是当前最好的 CO2 screening metric，但在模型池继续扩展前，应称为 secondary selection signal。
- 标准 baseline 说明 FCTV 的必要性：`segrnn_forecaster` 的离线 CO2 forecasting 并不强，但闭环 CO2 tracking 明显好于它的 offline final-step CO2 MAE 所暗示的结果。
- 这支持论文叙事：forecast quality 必须通过 control-relevant timing、bias、sensitivity 和逐目标 transfer role 来评价，而不能只看普通离线 forecasting rank。

剩余 baseline 缺口：

- 可行时补至少一个 decomposition / frequency-style baseline。
- 用 family-level ablation 区分框架效应和模块效应。

## 24. 2026-04-28 Frequency Baseline 与归因报告

当前轮剩余的 baseline 和归因任务已经完成。

实现更新：

- 新增 [frequency_forecaster.py](c:/repositories/strawberry/agc_mpc/models/frequency_forecaster.py)，作为轻量 frequency-style conditional baseline。
  - 它从历史状态序列中提取低频 FFT 模式。
  - 它把 frequency context 与未来天气、未来请求控制输入融合。
  - 它是当前仓库内协议一致的 frequency-style baseline，不是 Autoformer / FEDformer / TimesNet 的正式复现。
- 已将 `frequency_baseline` / `frequency_forecaster` 接入：
  - [compare_training_regimes.py](c:/repositories/strawberry/agc_mpc/compare_training_regimes.py)
  - [control_main.py](c:/repositories/strawberry/agc_mpc/control_main.py)
  - [control_relevant_validation.py](c:/repositories/strawberry/agc_mpc/control_relevant_validation.py)
  - [analyze_forecast_to_control_transfer.py](c:/repositories/strawberry/agc_mpc/analyze_forecast_to_control_transfer.py)
- 已生成归因说明：
  - `results/forecasting/analysis/forecast_to_control_attribution_reference.md`

Frequency baseline 结果：

- 离线预测：
  - Full MAE：`Tair=1.253`、`Rhair=4.624`、`CO2air=90.101`
  - Final MAE：`Tair=1.383`、`Rhair=5.284`、`CO2air=91.544`
- 96-step `GradientMPC`：
  - objective `0.4338`
  - `Tair MAE=1.725`
  - `Rhair MAE=8.759`
  - `CO2air MAE=15.530`

更新后的 FCTV 模型池：

- 默认严格模型池现在是 `17` 个模型。
- 当前覆盖 DLinear、NLinear、frequency-style MLP、GRU、LSTM、SegRNN、纯 Transformer、Transformer-hybrid、PatchTST-style residual、iTransformer-style residual、CO2-aware residual，以及 PHF / control-aware fusion 变体。

`17` 模型池下的 FCTV 指标角色：

- `rhair_first_step_mae -> mpc_rhair_mae`
  - role：`primary_selection`
  - Spearman `0.711`
  - pairwise consistency `0.787`
- `co2_first_step_mae -> mpc_co2_mae`
  - role：`secondary_selection`
  - Spearman `0.516`
  - pairwise consistency `0.681`
- `co2_constraint_near_mae_proxy -> mpc_co2_mae`
  - role：`secondary_selection`
  - Spearman `0.522`
  - pairwise consistency `0.676`
- `tair_first_step_mae -> mpc_tair_mae`
  - role：`offline_or_diagnostic_only`
  - Spearman `-0.270`
  - pairwise consistency `0.412`
- `multiobjective_transfer_selection_score -> mpc_objective`
  - role：`weak_selection`
  - Spearman `0.267`
  - pairwise consistency `0.618`

归因结论：

- 当前证据支持 metric-mediated attribution，而不是简单说“某个框架更好”。
- framework effect 是存在的：例如 `segrnn_forecaster` 和 `frequency_forecaster` 的离线 CO2 预测较弱，但闭环 CO2 tracking 明显好于 final-step CO2 MAE 所暗示的结果。
- PHF / iTransformer 家族内部的 module effect 是 horizon-specific 的：late expert、horizon mixture、frozen-backbone mixture、control-aware fusion 会分别改变不同 FCTV 指标。
- 当前最稳妥的表述是：
  - 模型框架和模块会通过特定 forecast-side behavior 影响控制；
  - FCTV 用来识别哪些 behavior 对哪个控制目标有用；
  - 整体 objective 的结论仍必须通过最终闭环 MPC 验证。

剩余可选后续工作：

- 如果论文需要严格外部 baseline，可继续补正式 Autoformer / FEDformer / TimesNet 实现。
- 为了增强因果稳健性，可在多个 start index 上重复闭环 rollout。
- 如果要提出更强的 module-causality claim，需要把同类模块移植到一个以上 backbone 上做 controlled module swap。

## 25. 2026-04-28 FCTV 后续推进清单与指标来源解释

下一步不应该继续盲目堆模型，而是把“预测指标为什么能解释控制收益”这条逻辑链补完整。

P0：论文级方法论整理。

- 明确 FCTV 的定位：它不是一个新模型，而是 forecasting evaluation 和闭环 MPC validation 之间的筛选 / 诊断协议。
- 基于 `results/forecasting/analysis/forecast_to_control_transfer_reference.md` 写成方法章节：候选指标来源、验证方式、角色分类。
- 把当前 `17` 模型池结论写清楚：`Rhair first-step MAE` 最强，`CO2 first-step / constraint-near` 是辅助筛选，`Tair` 当前不能被 target-matched forecast error 解释，整体 objective 仍需闭环验证。

P1：补稳健性实验。

- 在多个 closed-loop start index 上重复 96-step rollout，验证当前 FCTV 关系不是某一个片段偶然得到的。
- 对每个 start index 重算 `mpc_tair_mae`、`mpc_rhair_mae`、`mpc_co2_mae` 和 `mpc_objective`。
- 重新统计 Spearman、pairwise consistency、top-k hit、leave-one-model robustness 和 leave-one-family robustness。
- 如果结论稳定，FCTV 才能从“当前实验现象”升级为“可复用验证方法”。

P1：补归因实验。

- 固定 backbone，只替换模块：例如 iTransformer residual、CO2 late adapter、frozen expert、horizon mixture、control-aware fusion。
- 固定模块思想，换 backbone：如果可行，把类似 CO2 late / fusion 思路迁移到一个以上 backbone。
- 目标是区分收益来自框架、模块，还是某些 FCTV 指标改善后间接导致控制改善。

P2：补外部 baseline。

- 如果论文需要更强外部对比，可补正式 Autoformer / FEDformer / TimesNet。
- 这不是当前最急任务，因为已有 `17` 个严格可比模型池；更急的是稳健性和归因。

P2：补展示材料。

- 做一张逻辑链图：模型 / 模块 -> forecast-side behavior -> FCTV metric -> closed-loop target。
- 做一张指标角色表：selection metric、secondary metric、diagnostic-only metric。
- 做一张反例图：final-step CO2 MAE 好不一定控制好，说明普通 forecast rank 不够。

指标来源解释口径：

- 普通指标如 MAE、RMSE、R2 来自 forecasting / regression tradition，主要回答“预测整体拟合得好不好”。
- R2 是统计回归中的拟合优度指标，定义为 `R2 = 1 - SSE / SST`，衡量模型解释目标变量方差的比例。
- 这些普通指标不能直接回答“预测器用于 MPC 后控制收益是否更好”，因为 MPC 只执行 receding horizon 中最前面的控制动作。
- FCTV 指标来自 MPC 执行机制、控制目标结构和优化器敏感性要求。
- `first-step MAE` 和 `control_horizon MAE` 来自 receding-horizon MPC 的执行机制，因为控制器每次最依赖即将执行的短时域预测。
- `bias` 来自控制偏差风险，因为系统性偏高或偏低会让 MPC 选择方向错误的控制动作。
- `constraint-near MAE` 来自约束 / 设定点附近的控制风险，因为靠近约束时的小误差比远离约束时的小误差更可能改变控制决策。
- `gradient diagnostics` 来自 GradientMPC 的优化需求，用来判断预测模型是否对未来控制输入有合理敏感性。
- 当前论文表述应为：先由 MPC 机制提出候选指标，再通过跨模型 transfer validation 验证哪些指标真的能预测闭环收益，而不是事后凭结果凑指标。

## 26. 2026-04-28 FCTV 方法报告、多起点工具与展示材料

第 25 节的后续清单已经从开放任务推进为具体的方法报告、可复现实验工具和展示资产。

已完成 P0 方法论整理：

- 新增 `results/forecasting/analysis/forecast_to_control_transfer_method_reference.md`。
- 方法报告明确 FCTV 是 offline forecasting 和闭环 MPC validation 之间的筛选 / 诊断协议。
- 报告解释了候选指标来源：receding-horizon 执行机制、短时域 bias 风险、constraint-near 风险，以及 GradientMPC 对控制敏感性的要求。
- 报告记录当前 `17` 模型池结论：
  - `rhair_first_step_mae` 是解释 `mpc_rhair_mae` 的最强逐目标信号。
  - `co2_first_step_mae` 和 `co2_constraint_near_mae_proxy` 是 CO2 的辅助筛选指标。
  - `tair_first_step_mae` 目前不能可靠选择 `mpc_tair_mae` 更好的模型。
  - `multiobjective_transfer_selection_score` 对整体 objective 仍然只是 weak selection。

已完成 P1 稳健性工具：

- 新增 `run_fctv_multistart_control.py`，用于在多个 closed-loop start index 上重复 96-step `GradientMPC` rollout。
- 新增 `analyze_fctv_multistart_transfer.py`，用于把每个 start index 的闭环指标替换回 FCTV 分析并重新计算 transfer 统计。
- `AGCConfig` 新增可选 `control_output_tag`。
- `AGCClosedLoopSimulator` 现在会在 summary 中记录 `start_idx`，并用 `control_output_tag` 避免多起点 rollout 的图和 summary 互相覆盖。
- `control_main.py` 的 suite summary 现在记录 `start_idx` 和 `output_tag`。

执行说明：

- 本轮没有直接跑完整 multi-start robustness benchmark，因为这需要在 `17` 模型池上执行大量昂贵的 96-step GradientMPC rollout。
- 当前已经具备可复现实验命令路径：
  - `python agc_mpc/run_fctv_multistart_control.py --start-indices 0 96 192 --steps 96`
  - `python agc_mpc/analyze_fctv_multistart_transfer.py --suite-json <generated_suite_json>`

已完成 P2 展示材料：

- 新增 `plot_fctv_presentation_assets.py`。
- 已生成：
  - `results/forecasting/figures/comparisons/fctv_presentation_reference_logic_chain.png`
  - `results/forecasting/figures/comparisons/fctv_presentation_reference_metric_roles.png`
  - `results/forecasting/figures/comparisons/fctv_presentation_reference_co2_counterexample.png`

归因状态：

- 当前能够支持的仍然是 metric-mediated attribution，而不是笼统的框架因果结论。
- 现有 PHF / iTransformer 变体提供同家族模块证据，标准 baseline 提供框架对照。
- 更强的因果归因仍需要 multi-start rollout 和跨一个以上 backbone 的 controlled module swap。

验证情况：

- 已基于当前 transfer JSON 生成新的展示资产。
- 由于当前环境阻止 `__pycache__` 字节码替换，改用 AST parsing 验证本轮修改的 Python 文件语法。

## 27. 2026-04-28 剩余可跑模型与实验执行规则

当前 `17` 模型 FCTV 池之后仍然存在的可运行缺口：

- 已被 `control_main.py` 支持、也已有 checkpoint，但尚未进入当前严格 `17` 模型 FCTV 池：
  - `itransformer_co2_residual`
  - `itransformer_co2_frozen_expert`
  - `itransformer_co2_teacher_distill`
  - `itransformer_co2_protected_expert`
  - `itransformer_co2_protected_terminal`
  - `itransformer_co2_wavelet_residual`
  - `itransformer_co2_wavelet_blend`
- 也可以运行，但对当前严格模型池优先级较低：
  - `dlinear_baseline`
  - `transformer_hybrid_baseline`
  - `transformer_baseline`
- 除非协议对齐，否则继续排除：
  - `diffmpc_style_transformer`，因为它的历史协议与当前严格 288-step AGC control-validation 协议不一致。
- 尚未作为正式外部 baseline 实现：
  - Autoformer / FEDformer / TimesNet。

立即可跑的实验优先级：

1. 对当前 FCTV 池缺失、但已有 checkpoint 的 CO2 / PHF 变体补跑 96-step `GradientMPC` 闭环检查。
2. 如果这些闭环运行完成，用扩展模型池重算 `control_relevant_validation.py` 和 `analyze_forecast_to_control_transfer.py`。
3. 单起点扩展池完成后，再对最重要 predictor 做 multi-start robustness。

规则澄清：

- 以后推进实验时，不要默认把“跑模型”视为应该回避的事情。
- 如果用户要求推进实验，且模型、checkpoint、脚本都存在，应直接运行。
- 如果完整运行成本很高，先选择有技术理由的子集运行，报告已运行内容，并留下剩余精确命令。

## 28. 2026-04-28 扩展 24 模型 FCTV 实际运行

第 27 节列出的缺失 checkpointed CO2 / PHF 变体已经实际补跑，没有停留在计划层面。

新的 96-step `GradientMPC` 闭环结果：

- `itransformer_co2_residual`：objective `0.0557`，`Tair MAE=0.936`，`Rhair MAE=1.503`，`CO2air MAE=6.421`
- `itransformer_co2_frozen_expert`：objective `0.0649`，`Tair MAE=0.917`，`Rhair MAE=2.263`，`CO2air MAE=20.140`
- `itransformer_co2_teacher_distill`：objective `0.3502`，`Tair MAE=2.789`，`Rhair MAE=6.877`，`CO2air MAE=27.338`
- `itransformer_co2_protected_expert`：objective `0.0606`，`Tair MAE=0.880`，`Rhair MAE=1.441`，`CO2air MAE=14.206`
- `itransformer_co2_protected_terminal`：objective `0.3837`，`Tair MAE=3.380`，`Rhair MAE=6.179`，`CO2air MAE=27.089`
- `itransformer_co2_wavelet_residual`：objective `0.0639`，`Tair MAE=1.075`，`Rhair MAE=2.142`，`CO2air MAE=7.776`
- `itransformer_co2_wavelet_blend`：objective `0.0771`，`Tair MAE=1.023`，`Rhair MAE=1.928`，`CO2air MAE=8.020`

已生成 / 更新输出：

- `results/control/summaries/predictor_suite_missing_co2_phf_reference_96steps.json`
- `results/forecasting/analysis/control_relevant_validation_reference.{json,csv,md}`
- `results/forecasting/analysis/forecast_to_control_transfer_reference.{json,csv,md}`
- `results/forecasting/analysis/forecast_to_control_transfer_robustness_reference.csv`
- `results/forecasting/figures/comparisons/` 下的 FCTV comparison、robustness、summary 和 presentation figures

更新后的 24 模型 FCTV 结论：

- 当前模型池包含 `24` 个模型。
- `rhair_first_step_mae -> mpc_rhair_mae` 仍是最强逐目标信号，但角色降为 `secondary_selection`：
  - Spearman `0.592`
  - pairwise consistency `0.732`
  - leave-one-model Spearman minimum `0.537`
- `co2_first_step_mae -> mpc_co2_mae` 在扩展池中不再是稳定 selector：
  - role：`offline_or_diagnostic_only`
  - Spearman `0.168`
  - pairwise consistency `0.549`
- `co2_constraint_near_mae_proxy -> mpc_co2_mae` 同样不再稳定：
  - role：`offline_or_diagnostic_only`
  - Spearman `0.015`
  - pairwise consistency `0.507`
- `tair_first_step_mae -> mpc_tair_mae` 仍不可靠：
  - Spearman `-0.123`
  - pairwise consistency `0.464`
- `multiobjective_transfer_selection_score -> mpc_objective` 仍不适合作为整体 objective selector：
  - Spearman `0.167`
  - pairwise consistency `0.564`
- `rhair_first_step_mae -> mpc_objective` 是当前最强的整体 objective 辅助信号：
  - role：`objective_secondary_selection`
  - Spearman `0.507`
  - pairwise consistency `0.703`

解释更新：

- 早先 `17` 模型池中的 CO2 screening 结论具有模型池依赖性；补入缺失 CO2 / PHF 变体后，CO2 first-step 和 constraint-near transfer 明显变弱。
- 这对论文叙事是有价值的证据：FCTV 必须在明确模型池范围下报告 metric role，不能过度宣称 universal transfer。
- 新增模型中闭环 CO2 最好的是 `itransformer_co2_residual`，`CO2air MAE=6.421`，接近 `control-aware fusion` (`6.415`) 和 `late_frozen_expert` (`6.298`)，但整体 objective 明显更好。
- `itransformer_co2_protected_expert` 是新增变体里 objective 最好的模型 (`0.0606`)，且 `Rhair MAE=1.441` 很强，值得进入控制侧讨论。
- 当前扩展单起点池之后的立即下一步是 multi-start robustness，而不是继续加新架构。

## 29. 2026-04-28 初始 Multi-Start FCTV 稳健性运行

已经完成一组代表性的 multi-start robustness run，没有把稳健性继续停留在未来计划。

执行范围：

- `10` 个 predictor
- start indices：`0`、`96`、`192`
- rollout 长度：`96` steps
- controller：`GradientMPC`

predictor 子集：

- `current_hybrid_transformer`
- `transformer_hybrid_residual`
- `segrnn_forecaster`
- `frequency_forecaster`
- `itransformer_co2_residual`
- `itransformer_co2_protected_expert`
- `itransformer_co2_late_residual`
- `itransformer_co2_late_frozen_expert`
- `itransformer_co2_control_aware_fusion`
- `itransformer_co2_horizon_mixture`

已生成输出：

- `results/control/summaries/fctv_multistart_gradient_mpc_reference_96steps_starts_0_96_192.json`
- `results/forecasting/analysis/forecast_to_control_transfer_multistart_reference.{json,csv,md}`
- 每个 start 的 transfer report：
  - `forecast_to_control_transfer_multistart_reference_start00000.*`
  - `forecast_to_control_transfer_multistart_reference_start00096.*`
  - `forecast_to_control_transfer_multistart_reference_start00192.*`
- `results/forecasting/figures/comparisons/forecast_to_control_transfer_multistart_reference.png`

重要执行说明：

- 长时间命令在完成并保存 suite JSON 后超过工具超时时间，因此 shell 返回 timeout status `124`。
- 输出文件已经存在且完整，analyzer 成功处理了 start indices `[0, 96, 192]`。

Multi-start 指标结论：

- `co2_first_step_mae -> mpc_co2_mae` 跨 start 不稳定：
  - start `0`：`secondary_selection`，Spearman `0.498`，pairwise `0.705`
  - start `96`：`offline_or_diagnostic_only`，Spearman `-0.146`，pairwise `0.409`
  - start `192`：`offline_or_diagnostic_only`，Spearman `-0.243`，pairwise `0.432`
- `rhair_first_step_mae -> mpc_rhair_mae` 也不稳定：
  - start `0`：`secondary_selection`，Spearman `0.418`，pairwise `0.667`
  - start `96`：`offline_or_diagnostic_only`，Spearman `-0.103`，pairwise `0.444`
  - start `192`：`offline_or_diagnostic_only`，Spearman `0.091`，pairwise `0.578`
- `multiobjective_transfer_selection_score -> mpc_objective` 仍然只是 weak 或 diagnostic：
  - start `0`：`weak_selection`，Spearman `0.285`，pairwise `0.600`
  - start `96`：`offline_or_diagnostic_only`，Spearman `0.188`，pairwise `0.556`
  - start `192`：`weak_selection`，Spearman `0.285`，pairwise `0.600`
- `tair_first_step_mae -> mpc_tair_mae` 仍不可靠。

Multi-start 模型侧发现：

- `itransformer_co2_residual` 是测试子集中最稳定的 CO2 闭环 tracking 模型：
  - start `0`：`CO2air MAE=6.331`，objective `0.0558`
  - start `96`：`CO2air MAE=11.074`，objective `0.0654`
  - start `192`：`CO2air MAE=10.701`，objective `0.0465`
- 每个 start 下 objective 最优模型：
  - start `0`：`current_hybrid_transformer`，objective `0.0442`
  - start `96`：`current_hybrid_transformer`，objective `0.0517`
  - start `192`：`transformer_hybrid_residual`，objective `0.0235`

解释更新：

- multi-start 结果强化了限制结论：FCTV metric role 不仅依赖模型池，也依赖 rollout segment。
- FCTV 应表述为带明确范围的诊断协议，而不是 universal offline selector。
- 当前最强的近期模型结论不是某个 FCTV 指标能通用选出 winner，而是 `itransformer_co2_residual` 值得重新关注，因为它是稳健的 CO2 闭环 tracker。
- 下一步实验优先级是完整 24 模型 multi-start robustness，或围绕 `itransformer_co2_residual`、`current_hybrid_transformer`、`transformer_hybrid_residual` 和主要 PHF/fusion 变体做更小的 repeated-start suite。

## 30. 2026-04-28 扩展 16 模型 Multi-Start FCTV 稳健性运行

初始 `10` 模型 multi-start 子集已经扩展到 `16` 个 predictor，新增：

- `itransformer_residual`
- `patchtst_residual`
- `transformer_forecaster`
- `nlinear_forecaster`
- `dlinear_forecaster`
- `itransformer_co2_wavelet_residual`

执行说明：

- 第二个长时间命令同样在完成并保存 suite JSON 后超过工具超时，返回 status `124`。
- 保存的 suite 是完整的，并已与前一个 10 模型 suite 合并。

已生成输出：

- `results/control/summaries/fctv_multistart_gradient_mpc_reference_96steps_6predictors_8e102971d9_starts_0_96_192.json`
- `results/control/summaries/fctv_multistart_gradient_mpc_reference_96steps_16predictors_starts_0_96_192.json`
- `results/forecasting/analysis/forecast_to_control_transfer_multistart16_reference.{json,csv,md}`
- 每个 start 的 `forecast_to_control_transfer_multistart16_reference_start*.{json,csv,md}` 和 robustness CSV
- `results/forecasting/figures/comparisons/forecast_to_control_transfer_multistart16_reference.png`
- `results/forecasting/analysis/fctv_multistart_model_rankings_reference.{csv,md}`
- `results/forecasting/figures/comparisons/fctv_multistart_model_rankings_reference.png`

16 模型 multi-start 指标结论：

- `co2_first_step_mae -> mpc_co2_mae` 仍然具有 segment dependence：
  - start `0`：`secondary_selection`，Spearman `0.366`，pairwise `0.630`
  - start `96`：`offline_or_diagnostic_only`，Spearman `-0.263`，pairwise `0.395`
  - start `192`：`offline_or_diagnostic_only`，Spearman `-0.243`，pairwise `0.412`
- `rhair_first_step_mae -> mpc_rhair_mae` 进一步变弱：
  - start `0`：`weak_selection`，Spearman `0.282`，pairwise `0.617`
  - start `96`：`offline_or_diagnostic_only`，Spearman `-0.068`，pairwise `0.458`
  - start `192`：`offline_or_diagnostic_only`，Spearman `0.174`，pairwise `0.583`
- `multiobjective_transfer_selection_score -> mpc_objective` 不稳定：
  - start `0`：`weak_selection`，Spearman `0.338`，pairwise `0.617`
  - start `96`：`offline_or_diagnostic_only`，Spearman `-0.074`，pairwise `0.458`
  - start `192`：`offline_or_diagnostic_only`，Spearman `0.144`，pairwise `0.567`
- `tair_first_step_mae -> mpc_tair_mae` 仍不可靠。

16 模型 multi-start 模型侧结论：

- `itransformer_co2_residual` 仍是最稳定的 CO2 闭环 tracking 模型：
  - start `0`：CO2 最优，`CO2air MAE=6.331`，objective `0.0558`
  - start `96`：CO2 最优，`CO2air MAE=11.074`，objective `0.0654`
  - start `192`：CO2 最优，`CO2air MAE=10.701`，objective `0.0465`
- 每个 start 下整体 objective 最优模型：
  - start `0`：`current_hybrid_transformer`，objective `0.0442`
  - start `96`：`current_hybrid_transformer`，objective `0.0517`
  - start `192`：`transformer_hybrid_residual`，objective `0.0235`
- 额外重要的 segment-specific 发现：
  - start `192`：`dlinear_forecaster` 达到 `CO2air MAE=11.316`，objective `0.0449`
  - start `192`：`itransformer_residual` 达到 `CO2air MAE=11.644`，objective `0.0360`

解释更新：

- 16 模型 multi-start 结果确认：当前没有任何 FCTV forecast-side metric 是稳定的 universal selector。
- FCTV 仍然适合作为诊断协议，用来识别 mismatch 和 segment dependence。
- 模型叙事现在应该强调稳健闭环证据：
  - `current_hybrid_transformer` 在 starts `0` 和 `96` 上仍是最强 objective-oriented baseline。
  - `transformer_hybrid_residual` 在 start `192` 上 objective 最强。
  - `itransformer_co2_residual` 在扩展 multi-start 子集中始终是最强 CO2 tracker。
- multi-start model ranking figure 已生成，用于直接对比不同 start 下的 objective 和 CO2 MAE。

## 31. 2026-04-29 FCTV 周报汇报图

已生成面向导师周报的 FCTV 结果链条汇总图。

新增脚本和输出：

- `agc_mpc/plot_fctv_weekly_metric_degradation.py`
- `results/forecasting/figures/comparisons/fctv_weekly_metric_degradation_summary.png`

图中要传达的信息：

- 早期 `17` 模型 CO2-focused FCTV 阶段确实出现了有用的筛选信号。
- 扩展到 `24` 模型池后，CO2 first-step 和 constraint-near 指标退化为 diagnostic-only 角色。
- 进一步扩展到 `16` 模型、starts `0`、`96`、`192` 后，主要 forecast-side metrics 表现出明显的模型池依赖和片段依赖。
- 这张图应用于汇报当前结论：离线预测指标不能可靠筛选闭环控制收益；FCTV 更适合作为诊断框架，闭环 MPC 验证仍然必要。

## 32. 2026-05-12 论文式 FCTV 阶段

探索性 FCTV 阶段已经收束。下一阶段使用论文式固定协议，而不是机会式追加实验。

新增长期维护文档：

- `agc_mpc/FCTV_EXPERIMENT_DESIGN.md`
- `agc_mpc/FCTV_EXPERIMENT_DESIGN.zh-CN.md`
- `agc_mpc/FCTV_METHOD_SECTION.md`
- `agc_mpc/FCTV_METHOD_SECTION.zh-CN.md`

新增可执行 benchmark 入口：

- `agc_mpc/run_fctv_final_closed_loop_benchmark.py`

本周决策：

- A 已在文档层面完成：固定 FCTV 论文问题、模型池、benchmark 协议、指标族和实验矩阵。
- B 已执行：最终 benchmark 为 `16` 个 predictor、starts `0`、`96`、`192`、`288`、`384` 生成了 `80` 条闭环记录。
- C 已完成草稿：FCTV 方法章节定义预测侧指标、闭环指标、Spearman 相关、两两模型排序一致率、top-k overlap 和稳健性检查。
- 以后不能默认回避模型和闭环实验。只要研究问题需要且计算时间允许，就应该运行；如果暂缓，必须记录计算成本和准确命令。

生成的最终 benchmark 输出：

- `results/control/summaries/fctv_multistart_gradient_mpc_reference_96steps_16predictors_25890932c3_starts_0_96_192_288_384.json`
- `results/forecasting/analysis/forecast_to_control_transfer_final_reference.{json,csv,md}`
- `results/forecasting/figures/comparisons/forecast_to_control_transfer_final_reference.png`
- `results/forecasting/analysis/fctv_final_multistart_model_rankings_reference.{csv,md}`
- `results/forecasting/figures/comparisons/fctv_final_multistart_model_rankings_reference.png`

最终 5 起点结果：

- 预测侧 transfer 指标仍然具有 start dependence，不能作为稳定 universal selector。
- `current_hybrid_transformer` 跨 start 平均 objective 最好：`0.0662 +/- 0.0269`。
- `itransformer_co2_residual` 跨 start 平均 CO2 闭环 tracking 最好：`CO2air MAE = 10.215 +/- 2.043`，同时平均 objective 第二：`0.0701 +/- 0.0234`。

下一阶段队列：

- F：基于最终 FCTV 设计和已有闭环证据，准备面向导师的阶段汇报。
- E：等 tracking-control benchmark 稳定后，启动 economic/resource-aware MPC formulation。

## 33. 2026-05-12 F 和 E 阶段执行

已完成任务 F：

- 新增 `agc_mpc/FCTV_STAGE_REPORT.md`。
- 新增 `agc_mpc/FCTV_STAGE_REPORT.zh-CN.md`。
- 该报告把 FCTV 结果表述为受控负结果 / 诊断结果，而不是项目失败。
- 报告给出面向导师的实验链条：CO2-focused 指标归纳、扩大模型池验证、多目标验证、多起点闭环验证。

已完成任务 E 的第一步可执行实现：

- 在 `agc_mpc/ECONOMIC_RESOURCE_MPC.md` 中新增 economic/resource-aware MPC formulation。
- 新增中文镜像 `agc_mpc/ECONOMIC_RESOURCE_MPC.zh-CN.md`。
- 在 `AGCConfig` 和 `PredictiveControlAdapter.control_cost()` 中新增默认关闭的 economic/resource objective 项。
- 在闭环 summary 中新增 `resource_proxy_mean`。
- 新增 `agc_mpc/run_economic_resource_mpc_probe.py`。
- 新增 `agc_mpc/analyze_economic_resource_probe.py`。

重要兼容规则：

- `economic_resource_weight = 0.0` 是默认值，因此之前的 FCTV 和 tracking-only MPC benchmark 仍保持可比。

已执行 E 阶段 smoke/probe：

- Tracking-only probe：`fctv_multistart_gradient_mpc_reference_24steps_2predictors_c5d60ca7a5_tracking_probe_w000_starts_0.json`。
- Economic/resource probe：`fctv_multistart_gradient_mpc_reference_24steps_2predictors_c5d60ca7a5_economic_probe_w015_starts_0.json`。
- 对比输出：`results/control/summaries/economic_resource_probe_comparison.{csv,md}` 和 `results/control/figures/economic_resource_probe_comparison.png`。

Probe 结果：

- `current_hybrid_transformer`：resource proxy `0.354 -> 0.332`（`-6.0%`），CO2 MAE `10.964 -> 12.380`。
- `itransformer_co2_residual`：resource proxy `0.377 -> 0.357`（`-5.3%`），CO2 MAE `2.938 -> 4.899`。

解释：

- Resource-aware MPC 代码路径已经跑通，并且会改变优化动作。
- 第一版 resource weight 已产生可量化的 resource-tracking trade-off，但这只是 24-step、single-start probe。
- 下一轮严谨 E 阶段实验应扫描 resource weights，并使用 96-step、multi-start rollouts。

补充 top-5 E 阶段 probe：

- Predictors：`current_hybrid_transformer`、`itransformer_co2_residual`、`segrnn_forecaster`、`transformer_forecaster`、`transformer_hybrid_residual`。
- Profiles：`tracking_top5_w000` vs `economic_top5_w015`。
- 输出：`results/control/summaries/economic_resource_top5_start0_24steps_comparison.{csv,md}` 和 `results/control/figures/economic_resource_top5_start0_24steps_comparison.png`。

Top-5 probe 解释：

- `transformer_forecaster` 的 resource proxy 降幅最大（`-8.6%`），同时 CO2 退化相对较小（`8.051 -> 8.486`）。
- `itransformer_co2_residual` 加入 economic term 后仍保留最好的绝对 CO2 tracking（`4.899`），但相比 tracking-only CO2 MAE（`2.938`）退化更明显。
- `transformer_hybrid_residual` 的 resource proxy 反而上升（`+2.3%`），说明当前 economic weight 不会让所有 predictor 都同向降低资源代理。

96-step top-3 resource-weight sweep：

- Predictors：`current_hybrid_transformer`、`itransformer_co2_residual`、`transformer_forecaster`。
- Starts：`0`、`96`、`192`。
- Weights：`0.00`、`0.05`、`0.15`、`0.30`。
- 输出：`results/control/summaries/economic_resource_sweep_top3_reference.{csv,md}` 和 `results/control/figures/economic_resource_sweep_top3_reference.png`。

Sweep 结论：

- `w=0.05` 是当前有价值区间：resource proxy 下降约 `6%` 到 `10%`，平均 CO2 退化约 `2%` 到 `4%`。
- `current_hybrid_transformer` 有最好的低权重 trade-off：resource proxy `-9.8%`，CO2 MAE `+2.1%`。
- `itransformer_co2_residual` 仍是绝对 CO2 tracking 最好的模型，但在高 resource weight 下更脆弱。
- `w=0.15` 和 `w=0.30` 虽然有更强 resource reduction，但会导致明显更大的 CO2 退化。

## 34. 已计划的主线真实 AGC 资源 / 经济验证

状态：已在 `2026-05-12` 执行完成；完成产物和结论见第 35 节。

该 Plan 模式任务不应继续把 FCTV 当作研究方向推进。FCTV 已经作为诊断 / 负结果分支收束。

主线提醒：

- 毕业设计主线仍然是 `面向控制的温室多步预测 + 闭环 MPC`。
- 下一步是把一个或少数几个已选预测模型推进到主线最终验证：使用 AGC 数据集里的真实资源 / 经济字段做验证。
- 不要默认跑所有模型。目标是框架级最终验证，不是再做大模型池 leaderboard。

核心思想：

- 用真实 AGC 资源和经济数据替代当前 action-level `resource_proxy` 叙事：
  - `Resources.csv`：`Heat_cons`、`ElecHigh`、`ElecLow`、`CO2_cons`、`Irr`、`Drain`
  - `Production.csv`：`ProdA`、`ProdB`、采收穗 / 果数量和重量
  - `TomQuality.csv`：`TSS/Brix`、flavour、acid、juice、bite、fruit weight
  - `Economics.pdf`：heat/electricity/CO2 价格、作物维护成本、按 Brix 和日期变化的番茄价格

推荐模型集：

- 必选：
  - `current_hybrid_transformer`：整体闭环 objective 最强 baseline
  - `itransformer_co2_residual`：proposed / 主线 CO2-aware 模型，也是最强闭环 CO2 tracker
- 如果时间允许，可选：
  - `transformer_forecaster`：economic probe 中表现出 resource-trade-off 潜力的候选

Plan 模式任务 1：检查并编码 AGC economics。

- 读取 `AutonomousGreenhouseChallenge_edition2/Economics.pdf`。
- 提取 net profit 公式：
  - `Net Profit = Income - Fixed costs - Variable costs`
- 编码 variable cost 规则：
  - peak electricity：`0.08 EUR/kWh`
  - off-peak electricity：`0.04 EUR/kWh`
  - heat：`0.0083 EUR/MJ`
  - CO2：前 `12 kg/m2` 为 `0.08 EUR/kg`，之后为 `0.2 EUR/kg`
  - crop maintenance：`0.0085 EUR per stem/m2 per day`
- 编码 tomato income 规则：
  - Class A：全价
  - Class B：半价
  - 番茄价格由日期和 Brix 决定，使用 `Economics.pdf` 中的价格表
- 如果 Brix 插值或作物维护成本无法完全复现，必须明确写出近似假设。

Plan 模式任务 2：实现真实 AGC economics analyzer。

- 新增脚本：`agc_mpc/analyze_agc_real_economics.py`。
- 输入：
  - AGC dataset root
  - compartment list，默认六个 AGC compartment 全部纳入
- 输出：
  - `results/control/summaries/agc_real_economics_by_compartment.csv`
  - `results/control/summaries/agc_real_economics_by_compartment.md`
  - `results/control/figures/agc_real_economics_by_compartment.png`
- 指标：
  - total heat consumption 和 heat cost
  - total peak/off-peak electricity 和 electricity cost
  - total CO2 consumption 和 CO2 cost
  - irrigation 和 drain totals
  - tomato Class A/B production
  - estimated income
  - estimated variable cost
  - approximate net profit
  - 如果可行，计算每 kg tomato resource use
- 目的：
  - 在评估我们 MPC rollout 之前，先建立真实 AGC resource/economic baseline。

Plan 模式任务 3：从 AGC 数据校准真实资源成本估计器。

- 新增脚本：`agc_mpc/calibrate_agc_resource_cost_model.py`。
- 使用 `GreenhouseClimate.csv`、`Weather/Weather.csv` 和 `Resources.csv`。
- 目标变量：
  - daily 或对齐后的 `Heat_cons`
  - `ElecHigh + ElecLow`
  - `CO2_cons`
  - `Irr`
- 候选解释变量：
  - `t_heat_sp`、`t_vent_sp`、`co2_sp`、`assim_sp`、`window_pos_lee_sp`、`water_sup_intervals_sp_min`
  - `Tout`、`Iglob`、`PARout`，以及可用 time features
- 优先使用简单可解释模型：
  - linear regression / ridge regression
  - 可行时使用非负系数
- 输出：
  - `results/control/summaries/agc_resource_cost_model_coefficients.csv`
  - `results/control/summaries/agc_resource_cost_model_validation.md`
  - `results/control/figures/agc_resource_cost_model_validation.png`
- 目的：
  - 把 MPC 生成的 action trajectories 转换为估计的真实 AGC resource cost，而不再只是 normalized action proxy。

Plan 模式任务 4：用真实资源估计成本评估主线模型。

- 新增脚本：`agc_mpc/evaluate_mainline_real_resource_control.py`。
- 如果已有 rollout summaries/traces 足够，直接复用；如果缺少 action 轨迹，则重新跑短闭环 rollout。
- 必选模型：
  - `current_hybrid_transformer`
  - `itransformer_co2_residual`
- 可选模型：
  - `transformer_forecaster`
- 协议：
  - `GradientMPC`
  - `Reference`
  - `96` steps
  - 如果可行，starts `0`、`96`、`192`、`288`、`384`
  - 必要时比较 tracking-only 和一个低 resource-aware setting
- 输出：
  - `results/control/summaries/mainline_real_resource_model_comparison.csv`
  - `results/control/summaries/mainline_real_resource_model_comparison.md`
  - `results/control/figures/mainline_real_resource_model_comparison.png`
- 指标：
  - closed-loop objective
  - `Tair`、`Rhair`、`CO2air` MAE
  - estimated heat cost
  - estimated electricity cost
  - estimated CO2 cost
  - estimated irrigation
  - estimated total variable resource cost
  - 相对 `current_hybrid_transformer` 的 cost/tracking trade-off

Plan 模式任务 5：生成面向论文的最终验证表述。

- 新增结果说明：
  - `results/control/summaries/mainline_real_resource_validation_conclusion.md`
- 结论边界：
  - 可以声明：已选预测模型可以在闭环 MPC 中用 real-AGC-resource-calibrated cost framework 进行评估。
  - 可以声明：可以比较少数模型的 estimated resource cost 和 tracking trade-off。
  - 不可以声明：真实整季 net profit 得到提升，因为当前 MPC rollout 不包含作物 / 产量动态模型。
- 面向论文的表述应为：
  - `current_hybrid_transformer` 是最强整体 tracking baseline。
  - `itransformer_co2_residual` 是最强 CO2-aware closed-loop tracker。
  - 真实 AGC resource/economic 数据可用于估计已选 MPC rollout 的资源影响。
  - 最终验证把 forecasting model 主线连接到 resource-aware greenhouse control，但不夸大为真实商业利润提升。

下一轮 Plan 模式推荐执行顺序：

1. 检查所有 AGC resource/production/quality 字段和 economics PDF。
2. 实现 `analyze_agc_real_economics.py`，复现各 compartment 的 resource/economic summary。
3. 实现第一版 resource-cost estimator，并用记录式 AGC resource consumption 验证。
4. 用该 estimator 评估 `current_hybrid_transformer` 和 `itransformer_co2_residual`。
5. 生成主线最终 comparison table、figure 和 conclusion note。
6. 更新 `CONTEXT.md` 和 `CONTEXT.zh-CN.md`。
7. 如果用户要求，按小段 commit/push。

## 35. 2026-05-12 主线真实 AGC 资源 / 经济验证

已完成计划中的真实 AGC 资源 / 经济验证阶段。

新增脚本：

- `agc_mpc/analyze_agc_real_economics.py`
- `agc_mpc/calibrate_agc_resource_cost_model.py`
- `agc_mpc/evaluate_mainline_real_resource_control.py`

Simulator 更新：

- `AGCClosedLoopSimulator` 现在会为每条 rollout 保存 trace JSON，包含时间戳、预测 / 参考目标、执行动作、记录动作、objective、动作变化和 resource proxy。
- 真实资源估计需要这些 trace，因为旧的 summary-only rollout 没有足够的动作序列信息。

真实 AGC economics baseline：

- 已编码 `Economics.pdf` 规则：
  - `Net Profit = Income - Fixed costs - Variable costs`
  - peak electricity `0.08 EUR/kWh`
  - off-peak electricity `0.04 EUR/kWh`
  - heat `0.0083 EUR/MJ`
  - CO2 前 `12 kg/m2` 为 `0.08 EUR/kg`，之后为 `0.20 EUR/kg`
  - crop maintenance `0.0085 EUR per stem/m2 per day`
  - Class A 番茄全价，Class B 番茄半价
  - 使用 PDF 表格中的日期和 Brix 相关番茄价格
- 输出：
  - `results/control/summaries/agc_real_economics_by_compartment.csv`
  - `results/control/summaries/agc_real_economics_by_compartment.md`
  - `results/control/figures/agc_real_economics_by_compartment.png`
- 近似 net-profit 排名：
  - `Automatoes`: `6.05 EUR/m2`
  - `AICU`: `5.85 EUR/m2`
  - `Reference`: `3.60 EUR/m2`
  - `IUACAAS`: `3.29 EUR/m2`
  - `Digilog`: `3.12 EUR/m2`
  - `TheAutomators`: `2.60 EUR/m2`

资源估计器：

- 在 AGC 日尺度记录上拟合了简单的非负系数 ridge 模型。
- 输入包括记录 setpoints、weather 和 derived drive terms 的日尺度汇总。
- 目标：
  - `Heat_cons`
  - `ElecHigh + ElecLow`
  - `CO2_cons`
  - `Irr`
- 输出：
  - `results/control/summaries/agc_resource_cost_model_coefficients.csv`
  - `results/control/summaries/agc_resource_cost_model_validation.md`
  - `results/control/summaries/agc_resource_cost_model.json`
  - `results/control/figures/agc_resource_cost_model_validation.png`
- 验证结果：
  - heat MAE `0.5657`，R2 `0.620`
  - electricity MAE `0.2816`，R2 `0.835`
  - CO2 MAE `0.0102`，R2 `0.731`
  - irrigation MAE `1.0140`，R2 `0.448`

已执行闭环 runs：

- 必选模型：
  - `current_hybrid_transformer`
  - `itransformer_co2_residual`
- 协议：
  - `GradientMPC`
  - `Reference`
  - `96` steps
  - starts `0`、`96`、`192`、`288`、`384`
  - tracking-only profile `real_resource_w000`
  - low resource-aware profile `real_resource_w005`
- 这些是使用已有 checkpoint 的真实 MPC rollout，不是只补文档。
- 生成 suites：
  - `results/control/summaries/fctv_multistart_gradient_mpc_reference_96steps_2predictors_c5d60ca7a5_real_resource_w000_starts_0_96_192_288_384.json`
  - `results/control/summaries/fctv_multistart_gradient_mpc_reference_96steps_2predictors_c5d60ca7a5_real_resource_w005_starts_0_96_192_288_384.json`

主对比输出：

- `results/control/summaries/mainline_real_resource_model_comparison.csv`
- `results/control/summaries/mainline_real_resource_model_comparison_details.csv`
- `results/control/summaries/mainline_real_resource_model_comparison.md`
- `results/control/figures/mainline_real_resource_model_comparison.png`
- `results/control/summaries/mainline_real_resource_validation_conclusion.md`

主要结果：

- `real_resource_w000 + current_hybrid_transformer`：
  - objective `0.0660`
  - `CO2air MAE = 29.472`
  - estimated resource cost `0.0127 EUR/m2`
- `real_resource_w000 + itransformer_co2_residual`：
  - objective `0.0695`
  - `CO2air MAE = 10.168`
  - estimated resource cost `0.0094 EUR/m2`
- `real_resource_w005 + current_hybrid_transformer`：
  - objective `0.0841`
  - `CO2air MAE = 29.929`
  - estimated resource cost `0.0114 EUR/m2`
- `real_resource_w005 + itransformer_co2_residual`：
  - objective `0.0879`
  - `CO2air MAE = 10.980`
  - estimated resource cost `0.0085 EUR/m2`

解释：

- `current_hybrid_transformer` 仍是 tracking-only mean objective 最强的整体 tracking baseline。
- `itransformer_co2_residual` 仍是最强 CO2-aware closed-loop tracker，并且在 selected-model real-resource 对比中有更低的估计资源成本。
- 低 resource-aware 设置 `w=0.05` 会降低两个已选模型的估计资源成本，但会提高 objective，并轻微恶化 CO2 tracking。
- 这支持面向论文的表述：已选 forecasting 模型可以在 real-AGC-resource-calibrated cost framework 下做闭环 MPC 评估。

边界：

- 不能声称真实整季 net profit 提升。
- 当前 MPC rollout 没有作物 / 产量 / 品质动态模型。
- 合法结论是：对已选闭环 MPC rollout 做 estimated resource-cost 和 tracking trade-off 对比。

下一步实际工作：

- 围绕这次最终验证写论文结果小节。
- 如果还需要补实验，应保持很窄：只围绕两个已选模型做 `w=0.02`、`w=0.05`、`w=0.08` 的敏感性检查。

## 36. 2026-05-12 主线真实资源后续任务完成

已完成真实资源验证阶段的剩余后续任务。

新增闭环 runs：

- 模型：
  - `current_hybrid_transformer`
  - `itransformer_co2_residual`
- 协议：
  - `GradientMPC`
  - `Reference`
  - `96` steps
  - starts `0`、`96`、`192`、`288`、`384`
- 新增 resource weights：
  - `real_resource_w002` = `w=0.02`
  - `real_resource_w008` = `w=0.08`
- 这些是使用已有 checkpoint 的真实 MPC rollout。

生成的 sensitivity suites：

- `results/control/summaries/fctv_multistart_gradient_mpc_reference_96steps_2predictors_c5d60ca7a5_real_resource_w002_starts_0_96_192_288_384.json`
- `results/control/summaries/fctv_multistart_gradient_mpc_reference_96steps_2predictors_c5d60ca7a5_real_resource_w008_starts_0_96_192_288_384.json`

生成的 sensitivity 分析：

- `results/control/summaries/mainline_real_resource_sensitivity.csv`
- `results/control/summaries/mainline_real_resource_sensitivity_details.csv`
- `results/control/summaries/mainline_real_resource_sensitivity.md`
- `results/control/figures/mainline_real_resource_sensitivity.png`

生成的最终汇报和写作材料：

- `agc_mpc/plot_mainline_real_resource_final_summary.py`
- `results/control/figures/mainline_real_resource_final_summary.png`
- `results/control/summaries/agc_resource_cost_model_coefficient_diagnosis.md`
- `results/control/summaries/mainline_real_resource_thesis_result_section.md`
- 已更新 `results/control/summaries/mainline_real_resource_validation_conclusion.md`

完整 sensitivity 结果：

- `w=0.00 + current_hybrid_transformer`：
  - objective `0.0660`
  - `CO2air MAE = 29.472`
  - estimated resource cost `0.0127 EUR/m2`
- `w=0.00 + itransformer_co2_residual`：
  - objective `0.0695`
  - `CO2air MAE = 10.168`
  - estimated resource cost `0.0094 EUR/m2`
- `w=0.02 + current_hybrid_transformer`：
  - objective `0.0743`
  - `CO2air MAE = 29.808`
  - estimated resource cost `0.0123 EUR/m2`
- `w=0.02 + itransformer_co2_residual`：
  - objective `0.0778`
  - `CO2air MAE = 10.297`
  - estimated resource cost `0.0096 EUR/m2`
- `w=0.05 + current_hybrid_transformer`：
  - objective `0.0841`
  - `CO2air MAE = 29.929`
  - estimated resource cost `0.0114 EUR/m2`
- `w=0.05 + itransformer_co2_residual`：
  - objective `0.0879`
  - `CO2air MAE = 10.980`
  - estimated resource cost `0.0085 EUR/m2`
- `w=0.08 + current_hybrid_transformer`：
  - objective `0.0941`
  - `CO2air MAE = 30.180`
  - estimated resource cost `0.0111 EUR/m2`
- `w=0.08 + itransformer_co2_residual`：
  - objective `0.0931`
  - `CO2air MAE = 11.660`
  - estimated resource cost `0.0076 EUR/m2`

Sensitivity 解读：

- `current_hybrid_transformer` 在 `w=0.00` 下仍是 tracking-only objective 最强 baseline。
- `itransformer_co2_residual` 在所有已测试 resource weights 下仍是最强 CO2-aware closed-loop tracker。
- 在这个 selected-model 对比中，`itransformer_co2_residual` 的估计资源成本也低于 `current_hybrid_transformer`。
- 提高 resource weight 会降低估计资源成本，但会提高优化 objective，并逐步恶化 `CO2air` tracking。
- `w=0.05` 是最有说服力的折中设置：
  - 它比 `w=0.02` 有更清楚的成本降低
  - 又避免了 `w=0.08` 中更大的 CO2 退化
- `w=0.08` 可作为更强 resource-saving 点，但应写成高 trade-off 设置，而不是默认推荐。

资源估计器系数诊断：

- heat 和 electricity 估计可用于粗粒度 resource-cost 对比。
- CO2 估计可用于已选 rollout 对比，但不能写成 mechanistic carbon-balance model。
- irrigation 验证质量较弱，只应作为辅助背景指标。

最终论文表述边界：

- 可以声明：
  - 已选 forecasting 模型可以在 closed-loop MPC 中用 real-AGC-resource-calibrated cost estimates 评估
  - `current_hybrid_transformer` 是最强整体 tracking baseline
  - `itransformer_co2_residual` 是最强 CO2-aware closed-loop tracker，并且在 selected comparison 中有较好的估计资源成本
  - 低 resource-aware MPC 权重揭示了可量化的 tracking-resource trade-off
- 不可以声明：
  - 真实整季商业 net-profit 提升
  - yield 或 quality 提升，因为当前 rollout 没有 crop/yield/quality 动态模型

下一步任务：

- 从实验推进转为论文写作和最终汇报材料组装。
- 除非论文论证明确需要，否则不要再扩成新的大模型 leaderboard。

## 37. 2026-05-13 导师汇报图

已用真实实验输出生成两张中文导师汇报图。

新增脚本：

- `agc_mpc/plot_supervisor_report_figures_cn.py`

生成图：

- `results/control/figures/supervisor_fig1_model_selection_cn.png`
  - 使用最终 16 模型、5 起点闭环验证表。
  - 展示代表模型在平均闭环 objective 和平均 `CO2air` MAE 上的位置。
  - 突出最终保留的两个主模型：
    - `current_hybrid_transformer`：均衡模型 / 整体 tracking baseline 最强。
    - `itransformer_co2_residual`：CO2 专项模型 / CO2 闭环 tracker 最强。
  - 左上角已经明确写出两个模型的原始名字：
    - 均衡模型：`current_hybrid_transformer`
    - CO2 专项模型：`itransformer_co2_residual`
- `results/control/figures/supervisor_fig2_resource_economic_cn.png`
  - 使用 `mainline_real_resource_sensitivity.csv`。
  - 展示估计资源成本随资源惩罚权重变化的趋势。
  - 展示更强资源惩罚带来的 CO2 tracking 代价。
  - 对比 `w=0.05` 下估计 heat、electricity、CO2 和 irrigation 消耗。
  - 展示 CO2 tracking 与估计资源成本的权衡。
- `results/control/figures/supervisor_fig2_w005_tradeoff_cn.png`
  - 用于解释为什么 `w=0.05` 是推荐折中设置的一张简化单图。
  - 实线表示估计资源成本，虚线表示 `CO2air` tracking 误差。
  - 图中突出说明资源惩罚越大，估计成本越低，但 CO2 tracking 误差越高。
  - 明确标注 `w=0.05` 为折中点，`w=0.08` 为更省资源但控制代价更大的点。
- `results/control/figures/supervisor_fig2_combined_tradeoff_resource_cn.png`
  - 这是 2026-05-13 按导师汇报需求整理后的最终图二。
  - 一张图里包含两个子图：
    - 左图：解释为什么选择 `w=0.05`，也就是资源成本和 CO2 控制误差之间的折中。
    - 右图：只展示 `w=0.05` 时两个模型的估计资源消耗对比。
  - 右图使用相对消耗，令 `current_hybrid_transformer` 等于 `1.00`，更方便直接说明 heat、electricity、CO2、irrigation 的差别。
  - 左上角也已经写出两个保留模型的原始名字。

解释提醒：

- 这两张图比较的是闭环 tracking 和估计资源成本。
- 不能用来声称真实整季 net-profit 提升，因为当前没有 crop/yield/quality 动态模型。

## 38. 已计划的经济 baseline 扩展：全时段锚定 MPC 与 AGC 资源基线

状态：已在 `2026-05-13` 计划。建议作为新对话中的下一轮 Plan 模式任务执行。

不要重新开启大模型搜索。下一步应该把已经选定的经济 / 资源 MPC 验证，扩展成更有说服力的 baseline 对比。

核心论文问题：

- 加入经济 / 资源项之后，已选控制器能不能在更长时间范围内降低估计资源成本，同时把温室气候控制退化控制在可接受范围？
- 我们的反事实 MPC 资源消耗估计，和真实 AGC `Reference` / AI 队伍在同一时间段内的真实资源消耗相比，处在什么水平？
- 只要还没有作物 / 产量 / 品质动态模型，就仍然不能声称真实整季商业净收益提升。

推荐三阶段计划：

### 阶段 1：全时段锚定闭环 MPC + 资源成本对比

目的：

- 不再只依赖少数几个 `96` 步片段。
- 用反复滚动优化覆盖测试期。
- 让资源成本结论不那么依赖手动挑选的起点。

实验协议：

- 每个决策点都用真实观测到的 AGC 状态 / 历史作为锚点。
- 每个锚点执行：
  - 从真实数据构造 `x_past`
  - MPC 优化未来 `96` 步
  - 只执行前 `1` 步或前 `24` 步
  - 锚点向前移动对应执行长度
  - 用下一段真实状态 / 历史重新锚定
- 一直重复，直到覆盖选定测试时间段。
- 推荐初始执行长度：
  - 主实验：`24` 步，因为计算量可控，并且和当前 `96` 步验证尺度一致
  - 可选严格检查：在较短片段上执行 `1` 步滚动，如果算力允许再做

必做控制器组合：

- `current_hybrid_transformer`，`w=0.00`
- `current_hybrid_transformer`，`w=0.05`
- `itransformer_co2_residual`，`w=0.00`
- `itransformer_co2_residual`，`w=0.05`

时间允许时的可选敏感性组合：

- `current_hybrid_transformer`，`w=0.02` 和 `w=0.08`
- `itransformer_co2_residual`，`w=0.02` 和 `w=0.08`

不要再加入新预测模型，除非论文论证明确需要。当前已选模型故事已经足够：

- `current_hybrid_transformer`：均衡模型，整体 tracking baseline 最强
- `itransformer_co2_residual`：CO2 专项模型，CO2-aware tracker 最强

建议实现：

- 新增 `agc_mpc/run_full_period_anchored_resource_mpc.py`。
- 复用 `AGCClosedLoopSimulator`、`PredictiveControlAdapter`、已选 checkpoint 和当前资源成本估计器。
- 如果现有 simulator 更适合固定 starts，就只加一个薄的调度器生成锚点和执行窗口，不要重写控制器内部。
- 保存两类结果：
  - 每个执行片段一行
  - 每个控制器组合一个聚合结果

建议输出：

- `results/control/summaries/full_period_anchored_resource_mpc_segments.csv`
- `results/control/summaries/full_period_anchored_resource_mpc_summary.csv`
- `results/control/summaries/full_period_anchored_resource_mpc_summary.md`
- `results/control/figures/full_period_anchored_resource_mpc_tradeoff.png`
- `results/control/figures/full_period_anchored_resource_mpc_cumulative_cost.png`

必须报告的控制指标：

- objective
- `Tair` MAE
- `Rhair` MAE
- `CO2air` MAE
- 如果已有边界信息，可选报告违反边界比例

必须报告的估计资源指标：

- 估计加热用量 / 加热成本
- 估计电力用量 / 电力成本
- 估计 CO2 用量 / CO2 成本
- 估计灌溉量
- 估计总资源成本

必须报告的权衡指标：

- 相对 `w=0.00` 的资源成本变化
- 相对 `w=0.00` 的 CO2 误差变化
- 相对 `w=0.00` 的 objective 变化
- 覆盖测试期内的累计估计资源成本

如果结果一致，可以声明：

- 经济 / 资源感知 MPC 在更长的锚定测试期仿真中带来了可测量的估计资源成本下降。
- 这种下降伴随明确的气候控制代价。
- 如果 `w=0.05` 能降低估计成本，同时避免 `w=0.08` 的更强 CO2 退化，那么 `w=0.05` 可以作为主要折中设置。

禁止声明：

- 仅凭阶段 1，不能声明真实净收益提升。
- 仅凭阶段 1，不能声明产量或品质提升。

### 阶段 2：AGC Reference / Automatoes / AICU 同时间段资源 baseline 对比

目的：

- 让最终 baseline 对比更接近温室控制论文和 AGC 比赛实践。
- 在同一时间窗口内，把我们的反事实 MPC 资源估计，和真实 AGC 资源记录进行对比。

baseline 集合：

- 必做：
  - AGC `Reference`
  - AGC `Automatoes`
  - AGC `AICU`
- 可选：
  - `Digilog`
  - `IUACAAS`
  - `TheAutomators`

对比规则：

- 使用和阶段 1 全时段锚定 MPC 相同的日期 / 时间索引。
- 除非建立了作物 / 产量 / 品质动态模型，否则只比较资源维度。
- 必须清楚标注：
  - AGC baselines：真实执行后的真实资源消耗和真实产量结果
  - 我们的 MPC：反事实锚定闭环仿真，资源消耗来自估计器

建议实现：

- 新增或扩展 `agc_mpc/analyze_agc_same_period_resource_baselines.py`。
- 输入：
  - AGC 数据集根目录
  - 阶段 1 选定的时间窗口
  - compartment / team 列表
  - 阶段 1 的 MPC summary 文件
- 输出：
  - AGC 同期真实资源总量
  - MPC 同期估计资源总量
  - 归一化相对对比表

建议输出：

- `results/control/summaries/agc_same_period_resource_baselines.csv`
- `results/control/summaries/agc_same_period_resource_baselines.md`
- `results/control/figures/agc_same_period_resource_baselines.png`
- `results/control/figures/agc_same_period_resource_cost_vs_control.png`

AGC 真实 baseline 必须报告：

- 加热消耗
- 高价 / 低价电力消耗
- CO2 消耗
- 灌溉量
- 按 AGC 官方规则估计的资源成本
- 真实产量和净收益可以作为背景信息展示，但不能和我们的估计结果直接混排成同一排名

我们的 MPC 必须报告：

- 估计 heat / electricity / CO2 / irrigation
- 估计资源成本
- 阶段 1 的 tracking 指标

推荐表格措辞：

- `Reference`：真实人工专家温室执行
- `Automatoes` / `AICU`：真实 AGC AI 队伍温室执行
- `Our MPC`：反事实锚定闭环仿真，资源消耗为估计值

如果结果合理，可以声明：

- 本文框架可以在匹配时间窗口内，把已选 MPC 控制器与真实 AGC baseline 做资源成本维度的对比。
- 这提供了一个不夸大产量 / 品质影响的 baseline 对比方式。

禁止声明：

- 不能按真实 net profit 声称我们的 MPC 超过 AGC 队伍。
- 不能把 AGC 的真实利润和我们的“资源成本估计”混成同一种经济排名。

### 阶段 3：完整净收益模型只作为未来工作

目的：

- 给出通向完整经济控制论文的路线，但不把当前毕业论文范围撑得过大。

为什么只能作为未来工作：

- 真实净收益不只需要资源成本，还需要产量、品质、采收时间和价格响应。
- 当前 MPC rollout 没有模拟作物生长、番茄产量、糖度 / 品质，也没有模拟不同控制动作对售价和产量的反事实影响。

未来需要的模型组件：

- 作物生长 / 生物量 / 产量动态模型
- 采收时间和 Class A / Class B 产量模型
- 品质或 Brix 预测模型
- 对 heat、electricity、CO2、irrigation 更强验证的资源模型
- 使用 AGC 官方收入和成本规则的经济计算器
- 对估计产量和品质的不确定性分析

未来完整对比对象：

- 真实 AGC `Reference` 和 AI 队伍
- tracking-only MPC
- resource-aware MPC
- 带作物 / 产量模型的 economic MPC
- 可选简单规则控制器

当前论文允许写：

- “本文在 AGC 校准的经济规则下评估资源成本与气候控制之间的权衡。”
- “完整净收益对比需要经过验证的作物 / 产量 / 品质响应模型，因此作为未来工作。”

当前论文禁止写：

- “本文控制器提高了真实净收益。”
- “本文控制器提高了产量或番茄品质。”
- “本文控制器在经济上超过 AGC 队伍。”

下一轮 Plan 模式推荐执行顺序：

1. 阅读 `CONTEXT.md`、`CONTEXT.zh-CN.md`、`ECONOMIC_RESOURCE_MPC.md`、`ECONOMIC_RESOURCE_MPC.zh-CN.md`。
2. 检查已有资源估计器输出和当前真实资源敏感性结果。
3. 先实现阶段 1 的全时段锚定 runner，执行长度先用 `24`。
4. 跑四个必做控制器组合。
5. 分析并画出阶段 1 结果。
6. 实现阶段 2 的 AGC 同期 baseline 分析器。
7. 生成 AGC-vs-MPC 匹配时间窗口资源对比表和图。
8. 写一段面向论文的结论说明，明确区分资源成本对比和真实净收益声明。
9. 更新 `CONTEXT.md` 和 `CONTEXT.zh-CN.md`。
10. 如果用户要求，按小批量 commit / push。

## 39. 2026-05-13 全时段锚定资源 MPC 实现与 smoke check

已经实现第 38 节所需工具，并完成两段 smoke 验证。

新增脚本：

- `agc_mpc/run_full_period_anchored_resource_mpc.py`
  - 生成非重叠锚点 `0, 24, 48, ...`。
  - 复用已选 checkpoint、`AGCClosedLoopSimulator`、`GradientMPCController`、`PredictiveControlAdapter` 和已经校准的 `agc_resource_cost_model.json`。
  - 输出每个执行片段一行，以及每个 `(predictor, resource_weight)` 一行聚合结果。
  - 保留片段 trace / summary；长时运行默认不生成每个片段的单独图，避免产生成千上百张图。
- `agc_mpc/analyze_agc_same_period_resource_baselines.py`
  - 读取阶段 1 的 segment rows。
  - 用阶段 1 的时间窗口汇总 `Reference`、`Automatoes`、`AICU` 的真实 AGC `Resources.csv` 记录。
  - 对比 AGC 真实执行资源消耗和 MPC 反事实估计资源消耗。

Simulator / config 更新：

- 新增 `AGCConfig.control_save_rollout_figures`，默认值为 `True`。
- `AGCClosedLoopSimulator` 会遵守该开关，因此长时间锚定运行可以跳过单个 rollout 图。

已执行 smoke 命令：

```powershell
C:\Users\wokis\.conda\envs\strawberry_env\python.exe agc_mpc\run_full_period_anchored_resource_mpc.py --max-segments 2
C:\Users\wokis\.conda\envs\strawberry_env\python.exe agc_mpc\analyze_agc_same_period_resource_baselines.py --segments-csv agc_mpc\results\control\summaries\full_period_anchored_resource_mpc_segments_smoke_2segments_3eb7671f.csv --prefix agc_same_period_resource_baselines_smoke_2segments_3eb7671f
```

Smoke 输出：

- `results/control/summaries/full_period_anchored_resource_mpc_segments_smoke_2segments_3eb7671f.csv`
- `results/control/summaries/full_period_anchored_resource_mpc_summary_smoke_2segments_3eb7671f.csv`
- `results/control/summaries/full_period_anchored_resource_mpc_summary_smoke_2segments_3eb7671f.md`
- `results/control/figures/full_period_anchored_resource_mpc_tradeoff_smoke_2segments_3eb7671f.png`
- `results/control/figures/full_period_anchored_resource_mpc_cumulative_cost_smoke_2segments_3eb7671f.png`
- `results/control/summaries/agc_same_period_resource_baselines_smoke_2segments_3eb7671f.csv`
- `results/control/summaries/agc_same_period_resource_baselines_smoke_2segments_3eb7671f.md`
- `results/control/figures/agc_same_period_resource_baselines_smoke_2segments_3eb7671f.png`
- `results/control/figures/agc_same_period_resource_baselines_smoke_2segments_3eb7671f_cost_vs_control.png`

Smoke 验证：

- 阶段 1 segment rows：`8`。
- 阶段 1 summary rows：`4`。
- 阶段 2 baseline rows：`7`。
- 必要的 segment 和 baseline 列都存在。
- 聚合资源成本与 segment 行求和一致，最大绝对误差约 `1e-16`。
- Smoke 时间窗口：`2020-05-06 06:25` 到 `2020-05-06 10:20`。

Smoke 结果摘要：

- `current_hybrid_transformer`，`w=0.05`：估计资源成本相对同模型 `w=0.00` smoke baseline 下降 `15.8%`。
- `itransformer_co2_residual`，`w=0.05`：估计资源成本相对同模型 `w=0.00` smoke baseline 下降 `10.4%`。
- 这些只是 smoke 结果，不能作为最终论文结论。

计算量备注：

- 两段 smoke 共执行 `8` 个锚定 24-step rollout，耗时约 `6 min 52 s`。
- 完整计划是 `283` 个 segment、`4` 个控制器组合，也就是 `1132` 个锚定 24-step rollout。
- 按 smoke 吞吐估计，完整前台运行很可能是多小时 / 过夜任务。

有计算窗口时的正式完整运行命令：

```powershell
C:\Users\wokis\.conda\envs\strawberry_env\python.exe agc_mpc\run_full_period_anchored_resource_mpc.py
C:\Users\wokis\.conda\envs\strawberry_env\python.exe agc_mpc\analyze_agc_same_period_resource_baselines.py
```

声明边界不变：

- 可以声明：匹配时间窗口下的资源成本与气候控制权衡对比。
- 不能声明：真实净收益提升、产量提升、品质提升，或经济上超过 AGC 队伍。

## 40. 2026-05-14 全时段锚定资源 MPC 全量实验完成

已经完成第 38 节的过夜全量锚定实验。

运行日志：

- 通过 PowerShell 后台启动器在 `2026-05-13 22:40` 启动。
- 输出大约在 `2026-05-14 14:07` 完成。
- 近似 wall time：`15 h 27 min`。
- 日志文件：
  - `results/control/summaries/full_period_anchored_resource_mpc_fullrun_20260513_224001.out.log`
  - `results/control/summaries/full_period_anchored_resource_mpc_fullrun_20260513_224001.err.log`
- stderr 只有 PyTorch warnings，没有观察到致命错误。

全量实验协议：

- 隔间：`Reference`
- 锚定方式：每个片段都用真实 AGC test history 重新锚定
- 执行长度：`24` steps
- 片段起点：`0, 24, 48, ...`
- 每个控制器组合的片段数：`283`
- 总 segment rows：`1132`
- 覆盖时间窗口：`2020-05-06 06:25` 到 `2020-05-29 20:20`
- 控制器组合：
  - `current_hybrid_transformer`，`w=0.00`
  - `current_hybrid_transformer`，`w=0.05`
  - `itransformer_co2_residual`，`w=0.00`
  - `itransformer_co2_residual`，`w=0.05`

生成的正式输出：

- `results/control/summaries/full_period_anchored_resource_mpc_segments.csv`
- `results/control/summaries/full_period_anchored_resource_mpc_summary.csv`
- `results/control/summaries/full_period_anchored_resource_mpc_summary.md`
- `results/control/summaries/full_period_anchored_resource_mpc_suite.json`
- `results/control/figures/full_period_anchored_resource_mpc_tradeoff.png`
- `results/control/figures/full_period_anchored_resource_mpc_cumulative_cost.png`
- `results/control/summaries/agc_same_period_resource_baselines.csv`
- `results/control/summaries/agc_same_period_resource_baselines.md`
- `results/control/figures/agc_same_period_resource_baselines.png`
- `results/control/figures/agc_same_period_resource_cost_vs_control.png`

验证检查：

- `full_period_anchored_resource_mpc_segments.csv`：`1132` 行。
- `full_period_anchored_resource_mpc_summary.csv`：`4` 行。
- `agc_same_period_resource_baselines.csv`：`7` 行。
- 每个控制器组合正好有 `283` 个片段。
- segment 求和与 summary 聚合的最大绝对误差约 `1.26e-14`。

全时段锚定 MPC 结果：

| predictor | w | objective | CO2 MAE | estimated resource cost | cost vs w=0 | CO2 vs w=0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `current_hybrid_transformer` | `0.00` | `0.0632` | `27.659` | `0.3455 EUR/m2` | `0.0%` | `0.0%` |
| `current_hybrid_transformer` | `0.05` | `0.0794` | `27.811` | `0.3215 EUR/m2` | `-6.9%` | `+0.6%` |
| `itransformer_co2_residual` | `0.00` | `0.0710` | `15.112` | `0.3423 EUR/m2` | `0.0%` | `0.0%` |
| `itransformer_co2_residual` | `0.05` | `0.0897` | `15.706` | `0.3133 EUR/m2` | `-8.5%` | `+3.9%` |

同时间段 AGC 资源 baseline 对比：

| case | source | estimated / real resource cost | vs real Reference | CO2 MAE |
| --- | --- | ---: | ---: | ---: |
| `Reference` | 真实 AGC 执行资源 | `0.3425 EUR/m2` | `0.0%` | n/a |
| `Automatoes` | 真实 AGC 执行资源 | `0.3674 EUR/m2` | `+7.3%` | n/a |
| `AICU` | 真实 AGC 执行资源 | `0.2579 EUR/m2` | `-24.7%` | n/a |
| `current_hybrid_transformer`, `w=0.00` | MPC 反事实估计资源 | `0.3455 EUR/m2` | `+0.9%` | `27.659` |
| `current_hybrid_transformer`, `w=0.05` | MPC 反事实估计资源 | `0.3215 EUR/m2` | `-6.1%` | `27.811` |
| `itransformer_co2_residual`, `w=0.00` | MPC 反事实估计资源 | `0.3423 EUR/m2` | `-0.1%` | `15.112` |
| `itransformer_co2_residual`, `w=0.05` | MPC 反事实估计资源 | `0.3133 EUR/m2` | `-8.5%` | `15.706` |

解读：

- 全时段锚定结果与此前短 probe 和 multi-start 资源检查一致：`w=0.05` 会降低两个已选 MPC 控制器的估计资源成本。
- `current_hybrid_transformer` 获得 `6.9%` 的估计成本下降，同时相对自身 `w=0.00` 锚定 baseline 只有 `0.6%` 的 CO2 MAE 增加。
- `itransformer_co2_residual` 获得 `8.5%` 的估计成本下降，但 CO2 MAE 增加更明显，为 `3.9%`。
- `itransformer_co2_residual` 在这次全时段锚定对比中仍然是更好的 CO2 tracking 控制器。
- 同期 AGC 表应该写作资源成本参照，而不是真实经济排名，因为 AGC 行是真实执行资源，MPC 行是没有 crop/yield/quality 动态的反事实估计。

当前允许的论文声明：

- 已选 MPC 控制器可以在匹配时间窗口下与真实 AGC baseline 做资源成本维度对比。
- 低资源权重 `w=0.05` 在长时间锚定测试窗口中带来了可测量的估计资源成本下降，同时伴随明确的 CO2 tracking 权衡。

当前禁止的论文声明：

- 不能声明真实商业净收益提升。
- 不能声明产量或品质提升。
- 不能声明 MPC 控制器在经济上超过 AGC 队伍。

## 41. 2026-05-19 AGC 全队伍同期资源 baseline 完成

已经完成“方法一”：不再运行新的 MPC 模型，直接计算 AGC 全部队伍在同一时间窗口内的真实资源 baseline。

新增脚本：

- `agc_mpc/analyze_agc_same_period_all_teams_resource_baselines.py`

目的：

- 直接计算 6 个 AGC 队伍在 full-period anchored MPC 同一时间窗口内的真实资源消耗。
- 增加真实队伍单位产量资源强度。
- 增加真实队伍同窗口产量和近似 variable-cost 经济背景。
- MPC 行只作为反事实估计资源参照，不进入真实经济排名。

生成输出：

- `results/control/summaries/agc_same_period_all_teams_resource_baselines.csv`
- `results/control/summaries/agc_same_period_all_teams_resource_baselines.md`
- `results/control/summaries/agc_same_period_all_teams_resource_intensity.csv`
- `results/control/summaries/agc_same_period_all_teams_economic_context.csv`
- `results/control/summaries/agc_same_period_all_teams_economic_context.md`
- `results/control/figures/agc_same_period_all_teams_resource_baselines.png`
- `results/control/figures/agc_same_period_all_teams_resource_intensity.png`
- `results/control/figures/agc_same_period_all_teams_economic_context.png`

同期资源成本排序：

| case | source | resource cost | vs Reference | vs AICU |
| --- | --- | ---: | ---: | ---: |
| `IUACAAS` | real AGC | `0.1991 EUR/m2` | `-41.9%` | `-22.8%` |
| `AICU` | real AGC | `0.2579 EUR/m2` | `-24.7%` | `0.0%` |
| `itransformer_co2_residual`, `w=0.05` | MPC estimated | `0.3133 EUR/m2` | `-8.5%` | `+21.5%` |
| `current_hybrid_transformer`, `w=0.05` | MPC estimated | `0.3215 EUR/m2` | `-6.1%` | `+24.7%` |
| `Reference` | real AGC | `0.3425 EUR/m2` | `0.0%` | `+32.8%` |
| `Automatoes` | real AGC | `0.3674 EUR/m2` | `+7.3%` | `+42.5%` |
| `TheAutomators` | real AGC | `0.3654 EUR/m2` | `+6.7%` | `+41.7%` |
| `Digilog` | real AGC | `0.4335 EUR/m2` | `+26.6%` | `+68.1%` |

真实队伍同期经济背景：

| compartment | income | variable cost excl fixed | margin excl fixed | tomato kg/m2 |
| --- | ---: | ---: | ---: | ---: |
| `TheAutomators` | `10.946` | `1.467` | `9.479` | `5.086` |
| `AICU` | `10.337` | `1.176` | `9.161` | `5.010` |
| `Reference` | `10.224` | `1.975` | `8.249` | `4.960` |
| `Digilog` | `9.728` | `1.494` | `8.234` | `5.073` |
| `Automatoes` | `9.392` | `1.561` | `7.831` | `5.107` |
| `IUACAAS` | `7.931` | `1.178` | `6.753` | `4.067` |

解读：

- 全队伍表修正了此前简化表述：在原本挑选的三个 baseline 队伍中，`AICU` 的同期资源成本最低；但扩展到 6 个队伍后，`IUACAAS` 的记录资源成本更低。
- 不能因为 `IUACAAS` 资源成本最低就把它写成最优经济 baseline，因为它同窗口产量和收入明显更低。
- 如果同时考虑资源和产量背景，`AICU` 仍然是更强的低资源真实队伍参照。
- 我们最好的 MPC 估计资源结果低于真实 `Reference`，但仍高于真实 `AICU` 和 `IUACAAS`。
- 这支持更稳妥的论文表述：本文 MPC 相比人工专家 `Reference` 降低了估计资源成本，但没有达到最省资源 AGC 队伍水平；没有作物 / 产量 / 品质动态模型时不能做真实经济排名。

边界：

- 真实 AGC 队伍经济背景只适用于真实记录队伍。
- MPC 行仍然只是反事实资源估计。
- 不要把 MPC 放入真实队伍的 production / margin 排名。

## 42. 2026-05-19 资源 baseline 汇报图

已经为全时段锚定 MPC 和 AGC 全队伍同期资源 baseline 结果生成汇报图。

新增脚本：

- `agc_mpc/plot_resource_baseline_report_figures.py`

生成图：

- `results/control/figures/resource_report_fig1_mpc_tradeoff.png`
  - 展示 full-period anchored MPC 在 `w=0.05` 下的估计资源成本下降。
  - 同时展示对应的 CO2 tracking 代价。
- `results/control/figures/resource_report_fig2_all_team_resource_baseline.png`
  - 按同期资源成本排序展示所有真实 AGC 队伍，以及两个 `w=0.05` MPC 估计资源结果。
  - 图中加入真实 `Reference` 和 `AICU` 参考线。
- `results/control/figures/resource_report_fig3_real_team_resource_intensity.png`
  - 展示真实 AGC 队伍单位 kg 番茄资源成本。
  - 同时展示 heat、CO2、irrigation 的单位 kg 番茄物理资源强度。
- `results/control/figures/resource_report_fig4_real_team_economic_context.png`
  - 展示真实 AGC 同窗口 income、剔除 fixed plant cost 的 variable cost、以及剔除 fixed plant cost 的 margin。
  - 因为当前没有 crop/yield/quality 动态模型，MPC 不进入这张真实经济背景图。
- `results/control/figures/resource_report_fig5_summary_dashboard.png`
  - 单页 dashboard，同时展示 MPC 成本下降、AGC 资源 baseline、CO2 权衡和真实队伍经济背景。

汇报提醒：

- 先用 Figure 5 作为总览页。
- 再用 Figures 1-4 分层解释各部分结果。
- 必须保留边界：资源成本对比成立；MPC 不能进入真实 net-profit 排名。
