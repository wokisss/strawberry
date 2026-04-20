# CO2_SPECIALIST_REPORT.zh-CN.md

中文对齐翻译版本。
英文主版本： [CO2_SPECIALIST_REPORT.md](c:/repositories/strawberry/agc_mpc/CO2_SPECIALIST_REPORT.md)
最近同步时间：`2026-04-14`

## 1. 这份报告覆盖什么

这份报告只回答一个问题：

在 [CO2_PAPERS_AND_DIRECTION.md](c:/repositories/strawberry/agc_mpc/CO2_PAPERS_AND_DIRECTION.md) 里列出的“直接做温室 `CO2` 预测”的论文中，哪些方法最值得先实现，哪些已经实现了，当前效果怎么样，以及下一步应该怎样并回当前 `agc_mpc` 主线。

这里的“已实现”指的是：

- 已经在当前 `AGC` 数据接口下变成可运行模型
- 已经按正式 fair-budget benchmark 跑过
- 已经整理出模型结构、原理、迁移价值和下一步优先级

## 2. 核心结论

目前已经落地了 3 条独立的 `CO2air` 专项预测线：

1. `co2_env_lstm`
2. `co2_vmd_lstm_fusion`
3. `co2_wavelet_gru_attn`

当前 fair-budget 结果排序：

1. `co2_wavelet_gru_attn`
   - Full `R2 = 0.7519`
   - Full `MAE = 45.209`
   - Final `R2 = 0.6159`
   - Final `MAE = 58.292`
2. `co2_vmd_lstm_fusion`
   - Full `R2 = 0.6863`
   - Full `MAE = 52.298`
   - Final `R2 = 0.6003`
   - Final `MAE = 59.697`
3. `co2_env_lstm`
   - Full `R2 = 0.3065`
   - Full `MAE = 74.157`
   - Final `R2 = -0.4852`
   - Final `MAE = 118.800`

直接解读：

- 只靠“环境因子 + LSTM”的路线不够强。
- `CO2` 更适合多尺度分解和自适应融合。
- 这与文献结论一致。
- 当前最强的独立 CO2 方向是 `wavelet / multi-scale + GRU + adaptive attention`。
- 直接端到端多目标迁移失败了，但更解耦的 horizon-wise frozen-expert 并回已经成功。
- 当前最强多目标 CO2 结果是 `itransformer_co2_horizon_mixture`，Full `CO2air MAE = 43.910`，Final `CO2air MAE = 47.661`。
- 这个离线 leader 的第一次闭环转化失败了；后续 frozen-backbone 版本恢复了控制梯度，形成了更适合 MPC 的安全折中。

## 3. 已落地文件

### 3.1 模型文件

- [co2_specialist_forecasters.py](c:/repositories/strawberry/agc_mpc/models/co2_specialist_forecasters.py)

已实现模型：

- `ConditionalCO2LSTMForecaster`
- `ConditionalCO2VMDLSTMFusionForecaster`
- `ConditionalCO2WaveletGRUAttnForecaster`

### 3.2 Benchmark 入口

- [benchmark_co2_specialist_forecasters.py](c:/repositories/strawberry/agc_mpc/benchmark_co2_specialist_forecasters.py)

协议：

- 数据集：`AGC`
- regime：`joint_all + Reference eval`
- 目标：只保留 `CO2air`
- 预算：`batch_size=256`, `epochs=200`, `lr=1e-4`, `patience=15`

### 3.3 画图入口

- [plot_co2_specialist_forecasters.py](c:/repositories/strawberry/agc_mpc/plot_co2_specialist_forecasters.py)

图输出目录：

- [results/forecasting/figures/co2_specialists](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/co2_specialists)

## 4. 论文到模型的映射

这里的“论文映射”要分两层看：

1. 原论文真正想解决什么问题
2. 我们怎样把那个思想翻译成当前仓库里的可运行结构

需要明确：

- 下面这些实现是 paper-inspired 的工程落地版
- 不是完整逐项复现实验
- `WT`、`VMD`、`SSA`、`DBO` 或精确 wavelet 工具链，在当前仓库里都是以适配现有训练栈的方式做近似实现

## 5. 逐篇论文汇报

### 5.1 Prediction of CO2 Concentration via Long Short-Term Memory Using Environmental Factors in Greenhouses

来源：

- [Horticultural Science and Technology / DOI 10.7235/HORT.20200019](https://www.hst-j.org/articles/xml/ozK9/)
- [KCI record](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002578287)

这篇论文做什么：

- 它直接预测温室 `CO2`。
- 它不是只把 `CO2` 当成多输出中的一个通道。
- 场景是芒果温室。
- 输入包括温度、湿度、太阳辐射、气压、土温、土壤湿度以及历史 `CO2`。
- 输出是未来 `2 h` 的 `CO2`。

核心思想：

- `CO2` 值得单独建模。
- `LSTM` 这类递归模型可以吸收环境和温室运行方式造成的滞后效应。
- 历史 `CO2` 本身就是最强信号之一。

高层结构：

1. 输入一段历史环境序列
2. 用 `LSTM` 编码时间依赖
3. 解码未来 `CO2`

这篇论文真正告诉我们的：

- 关键不是“LSTM 无敌”
- 关键是 `CO2` 应该有专门建模路径
- 如果没有控制量或运行日志，模型很容易低估补碳驱动的峰值

在仓库里的实现：

- 对应模型：`co2_env_lstm`
- 文件：[co2_specialist_forecasters.py](c:/repositories/strawberry/agc_mpc/models/co2_specialist_forecasters.py)

当前落地结构：

1. `x_past` 进入 `LSTM` encoder
2. `w_future + u_future` 进入未来条件嵌入
3. decoder `LSTM` 生成未来隐状态
4. 模型预测的是“最后观测 `CO2` 之上的增量”

为什么最后观测锚点重要：

- 直接回归绝对 `CO2` 轨迹不稳定
- `CO2` 强自回归
- 最后一个观测值是必要锚点
- 这和自回归、`NARX` 风格建模是一致的

当前表现如何解读：

- 它是 3 个独立 CO2 模型里最弱的
- 说明单一、朴素的递归主干不足以解决温室 `CO2`
- 但它依然是一个干净的专项 baseline

我们能借什么：

1. 单目标 `CO2` 建模是合理的
2. 必须保留自回归锚点
3. 纯 `LSTM` 线适合做 teacher、baseline 或 ablation

如何并回主线：

- 不建议直接替换主多目标预测器
- 更适合作为干净的 CO2 专项 baseline 或辅助 expert

优先级：

- 中

### 5.2 Time-serial analysis of deep neural network models for prediction of climatic conditions inside a greenhouse

来源：

- [ScienceDirect / DOI 10.1016/j.compag.2020.105402](https://www.sciencedirect.com/science/article/pii/S0168169919317326)
- [KIST abstract page](https://pubs.kist.re.kr/handle/201004/118578)

这篇论文做什么：

- 它不只预测 `CO2`
- 它联合比较温度、湿度和 `CO2`
- 它对比 `ANN`、`NARX` 和 `RNN-LSTM`
- 重点是这些方法在时间序列预测场景下的表现

最重要结论：

- 在温室这种有强时序滞后的系统里，纯前馈 `ANN` 不够
- `NARX` 和 `RNN-LSTM` 更匹配这种动力学
- `RNN-LSTM` 是研究中最稳定的方法族
- `CO2` 明显比温度更难预测

为什么重要：

- 它说明递归记忆在温室预测里仍然关键
- 它提醒我们不要默认更大的通用结构一定更好
- 它再次确认 `CO2` 需要比简单变量更强的时间建模

我们如何借用：

- 我们没有完整复现 `ANN / NARX / RNN-LSTM` 对照栈
- 而是把它的实用结论浓缩进 `co2_env_lstm` 这条线
- 核心问题是：一个干净的 CO2-only recurrent baseline 能走到哪里

后续还能借什么：

1. 在 CO2 建模里保留 recurrent branch
2. 如有需要，可再加更显式的自回归输入
3. 不要过早放弃递归专项模型

优先级：

- 高，但更多是结构依据，不一定是最终最强架构

### 5.3 Multi-model fusion method for predicting CO2 concentration in greenhouse tomatoes

来源：

- [ScienceDirect / DOI 10.1016/j.compag.2024.109623](https://www.sciencedirect.com/science/article/pii/S0168169924010147)

这篇论文做什么：

- 它直接预测番茄温室 `CO2`
- 它明确认为单一模型不足以处理非平稳且带噪声的 `CO2`
- 所以走的是“先分解，再建模，再融合”的路线

从摘要和亮点能稳定提炼出的结构：

1. `WT` 去噪
2. `VMD` 做多尺度分解
3. `LSTM` 建模分解后的成分
4. `attention` 强调重要时间内容
5. 最后融合成 `CO2` 预测

核心原理：

- `CO2` 不是单一时间尺度变量
- 它混合了慢的昼夜趋势、中等尺度的通风或补碳变化，以及尖锐局部扰动
- 这些频带应该分开建模，再自适应融合

为什么这和我们的问题直接相关：

- 我们已经观察到 `CO2air` 平均指标看起来还行，但 rollout 窗口会严重漂移
- 这正是混合尺度没有处理好时的典型失败模式

在仓库里的实现：

- 对应模型：`co2_vmd_lstm_fusion`

当前落地结构：

1. 用平滑滤波近似 trend/detail 分解
2. trend 和 detail 分别进入 `LSTM` encoder
3. 把未来天气和未来控制变成 query token
4. 分别对两个分支做 attention
5. 用动态门控做分支融合
6. 最后预测“最后观测 `CO2` 之上的增量”

为什么这是合理的 paper-inspired 翻译：

- 当前仓库没有完整 `WT + VMD` 工具链
- 但关键骨架 `decomposition + LSTM + attention + fusion` 被保留下来了
- 对当前工程来说，这是最实用的近似版

如何解读当前表现：

- 它明显优于纯 `LSTM`
- 说明分解和融合确实有效
- 但它还不是当前最强的独立 CO2 线

我们能借什么：

1. 保留多尺度分解
2. 不要只用单个 encoder 吃掉所有 CO2 模式
3. 用动态融合而不是固定权重

如何并回主线：

- 最适合做当前多目标 predictor 里的 `CO2 residual expert`
- 不适合直接替换整个多目标主模型

优先级：

- 很高

### 5.4 Prediction of CO2 concentration in mushroom greenhouse via optimized long and short term memory algorithm

来源：

- [Scientific Reports / DOI 10.1038/s41598-025-86394-0](https://www.nature.com/articles/s41598-025-86394-0)
- [PMC open version](https://pmc.ncbi.nlm.nih.gov/articles/PMC12485007/)

这篇论文做什么：

- 它预测食用菌温室中的 `CO2`
- 它不只是换 backbone
- 它把分解和优化一起引进来
- 对比模型包括：
  - `LSTM`
  - `EMD-LSTM`
  - `VMD-LSTM`
  - `VMD-SSA-LSTM`
  - `VMD-DBO-LSTM`

很清楚的建模链条：

1. 先把 `CO2` 序列分解成多个成分
2. 再让 `LSTM` 去学这些分解成分
3. 再用 `SSA` 或 `DBO` 这类优化算法找更好的超参数
4. 最终得到更强的预测器

这篇论文带来的两层启发：

第一层：

- `VMD` 风格分解对 `CO2` 有帮助

第二层：

- 性能提升不只来自 backbone
- 也来自分解配置和超参数搜索

为什么这点重要：

- 它提醒我们不要只盯着结构替换
- 对 `CO2` 来说，hidden size、learning rate、分解粒度、horizon 权重都可能很敏感

它在当前仓库里的体现：

- 它是 `co2_vmd_lstm_fusion` 的主要设计依据之一
- 目前还没有直接把 `SSA` 或 `DBO` 搜索落地到仓库里

下一步能借什么：

1. 给分解融合线加结构化超参数搜索
2. 重点搜索：
   - 分解粒度
   - hidden size
   - learning rate
   - horizon-aware loss weighting
3. 先做离线搜索，不急着放进主训练循环

优先级：

- 高

### 5.5 Wavelet-decoupled GRU with adaptive attention for multi-step carbon dioxide concentration prediction in intelligent glass greenhouse

来源：

- [ScienceDirect / DOI 10.1016/j.atech.2025.101653](https://www.sciencedirect.com/science/article/pii/S2772375525008846)

这篇论文做什么：

- 这是当前最贴近我们问题的一篇
- 它就是为多步温室 `CO2` 预测设计的
- 它直接针对长预测窗口中的误差累积问题

摘要和亮点给出的结构轮廓很清楚：

1. 前端做 wavelet 或频率解耦
2. 用 `GRU` 处理分解后的多尺度特征
3. 用可随位置调整的多头注意力做多步预测
4. 明确关注 `1 h / 2 h / 4 h / 8 h` 场景

核心原理：

- 长时域 `CO2` 预测变差，不只是因为模型弱
- 更因为：
  - 慢周期和突发扰动混在一起
  - 不同频带的重要性会随 horizon 改变
- 所以模型应该：
  - 先分开时间频带
  - 再随 horizon 改变融合权重

为什么它现在最适合我们：

- 我们当前问题正是 `CO2air` 在后段预测步明显变差
- 这篇论文针对的是多步窗口，而不是点预测
- 它明确把后段 horizon 当成与前段不同的问题

在仓库里的实现：

- 对应模型：`co2_wavelet_gru_attn`

当前落地结构：

1. 用平滑滤波近似构造 `low / mid / high` 三个时间频带
2. 每个频带单独进入一个 `GRU` encoder
3. 用天气、控制量和 horizon ratio 组成未来 query token
4. 分别对三个频带做 attention
5. 用 softmax 自适应权重做融合
6. 最后预测“最后观测 `CO2` 之上的增量”

为什么这条线当前最强：

- 它同时抓住了“多尺度”和“horizon-aware”两个关键点
- 这正是当前 `CO2` 最需要的能力

如何解读当前表现：

- 它是 3 个独立 CO2 模型里最强的
- Full `MAE = 45.209`，已经优于当前多目标 `itransformer_co2_late_residual` 的 Full `MAE = 47.797`
- 但 Final `MAE = 58.292` 仍然偏高，说明 horizon 末端仍有优化空间

我们能借什么：

1. 保留显式多尺度频带分支
2. 把 `GRU` 继续作为严肃的 CO2 backbone 候选
3. 让融合权重显式依赖 horizon

如何并回主线：

- 这是当前最优先并回多目标 CO2 residual 线的思路
- 最自然的做法是用 wavelet-inspired 多尺度专项分支替换当前 CO2 adapter

优先级：

- 很高，目前是第一优先级

## 6. 下一步应该怎样分批落实

### 第一批：已经落实

目标：

- 把最相关的直接温室 `CO2` 预测方法翻译成可运行代码

已完成：

1. `co2_env_lstm`
2. `co2_vmd_lstm_fusion`
3. `co2_wavelet_gru_attn`

### 第二批：多目标并回

目标：

- 把当前最强的独立 CO2 思路并回多目标主线

实际进展：

1. 直接 wavelet residual integration 失败
2. 直接 wavelet blend integration 失败
3. frozen-expert integration 改善了 CO2，但 full/final error profile 仍然分裂
4. horizon-wise protected fusion 成功
5. frozen-backbone horizon-wise fusion 恢复了控制安全的短步行为

当前最好并回方案：

- `itransformer_co2_horizon_mixture`
- Full `CO2air MAE = 43.910`
- Final `CO2air MAE = 47.661`

为什么这个方案更有效：

- 保持独立 wavelet-GRU expert 冻结
- 只对 `CO2air` 通道做修正
- 早中段使用受保护的 expert correction
- 末端 horizon 回拉到 late-residual 的更强尾部行为

控制转化验证：

- `itransformer_co2_horizon_mixture` 是离线 leader，但转化到 MPC 很差，`GradientMPC` 的 `CO2air MAE = 28.696`。
- 主要原因是第一步行为：simulator 用第一步预测推进状态，而这个离线 leader 虽然改善了 full/final 指标，却伤害了控制对齐窗口里的第一步误差。
- `itransformer_co2_frozen_backbone_horizon_mixture` 冻结 late-residual backbone，只训练 horizon gate，并为 MPC 保留输入梯度。
- 从 frozen-backbone forward 路径移除 `torch.no_grad()` 后，`GradientMPC` 梯度恢复。
- frozen-backbone 版本达到 `GradientMPC` `CO2air MAE = 10.000`，接近 `late_residual`，比离线 leader 控制安全得多，但 CO2 控制仍然弱于 `late_frozen_expert`。

### 第三批：高优先级

目标：

- 把文献里“超参数优化对 CO2 很重要”的结论引进来

建议做法：

1. 不要一开始就直接实现完整 `SSA / DBO`
2. 先做轻量自动搜索：
   - hidden size
   - decomposition kernel / granularity
   - attention heads
   - horizon-weighted loss
3. 如果收益真实，再考虑补更正式的搜索机制

### 第四批：研究升级线

目标：

- 从纯单目标黑盒预测，升级到 carbon-balance 灰盒建模

方向：

1. `CO2 dosing`
2. ventilation exchange
3. canopy uptake / photosynthesis
4. respiration

这不是最先该做的，因为当前更紧迫的问题仍然是把 forecasting 强度和闭环转化做出来。

## 7. 当前最值得汇报的版本

最适合直接写进周报的表述是：

- 我们已经把文献里最有价值的几类温室 `CO2` 预测方法落成了 3 条独立 benchmark 线。
- 结果表明，纯 `LSTM` 不够，`CO2` 更受益于多尺度分解和自适应融合。
- 当前最强的方法是 `wavelet-inspired + GRU + adaptive attention`。
- 这条线在当前 fair-budget AGC benchmark 下达到 Full `CO2air MAE = 45.209`。
- 因此下一步应该优先把这条独立 CO2 专项逻辑并回当前多目标主线，而不是继续盲目更换 generic backbone。

多目标并回后的更新：

- 朴素端到端并回 standalone wavelet expert 失败了。
- 解耦并回成功了：frozen expert + protected horizon-wise correction + terminal pullback 目前给出了最强离线 `CO2air` 结果。
- 当前最强汇报线是 `itransformer_co2_horizon_mixture`：Full `CO2air MAE = 43.910`，Final `CO2air MAE = 47.661`。
- 面向控制汇报时，要把它和更安全的后续版本分开：`itransformer_co2_frozen_backbone_horizon_mixture` 在保留冻结模块输入梯度后，达到 `GradientMPC` `CO2air MAE = 10.000`。

## 8. 2026-04-07 多目标并回说明

在独立 CO2 specialist benchmark 之后，又测试了两种直接并回多目标主线的方案：

1. `itransformer_co2_wavelet_residual`
2. `itransformer_co2_wavelet_blend`

正式结果：

- `itransformer_co2_wavelet_residual`
  - `CO2air`: Full `R2=0.5182`, MAE `65.984`
- `itransformer_co2_wavelet_blend`
  - `CO2air`: Full `R2=0.5813`, MAE `64.666`

它们都差于：

- `itransformer_residual`: Full `CO2air MAE = 51.161`
- `itransformer_co2_late_residual`: Full `CO2air MAE = 47.797`
- 独立 `co2_wavelet_gru_attn`: Full `CO2air MAE = 45.209`

当前结论：

- 独立 specialist 本身是强的
- 但朴素的端到端多目标并回会破坏它的优势
- 下一步更合理的方向，不是继续马上改另一版分支，而是尝试更解耦的迁移方式，例如 frozen-expert fusion、蒸馏，或 teacher-guided auxiliary loss

## 9. 2026-04-14 成功的解耦并回

在直接并回失败后，又测试了多种解耦变体：

- `itransformer_co2_frozen_expert`
- `itransformer_co2_late_frozen_expert`
- `itransformer_co2_teacher_distill`
- `itransformer_co2_recoupled_expert`
- `itransformer_co2_protected_expert`
- `itransformer_co2_protected_terminal`
- `itransformer_co2_horizon_mixture`
- `itransformer_co2_frozen_backbone_horizon_mixture`

最佳结果：

- `itransformer_co2_horizon_mixture`
  - `Tair`: Full `R2=0.9508`, MAE `0.604`; Final `R2=0.9374`, MAE `0.689`
  - `Rhair`: Full `R2=0.8958`, MAE `3.882`; Final `R2=0.8615`, MAE `4.568`
  - `CO2air`: Full `R2=0.7868`, MAE `43.910`; Final `R2=0.7468`, MAE `47.661`

解读：

- 这是当前 fair-budget 下第一个统一 `CO2air` 全时域 leader 和末步 leader 的模型。
- 它把 standalone `co2_wavelet_gru_attn` 的 Full MAE `45.209` 成功迁移并进一步提升到了多目标场景。
- 它也刷新了之前多目标最强 Full MAE `44.727` 和 Final MAE `50.139`。
- 主要剩余弱点是 `Rhair` 仍然略弱于 `itransformer_residual`。
- 第一轮闭环检查没有顺利转化：`GradientMPC` + `itransformer_co2_horizon_mixture` 的 `CO2air MAE = 28.696`，明显差于之前 `late_frozen_expert` 的控制侧 CO2 结果。
- 后续 frozen-backbone 版本的 Full `CO2air MAE = 46.334`，Final `CO2air MAE = 50.139`，所以它不是离线 leader。
- 它的价值是控制安全性：保留穿过冻结 backbone 和 expert 的输入梯度后，`GradientMPC` 达到 objective `0.0718`，`Tair MAE=1.158`，`Rhair MAE=1.615`，`CO2air MAE=10.000`。
- 这比 `late_residual` 的 CO2 控制略好一点（`10.125` 到 `10.000`），但仍然没有超过 `late_frozen_expert` 的 CO2 控制（`6.298`）。

研究结论：

- 文献中真正有效的思路不是“把 wavelet expert 直接塞进整个端到端模型”。
- 真正有效的是“保留 specialist 作为稳定 CO2 teacher/expert，再控制何时、何处信任它”。
- horizon-dependent trust 现在是已经验证的最强迁移机制。
- 但离线 horizon-dependent trust 不会自动变成 control-safe。
- 下一步研究是 control-aware CO2 fusion：保留 `late_frozen_expert` 的短时域可控性，同时保留 horizon-mixture 家族的离线末端收益。
