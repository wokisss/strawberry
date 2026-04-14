# CO2_PAPERS_AND_DIRECTION.zh-CN.md

中文对齐翻译版本。
英文主版本： [CO2_PAPERS_AND_DIRECTION.md](c:/repositories/strawberry/agc_mpc/CO2_PAPERS_AND_DIRECTION.md)
最近同步时间：`2026-04-07`

## 目的

这份笔记只聚焦温室 `CO2` 预测与控制。

它回答两个实际问题：

1. 当论文报告 `MAE` 时，它是在归一化数据上计算的，还是在物理量单位上计算的？
2. 如果我们想改进 `agc_mpc` 里的 `CO2air`，哪些论文最值得先读，哪些思路最值得先借用？

这不是为了排名而排名。
目标是提炼出真正能迁移到当前 `AGC` 工作流中的技术方向。

## A. 如何判断论文里的 MAE 是否是归一化误差

在温室论文里，通常有三种常见情况。

1. 训练时做了归一化，但最终误差是在反归一化之后报告的。
   - 这类指标通常带有物理单位，例如 `ppm`、`degC` 或 `%RH`。

2. 论文直接在归一化目标上报告误差。
   - 这类指标通常是很小的小数，比如 `0.0117`，而且往往不带物理单位。

3. 公开摘要页没有给出足够细节。
   - 这种情况下，在没有看全文之前，不要断言 `MAE` 是归一化还是非归一化。

实用判断规则：

- 如果误差写成 `ppm`，通常就是物理单位误差。
- 如果数值很小且没有单位，而目标本身在几百 `ppm` 量级，那大概率是归一化误差。
- 如果公开页面只给了 `R2`，那 `MAE` 是否归一化仍然是不确定状态。

## B. CO2 专项论文清单

### B1. 直接做温室 CO2 预测的论文

| 论文 | 解决什么问题 | 主方法 | 指标状态 | 我们能借什么 | 优先级 |
| --- | --- | --- | --- | --- | --- |
| [Prediction of CO2 Concentration via Long Short-Term Memory Using Environmental Factors in Greenhouses](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002578287) | 用环境因子预测温室 `CO2`，向前预测 `2 h` | `LSTM` | 公开摘要主要给的是 `R2`，仅凭摘要无法确认 `MAE` 状态 | `CO2` 可以单独作为专项目标建模，而不是只作为通用气候模型的共享 head | 中 |
| [Time-serial analysis of deep neural network models for prediction of climatic conditions inside a greenhouse](https://doi.org/10.1016/j.compag.2020.105402) | 联合预测 `temperature / humidity / CO2` | `ANN`、`NARX`、`RNN-LSTM` | 公开结果页用 `ppm` 报告 `CO2` 误差，因此属于物理单位误差 | `CO2` 比温度更难，递归模型在温室动力学里仍然重要 | 高 |
| [Multi-model fusion method for predicting CO2 concentration in greenhouse tomatoes](https://doi.org/10.1016/j.compag.2024.109623) | 预测番茄温室 `CO2` 浓度 | `WT + VMD + LSTM + attention + fusion` | 公开摘要给出 `MAE = 0.0117` 和 `RMSE = 0.0194`，但没有物理单位，极可能是归一化误差 | `CO2` 更适合分解与融合，而不是单一 backbone | 很高 |
| [Prediction of CO2 concentration in mushroom greenhouse via optimized long and short term memory algorithm](https://doi.org/10.1038/s41598-025-86394-0) | 预测食用菌温室 `CO2` | `VMD-SSA-LSTM`、`VMD-DBO-LSTM` | 公开摘要直接给出 `MAE = 2.6365 ppm`，因此是物理单位误差 | 即便 backbone 仍是递归模型，`CO2` 也明显受益于分解与优化 | 高 |
| [Wavelet-decoupled GRU with adaptive attention for multi-step carbon dioxide concentration prediction in intelligent glass greenhouse](https://doi.org/10.1016/j.atech.2025.101653) | 多步温室 `CO2` 预测，最长到 `8 h` | 类 wavelet 解耦 + `GRU` + adaptive attention | 公开页面用 `ppm` 报告误差；训练很可能用缩放，但最终指标是物理单位 | 强烈支持对 `CO2` 做多尺度分解和自适应加权 | 很高 |

### B2. CO2 控制与优化论文

| 论文 | 解决什么问题 | 主方法 | 为什么重要 |
| --- | --- | --- | --- |
| [Model-based control of CO2 concentration in greenhouses at ambient levels increases cucumber yield](https://doi.org/10.1016/j.agrformet.2006.12.002) | 环境浓度附近的 `CO2` 控制 | 基于作物吸收建模的 model-based control | 提醒我们最终目标不只是把 `ppm` 预测准，还包括支持补碳策略和作物吸收估计 |
| [Model predictive control of a Venlo-type greenhouse system considering electrical energy, water and carbon dioxide consumption](https://doi.org/10.1016/j.apenergy.2021.117163) | 联合能耗、水耗和 `CO2` 消耗控制 | `MPC` | 如果后续把 `CO2` 从纯预测目标移入控制代价，这篇很重要 |
| [Intelligent control and energy optimization in controlled environment agriculture via nonlinear model predictive control of semi-closed greenhouse](https://doi.org/10.1016/j.apenergy.2022.119334) | 联合控制 `temperature / humidity / CO2 / light` | 基于能量与质量平衡的 `NMPC` | 强烈支持把温室建模成耦合的能量 + 质量系统，而不是只做黑盒预测 |
| [CO2 enrichment in greenhouse production: Towards a sustainable approach](https://doi.org/10.3389/fpls.2022.1029901) | `CO2` 富集策略综述 | review | 如果问题从预测精度转向高效、可持续的 `CO2` 使用，这是一个很好的入口 |

### B3. 灰盒与通量模型论文

| 论文 | 解决什么问题 | 主方法 | 为什么重要 |
| --- | --- | --- | --- |
| [An autocalibrating model for simulating and measuring net canopy photosynthesis using a standard greenhouse climate computer](https://doi.org/10.1016/0168-1699(91)90019-6) | 估计温室内冠层净光合 | `CO2` 平衡模型 + black-box 光合模型 | 这是最清晰的 `CO2 balance + black-box` 灰盒路线先例之一 |
| [Estimation of net photosynthesis of a greenhouse canopy using a mass balance method and mechanistic models](https://doi.org/10.1016/0168-1923(94)90106-6) | 从温室 `CO2` 平衡估计冠层光合 | mass balance + mechanistic models | 支持把 `CO2` 与冠层吸收和通风交换一起建模，而不是只当作普通标量时间序列 |
| [Validation of a Photosynthesis Model through the Use of the CO2 Balance of a Greenhouse Tomato Canopy](https://doi.org/10.1006/anbo.1999.0938) | 用温室 `CO2` 平衡验证光合模型 | `CO2` 平衡 + 植物生理模型 | 再次强化 `CO2` 应与植株吸收过程关联建模 |

## C. 这些论文共同说明了什么

在直接 `CO2` 预测论文中，有两个稳定模式。

1. `CO2` 比 `Tair` 更非平稳，也更依赖具体运行 regime。
2. 对 `CO2` 有效的方法通常至少加入下面其中之一：
   - 分解 / 去噪 / 多尺度处理
   - 动态融合 / 自适应加权

这和我们在 `AGC` 里的观察一致：

- `CO2air` 的全局平均指标可能还可以
- 但局部 rollout 窗口仍然可能严重漂移

所以，下一步不应该只是“换一个更大的通用 transformer”。更现实的选择是下面两条路线之一。

## D. 适用于 `agc_mpc` 的两条现实路线

### 路线 1：CO2 专项 forecasting 分支

保留当前多目标设定，但为 `CO2` 增加一个比现有 residual 变体更专项的分支。

从文献看，最合理的组成包括：

1. 在序列建模前先做分解
   - `WT`
   - `VMD`
   - 或其他多尺度拆分

2. 为 `CO2` 分支选更合适的 backbone
   - `GRU`
   - `LSTM`
   - `GRU/LSTM + attention`

3. 做 variable-weight fusion
   - 按目标变量加权
   - 按 horizon 加权
   - 按上下文加权

这条路线最容易并入当前 forecasting 代码库。

### 路线 2：Energy-Water-Carbon 灰盒模型

不要只把 `CO2air` 当作另一个输出通道，而是把温室定义成一个耦合系统：

- 能量流
- 水分流
- 碳流

然后构建灰盒 predictor：

- 已知部分用机理平衡方程
- 不完整部分用 black-box residual model 去补

对 `CO2` 来说，自然的潜变量包括：

- `CO2 dosing`
- ventilation exchange
- canopy net uptake / photosynthesis
- respiration terms

这条路线更偏研究，也更温室原生。

## E. 建议阅读顺序

如果当前目标是尽快改进 `CO2air` forecasting：

1. [Multi-model fusion method for predicting CO2 concentration in greenhouse tomatoes](https://doi.org/10.1016/j.compag.2024.109623)
2. [Wavelet-decoupled GRU with adaptive attention for multi-step carbon dioxide concentration prediction in intelligent glass greenhouse](https://doi.org/10.1016/j.atech.2025.101653)
3. [Prediction of CO2 concentration in mushroom greenhouse via optimized long and short term memory algorithm](https://doi.org/10.1038/s41598-025-86394-0)
4. [Time-serial analysis of deep neural network models for prediction of climatic conditions inside a greenhouse](https://doi.org/10.1016/j.compag.2020.105402)

如果目标是走向更强的温室原生 CO2 建模线：

1. [An autocalibrating model for simulating and measuring net canopy photosynthesis using a standard greenhouse climate computer](https://doi.org/10.1016/0168-1699(91)90019-6)
2. [Model-based control of CO2 concentration in greenhouses at ambient levels increases cucumber yield](https://doi.org/10.1016/j.agrformet.2006.12.002)
3. [Intelligent control and energy optimization in controlled environment agriculture via nonlinear model predictive control of semi-closed greenhouse](https://doi.org/10.1016/j.apenergy.2022.119334)

## 总结

对于 `CO2air`，文献并不支持“再换一个通用 backbone 就能解决问题”的说法。

更强的方向是：

1. `CO2` 专项 `decomposition + sequence model + dynamic fusion`
2. `CO2 balance + photosynthesis + control` 的灰盒建模

如果继续沿用当前 `agc_mpc` 架构，最快的下一步是路线 1。
如果目标是做更原创、更温室原生的研究线，路线 2 更强。