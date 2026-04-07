# CO2 专项文献与方向整理

## 目的

这份笔记只聚焦温室里的 `CO2` 预测与控制问题。

它主要回答两个实际问题：

1. 我们查到的论文里，`MAE` 到底是对归一化数据算的，还是对物理量算的？
2. 如果后续想专门改进 `agc_mpc` 里的 `CO2air`，哪些论文最值得读，哪些思路最值得借？

这不是榜单。
目标是提炼能真正迁移到当前 `AGC` 流程里的技术方向。

## A. 怎么判断论文里的 MAE 是不是归一化的

在温室论文里，常见情况通常有三类：

1. 训练前做了归一化，但最终误差是在反归一化后报告的。
   - 这类指标通常会带物理单位，例如 `ppm`、`degC`、`%RH`。

2. 论文直接在归一化目标上报告误差。
   - 这类指标往往是很小的小数，例如 `0.0117`，而且通常不带物理单位。

3. 公开摘要页没有给足够细节。
   - 这种情况下，不能在没看到全文前就断言 `MAE` 是否归一化。

实用判断规则：

- 如果指标写成 `ppm`，通常应该理解为物理单位误差。
- 如果指标是很小的小数、又不带单位，而目标本身显然在几百 `ppm` 量级，那大概率是归一化误差。
- 如果公开页面只给了 `R2`，那 `MAE` 是否归一化就是未确定状态。

## B. CO2 专项论文清单

### B1. 直接做温室 CO2 预测的论文

| 论文 | 解决什么问题 | 主方法 | 指标状态 | 能借什么 | 优先级 |
| --- | --- | --- | --- | --- | --- |
| [Prediction of CO2 Concentration via Long Short-Term Memory Using Environmental Factors in Greenhouses](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002578287) | 用环境因子预测温室 `CO2`，`2 h` ahead | `LSTM` | 公开摘要主要给 `R2`，单靠摘要无法确认 `MAE` 是否归一化 | `CO2` 可以单独当成专门目标来建模，而不只是通用气候模型里的一个共享 head | 中 |
| [Time-serial analysis of deep neural network models for prediction of climatic conditions inside a greenhouse](https://doi.org/10.1016/j.compag.2020.105402) | 联合预测 `temperature / humidity / CO2` | `ANN`、`NARX`、`RNN-LSTM` | 公开结果页明确给了 `CO2` 的 `ppm` 误差，因此是物理单位误差 | `CO2` 本来就比温度难；递归类模型在温室动力学里仍有价值 | 高 |
| [Multi-model fusion method for predicting CO2 concentration in greenhouse tomatoes](https://doi.org/10.1016/j.compag.2024.109623) | 温室番茄 `CO2` 浓度预测 | `WT + VMD + LSTM + attention + fusion` | 公开摘要给 `MAE = 0.0117`、`RMSE = 0.0194`，但没有物理单位；大概率是归一化误差 | `CO2` 更适合“分解 + 融合”，而不是单一 backbone | 很高 |
| [Prediction of CO2 concentration in mushroom greenhouse via optimized long and short term memory algorithm](https://doi.org/10.1038/s41598-025-86394-0) | 食用菌温室 `CO2` 预测 | `VMD-SSA-LSTM`、`VMD-DBO-LSTM` | 公开摘要直接给 `MAE = 2.6365 ppm`，因此是物理单位误差 | 即便 backbone 仍是 recurrent，`CO2` 也会明显受益于 decomposition 与优化 | 高 |
| [Wavelet-decoupled GRU with adaptive attention for multi-step carbon dioxide concentration prediction in intelligent glass greenhouse](https://doi.org/10.1016/j.atech.2025.101653) | 智能玻璃温室多步 `CO2` 预测，最长到 `8 h` | wavelet 类解耦 + `GRU` + adaptive attention | 公开页给的是 `ppm` 误差；训练很可能有归一化，但最终指标是物理单位 | 强烈支持 `CO2` 应按多尺度解耦 + 自适应权重来处理 | 很高 |

### B2. CO2 控制与优化论文

| 论文 | 解决什么问题 | 主方法 | 为什么重要 |
| --- | --- | --- | --- |
| [Model-based control of CO2 concentration in greenhouses at ambient levels increases cucumber yield](https://doi.org/10.1016/j.agrformet.2006.12.002) | 环境浓度附近的 `CO2` 控制 | 基于作物吸收模型的 model-based control | 提醒我们最终目标不只是预测 `ppm`，还包括支持 `CO2` 供给策略和植株吸收估计 |
| [Model predictive control of a Venlo-type greenhouse system considering electrical energy, water and carbon dioxide consumption](https://doi.org/10.1016/j.apenergy.2021.117163) | 联合考虑 energy、water、`CO2` 消耗 | `MPC` | 如果后面把 `CO2` 从纯预测目标移到控制 cost 里，这篇很重要 |
| [Intelligent control and energy optimization in controlled environment agriculture via nonlinear model predictive control of semi-closed greenhouse](https://doi.org/10.1016/j.apenergy.2022.119334) | 联合控制 `temperature / humidity / CO2 / light` | 基于 energy 和 mass balance 的 `NMPC` | 强烈支持把温室建成耦合的 energy + mass balance，而不是纯黑盒 forecast |
| [CO2 enrichment in greenhouse production: Towards a sustainable approach](https://doi.org/10.3389/fpls.2022.1029901) | `CO2` enrichment 策略综述 | review | 如果问题从“预测得准不准”转向“如何高效、可持续地用 CO2”，这是很好的总入口 |

### B3. 灰盒与通量模型论文

| 论文 | 解决什么问题 | 主方法 | 为什么重要 |
| --- | --- | --- | --- |
| [An autocalibrating model for simulating and measuring net canopy photosynthesis using a standard greenhouse climate computer](https://doi.org/10.1016/0168-1699(91)90019-6) | 温室内冠层净光合估计 | `CO2` balance model + black-box photosynthesis model | 这是最清晰的 `CO2 balance + black-box` 灰盒路线先例 |
| [Estimation of net photosynthesis of a greenhouse canopy using a mass balance method and mechanistic models](https://doi.org/10.1016/0168-1923(94)90106-6) | 用温室 `CO2` balance 估计冠层光合 | mass balance + mechanistic models | 支持把 `CO2` 与冠层吸收、通风交换联系起来，而不是只把它当作普通标量时序 |
| [Validation of a Photosynthesis Model through the Use of the CO2 Balance of a Greenhouse Tomato Canopy](https://doi.org/10.1006/anbo.1999.0938) | 用温室 `CO2` balance 验证光合模型 | `CO2` balance + 生理模型 | 再次强化了 `CO2` 应与作物吸收过程联合建模的思路 |

## C. 这些论文共同指向什么

从直接做 `CO2` 预测的论文里，可以稳定看出两点：

1. `CO2` 比 `Tair` 更非平稳，也更依赖具体 regime。
2. 对 `CO2` 有效的方法，通常都会加入以下至少一种：
   - 分解 / 去噪 / 多尺度处理
   - 动态融合 / 自适应加权

这和当前 `AGC` 里的观察是一致的：

- `CO2air` 在全局平均指标上可能还行
- 但局部 rollout 窗口里仍然可能漂得很严重

所以，下一步不应该只是“换一个更大的 generic transformer”。
更合理的是下面两条路线之一。

## D. 对 `agc_mpc` 最现实的两条路线

### 路线 1：CO2 专项 forecasting 分支

保留当前 3-target 设定，但为 `CO2` 增加一个比现有 residual variant 更专门的分支。

从文献看，最合理的组成包括：

1. 序列建模前先做 decomposition
   - `WT`
   - `VMD`
   - 或其他多尺度拆分

2. 为 `CO2` 分支选更合适的时间 backbone
   - `GRU`
   - `LSTM`
   - `GRU/LSTM + attention`

3. 做 variable-weight fusion
   - 按目标变量加权
   - 按 horizon 加权
   - 按上下文加权

这条路线最容易接到当前 forecasting 代码里。

### 路线 2：energy-water-carbon 灰盒模型

不要只把 `CO2air` 当成另一个输出通道，而是把温室动力学定义为一个耦合系统：

- energy flow
- water flow
- carbon flow

然后做灰盒 predictor：

- 能写清楚的部分用机理平衡方程
- 写不清楚的部分用 black-box residual 去补

对 `CO2` 来说，最自然的 latent 量包括：

- `CO2 dosing`
- ventilation exchange
- canopy net uptake / photosynthesis
- respiration 等项

这条路线研究味更强，也更贴近温室真实过程。

## E. 建议阅读顺序

如果当前目标是尽快改进 `CO2air` forecasting：

1. [Multi-model fusion method for predicting CO2 concentration in greenhouse tomatoes](https://doi.org/10.1016/j.compag.2024.109623)
2. [Wavelet-decoupled GRU with adaptive attention for multi-step carbon dioxide concentration prediction in intelligent glass greenhouse](https://doi.org/10.1016/j.atech.2025.101653)
3. [Prediction of CO2 concentration in mushroom greenhouse via optimized long and short term memory algorithm](https://doi.org/10.1038/s41598-025-86394-0)
4. [Time-serial analysis of deep neural network models for prediction of climatic conditions inside a greenhouse](https://doi.org/10.1016/j.compag.2020.105402)

如果目标是往更像论文主线的 `CO2` 建模方向走：

1. [An autocalibrating model for simulating and measuring net canopy photosynthesis using a standard greenhouse climate computer](https://doi.org/10.1016/0168-1699(91)90019-6)
2. [Model-based control of CO2 concentration in greenhouses at ambient levels increases cucumber yield](https://doi.org/10.1016/j.agrformet.2006.12.002)
3. [Intelligent control and energy optimization in controlled environment agriculture via nonlinear model predictive control of semi-closed greenhouse](https://doi.org/10.1016/j.apenergy.2022.119334)

## 总结

对于 `CO2air`，现有文献并不支持“再换一个 generic backbone 就能解决问题”。

更强的方向是：

1. `CO2` 专项 `decomposition + sequence model + dynamic fusion`
2. `CO2 balance + photosynthesis + control` 的 gray-box 建模

如果继续沿用当前 `agc_mpc` 架构，最快的下一步是路线 1。
如果想做更原创、更温室原生的研究线，路线 2 更强。
