# Thesis Literature Library

中文对齐翻译版本。
英文主版本：[THESIS_LITERATURE_LIBRARY.md](c:/repositories/strawberry/agc_mpc/THESIS_LITERATURE_LIBRARY.md)
最近同步时间：`2026-04-20`

## 目的

本文档是 `agc_mpc` 论文主线的文献库。

它比之前的 control-relevant MPC 笔记更宽，不只保存 control-relevant MPC 文献。以后凡是论文可能引用的文献，都可以维护到这里，包括：

- 温室多步预测
- 温室气候控制与 economic MPC
- CO2 forecasting、碳平衡、CO2 enrichment
- 通用 time-series forecasting 架构
- control-relevant identification、validation、prediction-control correlation
- uncertainty、robustness、probabilistic forecasting、resource-aware control

这不是排行榜。只有当任务、horizon、目标变量、数据集设定和控制目标足够接近时，数值指标才可以直接比较。

## 阅读指南

| 标签 | 含义 | 使用方式 |
| --- | --- | --- |
| Most comparable | 和我们相近的温室多变量 forecasting 或温室气候控制。 | 用在主文献对比和动机部分。 |
| Partially comparable | 温室论文，但 horizon 更短、目标更少或目标不同。 | 用作方向证据，不做直接数值 benchmark。 |
| Reference only | 通用时序、控制理论或建筑/HVAC 文献。 | 用来支撑架构、验证指标或控制方法。 |

## 当前论文故事

1. 温室 forecasting 不是 plain Transformer 排行榜问题。近期温室和通用时序文献都支持 hybrid、residual、decomposition、normalization、attention/RNN 和 horizon-aware fusion。
2. CO2 比 `Tair` 更难、更依赖运行工况；它需要 decomposition、多尺度建模、自适应融合，最终还应走向 carbon-balance gray-box modeling。
3. 离线 forecasting 更好不一定带来闭环 MPC 更好。在固定设置里二者常呈正相关，但跨模型结构选择时必须做 control-relevant validation。
4. 当前 PHF 主线应写成 forecasting improvement + control-relevant validation，而不是声称离线 forecast leader 自动成为 control leader。
5. 成熟温室 MPC 论文最终需要 uncertainty、constraints、resource/economic cost 和 closed-loop validation，不只是 point forecast accuracy。

## 关键术语

| 术语 | 含义 | 为什么重要 |
| --- | --- | --- |
| Controlled environment agriculture, CEA | 温室、植物工厂等受控环境农业。 | 论文最终领域是温室气候预测与控制。 |
| Multi-step forecasting | 一次性或递归预测多个未来时间步。 | 当前 AGC 默认 horizon 是 `24 x 5 min = 2 h`。 |
| Exogenous variables | 模型不能控制但会影响未来的变量，例如天气。 | `Weather.csv` 提供未来天气，应进入预测和 MPC。 |
| Control inputs | 控制器选择的变量，例如设定值。 | `u_future` 表示未来计划控制动作。 |
| Residual model | 在稳定 baseline 上学习校正量的模型。 | 当前强主线使用稳定主路径加 residual / specialist correction。 |
| Decomposition | 把序列拆成 trend、seasonal、frequency 或 multi-scale 分量。 | CO2 和温室信号通常同时有慢周期和短时扰动。 |
| Horizon-aware fusion | 不同预测步使用不同模型信任度。 | PHF 使用 horizon-dependent trust 和 terminal pullback。 |
| Model predictive control, MPC | 在有限未来时域预测系统行为，优化控制动作，只执行第一步，然后滚动重复。 | 当前控制主线就是 MPC。 |
| Receding horizon | MPC 每轮只执行第一步优化动作，再重新求解。 | first-step 和 short-horizon error 会强烈影响闭环。 |
| Control-relevant validation | 按模型控制用途验证，而不是只看普通预测拟合。 | 这直接支撑 `control_relevant_validation.py`。 |
| Oracle gap | 真实 MPC 与使用完美未来信息的理想控制器之间的差距。 | 用来量化预测误差仍造成多少控制损失。 |

## 与文献相比的项目定位

本节合并旧文档 [LITERATURE_COMPARISON.md](c:/repositories/strawberry/agc_mpc/LITERATURE_COMPARISON.md) 中仍有价值的论文定位内容。旧文档现在只作为 snapshot；本文献库是后续论文写作时维护的主版本。

### 当前 AGC 设定

| 项目 | 当前设定 |
| --- | --- |
| Project | `agc_mpc` |
| 主数据集 | `AutonomousGreenhouseChallenge_edition2` |
| 任务 | 面向控制的多步条件预测 |
| 输入 | `x_past / w_future / u_future` |
| 输出 | `Tair / Rhair / CO2air / Tot_PAR`，近期 fair-budget 工作经常聚焦 `Tair / Rhair / CO2air` |
| 默认 horizon | `24 x 5 min = 2 h` |
| Forecasting baselines | `GRU`, `DLinear`, `SegRNN`, `Transformer`, `Transformer-hybrid`，以及 residual 和 CO2 specialist 变体 |
| 控制设定 | surrogate 闭环 benchmark，包含 `Recorded / GradientMPC / CEMMPC` |

### 早期 AGC baseline 定位

旧 AGC baseline 的 final-step 结果如下：

| Model | Tair R2 / MAE | Rhair R2 / MAE | CO2air R2 / MAE | Tot_PAR R2 / MAE |
| --- | --- | --- | --- | --- |
| `DLinear` | `0.9526 / 0.729` | `0.8184 / 4.209` | `0.7928 / 51.481` | `0.9779 / 31.295` |
| `Transformer` | `0.9413 / 0.823` | `0.7454 / 4.919` | `0.8242 / 47.229` | `0.9859 / 24.964` |
| `Transformer-hybrid` | `0.9480 / 0.770` | `0.6927 / 5.306` | `0.7434 / 58.318` | `0.9846 / 28.509` |

解读：

- `Tair` 已经较强。
- 对 2 h 多步任务来说，`CO2air` 已经达到可接受到较强的水平。
- `Rhair` 是最弱目标。
- `Tot_PAR` 离线很强，但离线光照预测强并不自动意味着闭环控制最好。
- 后续 CO2 specialist 工作已经刷新 CO2 frontier，因此这些旧数值只适合做历史定位，不应作为当前最终 benchmark。

### 旧 Strawberry 与 AGC

旧 Strawberry 数据集仍可作为 stress-test 参考，但 AGC 更适合作为论文平台，因为它包含未来天气、未来控制计划、执行器反馈、多隔间和资源信号。

共同变量上的代表性 2 h 对比：

| Setting | Temperature final MAE | Humidity final MAE | CO2 final MAE | Temperature final R2 | Humidity final R2 | CO2 final R2 |
| --- | --- | --- | --- | --- | --- | --- |
| old Strawberry `Transformer-hybrid` | `3.36` | `6.78` | `105.88` | `0.796` | `0.840` | `0.073` |
| AGC `DLinear` | `0.76` | `4.46` | `54.73` | `0.949` | `0.798` | `0.776` |
| AGC `Transformer` | `0.82` | `4.92` | `47.23` | `0.941` | `0.745` | `0.824` |
| AGC `Transformer-hybrid` | `0.77` | `5.31` | `58.32` | `0.948` | `0.693` | `0.743` |

论文解读：

- 在相同 2 h framing 下，旧 Strawberry 设定明显更弱。
- AGC 不是完美数据集，但更适合面向控制的 benchmark。
- 选择 AGC 的核心原因不是它分数完美，而是它的数据接口更贴近 MPC。

### 文献定位结论

可以稳妥主张：

- 从旧 Strawberry 切换到 AGC 是合理的。
- `DLinear` 很强符合温室 forecasting 和 long-time-series forecasting 文献，不反常。
- 当前 AGC forecasting 结果没有明显坏掉，也没有远低于可比温室文献的大致范围。

仍然薄弱的地方：

- 湿度预测仍弱于一些专项温室 forecasting 论文。
- 闭环控制仍是早期 surrogate benchmark。
- 还没有 uncertainty-aware forecasting/control。
- 还没有成熟的 economic/resource objective。

可能限制因素：

- 相比专项 forecasting 论文，超参数搜索仍有限。
- target-specific loss balancing 不足。
- horizon-aware loss shaping 不足。
- 还没有 probabilistic forecast head。
- 还没有显式 humidity-focused specialist branch。
- 还没有 compartment adapter 或 transfer-learning layer。
- 当前 control rollout 中 actuator/VIP/physics transition 还不够丰富。

给导师汇报时可以这样说：

> 当前 AGC 结果还不是 final-paper quality，但已经处在可比温室 forecasting 文献的大致性能带内。选择 AGC 的核心理由不是它已经给出完美分数，而是它更匹配 control-oriented 任务：未来天气、未来控制计划、执行器反馈、多隔间和资源信号都可用。因此，即使建模栈还没有完全优化，AGC 也是 multi-step forecasting plus MPC 更合适的研究平台。

## A. 温室 Forecasting 文献

| 论文 | 任务 | 方法 | Baseline | 主要启发 | 链接 | 可比性 |
| --- | --- | --- | --- | --- | --- | --- |
| Ahn et al., 2024 | 温室 `temperature / RH / CO2`，`1 h` 与 `3 h` 预测 | `Autoformer` | `DLinear`, `LSTM`, `SegRNN` | 最接近我们的温室 forecasting 参考之一；`DLinear`、`SegRNN` 等简单模型并不弱，Transformer 不会天然占优。 | https://www.mdpi.com/2073-4395/14/3/417 | Most comparable |
| Li et al., 2024 | 温室空气温度与土壤温度，`30-480 min` | `Attention-LSTM` | `RNN`, `GRU`, `LSTM` | RNN 加 attention 在温室中短时域仍然有效。 | https://www.sciencedirect.com/science/article/pii/S0168169923009900 | Partially comparable |
| Mao et al., 2024 | 温室 `temperature / humidity / PAR`，`30-120 min` | `PSO-BiGRU-Attention-LightGBM` | `BiGRU-Attention`, `LightGBM`, 等权 ensemble | 强证据支持 hybrid 和 variable-weight fusion，而不是单一 backbone。 | https://www.sciencedirect.com/science/article/pii/S0168169924002096 | Partially comparable |
| Guo et al., 2024 | 温室 temperature 与 humidity 多步预测 | temporal-position-attention `LSTM` | LSTM-family comparisons | 说明通过任务特定 sequence modeling 可以取得较高湿度精度，但不包含 CO2，也不是闭环控制 surrogate。 | https://doi.org/10.1007/s00477-024-02840-x | Partially comparable |
| Wang et al., 2025 | 温室土壤温度，`3 h / 6 h / 24 h / 48 h` | `ReSSA-iTransformer` = `iTransformer + RevIN + SSA` | `LSTM`, `Informer`, `Autoformer`, `iTransformer` | 强 Transformer 变体通常带 normalization、decomposition 或任务特定改造。 | https://www.mdpi.com/2073-4395/15/1/223 | Reference/partial |
| Choi and Yang, 2025 | 温室 `temperature / RH / CO2` 概率预测，`3 h` | Probabilistic `1D CNN`, probabilistic `LSTM` | deterministic CNN/LSTM | 如果 forecasting 要服务控制，不确定性很重要。 | https://www.mdpi.com/2077-0472/15/23/2461 | Most comparable for uncertainty |
| Seri et al., 2025 | 温室微气候建模，强调变量耦合 | Directed `STGNN` | `RNN` | 当 actuator-climate 关系显式建模时，结构化耦合可能比纯时间 backbone 更重要。 | https://www.sciencedirect.com/science/article/pii/S0360132325009461 | Reference/partial |
| Cebolla-Alemany et al., 2026 | 屋顶温室空气温度，`5 / 10 / 15 min` | `Thermocast` 模块化 ensemble | 传统与 ensemble 回归器 | 超短时单目标温度结果可作上限参考，但不能直接对比 AGC 2 h 多输出任务。 | https://www.sciencedirect.com/science/article/pii/S2772375525009645 | Partially comparable |

### 详细笔记：Ahn et al., 2024

转述摘要：

这篇论文评估温室 temperature、relative humidity 和 CO2 的时序预测模型。它和我们的 AGC 任务高度相关，因为覆盖相同核心环境变量。重要信息是：温室时序预测并不会天然偏向 Transformer，线性模型和 recurrent baseline 仍然可以很强。

论文用途：

- 解释为什么 `DLinear` 在 AGC 中很强并不反常。
- 支撑稳定 baseline 加 residual/specialist correction，而不是盲目换大 backbone。

### 详细笔记：Mao et al., 2024

论文：
- Xiaojuan Mao et al., 2024
- `A variable weight combination prediction model for climate in a greenhouse based on BiGRU-Attention and LightGBM`
- 链接：https://www.sciencedirect.com/science/article/pii/S0168169924002096

论文内容：

| 项目 | 内容 |
| --- | --- |
| Greenhouse | 中国南京单个 Venlo-type greenhouse |
| Crop | cherry tomato |
| Time range | `2020-09-23` 到 `2021-06-06` |
| Sampling interval | `10 min` |
| Sample count | `37,008` |
| Inputs | 室内气候、室外气候、控制操作变量 |
| Targets | air temperature, air humidity, PAR |
| History length | `120 min` |
| Horizons | `30-120 min` |
| Single models | `GRU`, `BiGRU`, `BiGRU-Attention`, `XGBoost`, `LightGBM` |
| Ensemble models | equal-weight `BiGRU-Attention-LightGBM`, variable-weight `PSO-BiGRU-Attention-LightGBM` |
| Metrics | `RMSE`, `MAE`, `R2` |

为什么它的高 R2 不能直接和 AGC 比：

- 不包含 CO2。
- 单温室、horizon 更短。
- 只面向 forecasting，不需要 MPC surrogate rollout 一致性。
- 它本身已经是 sequence model、tabular learner 和 horizon-dependent weighting 的 hybrid。

论文用途：

- 支撑 hybrid/residual/fusion 设计。
- 支撑 horizon-dependent weighting。
- 不要把它的数值直接拿来 benchmark AGC `CO2air`。

### 详细笔记：Guo et al., 2024

论文：
- `Multi-Step Prediction of Greenhouse Temperature and Humidity Based on Temporal Position Attention LSTM`
- DOI：https://doi.org/10.1007/s00477-024-02840-x

转述摘要：

这篇论文使用 temporal-position-attention LSTM 和室内外变量进行温室 temperature / humidity 多步预测。它的价值在于说明，当架构针对温室任务调优时，湿度预测可以明显增强。但它不包含 CO2，也没有作为 control surrogate 做闭环评估，因此不能直接对标当前 AGC 控制任务。

论文用途：

- 讨论当前 `Rhair` 相对专项温室 forecasting 论文仍偏弱时引用。
- 支撑未来做 humidity-focused residual 或 specialist branch。
- 不要把它当成 MPC performance 的直接证据。

### 详细笔记：Choi and Yang, 2025

转述摘要：

这篇论文研究包含 CO2 在内的温室气候概率预测。它的价值不只是模型族，而是不确定性视角。对于控制，predictor 最好能给出 uncertainty 或 risk，因为 MPC 决策可能对未来扰动和约束违反敏感。

论文用途：

- 用于未来 probabilistic PHF 或 stochastic MPC。
- 用于说明 point forecast MAE 不是最终终点。

## B. 温室控制与 Economic MPC 文献

| 论文 | 任务 | 方法 | Baseline / comparison | 主要启发 | 链接 | 用途 |
| --- | --- | --- | --- | --- | --- | --- |
| Svensen et al., 2024 | 参数不确定性下的温室生产控制 | chance-constrained `SMPC` | nonlinear chance-constrained MPC setup | 控制论文竞争点是 uncertainty、constraints、tractability，不只是 predictor 名字。 | https://www.sciencedirect.com/science/article/pii/S0168169923009663 | uncertainty-aware control |
| Le and Bui, 2025 | 智能温室 `NMPC`，`30` 天仿真 | `NMPC` + `LSTM` disturbance forecast | feedback-only, forecast preview, ideal preview, cold/warm start | 很适合参考 forecast preview、ideal preview、warm start 对比。 | https://www.mdpi.com/2076-3417/15/14/7988 | control validation design |
| Mallick et al., 2025 | 预测不确定性下的温室气候控制 | RL-based MPC | robust/stochastic MPC 定位 | 代表 controller-learning 方向，应放在 surrogate benchmark 稳住之后。 | https://www.sciencedirect.com/science/article/pii/S2772375524003551 | future work |
| Kim and You, 2025 | 不确定性下的节能温室气候控制 | `GP-SMPC` + online learning | `NMPC`, `RMPC`, `DDRMPC` | 成熟控制工作会组合 uncertainty、online correction、energy 和 CO2 cost。 | https://www.sciencedirect.com/science/article/pii/S0306261925005719 | 终局控制故事 |
| Mansour et al., 2025 | 半封闭温室控制、鲁棒性、经济优化、迁移 | hierarchical MPC + DRL | robust/stochastic MPC, model-free DRL | 后期完整系统形态：上层经济层加下层 tracking control。 | https://www.sciencedirect.com/science/article/pii/S2772375525005581 | future architecture |
| Mahmood et al., 2021 | 温室温度控制和节能 | ML model + MPC | conventional control | 控制输出必须包括 energy/resource outcome，不能只看 tracking。 | https://www.sciencedirect.com/science/article/pii/S0959652621033588 | economic motivation |
| Chen and You, 2022 | 半封闭温室 `temperature / humidity / CO2 / light` 控制 | energy/mass-balance `NMPC` | case comparisons | 成熟 greenhouse NMPC 是多变量、多执行器、显式经济优化。 | https://www.sciencedirect.com/science/article/abs/pii/S0306261922006845 | greenhouse-native MPC |

## C. CO2 Forecasting、碳平衡与 CO2 Enrichment 文献

| 论文 | 任务 | 方法 | 指标备注 | 主要启发 | 链接 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- |
| LSTM with environmental factors for greenhouse CO2 | 温室 CO2 `2 h` ahead | `LSTM` | public abstract 主要报告 `R2`，MAE 状态不清楚 | CO2 应作为专项目标建模，而不只是共享 head。 | https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002578287 | Medium |
| Time-serial analysis of DNN models for greenhouse climate | 联合 `temperature / humidity / CO2` forecasting | `ANN`, `NARX`, `RNN-LSTM` | public page 报告 ppm 级 CO2 error | CO2 比 temperature 更难，recurrent model 仍然重要。 | https://doi.org/10.1016/j.compag.2020.105402 | High |
| Multi-model fusion for greenhouse tomato CO2 | 温室番茄 CO2 concentration | `WT + VMD + LSTM + attention + fusion` | public abstract 报告很小的无单位 MAE/RMSE，大概率是归一化指标 | CO2 受益于 decomposition 和 adaptive fusion。 | https://doi.org/10.1016/j.compag.2024.109623 | Very high |
| Mushroom greenhouse CO2 optimized LSTM | 蘑菇温室 CO2 | `VMD-SSA-LSTM`, `VMD-DBO-LSTM` | 报告 ppm 级 MAE | decomposition 加优化能明显改善 CO2 forecasting。 | https://doi.org/10.1038/s41598-025-86394-0 | High |
| Wavelet-decoupled GRU with adaptive attention | 最长 `8 h` 的温室 CO2 多步预测 | wavelet-like decoupling + `GRU` + adaptive attention | public page 报告物理 ppm error | 强支持 CO2 的多尺度 decomposition 和 adaptive weighting。 | https://doi.org/10.1016/j.atech.2025.101653 | Very high |
| Model-based CO2 control increases cucumber yield | 近环境浓度 CO2 控制 | model-based control with crop uptake | control/yield paper | CO2 控制最终应关联 dosing strategy 和 crop uptake。 | https://doi.org/10.1016/j.agrformet.2006.12.002 | High |
| MPC of Venlo greenhouse considering energy, water, CO2 | 联合资源控制 | MPC | control/economic paper | CO2 应进入 resource/economic objective，而不只是 forecast target。 | https://doi.org/10.1016/j.apenergy.2021.117163 | High |
| CO2 enrichment review | 温室 CO2 enrichment 综述 | review | review | 用于 sustainable CO2 use 和生理背景。 | https://doi.org/10.3389/fpls.2022.1029901 | Medium |
| Autocalibrating canopy photosynthesis model | 估计冠层净光合作用 | CO2 balance + black-box photosynthesis | gray-box | `CO2 balance + black-box residual` 的早期清晰先例。 | https://doi.org/10.1016/0168-1699(91)90019-6 | High |
| Net photosynthesis by mass balance | 用质量平衡估计冠层光合作用 | mass balance + mechanistic model | gray-box | 支撑把 CO2 和 canopy uptake、ventilation exchange 连接起来。 | https://doi.org/10.1016/0168-1923(94)90106-6 | High |
| Photosynthesis model validation using CO2 balance | 用温室 CO2 balance 验证光合作用模型 | CO2 balance + plant physiology | gray-box | 强化 greenhouse-native CO2 modeling。 | https://doi.org/10.1006/anbo.1999.0938 | High |

## D. 通用 Time-Series 架构文献

这些论文不是温室专项论文，但适合解释为什么把某些模型族引入 `agc_mpc`。

| 论文 | 任务 | 方法 | 主要启发 | 链接 | 用途 |
| --- | --- | --- | --- | --- | --- |
| Zeng et al., 2022 | 长时序预测 | `LTSF-Linear / DLinear` | 简单线性模型可以超过很多 Transformer；支撑稳定线性主路径。 | https://arxiv.org/abs/2205.13504 | DLinear baseline justification |
| PatchTST, 2023 | 长时序预测 | patch-based Transformer | Patching 改变 tokenization，经常改善 TS Transformer。 | https://openreview.net/forum?id=Jbdc0vTOcol | future residual candidate |
| iTransformer, 2024 | 多变量时序预测 | 按变量维度 inverted tokenization | 适合 `Tair / Rhair / CO2air / PAR` 等变量耦合。 | https://openreview.net/forum?id=JePfAI8fah | current residual backbone |
| TimeMixer, 2024 | 长时序预测 | multi-scale mixing | 支撑温室周期和扰动的多尺度分解 / 融合。 | https://openreview.net/forum?id=7oLshfEIC2 | future multi-scale route |
| SAMformer, 2024 | 多变量预测 | shallow Transformer + SAM | Transformer 弱可能与训练有关，不只是架构问题。 | https://openreview.net/forum?id=8kLzL5QBh2 | training-strategy citation |
| ETSformer, 2023 | 长时序预测 | level-growth-seasonality decomposition | 支撑分解式 Transformer 设计。 | https://openreview.net/forum?id=5m_3whfo483 | decomposition support |
| FreTS, 2023 | 时序预测 | frequency-domain MLP | 频域建模可以紧凑捕捉全局周期结构。 | https://openreview.net/forum?id=iif9mGCTfy | future frequency residual |
| TiDE, 2023 | 长时序预测 | dense encoder-decoder | 超过 Transformer 的也可能是 MLP-style 模型。 | https://arxiv.org/abs/2304.08424 | architecture baseline |
| Mamba, 2024 | sequence modeling | selective state space model | 有长记忆的动态系统可能适合 SSM-style modeling。 | https://arxiv.org/abs/2312.00752 | future dynamic backbone |
| Simple-Mamba, 2024 | 时序预测 | Mamba variant | Mamba 是否有效依赖任务和实现，不应盲目替换 DLinear。 | https://arxiv.org/abs/2403.11144 | cautionary citation |
| OneNet, 2023 | 在线时序预测 | online ensemble | 支撑动态组合不同 inductive bias 的模型。 | https://openreview.net/forum?id=Q25wMXsaeZ | expert fusion support |

## E. Control-Relevant MPC 与 Prediction-Control Correlation

### E1. 核心结论

一个模型离线预测更好，不必然意味着闭环 MPC 更好。

在固定任务、固定模型族、固定目标函数、固定 horizon 和固定调参条件下，预测精度和控制效果经常呈正相关。但是不存在通用单调规律，不能说普通 forecast MAE/MSE 更低或 R2 更高，就一定意味着 MPC 表现更好。MPC 表现取决于误差出现在 horizon 的哪个位置，模型是否保留控制敏感输入输出方向，优化器是否能看到有用梯度或敏感性，以及预测误差是否影响活跃约束和经济项。

### E2. Control-relevant MPC 文献表

| 论文 | 领域 | 主要信息 | 在我们论文里的作用 |
| --- | --- | --- | --- |
| [Huang, Malhotra, and Tamayo, 2003](https://doi.org/10.1016/S0009-2509(03)00077-0) | 过程工业 MPC 辨识与验证 | 模型应按多步 predictive control 用途验证，而不是只看普通拟合。 | control-relevant validation 的基础文献。 |
| [Misra, Darby, Panjwani, and Nikolaou, 2017](https://doi.org/10.3390/pr5030042) | 多变量 control-relevant experiment design | 准确模型还必须满足 integral controllability 等控制相关性质。 | 支撑“模型接近真实系统还不够”。 |
| [Badwe et al., 2010](https://doi.org/10.1016/j.jprocont.2009.12.006) | MPC model-plant mismatch diagnostics | 差模型不一定导致控制变差，mismatch 影响依赖 setpoint direction。 | 支撑非通用、方向依赖的 prediction-control relation。 |
| [Lawrynczuk, 2010](https://doi.org/10.1016/j.neucom.2009.12.015) | neural models for predictive control | 神经 predictor 应按 MPC 用途训练，尤其是 long-range prediction。 | 支撑 control-aware training/validation。 |
| [Potts, Romano, and Garcia, 2014](https://doi.org/10.1016/j.conengprac.2013.09.007) | MPC relevant identification | model-structure mismatch 下，多步 prediction error 方法更有用。 | 支撑真实系统不在模型类内时做 horizon-aware model selection。 |
| [Ludolfinger, Hamacher, and Martens, 2025](https://doi.org/10.1016/j.segy.2025.100202) | smart energy storage MPC | Forecast MSE 和 MPC revenue 可能不一致，perfect-forecast oracle 仍明显更强。 | 现代实证证据：prediction metrics alone 会误排 MPC model。 |
| [Wang, Mai, Li, and Ding, 2024](https://doi.org/10.3390/buildings14072212) | HVAC demand response MPC | 预测精度降低通常削弱 MPC，但低精度模型仍可能有用。 | 支撑固定设置内的条件性正相关。 |
| [Hou, Li, Nord, and Huang, 2022](https://doi.org/10.1016/j.enbuild.2021.111793) | weather uncertainty 下的 building HVAC MPC | Weather forecast correction 能恢复大部分理论 MPC 收益。 | 支撑 bias/error correction 是控制相关 forecast 操作。 |
| [Jain et al., 2018](https://doi.org/10.1186/s42162-018-0064-9) | occupancy prediction errors in HVAC MPC | occupancy 误差增大时，MPC 可能比简单静态日程还差。 | 支撑 prediction error robustness evaluation。 |
| [Esrafilian-Najafabadi and Haghighat, 2022](https://doi.org/10.1016/j.enbuild.2021.111808) | HVAC control 中 occupancy model selection | MAE/accuracy 与 control-oriented score 只有弱到中等相关。 | 支撑用 control score 替代纯 ML metrics。 |
| [Grant and Gehbauer, 2022](https://doi.org/10.26868/25746308.2022.C026) | weather forecast error 下的 dynamic facades MPC | forecast 不准确会增加 cost 和 glare，bias correction 可大幅消除损失。 | 支撑 signed bias 和 correction diagnostics。 |

### E3. 详细 control-relevant 笔记

#### Huang, Malhotra, and Tamayo, 2003

转述摘要：

这篇论文研究当模型要放进 MPC 中使用时，过程数据应该怎样过滤和验证。它不是问模型是否在普通意义上准确，而是问模型是否支持 MPC prediction horizon 内的最优多步预测。论文提出了多步预测滤波器，以及一种检测 predictive-control 相关变化的验证方案。

主要结论：

- 一个模型可能无法通过严格的普通 validation，但控制效果仍然可以接受。
- 一个模型也可能通过普通 validation，却无法带来满意控制效果。
- 验证指标应该服务于模型的最终用途。
- 对 MPC 来说，这个用途就是有限时域优化器中的多步预测。

论文可用表述：

> 按 Huang 等人的 control-relevant validation 观点，用于 MPC 的模型评估应绑定到控制器有限时域内的多步预测质量，而不能只依赖普通离线拟合。

#### Misra, Darby, Panjwani, and Nikolaou, 2017

转述摘要：

这篇 review 说明，多变量控制模型除了满足普通精度要求，还必须满足控制相关要求。它重点讨论用于估计模型阶次、以及辨识满足 integral controllability 的模型的实验设计。

主要结论：

- 标准 experiment design 不一定能产生最适合 controller design 的数据。
- 一个在普通误差意义上更接近真实过程的模型，仍可能不适合鲁棒控制。
- 多变量系统需要控制相关性质，而不仅是小的输出误差。

论文可用表述：

> 在多变量控制中，模型质量不能简化为输入输出拟合；被辨识模型还必须保留控制器所需的控制相关性质。

#### Badwe et al., 2010

转述摘要：

这篇论文用闭环数据研究 model-plant mismatch 如何影响 MPC 表现。它强调，控制差可能来自 mismatch，也可能来自扰动或调参，而且 mismatch 的影响不是均匀的，会依赖 setpoint 变化方向。

主要结论：

- 差模型不一定导致闭环控制变差。
- 控制变差也不一定只由模型误差导致。
- 模型 mismatch 的影响可能依赖 setpoint 方向。
- 需要闭环诊断来判断 mismatch 是否真的是性能下降原因。

论文可用表述：

> 模型 mismatch 需要通过闭环效应来理解，并且可能具有方向依赖性；因此预测误差必须和控制动作、设定值轨迹一起解释。

#### Lawrynczuk, 2010

转述摘要：

这篇论文把 neural model 训练和该模型之后在 MPC 中的用途联系起来。它认为，模型辨识阶段就应该考虑 MPC 会反复使用该模型做多步预测和优化，而不是只做一步预测。

主要结论：

- 用于 MPC 的 neural predictor 不应该只按普通 one-step predictor 训练。
- 递归多步误差会积累并误导 MPC。
- 训练和验证应该反映模型之后的 predictive-control 角色。

论文可用表述：

> 用于 MPC 的神经 surrogate model 应按照控制器实际使用的多步预测来训练和评估，而不是只看传统一步预测误差。

#### Potts, Romano, and Garcia, 2014

转述摘要：

这篇论文研究存在 model-structure mismatch 时的 MPC-relevant identification，也就是所选模型类无法精确表示真实系统。论文提出增强的多步 prediction error 方法，并强调 predictor 稳定性和鲁棒性。

主要结论：

- 当模型结构不完美时，MPC-relevant identification 尤其有用。
- 在 horizon 上优化扰动和预测行为可以改善控制相关拟合。
- predictor 稳定性很重要，不能只看误差大小。

论文可用表述：

> 当真实过程不在所选模型类中时，MPC 相关的多步辨识和验证比普通 prediction-error minimization 更合适。

#### Ludolfinger, Hamacher, and Martens, 2025

转述摘要：

这篇论文在 smart energy storage MPC 中比较负荷、光伏发电和电价预测模型。它同时评估 test-set prediction error 和 MPC revenue。结果显示，最低 MSE 的模型可以有最好的控制收益，但其他模型也显示出 prediction ranking 和 control ranking 不一致的情况。

主要结论：

- XGBoost 取得最低的报告 MSE 和最高的 realistic revenue gain。
- MSE 排名较差的线性模型仍取得有竞争力的 MPC revenue。
- 完美预测 oracle 的收益明显高于真实 forecast 设置。
- 标准 MSE 可能误判 forecast 对控制的价值。

论文可用表述：

> 最新 energy-storage MPC 证据表明，标准 forecast error metric 可能误判控制效用，并且即使测试误差看起来较强，真实控制器仍可能远离 perfect-forecast oracle。

#### Wang, Mai, Li, and Ding, 2024

转述摘要：

这篇论文比较 SVM、ANN、XGBoost 和 LightGBM 在 HVAC demand response MPC 中的 predictor 表现。它评估预测精度、预测时间和训练时间，然后改变模型精度水平来测试 MPC 表现如何变化。

主要结论：

- 在该论文的受控精度退化实验中，预测精度降低通常会削弱 MPC 表现。
- 即使预测精度相对较低，MPC 仍然值得使用。
- 预测速度和训练速度对 MPC 部署也重要。

论文可用表述：

> 在固定控制设置内，预测精度可以改善 MPC 表现，但跨不同模型结构选择 predictor 时仍需要 control-relevant validation。

#### Hou, Li, Nord, and Huang, 2022

转述摘要：

这篇论文研究天气 forecast 不确定性下的建筑 HVAC MPC。它引入一个简单的 forecast error model，在 MPC 优化前改善天气输入，并和传统 rule-based control、以及不做 forecast-error correction 的 MPC 对比。

主要结论：

- 天气 forecast error 会显著削弱 MPC 收益。
- 加入 error model 后，在该案例中恢复了大部分理论 energy-cost 和 comfort 收益。
- Bias/error correction 可以是一种控制相关 forecast 操作，而不只是预测改进。

论文可用表述：

> Forecast correction 应通过它对控制器 objective 和 constraints 的影响来评估，因为减少正确位置的误差可以恢复 MPC 收益。

#### Jain et al., 2018

转述摘要：

这篇论文研究 occupancy prediction error 如何影响 HVAC MPC。它使用 building thermal simulator 和真实 occupancy 数据，说明 occupancy 误差增加会同时破坏 energy use 和 comfort。论文还评估了 personal environmental comfort 层，说明它可以增强控制系统对预测误差的鲁棒性。

主要结论：

- 当 occupancy prediction error 增大时，MPC 可能比简单静态日程还差。
- 预测误差同时影响 energy 和 comfort。
- 鲁棒层可以缓解 forecast error 的影响。

论文可用表述：

> Forecast error 可能把 predictive control 推到比简单 baseline 更差，因此需要 robustness-aware validation，而不能只依赖离线指标。

#### Esrafilian-Najafabadi and Haghighat, 2022

转述摘要：

这篇论文比较不同 occupancy prediction formulation 和机器学习方法在 HVAC 控制中的作用。它不仅用 MAE 或 accuracy 评估模型，还用 thermal comfort、energy efficiency 和综合 control-oriented performance score 评估模型。

主要结论：

- 选择正确的 prediction formulation 可能比选择机器学习算法本身更重要。
- MAE 和 accuracy 与总体控制性能评分只有弱到中等相关。
- 只依赖机器学习指标，可能无法为 HVAC control 选出最好的模型。

论文可用表述：

> 在 HVAC 控制中，标准预测指标可能只与 comfort-energy 控制评分弱相关，因此 predictor selection 应加入下游控制标准。

#### Grant and Gehbauer, 2022

转述摘要：

这篇论文模拟 weather forecast error，并评估它们对 dynamic facades MPC 的影响。结果显示，不完美 forecast 会增加 cost 和 glare，而 bias correction 可以在该案例中消除大部分损失。

主要结论：

- Weather forecast error 会损害经济和舒适性相关结果。
- Bias correction 可能比原始 forecast 模型复杂度更重要。
- Forecast error 应通过下游 cost 和 comfort measures 来评估。

论文可用表述：

> 对 MPC 来说，修正系统性 forecast bias 可能比降低无差别的平均误差更有价值。

### E4. 什么时候 prediction-control correlation 通常为正

在以下条件下，预测精度和控制表现更可能正相关：

- 模型结构、控制器、目标函数、约束和优化算法固定
- 精度提升发生在进入 control objective 或 constraints 的变量上
- 精度提升发生在 receding-horizon 更新使用的 first step 或 control horizon 内
- 减少的是 signed bias，而不只是方差
- 输入输出敏感性和梯度符号被保留下来
- 预测速度仍然足够在线 MPC 使用
- 优化器无法利用模型伪影

### E5. 什么时候 prediction-control correlation 会变弱或变负

普通预测指标在以下情况下可能误导 MPC 模型选择：

- 模型改善 terminal 或 full-horizon MAE，但 worsens first-step 或 short-horizon error
- 误差发生在控制不敏感区域，导致离线指标夸大其重要性
- 小误差发生在高度控制敏感方向，导致离线指标低估其重要性
- 模型数值预测对，但输入输出敏感性错
- 模型给出平滑但有 bias 的 forecast，导致系统性控制错误
- 活跃约束依赖少数事件，而平均 MAE 把这些事件稀释掉
- 多个 exogenous forecast input 的误差在 MPC 中复合
- 控制器利用了不真实的梯度或平坦敏感性
- 计算延迟改变了实际可执行控制策略

这正是当前 PHF 结果最相关的情况。

## F. 推荐用于 `agc_mpc` 的指标

### 与 receding-horizon MPC 对齐的 forecast 指标

| 指标 | 定义 | 为什么重要 |
| --- | --- | --- |
| First-step MAE | 第 1 个 horizon step 的误差。 | 当前 simulator 使用第一步预测推进状态。 |
| Control-horizon MAE | 前 `N_c` 步误差均值，目前可用前 `6` 步。 | 前几步主导 receding-horizon 行为。 |
| Horizon-weighted MAE | 对控制相关步数给更高权重的误差。 | 比平等对待 24 步更合理。 |
| Segment MAE | 分 early、middle、late horizon 的误差。 | 能看出模型是否用 early accuracy 交换 terminal accuracy。 |
| Final-step MAE | 终端预测步误差。 | 仍然适合支撑长时域 forecasting claim。 |
| Signed bias | 每个 target 和 horizon segment 的平均有符号误差。 | 系统性高估 / 低估可能比零均值噪声更糟。 |
| Constraint-near error | 约束附近或目标极值附近的误差。 | MPC 往往最关心这些窗口。 |

### 控制敏感性指标

| 指标 | 定义 | 为什么重要 |
| --- | --- | --- |
| `dy/du` sensitivity | 小幅扰动控制输入后预测目标的变化。 | 检查执行器是否有正确建模效果。 |
| Cost-gradient magnitude | 控制 objective 对未来输入的平均绝对梯度。 | 检查 GradientMPC 是否能看到有用信号。 |
| Gradient sign consistency | 敏感性符号是否符合物理 / 控制预期。 | 符号错误会让优化器反向动作。 |
| Input-specific gradient share | 每个控制输入的梯度大小。 | 判断模型是否依赖合理执行器。 |
| Flatness / saturation score | 梯度接近零或被截断的比例。 | 解释 GradientMPC 不动作或不稳定。 |

### 闭环验证指标

| 指标 | 定义 | 为什么重要 |
| --- | --- | --- |
| MPC objective | 实际闭环评估的控制成本。 | 首要控制指标。 |
| Target MAE | `Tair`、`Rhair`、`CO2air` 的闭环 MAE。 | 把整体 objective 和 target-specific 行为拆开。 |
| Constraint violations | 约束违反次数和严重程度。 | 控制论文必须有。 |
| Resource/economic cost | 可用时加入能耗、CO2 施放、通风、灌溉或电价成本。 | 成熟温室 MPC 故事必须走向这里。 |
| Action activity | 控制输入的 total variation 或动作幅度。 | 检测过度激进或不动作的 controller。 |
| Recorded-policy gap | 与观测温室操作之间的差距。 | 在真实 cost 数据不完整时有用。 |
| Oracle gap | 与 perfect-forecast 或 ideal-preview MPC 的差距。 | 量化预测误差仍然损失多少性能。 |
| Robustness under forecast perturbation | biased / noisy forecast 下闭环退化程度。 | 把鲁棒性变成可实验验证的主张。 |

## G. 如何映射到当前结果

当前证据：

- `itransformer_co2_horizon_mixture` 是离线 CO2 forecasting leader：
  - CO2 Full MAE `43.910`
  - CO2 Final MAE `47.661`
- 它不是闭环 MPC leader：
  - `GradientMPC` objective `0.3713`
  - closed-loop CO2 MAE `28.696`
- `itransformer_co2_late_frozen_expert` 仍是当前最强 CO2 control baseline：
  - closed-loop CO2 MAE `6.298`
- `itransformer_co2_recoupled_expert` 仍是当前最强 overall objective baseline：
  - objective `0.0651`
- 初版 control-relevant validation 中，`late_residual`、`late_frozen_expert` 和 `frozen_backbone_horizon_mixture` 的综合排名领先 `horizon_mixture`。

解释：

- 文献正好预期了这种分裂。
- `horizon_mixture` 改善了 full/final offline forecasting，但当前 MPC 高度依赖 first-step 和 short-horizon behavior。
- 论文不能声称 offline leader 自动成为 control leader。
- 论文应该声称 PHF 改善了离线 CO2 forecasting，同时揭示 MPC 部署前必须引入 control-relevant validation。

## H. 可直接写进论文的段落

### Forecasting architecture 段落

近期温室 forecasting 文献表明，温室气候预测不是 plain Transformer 天然占优的任务。已有有效路线包括 linear model、带 attention 的 recurrent model、hybrid ensemble、decomposition-based architecture 和 variable-weight fusion。这说明，与其不断替换为更大的 generic backbone，更合理的路线是使用稳定主 predictor，并在其上加入 residual 或 specialist correction branch。

### CO2 段落

温室 CO2 forecasting 比温度预测更依赖运行工况，因为 CO2 浓度同时受到施放、通风交换、作物吸收和运行日程影响。因此，CO2 专项研究通常使用 decomposition、denoising、recurrent modeling 和 adaptive fusion。这支持 CO2 specialist line 和 PHF 设计：通过 protected、horizon-aware correction，把专门的 CO2 expert 并回多目标 predictor。

### Control-relevant validation 段落

MPC 的 predictor selection 不能只依赖普通 open-loop forecasting metrics。Control-relevant identification and validation 文献已经指出，一个模型在普通拟合意义上可以很准确，但仍可能不适合控制，因为控制器依赖多步预测、输入输出敏感性、活跃约束和 setpoint direction。最新 HVAC 和 energy-storage MPC 研究同样表明，MAE、MSE、accuracy 或 R2 等预测指标在固定设置内可能和控制表现相关，但跨不同模型结构或目标函数时可能误排模型。因此，本文同时使用离线 forecasting metrics 和 control-relevant metrics 评估 greenhouse predictors，包括 first-step error、short-horizon error、horizon-weighted error、control sensitivity 和 closed-loop MPC objective。

## I. 推荐引用角色

| 论文主张 | 最适合引用 |
| --- | --- |
| 温室 forecasting 不应写成 plain Transformer dominance。 | Ahn et al. 2024; Mao et al. 2024; Zeng et al. 2022 |
| Hybrid 和 variable-weight fusion 是合理温室预测路线。 | Mao et al. 2024; OneNet 2023; TimeMixer 2024 |
| CO2 受益于 decomposition 和 adaptive fusion。 | Multi-model CO2 fusion 2024; wavelet-decoupled GRU 2025; mushroom CO2 optimized LSTM 2025 |
| CO2 最终应连接 carbon balance 和 crop uptake。 | Acock et al. 1991; Nederhoff and Vegter 1994; model-based CO2 control 2007 |
| Control-relevant validation 是已有概念。 | Huang et al. 2003; Potts et al. 2014 |
| 普通 accuracy 不足以支撑多变量控制。 | Misra et al. 2017 |
| Model mismatch 的影响具有方向依赖性，且不总是单调。 | Badwe et al. 2010 |
| Neural MPC model 应按 predictive-control 角色训练 / 评估。 | Lawrynczuk 2010 |
| Forecast MSE 可能误判现代 energy system MPC 价值。 | Ludolfinger et al. 2025 |
| 固定设置下，预测精度仍然可能帮助控制。 | Wang et al. 2024 |
| Forecast correction 和 bias 对 MPC 很重要。 | Hou et al. 2022; Grant and Gehbauer 2022 |
| Occupancy / HVAC prediction metrics 可能只和控制评分弱相关。 | Jain et al. 2018; Esrafilian-Najafabadi and Haghighat 2022 |
| 成熟 greenhouse MPC 应包含 uncertainty/economic/resource terms。 | Chen and You 2022; Kim and You 2025; Svensen et al. 2024 |

## J. 后续文献任务

1. 为每篇最终会引用的论文补全 bibliographic metadata。
2. 补充显式比较 forecast quality 和 control performance 的 greenhouse-specific MPC 论文。
3. 补充 economic MPC、stochastic MPC 和 robust MPC 指标文献。
4. 补充 differentiable MPC 或 neural surrogate gradient-quality 文献。
5. 最终形成论文表格，拆分：
   - pure forecasting metrics
   - control-relevant validation metrics
   - closed-loop control metrics
   - resource/economic metrics
