# Recent Papers Survey

## Purpose

This note organizes recent forecasting and control papers that are relevant to the current `agc_mpc` project.

The focus is practical:

- what task each paper solves
- what model or controller it uses
- what baselines it compares against
- what we can actually learn from it for `AGC`

This is not a leaderboard.
Direct metric comparison is only valid when the task, horizon, variables, and data regime are close enough.

## Reading Guide

- `Most comparable`: greenhouse multi-variable forecasting or greenhouse climate control
- `Partially comparable`: greenhouse tasks with shorter horizons, fewer targets, or different objectives
- `Reference only`: general time-series papers that are useful for model design, but not directly comparable to our greenhouse setup

## A. Greenhouse Forecasting Papers

| 论文 | 任务 | 主模型 / 方法 | 对比模型 / baseline | 启发 | 链接 |
| --- | --- | --- | --- | --- | --- |
| Ahn et al., 2024 | 温室 `temperature / RH / CO2` 预测，`1 h` 与 `3 h` | `Autoformer` | `DLinear`, `LSTM`, `SegRNN` | 和我们最像的一篇。它提醒我们：在温室时序里，`DLinear` 和 `SegRNN` 并不弱，Transformer 不会天然占优。 | https://www.mdpi.com/2073-4395/14/3/417 |
| Li et al., 2024 | 温室空气温度与土壤温度多步预测，`30-480 min` | `Attention-LSTM` | `RNN`, `GRU`, `LSTM` | 说明温室里“RNN 主干 + 注意力”仍是有效路线，不是所有工作都在转向 plain Transformer。 | https://www.sciencedirect.com/science/article/pii/S0168169923009900 |
| Mao et al., 2024 | 温室 `temperature / humidity / PAR` 预测，`30-120 min` | `PSO-BiGRU-Attention-LightGBM` | `BiGRU-Attention`, `LightGBM`, 等权组合模型 | 这类工作真正强的地方是 hybrid / ensemble，而不是单一 backbone。对我们最直接的启发就是：若想真正超过 `DLinear`，更合理的是 residual / hybrid，而不是只堆更重的 Transformer。 | https://www.sciencedirect.com/science/article/pii/S0168169924002096 |
| Wang et al., 2025 | 温室土壤温度预测，`3 h / 6 h / 24 h / 48 h` | `ReSSA-iTransformer` = `iTransformer + RevIN + SSA` | `LSTM`, `Informer`, `Autoformer`, `iTransformer` | 就算走 Transformer 路线，强版本通常也带着分解、归一化和任务特定改造，不是 plain Transformer。 | https://www.mdpi.com/2073-4395/15/1/223 |
| Choi & Yang, 2025 | 温室 `temperature / RH / CO2` 概率预测，`3 h` | `Probabilistic 1D CNN`, `Probabilistic LSTM` | 对应 deterministic `1D CNN` 与 `LSTM` | 很适合提醒我们：如果 forecasting 真要服务控制，不确定性建模比继续堆 point forecaster 更重要。 | https://www.mdpi.com/2077-0472/15/23/2461 |
| Seri et al., 2025 | 温室微气候建模，强调变量耦合增强后的温度预测 | `Directed STGNN` | `RNN` | 如果后续把 `sp -> actuator -> climate` 做实，结构化耦合模型会比纯时间 backbone 更有价值。 | https://www.sciencedirect.com/science/article/pii/S0360132325009461 |
| Cebolla-Alemany et al., 2026 | 屋顶温室短时空气温度预测，`5 / 10 / 15 min` | `Thermocast` 模块化集成模型 | 多种传统与集成回归 baseline | 分数很高，但任务明显更简单，只做温度、超短时、非多变量控制 surrogate。适合作上限参考，不适合作直接对比分数。 | https://www.sciencedirect.com/science/article/pii/S2772375525009645 |

## B. Greenhouse Control Papers

| 论文 | 任务 | 主模型 / 方法 | 对比模型 / baseline | 启发 | 链接 |
| --- | --- | --- | --- | --- | --- |
| Svensen et al., 2024 | 温室生产系统控制，考虑参数不确定性 | `Chance-constrained SMPC` | 非线性 chance-constrained MPC 方案 | 控制论文真正竞争的不是 predictor 名字，而是 uncertainty formulation、约束处理和 tractability。 | https://www.sciencedirect.com/science/article/pii/S0168169923009663 |
| Le & Bui, 2025 | 智能温室 NMPC，`30` 天仿真，比较扰动预览与 warm start | `NMPC` + `LSTM disturbance forecast` | `feedback only`, `forecast preview`, `ideal preview`, cold/warm start | 非常适合做我们控制侧的参照：预测器可以是 preview module，而控制论文会专门比较 warm-start、forecast quality 和 preview quality。 | https://www.mdpi.com/2076-3417/15/14/7988 |
| Mallick et al., 2025 | 温室气候控制，面向预测不确定性的 RL-based MPC | `RL-based MPC` | 相对传统 `robust / stochastic MPC` 的定位 | 代表“控制器本身学习化”的方向，但应该排在基础 surrogate benchmark 稳住之后。 | https://www.sciencedirect.com/science/article/pii/S2772375524003551 |
| Kim & You, 2025 | 温室气候控制，考虑不确定性与能源效率 | `GP-SMPC` + online learning | `NMPC`, 并定位在 `RMPC / DDRMPC` 之上 | 说明真正成熟的控制论文会把 uncertainty、online correction、energy / CO2 cost 一起放进去。 | https://www.sciencedirect.com/science/article/pii/S0306261925005719 |
| Mansour et al., 2025 | 半封闭温室气候控制，鲁棒性 + 经济优化 + 迁移能力 | `Hierarchical MPC + DRL` | `robust/stochastic MPC`, `model-free DRL` | 代表后期更完整的系统形态：上层经济目标，下层跟踪控制。它更像论文终局，而不是当前 baseline 阶段的直接下一步。 | https://www.sciencedirect.com/science/article/pii/S2772375525005581 |
| Mahmood et al., 2021 | 温室温度控制与节能 | `ML model + MPC` | conventional control | 重要提醒：控制论文最终交付的不只是 tracking，也包括 energy / resource outcome。 | https://www.sciencedirect.com/science/article/pii/S0959652621033588 |
| Chen & You, 2022 | 半封闭温室 `temperature / humidity / CO2 / light` 控制与能耗优化 | `NMPC` + energy / mass balance model | 案例工况对照 | 这类工作说明真正成熟的 greenhouse NMPC 往往是多变量、多执行器、显式经济成本。 | https://www.sciencedirect.com/science/article/abs/pii/S0306261922006845 |

## C. General Time-Series Model References

These are not greenhouse-specific papers.
They are useful when deciding which architecture family is worth importing into `agc_mpc`.

| 论文 | 任务 | 主模型 / 方法 | 对比模型 / baseline | 启发 | 链接 |
| --- | --- | --- | --- | --- | --- |
| Zeng et al., 2022 | 通用长时序预测 | `LTSF-Linear / DLinear` | 多种 Transformer 系列 | 这篇是当前主线的理论起点。它直接质疑“Transformer 天然适合长时序预测”，并给出简单线性模型强于多种 Transformer 的结论。对我们而言，它支撑“DLinear 做稳定主路径，复杂模型只学残差”的策略。 | https://arxiv.org/abs/2205.13504 |
| PatchTST, 2023 | 通用长时序预测 | `PatchTST` | `DLinear` and multiple transformer baselines | 最核心启发是 patching：不要把每个时间点都当 token。对我们如果继续走 Transformer 路线，`PatchTST residual` 比 plain Transformer 更值得尝试。 | https://openreview.net/forum?id=Jbdc0vTOcol |
| iTransformer, 2024 | 通用多变量时序预测 | `iTransformer` | 多种 transformer 与 linear baseline | 它把 token 组织方式反过来，更强调变量维度关系。对温室这种 `Tair / Rhair / CO2air / PAR` 强耦合任务，这是非常直接的参考。 | https://openreview.net/forum?id=JePfAI8fah |
| TimeMixer, 2024 | 通用长时序预测 | `TimeMixer` 多尺度 mixing | 多种 Transformer / linear / MLP baseline | 强调多尺度分解与多预测器融合。对我们的直接启发是：温室里昼夜周期和短时扰动并存，多尺度 residual 分支很可能比单一 plain Transformer 更合适。 | https://openreview.net/forum?id=7oLshfEIC2 |
| SAMformer, 2024 | 通用多变量时序预测 | `Shallow Transformer + SAM` | `Transformer`, `TSMixer`, `iTransformer`, `PatchTST` 等 | 它的价值不是“又一个 Transformer”，而是指出 Transformer 在时序任务里常常输给线性模型的一个重要原因是训练不稳、陷入差的局部极值。对我们意味着：如果 residual 分支效果一般，不一定是结构错，也可能是训练策略不对。 | https://openreview.net/forum?id=8kLzL5QBh2 |
| ETSformer, 2023 | 通用长时序预测 | `level-growth-seasonality` decomposed Transformer | 多种 Transformer baseline | 说明“分解式 Transformer”本身就是成熟路线。它支持我们把趋势交给稳定主路径，把更难的季节 / 残差部分交给深层分支。 | https://openreview.net/forum?id=5m_3whfo483 |
| FreTS, 2023 | 通用时序预测 | `Frequency-domain MLP` | 多种 TSF baseline | 强调频域建模的两个好处：更完整的全局视角和更紧凑的能量分布。对温室这类昼夜周期很强的任务，频域 residual 分支是值得保留的后续候选。 | https://openreview.net/forum?id=iif9mGCTfy |
| TiDE, 2023 | 通用长时序预测 | `TiDE` dense encoder-decoder | `DLinear`, transformer families | 一个重要提醒：超过 `DLinear` 的不一定是 Transformer，也可能是更强的 MLP-style 模型。 | https://arxiv.org/abs/2304.08424 |
| Mamba, 2024 | 通用序列建模 | `Selective State Space Model` | attention / SSM families | 温室控制问题本质上很像动态系统，Mamba 可能比 plain Transformer 更贴近长历史状态传播。 | https://arxiv.org/abs/2312.00752 |
| Wang et al., 2024 | 通用时序预测中 Mamba 是否有效 | `Simple-Mamba` | 多种 TSF baseline | 说明 Mamba 有潜力，但是否真能超过 `DLinear` 仍然高度依赖任务与实现。对我们更合理的路线是 `DLinear + Mamba residual`，不是盲目纯 Mamba 替换。 | https://arxiv.org/abs/2403.11144 |
| OneNet, 2023 | 在线时序预测与概念漂移 | `Online ensembling network` | 两类不同归纳偏置模型的在线组合 | 这篇最重要的启发不是在线学习本身，而是“动态组合不同归纳偏置的模型”。它支持我们后续从固定 `base + residual` 迈向动态门控专家混合。 | https://openreview.net/forum?id=Q25wMXsaeZ |

## D. What This Means for `agc_mpc`

| 问题 | 结论 |
| --- | --- |
| 温室论文里是不是大家都在用 Transformer？ | 不是。近两年温室预测里常见强 baseline 仍包括 `DLinear / SegRNN / LSTM / GRU / Attention-LSTM / hybrid ensemble / CNN / GNN`。 |
| 为什么我们当前 `DLinear` 这么强？ | 这和文献是一致的，不反常。温室时序普遍有强趋势、强周期、慢动态和强外生驱动，`DLinear` 这类强归纳偏置模型天然占优。 |
| 继续堆 plain Transformer 值得吗？ | 不太值得。更合理的是 `patching / inversion / decomposition / normalization` 方向，或者直接做 hybrid residual。 |
| 真正可能超过当前 baseline 的方向是什么？ | `DLinear main path + residual branch`。当前最值得优先做的三个选型是：`Transformer-hybrid residual`、`iTransformer residual`、`PatchTST residual`。 |
| 为什么不是立刻开更多第四、第五条线？ | 因为当前最需要的是统一协议下的可比实验，而不是同时开太多新坑。先把前三个 residual 选型跑清楚，再决定是否继续扩到 `TimeMixer / FreTS / Mamba`。 |
| 控制论文真正的竞争点是什么？ | uncertainty、robust / stochastic formulation、economic objective、online correction，而不是只比 point forecast。 |
| 对我们下一步最直接的建议是什么？ | 先把这三个 residual 选型在同一 fair-budget 协议下跑出来，再把最强者接到更严格的 AGC closed-loop benchmark。 |

## E. Detailed Note: Mao et al., 2024

Paper:

- Xiaojuan Mao et al., 2024
- `A variable weight combination prediction model for climate in a greenhouse based on BiGRU-Attention and LightGBM`
- Link: https://www.sciencedirect.com/science/article/pii/S0168169924002096

### What the paper actually does

| 项目 | 内容 |
| --- | --- |
| Greenhouse | 单个中国南京 Venlo-type greenhouse |
| Crop | cherry tomato |
| Time range | `2020-09-23` to `2021-06-06` |
| Sampling interval | `10 min` |
| Sample count | `37,008` |
| Inputs | indoor + outdoor climate + control-operation variables |
| Targets | `air temperature`, `air humidity`, `PAR` |
| History length | `120 min` |
| Horizons | `30-120 min` |
| Single models | `GRU`, `BiGRU`, `BiGRU-Attention`, `XGBoost`, `LightGBM` |
| Ensemble models | equal-weight `BiGRU-Attention-LightGBM` and variable-weight `PSO-BiGRU-Attention-LightGBM` |
| Metrics | `RMSE`, `MAE`, `R2` |
| Data availability | `Data will be made available on request` |

### Why its `R2` is so high

The paper's high `R2` is plausible, but it does not mean the task is equivalent to our current `AGC` setup.

Main reasons:

1. The task is easier than our current setup.
   - It predicts only `temperature / humidity / PAR`
   - It does not include `CO2`
   - Its horizon is shorter overall than our current `2 h` control-oriented multi-output setup

2. The targets are structurally favorable.
   - `temperature` and `humidity` in a single greenhouse often have strong smoothness and daily regularity
   - `PAR` is harder, and its reported `R2` drops more obviously as the horizon grows

3. Its model is already a hybridized design.
   - `BiGRU-Attention` handles nonlinear sequential dynamics
   - `LightGBM` handles nonlinear tabular mapping
   - `PSO` optimizes horizon-dependent weights

4. Its setup is optimized around forecasting only.
   - It is not trying to be a control surrogate
   - It does not need future control rollout consistency

5. It is a single-greenhouse dataset.
   - This is generally easier than a multi-compartment benchmark with stronger operational diversity

### What we should learn from it

This paper does not mainly tell us:

- "BiGRU is better than Transformer"

What it more usefully tells us is:

- a hybrid predictor can outperform single models in greenhouse forecasting
- combining sequence models and tabular learners can work well
- variable weighting across horizons is meaningful
- strong humidity performance often comes from more engineered designs rather than plain backbones

For `agc_mpc`, the most relevant adaptation is:

- keep a stable main path
- add a nonlinear residual or auxiliary branch
- consider horizon-dependent fusion or weighting

That is one reason our current next-step direction is:

- `DLinear + residual branch`

rather than:

- another plain Transformer

## F. Suggested Next-Step Reading Order

1. Zeng et al., 2022
   - 先理解为什么 `DLinear` 这类线性模型在长时序预测里会这么强
2. Mao et al., 2024
   - 看 hybrid / ensemble 怎样把性能继续往上推
3. PatchTST, 2023
   - 看 patching 怎样改善 Transformer 的 token 组织方式
4. iTransformer, 2024
   - 看 inversion 怎样直接建模多变量耦合
5. TimeMixer, 2024
   - 看多尺度混合和分解对复杂周期任务的意义
6. SAMformer, 2024
   - 看为什么 Transformer 可能输在训练而不只是输在结构
7. Choi & Yang, 2025 and Kim & You, 2025
   - 再把视角切回 uncertainty 和 economic control

## Bottom Line

For this project, the strongest next direction is not:

- another plain Transformer

The stronger direction is:

- `DLinear` as the stable main path
- compare three residual candidates first:
  - `Transformer-hybrid residual`
  - `iTransformer residual`
  - `PatchTST residual`
- then connect the strongest one to control
- finally move toward uncertainty-aware and economic extensions
