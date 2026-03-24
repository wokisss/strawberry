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

| 论文 | 任务 | 主模型/方法 | 对比模型 / baseline | 启发 | 链接 |
| --- | --- | --- | --- | --- | --- |
| Ahn et al., 2024 | 温室 `temperature / RH / CO2` 预测，`1 h` 与 `3 h` | `Autoformer` | `DLinear`, `LSTM`, `SegRNN` | 和我们最像的一篇。结论很关键：温室时序里 `DLinear` 与 `SegRNN` 可以稳定强于 transformer 类基线，这和我们当前 `AGC` 观察一致。 | https://www.mdpi.com/2073-4395/14/3/417 |
| Li et al., 2024 | 温室空气温度与土壤温度多步预测，`30-480 min` | `Attention-LSTM` | `RNN`, `GRU`, `LSTM` | 说明温室场景里“RNN 主干 + 注意力”仍然是常见强基线，不是所有工作都转向 plain Transformer。 | https://www.sciencedirect.com/science/article/pii/S0168169923009900 |
| Mao et al., 2024 | 温室 `temperature / humidity / PAR` 预测，`30-120 min` | `PSO-BiGRU-Attention-LightGBM` | `BiGRU-Attention`, `LightGBM`, 等权组合模型 | 这类工作不是单 backbone 竞赛，而是 hybrid/ensemble。对我们最重要的启发是：如果想真正超过 `DLinear`，更像是做 residual/hybrid，而不是只加一个更重的 Transformer。 | https://www.sciencedirect.com/science/article/pii/S0168169924002096 |
| Wang et al., 2025 | 温室土壤温度预测，`3 h / 6 h / 24 h / 48 h` | `ReSSA-iTransformer` = `iTransformer + RevIN + SSA` | `LSTM`, `Informer`, `Autoformer`, `iTransformer` | 就算走 Transformer 路线，强版本通常也带着分解、归一化、task-specific 改造，不是 plain Transformer。 | https://www.mdpi.com/2073-4395/15/1/223 |
| Choi & Yang, 2025 | 温室 `temperature / RH / CO2` 概率预测，`3 h` | `Probabilistic 1D CNN`, `Probabilistic LSTM` | 对应 deterministic `1D CNN` 与 `LSTM` | 对我们非常有价值，因为它把 uncertainty 也纳入目标。后续如果我们要把 forecasting 真正接到 control，这类概率建模比单纯继续堆 point forecaster 更重要。 | https://www.mdpi.com/2077-0472/15/23/2461 |
| Seri et al., 2025 | 温室微气候建模，重点看变量耦合增强后的温度预测 | `Directed STGNN` | `RNN` | 说明当输入变量和耦合结构变复杂后，图结构模型可能开始占优。对我们意味着：如果后面把 `sp -> actuator -> climate` 做实，结构化模型会更有价值。 | https://www.sciencedirect.com/science/article/pii/S0360132325009461 |
| Cebolla-Alemany et al., 2026 | 屋顶温室短时空气温度预测，`5 / 10 / 15 min` | `Thermocast` 模块化集成模型 | 多种传统与集成回归基线 | 分数非常高，但任务明显更简单：只做温度、超短时、非多变量控制 surrogate。可当上限参考，不能拿来直接压我们的 `2 h` 多变量任务。 | https://www.sciencedirect.com/science/article/pii/S2772375525009645 |

## B. Greenhouse Control Papers

| 论文 | 任务 | 主模型/方法 | 对比模型 / baseline | 启发 | 链接 |
| --- | --- | --- | --- | --- | --- |
| Svensen et al., 2024 | 温室生产系统控制，考虑参数不确定性 | `Chance-constrained SMPC` | 与直接的非线性随机 chance-constrained MPC 方案对比 | 这类控制论文的重点不是 predictor 名字，而是 uncertainty formulation、约束处理和 tractability。对我们意味着：后期论文竞争点会在鲁棒/随机 MPC，而不只是 forecast MAE。 | https://www.sciencedirect.com/science/article/pii/S0168169923009663 |
| Le & Bui, 2025 | 智能温室 NMPC，40 天仿真，扰动预览与 warm-start 对比 | `NMPC` + `LSTM disturbance forecast` | `feedback only`, `LSTM forecast`, `ideal preview`, 各自 cold/warm start | 很适合作为我们控制层的参照物。它说明预测器可以作为 MPC 的 preview module，而控制论文会专门比较 warm-start、forecast quality、preview quality。 | https://www.mdpi.com/2076-3417/15/14/7988 |
| Mallick et al., 2025 | 温室气候控制，面向预测不确定性的 RL-based MPC | `RL-based MPC` | 相对传统 `robust / stochastic MPC` 路线定位 | 代表“控制器本身学习化”的方向。对我们而言，这更适合在 surrogate 和 benchmark 做稳之后再上，不应早于基础 MPC benchmark。 | https://www.sciencedirect.com/science/article/pii/S2772375524003551 |
| Kim & You, 2025 | 温室气候控制，考虑不确定性与能源效率 | `GP-SMPC` + online learning | `NMPC`, 并定位于 `RMPC / DDRMPC` 之上 | 和我们最终方向最接近的一篇之一。它清楚表明真正强的控制论文会把 uncertainty、online correction、energy/CO2 cost 一起放进去。 | https://www.sciencedirect.com/science/article/pii/S0306261925005719 |
| Mansour et al., 2025 | 半封闭温室气候控制，鲁棒性 + 经济优化 + 迁移能力 | `Hierarchical MPC + DRL` | 相对 `robust/stochastic MPC` 与 `model-free DRL` 定位 | 代表后期更完整的系统形态：上层经济目标，下层跟踪控制。对我们来说，这更像控制论文终局，而不是当前 baseline 阶段的直接下一步。 | https://www.sciencedirect.com/science/article/pii/S2772375525005581 |
| Mahmood et al., 2021 | 温室温度控制与节能 | `ML model + MPC` | conventional control | 虽然年份稍早，但非常适合提醒我们：控制论文最后要交付的不只是 tracking，也包括 energy/resource outcome。 | https://www.sciencedirect.com/science/article/pii/S0959652621033588 |
| Chen & You, 2022 | 半封闭温室 `temperature / humidity / CO2 / light` 控制与能耗优化 | `NMPC` + energy/mass balance model | 与案例工况对照 | 这类工作说明真正成熟的 greenhouse NMPC 往往是多变量、多执行器、显式经济成本。它是我们后期论文定位的重要参考。 | https://www.sciencedirect.com/science/article/abs/pii/S0306261922006845 |

## C. General Time-Series Model References

These are not greenhouse-specific papers.
They are useful when deciding which architecture family is worth importing into `agc_mpc`.

| 论文 | 任务 | 主模型/方法 | 对比模型 / baseline | 启发 | 链接 |
| --- | --- | --- | --- | --- | --- |
| PatchTST, 2023 | 通用长时序预测 | `PatchTST` | `DLinear` and multiple transformer baselines | 最核心启发是 patching：不要把每个时间点都当 token。对我们如果继续走 Transformer 路线，比 plain Transformer 更值得尝试。 | https://openreview.net/forum?id=Jbdc0vTOcol |
| iTransformer, 2024 | 通用多变量时序预测 | `iTransformer` | 多种 transformer 与 linear baseline | 它把 token 组织方式反过来，更强调变量维度关系。对我们的多变量 greenhouse 任务是直接相关的参考。 | https://openreview.net/forum?id=JePfAI8fah |
| TiDE, 2023 | 通用长时序预测 | `TiDE` dense encoder-decoder | `DLinear`, transformer families | 一个重要提醒：在更大 benchmark 上，超越 `DLinear` 的不一定是 Transformer，也可能是更强的 MLP-style 模型。 | https://arxiv.org/abs/2304.08424 |
| Mamba, 2024 | 通用序列建模 | `Selective State Space Model` | attention / SSM families | 对我们价值在于：温室控制问题本质上像动态系统，Mamba 可能比 plain Transformer 更贴合长历史状态传播。 | https://arxiv.org/abs/2312.00752 |
| Wang et al., 2024 | 通用时序预测中 Mamba 是否有效 | `Simple-Mamba` | 多种 TSF baseline | 说明 Mamba 在时序预测中有潜力，但是否真正超越 `DLinear` 仍依赖任务与实现。对我们更合理的路线是 `DLinear + Mamba residual`，而不是盲目纯 Mamba 替换。 | https://arxiv.org/abs/2403.11144 |

## D. What This Means for `agc_mpc`

| 问题 | 结论 |
| --- | --- |
| 温室论文里是不是大家都在用 Transformer？ | 不是。近两年温室预测里常见对比仍然是 `DLinear / SegRNN / LSTM / GRU / Attention-LSTM / hybrid ensemble / CNN / GNN`。 |
| 为什么我们当前 `DLinear` 这么强？ | 这和文献一致，不反常。温室时序普遍有强趋势、强周期、慢动态、强外生驱动，`DLinear` 这种强归纳偏置模型天然占优。 |
| 继续堆 plain Transformer 值得吗？ | 不太值得。更合理的是 `patching / inversion / decomposition / normalization` 方向，或者直接做 hybrid residual。 |
| 真正可能超过当前 baseline 的方向是什么？ | `DLinear main path + residual branch`，优先考虑 `Mamba / PatchTST / iTransformer` 这类更时序化的强分支。 |
| 控制论文真正的竞争点是什么？ | uncertainty、robust/stochastic formulation、economic objective、online correction，而不是只比 point forecast。 |
| 对我们下一步最直接的建议是什么？ | 先做 `hybrid residual forecaster`，再做 probabilistic version，最后把它接进更严格的 AGC closed-loop benchmark。 |

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
| Inputs | indoor + outdoor climate + control-operation variables; paper states the BiGRU-Attention input layer has `21` neurons |
| Targets | `air temperature`, `air humidity`, `PAR` |
| History length | paper states input step size is `120 min` |
| Horizons | paper reports `30-120 min` forecasting |
| Single models | `GRU`, `BiGRU`, `BiGRU-Attention`, `XGBoost`, `LightGBM` |
| Ensemble models | equal-weight `BiGRU-Attention-LightGBM`, and variable-weight `PSO-BiGRU-Attention-LightGBM` |
| Metrics | `RMSE`, `MAE`, `R2` |
| Data availability | `Data will be made available on request` |

### Why its R2 is so high

The paper's high `R2` is plausible, but it does **not** mean the task is equivalent to our current `AGC` setup.

Main reasons:

1. The task is easier than our current setup.
   - They predict only `temperature / humidity / PAR`
   - They do **not** include `CO2`
   - Their horizon is shorter overall than our `2 h` control-oriented multi-output framing

2. The targets are structurally favorable.
   - `temperature` and `humidity` in a single greenhouse often have strong smoothness and daily regularity
   - `PAR` is still harder, and their reported `R2` for `PAR` at `2 h` is notably lower than temperature/humidity

3. Their model is not a plain backbone.
   - It is a weighted ensemble of two very different learners:
     - `BiGRU-Attention` for sequential nonlinear dynamics
     - `LightGBM` for nonlinear tabular mapping
   - Then `PSO` is used to optimize the weight coefficients of the two models at different prediction times
   - So the reported result already includes hybridization and weight tuning

4. Their setup is optimized around prediction only.
   - The paper is forecasting-oriented
   - It is not trying to be a control surrogate with explicit future control rollout consistency

5. It is a single-greenhouse, single-system dataset.
   - This can make fit easier than a multi-compartment benchmark with more operational diversity

### Is its dataset similar to ours

Partly similar, but not equivalent.

| 维度 | Mao et al., 2024 | Our current `AGC` |
| --- | --- | --- |
| Domain | greenhouse climate prediction | greenhouse forecasting for control |
| Greenhouse count | single greenhouse | multiple compartments |
| Crop | cherry tomato | AGC challenge compartments |
| Sampling interval | `10 min` | `5 min` |
| Targets | `temperature / humidity / PAR` | `Tair / Rhair / CO2air / Tot_PAR` |
| Control usage | uses control-operation data as inputs | explicitly uses future candidate control `u_future` |
| Main goal | accurate short-term prediction | accurate multi-step prediction plus downstream MPC usefulness |
| CO2 target | no | yes |
| Closed-loop control evaluation | no | yes |

So the relationship is:

- similar enough to be informative
- not similar enough for strict apples-to-apples score comparison

The biggest differences are:

- we predict `CO2air`, they do not
- our interface is explicitly control-oriented
- we care about closed-loop behavior, they do not

### Can we reproduce it

Partially yes, exactly no.

What is reproducible in principle:

- the overall idea
- the model family:
  - `BiGRU-Attention`
  - `LightGBM`
  - equal-weight ensemble
  - `PSO`-optimized variable-weight ensemble
- the approximate training recipe disclosed in the paper:
  - BiGRU-Attention hidden units for different targets
  - Adam with learning rate `0.001`
  - LightGBM leaf node `35`, learning rate `0.1`
  - PSO particle number `50`, inertia weight `0.8`, max iteration `100`

What blocks exact reproduction:

1. The dataset is not public.
   - The paper says: `Data will be made available on request`

2. The split protocol is not fully standardized for external reproduction.
   - Even if we have the raw data, exact split/preprocessing details may still matter

3. Some implementation details are still under-specified.
   - for example exact feature engineering, exact multi-horizon training details, and how the final time-dependent ensemble weighting is operationalized in code

So the realistic answer is:

- exact paper-number reproduction is unlikely unless the authors share data and preprocessing details
- method-level reproduction is feasible

### What we should learn from it

This paper does **not** mainly tell us:

- "BiGRU is better than Transformer"

What it more usefully tells us is:

- a hybrid predictor can outperform single models in greenhouse forecasting
- combining a sequence model and a tree model can work well
- variable weighting across horizons is meaningful
- strong humidity performance often comes from more engineered, task-specific designs rather than plain backbones

For `agc_mpc`, the most relevant adaptation is not to literally copy this paper.
The more relevant lesson is:

- keep a stable main path
- add a nonlinear residual or auxiliary branch
- consider horizon-dependent fusion or weighting

That is one of the reasons our next-step direction should be closer to:

- `DLinear + residual branch`

than to:

- another plain Transformer

## F. Suggested Next-Step Reading Order

1. Ahn et al., 2024
   - 先理解为什么温室任务里 `DLinear` 会这么强
2. Mao et al., 2024
   - 看 hybrid/ensemble 怎么把性能继续往上推
3. Choi & Yang, 2025
   - 看 uncertainty 为什么对 greenhouse forecasting/control 关键
4. Kim & You, 2025
   - 看 uncertainty + energy objective 如何真正进入控制框架
5. PatchTST / iTransformer / Mamba
   - 再决定我们的新模型分支怎么选

## Bottom Line

For this project, the strongest next direction is not:

- another plain Transformer

The stronger direction is:

- `DLinear` as the stable main path
- a stronger nonlinear residual branch
- then uncertainty-aware and control-oriented extensions

That path is the most consistent with both:

- our current AGC results
- and the recent paper landscape
