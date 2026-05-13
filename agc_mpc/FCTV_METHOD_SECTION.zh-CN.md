# Forecast-To-Control Transfer Validation 方法章节

中文镜像版本。
对应英文主版：[FCTV_METHOD_SECTION.md](FCTV_METHOD_SECTION.md)

## 方法定位

本节定义 `Forecast-to-Control Transfer Validation`（FCTV），也就是用于检验预测侧模型质量是否能转化为闭环 MPC 质量的验证层。

该方法不默认认为“预测误差更低就一定控制更好”，而是把这个判断本身当成需要验证的假设。

在当前项目中，FCTV 有两个角色：

- 筛选检验：判断某个预测侧指标能否在不运行 MPC 的情况下对 predictor 排序。
- 诊断检验：解释为什么某些离线预测表现好的 predictor 仍然不能改善闭环控制。

当前证据更支持诊断角色，而不是稳定筛选角色。

## 验证对象

每个候选 predictor 接收相同的历史状态窗口、未来天气特征和未来请求控制输入，然后输出受控温室变量的多步预测：

- `Tair`
- `Rhair`
- `CO2air`

MPC 控制器在闭环 rollout 中使用该 predictor。因此最终比较分为两个层面：

- 预测侧：由离线预测轨迹计算误差。
- 控制侧：由闭环 MPC rollout 计算 tracking objective 和目标变量误差。

核心问题是预测侧排序和控制侧排序是否一致。

## 预测侧指标

FCTV 保留不运行 MPC 就能计算的指标：

- First-step MAE：下一步预测误差。
- Control-horizon MAE：控制器最直接使用的短时域平均预测误差。
- Control-horizon absolute bias：控制时域内 signed error 的绝对均值。
- Constraint-near MAE proxy：状态接近运行边界或参考带时的预测误差。
- Normalized composite score：把不同目标变量预测指标做尺度归一化后的综合分数。
- Gradient diagnostics：预测状态对未来控制输入的敏感性。

引入 first-step 和 control-horizon 指标，是因为 MPC 会反复滚动优化，短时域预测质量会被频繁使用。引入 constraint-near 指标，是因为接近参考值或运行边界时的预测错误更可能改变控制动作。引入 gradient diagnostics，是因为 MPC 需要 predictor 对候选控制输入有有效响应。

## 控制侧指标

闭环验证对所有 predictor 使用相同的控制器和 rollout 协议。

主要指标：

- `mpc_objective`：闭环 tracking/control objective。
- `mpc_tair_mae`：闭环 `Tair` tracking error。
- `mpc_rhair_mae`：闭环 `Rhair` tracking error。
- `mpc_co2_mae`：闭环 `CO2air` tracking error。

次要指标：

- control delta MAE
- action total variation

论文的主要结论应基于闭环指标，而不是只基于离线预测指标。

## 转化证据

FCTV 对每个预测侧指标和每个控制侧指标计算排序一致性。

### Spearman Rank Correlation

Spearman 相关系数量化两个排序是否同向变化。在这里，它问的是：

如果按某个预测指标给模型排序，这个排序是否和闭环 MPC 表现排序相似？

解释：

- 接近 `1`：强单调一致，该指标可能适合筛选。
- 接近 `0`：几乎没有排序关系，该指标更像诊断量。
- 小于 `0`：预测指标排序倾向于和控制指标排序相反。

可用筛选参考线不是数学定律。当前项目把 `0.2` 到 `0.4` 左右作为弱可用参考带，是因为样本量较小、模型族异质性强，而且温室 MPC 存在目标冲突。低于这个区间通常不足以做模型选择；落在这个区间只能作为谨慎的辅助筛选；只有在跨 start 和跨模型族时明显高于该区间，才适合声称具有可靠筛选能力。

### Pairwise Ordering Consistency

Pairwise ordering consistency 在这里称为“两两模型排序一致率”。

任意取两个模型，如果模型 A 的预测指标比模型 B 好，FCTV 检查 A 的闭环控制表现是不是也比 B 好。满足这一关系的模型对比例，就是两两模型排序一致率。

解释：

- `0.5`：接近随机排序。
- 高于 `0.6`：存在弱排序信号，可能有辅助价值。
- 高于 `0.7`：排序证据更强，但仍要求跨 start 和目标指标稳定。

这个统计量有实际意义，因为它直接对应模型选择问题：选择预测指标更好的模型，是否也更可能选到控制表现更好的模型。

### Top-K Overlap

Top-k overlap 检查预测侧排名靠前的模型，是否也出现在闭环排名靠前的模型中。

这很重要，因为实际模型选择通常更关心能否筛出候选短名单，而不是精确排列所有弱模型。

### 稳健性检查

FCTV 需要检查结论是否能跨以下条件保持：

- 不同 rollout start indices
- leave-one-model analysis
- 当 family 标签可用时，做 leave-one-family analysis
- `Tair`、`Rhair`、`CO2air` 的逐目标比较

如果某个预测指标只在一个 start、一个目标变量或一个狭窄本地模型族内有效，就不应称为可靠 selector。

## 当前实证模式

探索阶段得到的模式很清楚：

- 早期 CO2-focused 模型池中，短时域 CO2 指标看起来有筛选价值。
- 扩大到更广模型池后，CO2 预测指标失去稳定筛选能力。
- 推进到多目标、多起点分析后，transfer 表现出模型池依赖和 start 依赖。

代表性证据：

- 在扩展后的 24 模型分析中，`Rhair` first-step error 与 `Rhair` 闭环误差保持中等相关（`Spearman = 0.592`，两两模型排序一致率 `0.732`）。
- 在同一 24 模型分析中，CO2 first-step transfer 很弱（`Spearman = 0.168`，两两模型排序一致率 `0.549`），CO2 constraint-near transfer 接近随机（`Spearman = 0.015`，两两模型排序一致率 `0.507`）。
- 在最终 16 模型、5 起点分析中，CO2 first-step transfer 仍然具有 start dependence（starts `0`、`96`、`192`、`288`、`384` 分别为 `0.364`、`0.037`、`-0.149`、`0.243`、`-0.319`）。
- Multi-objective transfer score 同样不稳定（starts `0`、`96`、`192`、`288`、`384` 分别为 `0.406`、`0.235`、`0.174`、`0.362`、`0.141`）。

最终闭环模型证据：

- `current_hybrid_transformer` 在 5 个 start 上平均 objective 最好（`0.0662 +/- 0.0269`）。
- `itransformer_co2_residual` 的平均 CO2 闭环 tracking error 最好（`CO2air MAE = 10.215 +/- 2.043`），同时平均 objective 排名第二（`0.0701 +/- 0.0234`）。
- 这把模型结论和指标结论区分开：直接闭环 MPC 验证可以识别稳健 winner，但已测试的离线预测指标仍不能单独可靠筛出这些 winner。

这个模式说明，当前预测侧指标不应被用作确定性的闭环模型 selector。

## 面向论文的结论边界

可以严谨表述的结论是：

在当前跨模型族温室控制 benchmark 中，标准离线预测指标即使被改造成短时域和 constraint-near 指标，也不能可靠替代直接闭环 MPC 验证。

这个方法仍然有价值，因为它指出了 forecast-control 假设失效的位置：

- 目标变量依赖
- 模型族依赖
- rollout segment 依赖
- 预测误差和控制动作敏感性之间不匹配

因此，论文中应把 FCTV 表述为验证和诊断框架，而不是已经完成的通用 forecast-derived control score。
