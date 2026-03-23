# AGC 2019 数据字段整理与建模映射

## 1. 文档目的

本文档用于整理 `Autonomous Greenhouse Challenge, Second Edition (2019)` 数据集的结构、字段含义以及与本项目预测控制任务的映射关系。

目标是为后续 `diffmpc` 迁移提供统一的数据接口设计，重点服务于以下任务：

- 基于历史室内状态进行多步预测
- 显式利用未来天气预报和未来控制设定值
- 支持闭环 MPC / DPC
- 保留资源消耗、产量和品质作为后续扩展指标


## 2. 数据集背景

### 2.1 数据来源

该数据集来自：

- `Autonomous Greenhouse Challenge, Second Edition (2019)`

数据集记录了 5 个 AI 队伍和 1 个人工参考温室隔间在一个 6 个月樱桃番茄生产周期中的运行数据。

参与隔间：

- `AICU`
- `Automatoes`
- `Digilog`
- `IUACAAS`
- `TheAutomators`
- `Reference`

### 2.2 时间分辨率

- 主时序数据分辨率：`5 分钟`
- 时间字段：Excel serial date

示例：

- `43815.0 -> 2019-12-16 00:00:00`

### 2.3 数据层次

这套数据不是普通时序日志，而是一个多层控制系统日志，包含：

1. 室外天气
2. 室内状态
3. 控制请求设定值 `*_sp`
4. 实现后的控制参考值 `*_vip`
5. 执行器实际状态
6. 资源消耗
7. 产量与品质
8. 根区与营养液信息


## 3. 目录结构说明

建议主用目录：

- [AutonomousGreenhouseChallenge_edition2](c:/repositories/strawberry/AutonomousGreenhouseChallenge_edition2)

核心文件：

- [ReadMe.pdf](c:/repositories/strawberry/AutonomousGreenhouseChallenge_edition2/ReadMe.pdf)
- [Economics.pdf](c:/repositories/strawberry/AutonomousGreenhouseChallenge_edition2/Economics.pdf)
- 每个队伍目录下的：
  - `GreenhouseClimate.csv`
  - `GrodanSens.csv`
  - `CropParameters.csv`
  - `Resources.csv`
  - `Production.csv`
  - `LabAnalysis.csv`
  - `TomQuality.csv`
- 公共天气目录：
  - `Weather/Weather.csv`


## 4. 各数据表说明

### 4.1 `Weather/Weather.csv`

公共室外天气表，所有隔间共用。

主要字段：

- `Tout`: 室外温度
- `Rhout`: 室外相对湿度
- `AbsHumOut`: 室外绝对湿度
- `Iglob`: 全球辐射
- `PARout`: 室外 PAR
- `Pyrgeo`: 长波热辐射相关量
- `RadSum`: 辐射累计
- `Rain`: 是否下雨
- `Windsp`: 风速
- `Winddir`: 风向

建模角色：

- 主要属于 `w_future`


### 4.2 `GreenhouseClimate.csv`

最核心主表，包含：

- 室内状态
- 执行器状态
- 控制设定值
- realized setpoints
- 灌溉与部分排液相关量


### 4.3 `Resources.csv`

日尺度资源消耗表。

主要字段：

- `Heat_cons`
- `ElecHigh`
- `ElecLow`
- `CO2_cons`
- `Irr`
- `Drain`

建模角色：

- 不建议进入第一版短时预测模型输入
- 适合作为控制性能评估和经济指标


### 4.4 `Production.csv`

收获时刻记录的产量表。

主要字段：

- `ProdA`
- `ProdB`
- `avg_nr_harvested_trusses`
- `Truss development time`
- `Nr_fruits_ClassA`
- `Weight_fruits_ClassA`
- `Nr_fruits_ClassB`
- `Weight_fruits_ClassB`

建模角色：

- 更适合长期产量分析
- 不适合作为第一版 5 分钟级 climate MPC 主目标


### 4.5 `CropParameters.csv`

周尺度作物结构参数。

主要字段：

- `Stem_elong`
- `Stem_thick`
- `Cum_trusses`
- `stem_dens`
- `plant_dens`

建模角色：

- 后续扩展变量
- 当前第一版可暂不纳入主预测器


### 4.6 `TomQuality.csv`

双周尺度番茄品质数据。

主要字段：

- `Flavour`
- `TSS`
- `Acid`
- `%Juice`
- `Bite`
- `Weight`
- `DMC_fruit`

建模角色：

- 适合后续做长期结果层分析
- 当前不进入主时序预测控制模型


### 4.7 `LabAnalysis.csv`

灌溉液和排液的离子分析，双周尺度。

字段分两类：

- `irr_*`
- `drain_*`

建模角色：

- 更适合营养液与根区研究
- 当前主线先不纳入


### 4.8 `GrodanSens.csv`

根区 / 基质传感器数据，5 分钟分辨率。

主要字段：

- `EC_slab1`, `EC_slab2`
- `WC_slab1`, `WC_slab2`
- `t_slab1`, `t_slab2`

建模角色：

- 后续灌溉控制和根区状态建模可用
- 当前第一版作为可选扩展


## 5. `GreenhouseClimate.csv` 字段分层解释

### 5.1 室内状态变量

这些变量最适合用于：

- 历史输入 `x_past`
- 未来预测目标 `y_future`

字段：

- `Tair`: 室内空气温度
- `Rhair`: 室内相对湿度
- `CO2air`: 室内 CO2 浓度
- `HumDef`: 室内湿度亏缺
- `Tot_PAR`: 室内总 PAR
- `Tot_PAR_Lamps`: 来自灯的 PAR

辅助状态：

- `EC_drain_PC`
- `pH_drain_PC`


### 5.2 执行器实际状态

这些变量反映 greenhouse computer 最终执行出来的设备状态。

字段：

- `VentLee`: 背风侧开窗百分比
- `Ventwind`: 迎风侧开窗百分比
- `AssimLight`: HPS 灯开关状态
- `EnScr`: energy curtain 开度
- `BlackScr`: blackout curtain 开度
- `PipeLow`: 下层加热管温度
- `PipeGrow`: 生长回路加热管温度
- `co2_dos`: 实际 CO2 dosing
- `Water_sup`: 当天累计灌溉分钟数
- `Cum_irr`: 当天累计灌溉量

建模意义：

- 可以作为过去控制执行的反馈量放进 `x_past`
- 不建议直接作为未来规划输入


### 5.3 请求设定值 `*_sp`

`_sp` 表示算法或上位系统写入 process computer 的控制请求设定值。

字段：

- `co2_sp`
- `dx_sp`
- `t_rail_min_sp`
- `t_grow_min_sp`
- `Assim_sp`
- `scr_enrg_sp`
- `scr_blck_sp`
- `t_heat_sp`
- `t_vent_sp`
- `window_pos_lee_sp`
- `water_sup_int_sp_min`
- `int_blue_sp`
- `int_red_sp`
- `int_farred_sp`
- `int_white_sp`

建模意义：

- 这是最自然的未来控制输入 `u_future`


### 5.4 realized setpoints `*_vip`

`_vip` 表示 process computer 内部实际落实后的控制参考值，即 realized setpoints。

字段：

- `co2_vip`
- `dx_vip`
- `t_rail_min_vip`
- `t_grow_min_vip`
- `Assim_vip`
- `scr_enrg_vip`
- `scr_blck_vip`
- `t_heat_vip`
- `t_ventlee_vip`
- `t_ventwind_vip`
- `window_pos_lee_vip`
- `water_sup_int_vip_min`
- `int_blue_vip`
- `int_red_vip`
- `int_farred_vip`
- `int_white_vip`

建模意义：

- 这些值非常有研究价值
- 但第一版不建议直接作为 `u_future`
- 更适合做：
  - `sp -> vip -> actuator -> climate` 的层级分析
  - system identification
  - 控制系统中间层建模


## 6. 建模接口设计建议

为了与当前 `diffmpc` 框架兼容，建议统一成：

- `x_past`: 历史已观测输入
- `w_future`: 未来外生变量
- `u_future`: 未来控制输入
- `y_future`: 未来预测目标


## 7. 第一版推荐字段划分

### 7.1 `x_past`

推荐纳入：

- `Tair`
- `Rhair`
- `CO2air`
- `HumDef`
- `Tot_PAR`
- `Tot_PAR_Lamps`
- `VentLee`
- `Ventwind`
- `PipeLow`
- `PipeGrow`
- `AssimLight`
- `EnScr`
- `BlackScr`
- `co2_dos`
- `Cum_irr`

可选：

- `EC_drain_PC`
- `pH_drain_PC`


### 7.2 `w_future`

推荐纳入：

- `Tout`
- `Rhout`
- `AbsHumOut`
- `Iglob`
- `PARout`
- `Rain`
- `Windsp`
- `Winddir`


### 7.3 `u_future`

第一版建议使用 `*_sp`，不要直接使用 `*_vip`。

优先纳入：

- `t_heat_sp`
- `t_vent_sp`
- `co2_sp`
- `dx_sp`
- `Assim_sp`
- `scr_enrg_sp`
- `scr_blck_sp`
- `water_sup_int_sp_min`

第二层扩展：

- `t_rail_min_sp`
- `t_grow_min_sp`
- `window_pos_lee_sp`
- `int_blue_sp`
- `int_red_sp`
- `int_farred_sp`
- `int_white_sp`


### 7.4 `y_future`

第一版建议预测：

- `Tair`
- `Rhair`
- `CO2air`
- `Tot_PAR`

可选替换方案：

- 用 `HumDef` 替代 `Rhair`

如果后续发现湿度控制更适合 deficit 语义，那么可以改成：

- `Tair`
- `HumDef`
- `CO2air`
- `Tot_PAR`


## 8. 第一版不建议直接纳入主模型的字段

以下字段暂时建议不进入第一版时序预测模型：

- `*_vip`
- `Resources.csv` 的日尺度汇总
- `Production.csv`
- `TomQuality.csv`
- `LabAnalysis.csv`
- `CropParameters.csv`

原因：

- 时间分辨率不一致
- 直接并入会显著增加对齐复杂度
- 第一版主任务应先聚焦短时气候预测与控制


## 9. 第二版可扩展方向

### 9.1 控制结构建模

研究：

- `sp -> vip`
- `vip -> actuator state`
- `actuator -> indoor climate`

这比把所有列混在一起训练更接近真实 greenhouse control architecture。


### 9.2 根区和灌溉扩展

可加入：

- `GrodanSens.csv`
- `LabAnalysis.csv`

目标可以扩展到：

- 灌溉控制
- 根区状态预测
- 排液 EC / pH 管理


### 9.3 经济与长期指标扩展

可加入：

- `Resources.csv`
- `Production.csv`
- `TomQuality.csv`
- `Economics.pdf` 中的价格信息

目标可扩展为：

- economic MPC
- profit-aware control
- quality-aware control


## 10. 针对当前论文主线的建议

如果当前目标是尽快把 AGC 接入 `diffmpc`，建议按以下顺序推进：

1. 只用 `Weather.csv + GreenhouseClimate.csv`
2. 先做单队伍 `Reference`
3. 再扩展到多队伍联合训练
4. 再决定是否引入 `*_vip`
5. 最后再引入资源、产量和品质


## 11. 当前最推荐的第一版实验配置

### 主数据表

- `Weather/Weather.csv`
- `Reference/GreenhouseClimate.csv`

### 输入

- 历史室内状态和执行器反馈
- 未来天气
- 未来 setpoints

### 输出

- `Tair`
- `Rhair` 或 `HumDef`
- `CO2air`
- `Tot_PAR`

### 控制层

- 使用 `*_sp` 作为未来候选控制量
- 将 MPC/DPC 设计在 setpoint 层，而不是直接在 actuator 状态层


## 12. 一句话总结

AGC 2019 的关键优势不只是变量多，而是它天然包含了：

- 外生天气
- 室内状态
- 请求 setpoints
- realized setpoints
- 执行器状态
- 资源消耗
- 产量和品质

因此它非常适合做：

- 面向控制的多步预测
- 严格闭环的 greenhouse MPC / DPC
- 后续的 economic / robust / hierarchical control 扩展

