# DiffMPC: Differentiable Predictive Control for Smart Greenhouse

DiffMPC 是一个基于**可微预测控制 (DPC, Differentiable Predictive Control)** 的智能温室多变量联合控制系统仿真框架。本项目旨在解决高度复杂、非线性和强耦合的农业温室物理环境控制问题（如温度、湿度、二氧化碳浓度的联合控制优化）。

本项目展示了 DPC 在控制精度、多目标权衡能力、以及计算效率上对传统无梯度启发式优化算法（如 **PSO 粒子群优化**）的颠覆性优势，并提供了完整的 `Sim-to-Real` 仿真闭环验证。

---

## 🌟 核心亮点

1. **突破物理黑盒束缚 (物理引导的梯度寻优, PGG)**
   传统 MPC 面对物理环境模拟器时，往往因其不可导而退化为黑盒搜索（如遗传算法、粒子群算法）。本项目创新性地将温室的核心热力学和生物反应方程（光合作用、蒸腾作用）嵌入为一个 **“多变量物理引导梯度层 (PGG, Physics-Guided Gradients)”**。这使得 AI 可以直接通过 PyTorch 的反向传播（Backpropagation）机制，“一眼看穿”物理执行器干涉状态变量的导数方向，从而直接一步到位计算出最优解。

2. **高维解耦的多变量联合控制 (Temp, Hum, CO2)**
   系统采用真实温室数据集执行器 `['Heater', 'Ventilation', 'Fog', 'Lighting']`，深度考虑了执行器之间的**物理互斥与多重副作用**：
   - 开启**通风 (Ventilation)** 能快速降低 CO2 和湿度，但会造成温室剧烈失温；
   - 开启**起雾机 (Fog)** 能增加湿度，但附带强烈的蒸发冷却（降温）抑制效应；
   - 开启**补光灯 (Lighting)** 除了提供热量，其诱发的作物**光合作用**能作为消耗室内过剩 CO2 的绝佳手段。
   DPC 控制器通过全局梯度能够自主学会在开启大通风排废气的同时，同步开大加热器和补光灯进行精确的温度对冲补偿，展现出极其高级的“联合妥协与博弈”策略。

3. **基于 Transformer 的全局前瞻预测大脑**
   采用最新的大视界 Transformer Encoder-Decoder 架构彻底替换了原有的短视界 RNN/MLP 结构。利用 Self-Attention 和 Cross-Attention 机制，模型能够实现对大惯性环境极长延时（数十步开外）的非线性物理反馈进行精准的无衰减认知，极大提升了梯度回传的纵深与稳定性。

4. **100% 纯 PyTorch / GPU 硬件级极限加速**
   彻底重构底层物理沙盒和粒子群运算。所有算法模块（包括物理仿真反馈环境本身）全部实现在 PyTorch Tensor 计算图下进行。全程无 CPU-GPU 之间的数据 Copy 损耗，极大地压榨了 GPU（如 RTX 5070 Ti）的算力峰值。

---

## 🏗️ 项目架构

```text
c:\repositories\strawberry\diffmpc
│
├── main.py                     # 全局启动入口：加载数据、初始化环境与模型、并发执行滚动对比仿真
├── config.py                   # 系统的中央配置中心 (包含自动控制目标、惩罚权重、物理方程增益设定)
├── requirements.txt            # Python 依赖包列表
│
├── data_processing/            # 📊 数据工程流水线
│   └── processor.py            # 数据清洗、时序特征工程 (如 Hour_Sin/Cos)、MinMax 特征归一化及截断补齐
│
├── environment/                # 🌍 物理沙盒层 (纯 PyTorch Tensor 加速引擎)
│   └── physics_env.py          # 核心离线温室环境物理算子：建模气流热力学变化、湿度蒸发冷却与光合作用降 CO2 动力学
│
├── models/                     # 🧠 混合动力学模型与物理引导抽象层
│   ├── transformer_hybrid.py   # 基于 Transformer 时序 Encoder-Decoder 结合多物理领域专家 (MoE) 的全局前瞻状态预测大脑
│   └── decision_model.py       # PGG (Physics-Guided Gradients) 物理知识嵌入层：构建动作张量向下游状态投影的非线性可导计算图
│
├── controllers/                # 🎮 决策优化与工业网关层
│   ├── dpc_controller.py       # 🔥 (核心基石) DPC 可微控制器：基于 PyTorch Autograd + Adam/SGD 的滚动梯度寻优求解器
│   ├── pso_controller.py       # (竞品基准) 粒子群控制器：由 PyTorch 矩阵并性运算全速加压加速的无梯度瞎搜索对照组
│   ├── pwm_driver.py           # (工业转换器) PWM 转换网关：将 AI 数学期望输出的 (0-1) 模拟量离散转换为工业物理继电器的启/停数字脉冲开关组合
│   └── mdp_controller.py       # (历史遗留) 早期版本的离散动作空间 MDP 马尔可夫决策控制器验证文件
│
├── simulation/                 # 🚀 在线仿真推演核心引擎
│   ├── simulator.py            # Rolling Horizon (滚动优化) 调度主干。维护 Online 环境步进，与 DPC 预测进行环境偏差剥离运算
│   └── visualizer.py           # Multi-plots 特化可视化面板：挂载生成温度/湿度/CO2 贴合度、MAE指标卡以及 4D 高清执行器动作分步推演 5 联图
│
├── data/                       #  本地原始源数据池
│   └── (原始 Strawberry 环境时序 CSV、历史气象数据集集市)
│
└── results/                    # 🎯 分析输出归档库 (按演化阶段分层管理)
    ├── 01_early_exploration/   # 1月初期探索：MPC 仿真 & baseline 修复迭代
    │   ├── mpc_simulation_jan14/
    │   ├── mpc_simulation_jan20_21/
    │   └── baseline_fix_jan14/
    ├── 02_mpc_vs_mdp/          # MPC 与 MDP (马尔可夫决策控制) 对比
    ├── 03_mpc_baseline_optimized/ # 1月28日-2月 MPC 优化版本高频迭代存档
    ├── 04_mpc_pwm_refactored/  # PWM 离散化网关 & 代码重构版本
    ├── 05_dpc_vs_pso/          # DPC 可微控制器 vs PSO 粒子群 对比 (多变量升级)
    ├── 06_dpc_vs_sac/          # 🔥 DPC vs SAC 强化学习对比 (当前主线, 代码自动输出至此)
    ├── 07_predictor_diagnostic/ # 🔍 Transformer 预测大脑诊断图 (代码自动输出至此)
    └── 08_misc_figures/        # Figure_X、training_curve、diagnostic_output 等杂项
```

---

## 🚀 快速开始

### 环境依赖
- Python 3.10+
- PyTorch (支持 CUDA 12.x+)
- Numpy, Pandas, Scikit-learn
- Matplotlib

### 运行仿真

```bash
# 激活 Conda 环境
conda activate strawberry_env

# 一键启动完整的多变量模型训练与双边控制器 (DPC vs PSO) 真实滚动对比仿真
python main.py
```

执行后，终端将实时打印长达数百分钟模拟的在线仿真对战状态：
```text
...> 正在进行 300 步滚动仿真 (DPC vs PSO + PWM离散化)...
    步 50/300 | DPC_T=22.9°C (avg 350ms) | PSO_T=22.0°C (avg 99ms)
...
```

### 查看结果
执行完成后，仿真数据将会被送到 `visualizer.py` 生成一幅高清分析大图，并保存在 `results/` 目录下（如 `dpc_vs_pso_YYYYMMDD_HHMMSS.png`）。

该图像包含 **5 张子分析图**：
- **子图 1-3**：展示室内温度、湿度、CO2 的目标贴合情况（含 MAE 指标对标）。
- **子图 4-5**：详细披露了算法在不可见底层中对 `Heater`, `Ventilation`, `Fog`, `Lighting` 四大组件的高频微操执行线度。

---

## 📈 典型仿真结果展示与解析

在极速光照衰退并在湿度与 CO2 即将冲高失控的恶劣剧变环境中测试：

| 指标 (控制目标) | DPC (可微规划) | PSO (粒子群算法) | 优势提升 |
| :--- | :--- | :--- | :--- |
| **温度 MAE** (25°C) | **1.93 °C** | 2.51 °C | ✨ **+23%** |
| **CO2 MAE** (800ppm) | **159.1 ppm** | 249.2 ppm | ✨ **+36%** |
| **平均耗时** | 352 ms/step | **99 ms/step** | (计算代价高昂) |

**行为洞察：**
- 当遭遇 CO2 飙涨时危机时，由于缺乏梯度方向的“上帝视角”，粒子群算法 (PSO) 呈现剧烈的随机拉锯，只能在保住局部温度和放任废气飙升两难中陷入**局部最优**泥潭。
- 相反，**DPC 控制器利用 PGG 物理层导数，立刻洞悉破局之道**：它果断大规模开启通风口散除废气，同时计算出精确的热量流失散量，**并发火力全开 Heater 和 Lighting 热红外效应，实现了极为精准的温度“逆向拦截与对冲”！** 展现出无可比拟的全局掌控智力。
