# -*- coding: utf-8 -*-
"""
main.py
---------
温室 MPC 控制系统入口

重构后的简洁入口文件，按顺序调用各模块完成:
    数据处理 → 模型训练 → 仿真对比 (DPC vs PSO + PWM) → 结果可视化
"""

import warnings
import numpy as np
import torch
import sys

from config import Config
from data_processing.processor import DataProcessor
from models.segmented_hybrid import SegmentedHybridModel
from models.decision_model import DecisionControlModel
from controllers.dpc_controller import DPCController
from controllers.pso_controller import PSOController
from controllers.pwm_driver import PWMDriver, PWMSimulator
from environment.physics_env import PhysicsGreenhouseEnv
from training.trainer import Trainer
from simulation.simulator import Simulator
from simulation.visualizer import Visualizer

# 忽略 sklearn UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')


def setup_seed(seed):
    """固定随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    print("=" * 60)
    print("--- 智能温室控制系统 ---")
    print("--- DPC (可微规划) vs PSO (粒子群优化) ---")
    print("=" * 60)

    # 0. 配置
    cfg = Config()
    setup_seed(cfg.seed)
    print(f"---> 随机种子已固定: {cfg.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"---> 使用计算设备: {device}")

    # 1. 数据处理
    processor = DataProcessor(cfg)
    df = processor.load_and_preprocess()
    df = processor.merge_weather(df)
    df = processor.add_time_encoding(df)
    data_scaled = processor.prepare_features(df)
    datasets = processor.prepare_datasets(data_scaled)

    # 2. 模型初始化 + 训练
    print("\n---> 初始化混合预测模型...")
    model = SegmentedHybridModel(
        input_dim=len(processor.feature_order),
        future_dim=len(processor.future_indices),
        target_dim=3,
        forecast_horizon=cfg.horizon,
        hidden_dim=cfg.hidden_dim
    ).to(device)

    trainer = Trainer(model, config=cfg, device=device)
    trainer.train(datasets['X_train_p'], datasets['X_train_f'], datasets['y_train'])

    # 3. 决策模型 (共享同一个 DecisionControlModel)
    print("\n---> 初始化决策模型...")
    decision_model = DecisionControlModel(model, config=cfg)
    print(f"    多变量物理引导梯度层已挂载")

    # 4. 控制器初始化
    print("\n---> 初始化 DPC 控制器 (梯度优化)...")
    print(f"    lr={cfg.dpc_lr}, iterations={cfg.dpc_iterations}")
    dpc = DPCController(
        decision_model, processor.scaler, processor.target_indices,
        processor.future_indices, config=cfg
    )

    print(f"---> 初始化 PSO 控制器 (粒子群优化)...")
    print(f"    粒子数={cfg.pso_n_particles}, 代数={cfg.pso_n_generations}, "
          f"惯性={cfg.pso_w_inertia}, c1={cfg.pso_c1}, c2={cfg.pso_c2}")
    pso = PSOController(
        decision_model, processor.scaler, processor.target_indices,
        processor.future_indices, config=cfg
    )

    # 5. PWM 驱动器 (DPC 和 PSO 各自独立的 PWM 实例)
    print(f"\n---> 初始化 PWM 驱动器...")
    print(f"    周期={cfg.pwm_cycle}min, 最小吸合={cfg.pwm_min_on}min, 最小关断={cfg.pwm_min_off}min")

    pwm_driver_dpc = PWMDriver(
        cycle_minutes=cfg.pwm_cycle,
        min_on_minutes=cfg.pwm_min_on,
        min_off_minutes=cfg.pwm_min_off
    )
    pwm_sim_dpc = PWMSimulator(pwm_driver_dpc)

    pwm_driver_pso = PWMDriver(
        cycle_minutes=cfg.pwm_cycle,
        min_on_minutes=cfg.pwm_min_on,
        min_off_minutes=cfg.pwm_min_off
    )
    pwm_sim_pso = PWMSimulator(pwm_driver_pso)

    # 6. 仿真 (DPC vs PSO)
    print("\n---> 开始对比仿真 (DPC vs PSO + PWM)...")
    
    # [硬件加速] 使用 PyTorch 2.x 编译优化决策层，提升前向传播速度
    # 注意: Windows 原生暂不完全支持 Triton 后端，因此在 Windows 下跳过 torch.compile
    if hasattr(torch, 'compile') and sys.platform != 'win32':
        print("    [硬件加速] 正在编译模型计算图 (torch.compile)...")
        try:
            decision_model = torch.compile(decision_model)
            # 将编译后的模型重新挂载回控制器
            dpc.model = decision_model
            pso.model = decision_model
        except Exception as e:
            print(f"    [警告] torch.compile 失败，回退到急切模式: {e}")
    else:
        print("    [硬件加速] 当前平台为 Windows 或不支持 compile，跳过计算图编译。")

    sim = Simulator(
        dpc, pso, PhysicsGreenhouseEnv,
        processor.feature_order, processor.scaler, processor.target_indices,
        config=cfg, device=device,
        pwm_sim_dpc=pwm_sim_dpc, pwm_sim_pso=pwm_sim_pso
    )
    result = sim.run(datasets['X_test_p'], datasets['X_test_f'], config=cfg)

    # 7. 可视化
    print("\n---> 绘制结果...")
    viz = Visualizer(config=cfg)
    viz.plot_comparison(result)

    print("\nDONE")


if __name__ == "__main__":
    main()
