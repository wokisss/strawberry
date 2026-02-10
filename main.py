# -*- coding: utf-8 -*-
"""
main.py
---------
温室 MPC 控制系统入口

重构后的简洁入口文件，按顺序调用各模块完成:
    数据处理 → 模型训练 → 仿真对比 (含 PWM Sim-to-Real) → 结果可视化
"""

import warnings
import numpy as np
import torch

from config import Config
from data_processing.processor import DataProcessor
from models.segmented_hybrid import SegmentedHybridModel
from models.decision_model import DecisionControlModel
from controllers.dpc_controller import DPCController
from controllers.mdp_controller import LegacyMDPController
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
    print("--- 智能温室控制系统 (MPC vs MDP) ---")
    print("--- 模块化重构版 + Sim-to-Real PWM ---")
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
        forecast_horizon=cfg.horizon,
        hidden_dim=cfg.hidden_dim
    ).to(device)

    trainer = Trainer(model, config=cfg, device=device)
    trainer.train(datasets['X_train_p'], datasets['X_train_f'], datasets['y_train'])

    # 3. 控制器初始化
    print("\n---> 初始化控制器...")
    decision_model = DecisionControlModel(model, config=cfg)
    print(f"    物理引导梯度层 (Heater Gain={cfg.heater_gain}, Vent Gain={cfg.vent_gain})")

    dpc = DPCController(
        decision_model, processor.scaler, processor.target_idx,
        processor.future_indices, config=cfg
    )
    mdp = LegacyMDPController(
        target_temp=cfg.target_temp, vent_threshold=cfg.mdp_vent_threshold
    )

    # 4. PWM 驱动器 (Sim-to-Real)
    print(f"\n---> 初始化 PWM 驱动器...")
    print(f"    周期={cfg.pwm_cycle}min, 最小吸合={cfg.pwm_min_on}min, 最小关断={cfg.pwm_min_off}min")
    pwm_driver = PWMDriver(
        cycle_minutes=cfg.pwm_cycle,
        min_on_minutes=cfg.pwm_min_on,
        min_off_minutes=cfg.pwm_min_off
    )
    pwm_sim = PWMSimulator(pwm_driver)

    # 5. 仿真 (DPC + PWM vs MDP)
    print("\n---> 开始对比仿真 (DPC+PWM vs MDP)...")
    sim = Simulator(
        dpc, mdp, PhysicsGreenhouseEnv,
        processor.feature_order, processor.scaler, processor.target_idx,
        config=cfg, device=device, pwm_sim=pwm_sim
    )
    result = sim.run(datasets['X_test_p'], datasets['X_test_f'], config=cfg)

    # 6. 可视化
    print("\n---> 绘制结果...")
    viz = Visualizer(config=cfg)
    viz.plot_comparison(result)

    print("\nDONE")


if __name__ == "__main__":
    main()
