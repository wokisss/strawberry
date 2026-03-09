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
from models.transformer_hybrid import TransformerHybridModel
from models.decision_model import DecisionControlModel
from controllers.dpc_controller import DPCController
from controllers.sac_controller import SACController
from controllers.sac.sac_agent import SAC
from controllers.sac.replay_buffer import ReplayBuffer
from environment.gym_wrapper import GreenhouseGymEnv
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
    print("--- DPC (可微规划) vs SAC (柔性演员评判家) ---")
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
    print("\n---> 初始化 Transformer 混合预测模型...")
    model = TransformerHybridModel(
        input_dim=len(processor.feature_order),
        future_dim=len(processor.future_indices),
        target_dim=3,
        forecast_horizon=cfg.horizon,
        d_model=cfg.transformer_d_model,
        nhead=cfg.transformer_nhead,
        num_layers=cfg.transformer_num_layers,
        dim_feedforward=cfg.transformer_dim_feedforward,
        dropout=cfg.transformer_dropout
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

    # 5. ======================== SAC 离线训练 ========================
    print("\n---> 初始化 SAC 强化学习环境与代理...")
    print(f"    训练步数={cfg.sac_train_steps}, Batch={cfg.sac_batch_size}, LR={cfg.sac_lr}")
    
    gym_env = GreenhouseGymEnv( PhysicsGreenhouseEnv(np.zeros(3), config=cfg), config=cfg, 
                                scaler=processor.scaler, target_indices=processor.target_indices )
    
    sac_agent = SAC(gym_env.observation_space.shape[0], gym_env.action_space, cfg, device)
    memory = ReplayBuffer(cfg.sac_replay_size, cfg.seed)
    
    # 训练循环
    print("    [SAC] 开始离线训练...")
    updates = 0
    episodes = 0
    total_numsteps = 0
    
    while total_numsteps < cfg.sac_train_steps:
        episode_reward = 0
        episode_steps = 0
        done = False
        state, _ = gym_env.reset(seed=cfg.seed + episodes)
        
        while not done:
            if cfg.sac_train_steps > total_numsteps:
                action = sac_agent.select_action(state)
            else:
                break
                
            next_state, reward, terminated, truncated, _ = gym_env.step(action)
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
            done = terminated or truncated
            
            mask = 1 if episode_steps == gym_env.max_steps else float(not done)
            memory.push(state, action, reward, next_state, mask)
            state = next_state
            
            # 满足批次大小开始更新参数
            if len(memory) > cfg.sac_batch_size:
                sac_agent.update_parameters(memory, cfg.sac_batch_size, updates)
                updates += 1
        
        episodes += 1
        if episodes % 20 == 0 or total_numsteps >= cfg.sac_train_steps:
            print(f"        Episode: {episodes}, 交互总步数: {total_numsteps}, Ep. Reward: {episode_reward:.2f}")

    print(f"---> 离线训练完成，初始化 SAC 控制器...")
    sac_controller = SACController(
        sac_agent, processor.scaler, processor.target_indices,
        processor.future_indices, feature_order=processor.feature_order, config=cfg
    )

    # 6. PWM 驱动器 (DPC 和 SAC 各自独立的 PWM 实例)
    print(f"\n---> 初始化 PWM 驱动器...")
    print(f"    周期={cfg.pwm_cycle}min, 最小吸合={cfg.pwm_min_on}min, 最小关断={cfg.pwm_min_off}min")

    pwm_driver_dpc = PWMDriver(
        cycle_minutes=cfg.pwm_cycle,
        min_on_minutes=cfg.pwm_min_on,
        min_off_minutes=cfg.pwm_min_off
    )
    pwm_sim_dpc = PWMSimulator(pwm_driver_dpc)

    pwm_driver_sac = PWMDriver(
        cycle_minutes=cfg.pwm_cycle,
        min_on_minutes=cfg.pwm_min_on,
        min_off_minutes=cfg.pwm_min_off
    )
    pwm_sim_sac = PWMSimulator(pwm_driver_sac)

    # 7. 仿真 (DPC vs SAC)
    print("\n---> 开始对比仿真 (DPC vs SAC + PWM)...")
    
    # [硬件加速] 使用 PyTorch 2.x 编译优化决策层，提升前向传播速度
    # 注意: Windows 原生暂不完全支持 Triton 后端，因此在 Windows 下跳过 torch.compile
    if hasattr(torch, 'compile') and sys.platform != 'win32':
        print("    [硬件加速] 正在编译模型计算图 (torch.compile)...")
        try:
            decision_model = torch.compile(decision_model)
            # 将编译后的模型重新挂载回控制器 （SAC由于无模型，不用挂载决策模型）
            dpc.model = decision_model
        except Exception as e:
            print(f"    [警告] torch.compile 失败，回退到急切模式: {e}")
    else:
        print("    [硬件加速] 当前平台为 Windows 或不支持 compile，跳过计算图编译。")

    sim = Simulator(
        dpc, sac_controller, PhysicsGreenhouseEnv,
        processor.feature_order, processor.scaler, processor.target_indices,
        config=cfg, device=device,
        pwm_sim_dpc=pwm_sim_dpc, pwm_sim_sac=pwm_sim_sac
    )
    result = sim.run(datasets['X_test_p'], datasets['X_test_f'], config=cfg)

    # 8. 可视化
    print("\n---> 绘制结果...")
    viz = Visualizer(config=cfg)
    viz.plot_comparison(result)

    print("\nDONE")


if __name__ == "__main__":
    main()
