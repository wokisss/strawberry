# -*- coding: utf-8 -*-
"""
诊断脚本：观察 DPC 在单步优化中 action_param 的变化轨迹
对比 Adam vs SGD，观察收敛行为差异
"""

import warnings
import numpy as np
import torch
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

from config import Config
from data_processing.processor import DataProcessor
from models.segmented_hybrid import SegmentedHybridModel
from models.decision_model import DecisionControlModel

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_optimization(model, past_tensor, future_base, target_temp_norm, 
                     optimizer_type='adam', lr=0.2, iterations=100, 
                     w_track=20.0, w_energy=0.001, w_smooth=0.1,
                     init_action=None, prev_action=None, horizon=10):
    """运行单步优化并记录轨迹"""
    device = next(model.parameters()).device
    
    if init_action is None:
        init_action = [0.85, 0.0]
    if prev_action is None:
        prev_action = [0.9, 0.0]
    
    action_param = torch.tensor([[[init_action[0], init_action[1]]]],
                                device=device, requires_grad=True)
    prev_u = torch.tensor(prev_action, device=device).view(1, 1, 2)
    
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam([action_param], lr=lr)
    else:
        optimizer = torch.optim.SGD([action_param], lr=lr)
    
    for p in model.parameters():
        p.requires_grad = False
    model.train()
    
    trajectory = []
    losses = []
    
    for i in range(iterations):
        optimizer.zero_grad()
        
        u_soft = torch.clamp(action_param, min=0.0, max=1.0)
        u_expanded = u_soft.repeat(1, horizon, 1)
        f_weather = future_base[:, :, 2:]
        u_heater = u_expanded[:, :, 0:1]
        u_vent = u_expanded[:, :, 1:2]
        x_future_optim = torch.cat([u_heater, u_vent, f_weather], dim=2)
        
        pred_norm = model(past_tensor, x_future_optim, target_temp_norm=target_temp_norm)
        
        track_loss = torch.mean((pred_norm - target_temp_norm) ** 2)
        energy_loss = torch.mean(torch.abs(u_soft))
        smooth_loss = torch.mean((u_soft - prev_u.detach()) ** 2)
        
        total_loss = w_track * track_loss + w_energy * energy_loss + w_smooth * smooth_loss
        
        total_loss.backward()
        optimizer.step()
        
        h_val = u_soft[0, 0, 0].item()
        v_val = u_soft[0, 0, 1].item()
        raw_h = action_param[0, 0, 0].item()
        trajectory.append((h_val, v_val, raw_h))
        losses.append(total_loss.item())
    
    return trajectory, losses


def main():
    cfg = Config()
    setup_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    processor = DataProcessor(cfg)
    df = processor.load_and_preprocess()
    df = processor.merge_weather(df)
    df = processor.add_time_encoding(df)
    data_scaled = processor.prepare_features(df)
    datasets = processor.prepare_datasets(data_scaled)
    
    model = SegmentedHybridModel(
        input_dim=len(processor.feature_order),
        future_dim=len(processor.future_indices),
        forecast_horizon=cfg.horizon,
        hidden_dim=cfg.hidden_dim
    ).to(device)
    
    from training.trainer import Trainer
    trainer = Trainer(model, config=cfg, device=device)
    trainer.train(datasets['X_train_p'], datasets['X_train_f'], datasets['y_train'])
    
    decision_model = DecisionControlModel(model, config=cfg)
    
    # 目标温度归一化
    dummy = np.zeros((1, len(processor.scaler.scale_)))
    dummy[0, processor.target_idx] = cfg.target_temp
    target_temp_norm = processor.scaler.transform(dummy)[0, processor.target_idx]
    
    # 取第 150 步的数据 (温度接近目标时的典型场景)
    start = cfg.sim_start_idx + 150
    past_tensor = torch.FloatTensor(datasets['X_test_p'][start:start+1]).to(device)
    future_base = torch.FloatTensor(datasets['X_test_f'][start:start+1]).to(device)
    
    print("=" * 60)
    print("诊断: DPC 单步优化轨迹分析")
    print(f"初始动作: [0.85, 0.0], 上一步动作: [0.9, 0.0]")
    print("=" * 60)
    
    # 测试 1: Adam lr=0.2
    setup_seed(cfg.seed)
    traj_adam, loss_adam = run_optimization(
        decision_model, past_tensor, future_base, target_temp_norm,
        optimizer_type='adam', lr=0.2
    )
    
    # 测试 2: SGD lr=0.05
    setup_seed(cfg.seed)
    traj_sgd, loss_sgd = run_optimization(
        decision_model, past_tensor, future_base, target_temp_norm,
        optimizer_type='sgd', lr=0.05
    )
    
    # 测试 3: Adam lr=0.05
    setup_seed(cfg.seed)
    traj_adam_low, loss_adam_low = run_optimization(
        decision_model, past_tensor, future_base, target_temp_norm,
        optimizer_type='adam', lr=0.05
    )
    
    print("\n--- Adam lr=0.2 (当前设置) ---")
    print(f"  初始: heater={traj_adam[0][0]:.4f}, raw={traj_adam[0][2]:.4f}")
    print(f"  第10步: heater={traj_adam[9][0]:.4f}, raw={traj_adam[9][2]:.4f}")
    print(f"  第50步: heater={traj_adam[49][0]:.4f}, raw={traj_adam[49][2]:.4f}")
    print(f"  最终: heater={traj_adam[-1][0]:.4f}, raw={traj_adam[-1][2]:.4f}")
    print(f"  Loss 范围: {min(loss_adam):.6f} ~ {max(loss_adam):.6f}")
    print(f"  最优 Loss 在第 {loss_adam.index(min(loss_adam))} 步")
    # 检测振荡
    heaters = [t[0] for t in traj_adam]
    oscillation = sum(abs(heaters[i] - heaters[i-1]) for i in range(1, len(heaters)))
    print(f"  累计振荡幅度: {oscillation:.4f}")
    
    print("\n--- SGD lr=0.05 ---")
    print(f"  初始: heater={traj_sgd[0][0]:.4f}, raw={traj_sgd[0][2]:.4f}")
    print(f"  第10步: heater={traj_sgd[9][0]:.4f}, raw={traj_sgd[9][2]:.4f}")
    print(f"  第50步: heater={traj_sgd[49][0]:.4f}, raw={traj_sgd[49][2]:.4f}")
    print(f"  最终: heater={traj_sgd[-1][0]:.4f}, raw={traj_sgd[-1][2]:.4f}")
    print(f"  Loss 范围: {min(loss_sgd):.6f} ~ {max(loss_sgd):.6f}")
    print(f"  最优 Loss 在第 {loss_sgd.index(min(loss_sgd))} 步")
    heaters_sgd = [t[0] for t in traj_sgd]
    oscillation_sgd = sum(abs(heaters_sgd[i] - heaters_sgd[i-1]) for i in range(1, len(heaters_sgd)))
    print(f"  累计振荡幅度: {oscillation_sgd:.4f}")
    
    print("\n--- Adam lr=0.05 ---")
    print(f"  初始: heater={traj_adam_low[0][0]:.4f}, raw={traj_adam_low[0][2]:.4f}")
    print(f"  第10步: heater={traj_adam_low[9][0]:.4f}, raw={traj_adam_low[9][2]:.4f}")
    print(f"  第50步: heater={traj_adam_low[49][0]:.4f}, raw={traj_adam_low[49][2]:.4f}")
    print(f"  最终: heater={traj_adam_low[-1][0]:.4f}, raw={traj_adam_low[-1][2]:.4f}")
    print(f"  Loss 范围: {min(loss_adam_low):.6f} ~ {max(loss_adam_low):.6f}")
    print(f"  最优 Loss 在第 {loss_adam_low.index(min(loss_adam_low))} 步")
    heaters_al = [t[0] for t in traj_adam_low]
    oscillation_al = sum(abs(heaters_al[i] - heaters_al[i-1]) for i in range(1, len(heaters_al)))
    print(f"  累计振荡幅度: {oscillation_al:.4f}")
    
    # 关键对比
    print("\n" + "=" * 60)
    print("关键对比:")
    print(f"  Adam lr=0.2:  最优Loss={min(loss_adam):.6f}, 最终heater={traj_adam[-1][0]:.4f}, 振荡={oscillation:.2f}")
    print(f"  SGD  lr=0.05: 最优Loss={min(loss_sgd):.6f}, 最终heater={traj_sgd[-1][0]:.4f}, 振荡={oscillation_sgd:.2f}")
    print(f"  Adam lr=0.05: 最优Loss={min(loss_adam_low):.6f}, 最终heater={traj_adam_low[-1][0]:.4f}, 振荡={oscillation_al:.2f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
