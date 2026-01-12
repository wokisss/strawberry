# mpc_control_system.py
# 基于 Neural ODE 混合模型的 MPC 温室控制系统
# 包含：数据处理 -> 模型训练 -> MPC/MDP 策略对比 -> 结果可视化

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import sys
import os
import warnings

# 忽略 sklearn 的 UserWarning (关于 feature names 的警告)
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# 设置绘图风格和字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--> 使用计算设备: {device}")

class PhysicsGreenhouseEnv:
    """ 基于物理常识的简化温室环境 (用于仿真 ground truth) """
    def __init__(self, initial_temp):
        self.current_temp = initial_temp
        
    def step(self, action, outside_temp):
        # 物理参数 (针对分钟级仿真调整)
        k_insulation = 0.05  # 隔热系数 (越小保温越好)
        power_heater = 0.5   # 加热功率 (°C/min)
        eff_vent = 0.1       # 通风效率
        
        # 动力学方程
        # dT = -k*(Tin - Tout) + P_heat*u_heat - k_vent*u_vent*(Tin - Tout)
        delta_loss = - k_insulation * (self.current_temp - outside_temp)
        delta_heat = power_heater * action[0] 
        # 通风不仅带走热量，还试图将室内温度拉向室外温度
        delta_vent = - eff_vent * action[1] * (self.current_temp - outside_temp)
        
        # 更新状态 + 随机噪声
        self.current_temp += delta_loss + delta_heat + delta_vent + np.random.normal(0, 0.05)
        return self.current_temp

# ==============================================================================
# 1. 核心模型定义 (保留 baseline.py 中的 SegmentedHybridModel 和 Neural ODE)
# ==============================================================================

class ODEF(nn.Module):
    """ 神经微分方程 (Neural ODE) 导数提取器 """
    def __init__(self, input_dim, hidden_dim=64):
        super(ODEF, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, t, y):
        t_vec = torch.ones_like(y[..., :1]) * t
        cat_input = torch.cat([y, t_vec], dim=-1)
        return self.net(cat_input)

class SegmentedHybridModel(nn.Module):
    """ 分段混合模型 (三头专家系统) """
    def __init__(self, input_dim, future_dim, forecast_horizon, hidden_dim=32):
        super(SegmentedHybridModel, self).__init__()
        
        # 共享特征提取器
        self.past_conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.past_bigru = nn.GRU(input_size=64, hidden_size=hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.past_attention = nn.Linear(hidden_dim * 2, 1)
        self.future_gru = nn.GRU(input_size=future_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        
        # 分段输出头
        feature_size = hidden_dim * 2 + hidden_dim
        
        # 专家 A: 加热模式
        self.fc_heat = nn.Sequential(nn.Linear(feature_size, 32), nn.ReLU(), nn.Linear(32, forecast_horizon))
        # 专家 B: 通风模式
        self.fc_vent = nn.Sequential(nn.Linear(feature_size, 32), nn.ReLU(), nn.Linear(32, forecast_horizon))
        # 专家 C: 自然模式
        self.fc_natural = nn.Sequential(nn.Linear(feature_size, 32), nn.ReLU(), nn.Linear(32, forecast_horizon))

    def forward(self, x_past, x_future):
        # 历史特征提取
        x_p = x_past.permute(0, 2, 1)
        x_p = torch.relu(self.past_conv1(x_p))
        x_p = x_p.permute(0, 2, 1)
        gru_out_p, _ = self.past_bigru(x_p)
        weights_p = torch.softmax(self.past_attention(gru_out_p), dim=1)
        attended_p = torch.sum(weights_p * gru_out_p, dim=1)
        
        # 未来特征提取
        _, h_n = self.future_gru(x_future)
        future_features = h_n[-1]
        
        # 特征融合
        combined = torch.cat([attended_p, future_features], dim=1)
        
        # 专家预测
        pred_heat = self.fc_heat(combined)
        pred_vent = self.fc_vent(combined)
        pred_natural = self.fc_natural(combined)
        
        # 门控融合 (根据 Heater 和 Ventilation 的状态)
        # 假设 x_future 的前两列是 Heater, Ventilation
        heater_signal = x_future[:, :, 0].mean(dim=1, keepdim=True)
        vent_signal = x_future[:, :, 1].mean(dim=1, keepdim=True)
        
        w_heat = heater_signal
        w_vent = vent_signal
        w_natural = torch.clamp(1.0 - w_heat - w_vent, min=0.0)
        
        final_pred = (w_heat * pred_heat) + (w_vent * pred_vent) + (w_natural * pred_natural)
        return final_pred

# ==============================================================================
# 2. 控制器定义 (MPC & MDP)
# ==============================================================================

class MPC_Controller:
    """ 【新增】模型预测控制器 (Model Predictive Controller) """
    def __init__(self, model, scaler, target_idx, future_indices, horizon=10, target_temp=25.0):
        self.model = model
        self.scaler = scaler
        self.target_idx = target_idx
        self.future_indices = future_indices # 指示 x_future 中哪些列是可以控制的
        self.horizon = horizon
        self.target_temp = target_temp
        
        # 定义动作空间: [Heater, Ventilation]
        # 0: Heater, 1: Vent
        self.actions = [
            [0, 0], # 自然
            [1, 0], # 加热
            [0, 1], # 通风
            [1, 1]  # (可选) 同时开启，通常不合理但也是一种状态
        ]

    def get_optimal_action(self, current_past_tensor, current_future_base):
        """
        滚动时域优化：
        current_past_tensor: (1, seq_len, feat_dim) 当前的历史状态
        current_future_base: (1, horizon, feat_dim) 未来的基础环境数据(如天气、时间)，控制量待填
        """
        best_cost = float('inf')
        best_action = [0, 0]
        
        # 简单的随机射击法 (Shooting Method) / 穷举法 (因为动作空间很小)
        for action in self.actions:
            # 1. 构造“假想”的未来输入 x_future
            # 我们假设未来 horizon 步都执行这个 action (控制量保持一致)
            x_future_sim = current_future_base.clone()
            
            # 填入控制量 (假设前两列是 Heater, Vent)
            # 注意：这里需要确保 scaler 归一化后的值是匹配的。
            # 如果 scaler 是 0-1 归一化，且原始 Heater 是 0/1，那归一化后也是 0/1 (如果 min=0, max=1)
            # 这里简化处理，假设归一化后 1 代表开启
            x_future_sim[:, :, 0] = action[0] 
            x_future_sim[:, :, 1] = action[1]
            
            # 2. 调用模型预测未来轨迹
            with torch.no_grad():
                # model 输出的是归一化的预测值
                pred_norm = self.model(current_past_tensor, x_future_sim)
            
            # 3. 反归一化得到真实温度
            # 这种反归一化稍微麻烦，因为 scaler 是针对所有特征的
            # 我们构造一个 dummy array
            pred_val = self._inverse_transform_target(pred_norm.item(), self.target_idx)
            
            # 4. 计算代价函数 (Cost Function)
            # Cost = (T_pred - T_target)^2 + lambda * Energy
            temp_cost = (pred_val - self.target_temp) ** 2
            # 降低能耗惩罚权重，鼓励控制器在模型预测收益微弱时仍积极尝试
            energy_cost = 0.01 * (action[0] + action[1])
            
            total_cost = temp_cost + energy_cost
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_action = action
                
        return best_action, best_cost

    def _inverse_transform_target(self, val, col_idx):
        # 辅助函数：仅反归一化目标列
        dummy = np.zeros((1, len(self.scaler.scale_)))
        dummy[0, col_idx] = val
        return self.scaler.inverse_transform(dummy)[0, col_idx]

class LegacyMDPController:
    """ 【保留】旧的 MDP 控制器 (仅用于生成对比基准) """
    def __init__(self):
        # 简化的查表逻辑，模拟原 MDP 的行为
        # 状态: Rational(20-25), Cold(<20), Hot(>25)
        # 动作: 0(Wait), 1(Act) -> 这里映射回 [Heater, Vent]
        pass
        
    def get_action(self, current_temp):
        # 简单的基于规则的策略 (模拟 MDP 收敛后的结果)
        # 目标: 维持 22-25 度
        if current_temp < 22.0:
            return [1, 0] # 太冷 -> 开加热
        elif current_temp > 28.0:
            return [0, 1] # 太热 -> 开通风
        else:
            return [0, 0] # 适宜 -> 待机

# ==============================================================================
# 3. 数据处理与辅助函数
# ==============================================================================

def generate_ode_derivatives(df, target_cols):
    """ 计算物理导数特征 (简化版) """
    print(f"--> [ODE] 计算导数特征: {target_cols}")
    df_clean = df[target_cols].dropna()
    scaler = MinMaxScaler()
    data_np = scaler.fit_transform(df_clean.values)
    # 简单差分代替训练 ODE 网络以节省演示时间，实际使用可恢复 baseline.py 的完整逻辑
    derivs = np.gradient(data_np, axis=0)
    new_cols = [f"{c}_Deriv" for c in target_cols]
    return pd.DataFrame(derivs, index=df_clean.index, columns=new_cols)

def create_sequences(data, seq_length, forecast_horizon, future_indices, target_idx):
    xs_past, xs_future, ys = [], [], []
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        xs_past.append(data[i:(i + seq_length)])
        xs_future.append(data[i + seq_length : i + seq_length + forecast_horizon, future_indices])
        # 预测目标：序列的最后一个点 (MPC通常关注一段时间后的状态)
        ys.append(data[i + seq_length + forecast_horizon - 1, target_idx])
    return np.array(xs_past), np.array(xs_future), np.array(ys)

# ==============================================================================
# 4. 主流程
# ==============================================================================

def main():
    print("--- 启动智能温室控制系统 (MPC vs MDP) ---")
    
    # 1. 数据加载与预处理
    filename = 'Strawberry Greenhouse Environmental Control Dataset(version2).csv'
    if not os.path.exists(filename):
        print(f"错误: 未找到数据集 {filename}")
        return

    df = pd.read_csv(filename, encoding='latin1', sep=';', decimal=',', parse_dates=['Timestamp'], dayfirst=True, index_col='Timestamp')
    
    # 清洗列名
    df.columns = [c.replace('"', '').strip() for c in df.columns]
    
    # 处理开关量
    cols_to_binary = ['Heater', 'Ventilation', 'Lighting', 'Pump 1', 'Valve 1']
    for col in cols_to_binary:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 1 if str(x).lower() in ['on', 'yes', '1'] else 0)
    
    # 转换为数值并插值
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.resample('1min').mean().interpolate().ffill().bfill()
    
    # 特征工程
    outdoor_cols = [c for c in df.columns if 'Outdoor' in c or 'Solar' in c]
    df_derivs = pd.DataFrame() # 初始化为空，防止报错
    if outdoor_cols:
        df_derivs = generate_ode_derivatives(df, outdoor_cols)
        df = pd.concat([df, df_derivs], axis=1).dropna()
        
    print(f"--> 数据预处理完成，维度: {df.shape}")

    # 2. 准备数据集
    target_col = 'Temperature, °C'
    # 确保 Heater, Ventilation 在前两列，方便 MPC 操控
    feature_order = ['Heater', 'Ventilation', 'Lighting', 'Temperature, °C', 'Humidity, %', 'CO?, ppm'] + outdoor_cols + list(df_derivs.columns)
    # 过滤掉不存在的列
    feature_order = [f for f in feature_order if f in df.columns]
    
    df = df[feature_order]
    target_idx = feature_order.index(target_col)
    
    # 定义未来特征索引 (Future Inputs): 包含控制量、天气等
    # 在这个模型中，我们假设 x_future 包含了所有特征的"未来预报"
    # 但实际上 MPC 只能改变 Heater(idx=0) 和 Ventilation(idx=1)
    future_indices = list(range(len(feature_order))) 
    
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)
    
    seq_len = 60    # 过去 1 小时
    horizon = 10    # 预测未来 10 分钟
    
    X_past, X_future, y = create_sequences(data_scaled, seq_len, horizon, future_indices, target_idx)
    
    # 划分训练/测试集
    train_size = int(len(X_past) * 0.8)
    X_train_p, X_test_p = X_past[:train_size], X_past[train_size:]
    X_train_f, X_test_f = X_future[:train_size], X_future[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 转 Tensor
    X_train_p_t = torch.FloatTensor(X_train_p).to(device)
    X_train_f_t = torch.FloatTensor(X_train_f).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    
    # 3. 模型初始化与训练 (或加载)
    print("\n--> 初始化混合预测模型...")
    model = SegmentedHybridModel(input_dim=len(feature_order), future_dim=len(feature_order), forecast_horizon=1).to(device)
    # 注意: 为了 MPC 单步控制，这里 forecast_horizon 设为 1 (预测 horizon 分钟后的那一个点)
    
    # 简单训练循环 (模拟加载训练好的权重)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("--> 开始训练模型 (用于 MPC 预测核心)...")
    for epoch in range(50): # 增加训练轮数以尝试捕捉物理规律
        model.train()
        optimizer.zero_grad()
        # 这里为了简化，直接用最后一步的预测值训练
        outputs = model(X_train_p_t, X_train_f_t) 
        # outputs shape: (batch, 1) -> 需要 squeeze 吗？取决于定义。
        # SegmentedHybridModel输出维度是 (batch, forecast_horizon)
        loss = criterion(outputs, y_train_t.unsqueeze(1))
        loss.backward()
        optimizer.step()
        if (epoch+1) % 5 == 0:
            print(f"    Epoch {epoch+1}, Loss: {loss.item():.5f}")
            
    model.eval()
    print("--> 模型就绪。")

    # 4. 仿真实验：MPC vs MDP
    print("\n--> 开始对比仿真 (Simulation: MDP vs MPC)...")
    
    # 初始化控制器
    mpc = MPC_Controller(model, scaler, target_idx, future_indices, horizon=horizon, target_temp=25.0)
    mdp = LegacyMDPController()
    
    # 选取测试集的一段作为仿真区间 (例如 300 分钟)
    sim_steps = 300
    start_idx = 0 
    
    # 记录结果
    history_mdp = []
    history_mpc = []
    
    # 记录动作序列 [Heater, Ventilation]
    actions_mdp_log = []
    actions_mpc_log = []
    
    # 初始状态 (从测试集获取)
    # 注意：为了公平对比，我们需要一个"虚拟环境"。
    # 简单起见，我们假设：
    # T(t+1) = Model.predict(State(t), Action(t)) + Random_Noise
    # 这样 Model 既是控制器的大脑，也是仿真环境的物理引擎 (Digital Twin)
    
    current_state_mdp = X_test_p[start_idx:start_idx+1] # (1, 60, feat)
    current_state_mpc = X_test_p[start_idx:start_idx+1] # copy for mpc
    
    # 获取未来天气基准 (从测试集真实数据获取，假设只有 Heater/Vent 可变，其他是天气)
    future_base_seq = X_test_f[start_idx : start_idx + sim_steps] 
    
    # 初始化物理仿真环境
    # 获取初始温度 (反归一化)
    init_temp_norm = current_state_mpc[0, -1, target_idx]
    init_temp = mpc._inverse_transform_target(init_temp_norm, target_idx)
    env_mpc = PhysicsGreenhouseEnv(init_temp)
    env_mdp = PhysicsGreenhouseEnv(init_temp)
    
    print(f"--> 正在进行 {sim_steps} 步滚动优化仿真(物理引擎驱动)...")
    
    for t in range(sim_steps):
        # 1. 准备数据 Tensor
        state_mpc_tensor = torch.FloatTensor(current_state_mpc).to(device)
        
        # 获取当前时刻对应的未来基准 (用于 MPC 预测)
        # 这里的 future_base 包含了真实发生的"天气"，MPC 需要在这些天气基础上填入自己的动作
        current_future_base = torch.FloatTensor(future_base_seq[t]).unsqueeze(0).to(device) # (1, horizon, feat)
        
        # --- A. 运行 MPC ---
        opt_action, _ = mpc.get_optimal_action(state_mpc_tensor, current_future_base)
        actions_mpc_log.append(opt_action)
        
        # --- B. 运行 MDP ---
        # 获取当前 MDP 环境的温度 (反归一化)
        curr_temp_norm_mdp = current_state_mdp[0, -1, target_idx] # 序列最后一个点的温度
        curr_temp_mdp = mpc._inverse_transform_target(curr_temp_norm_mdp, target_idx)
        mdp_action = mdp.get_action(curr_temp_mdp)
        actions_mdp_log.append(mdp_action)
        
        # --- C. 环境演变 (Simulation Step) ---
        # 使用模型推演下一步状态 (Digital Twin Update)
        
        # 1) 推演 MPC 环境
        with torch.no_grad():
            # 获取当前室外温度 (用于物理计算)
            # 假设 feature_order 中包含 'Outdoor_Temp'，需要找到对应索引
            # 这里简化处理：假设室外温度是第 7 列 (根据 feature_order 逻辑推断)
            # feature_order = [H, V, L, Tin, Hin, CO2, Tout, ...]
            # 更好的方式是查找索引
            # 尝试查找室外温度列索引
            tout_idx = -1
            for idx, fname in enumerate(feature_order):
                if 'outdoor' in fname.lower() or 'solar' in fname.lower(): # 简单启发式
                    tout_idx = idx
                    break
            
            if tout_idx != -1 and tout_idx < current_state_mpc.shape[2]:
                curr_tout_norm = current_state_mpc[0, -1, tout_idx]
                curr_tout = mpc._inverse_transform_target(curr_tout_norm, tout_idx)
            else:
                # Fallback: 如果没找到室外温度数据，假定一个环境温度 (e.g. 15度)
                # 或者使用上一时刻的室内温度减去一点点? 不，还是常数安全。
                curr_tout = 15.0 

            
            # 物理引擎推演下一步真实温度
            next_temp_real = env_mpc.step(opt_action, curr_tout)
            
            # 归一化回去以便填入状态序列
            # 注意：这里需要单独归一化一个标量，稍微麻烦。我们构造一个 dummy array
            dummy_arr = np.zeros((1, len(scaler.scale_)))
            dummy_arr[0, target_idx] = next_temp_real
            next_temp_norm = scaler.transform(dummy_arr)[0, target_idx]
            
            # 更新 MPC 状态队列
            new_step_mpc = current_state_mpc[0, 1:, :].copy() # 移位
            next_real_features = X_test_p[start_idx + t + 1][-1, :].copy() # 获取下一时刻的真实天气特征
            next_real_features[target_idx] = next_temp_norm # 替换为物理引擎计算出的温度
            # 替换为执行的动作
            next_real_features[0] = opt_action[0]
            next_real_features[1] = opt_action[1]
            
            current_state_mpc = np.concatenate([new_step_mpc, next_real_features.reshape(1, -1)], axis=0).reshape(1, seq_len, -1)
            
            # 记录真实温度
            history_mpc.append(next_temp_real)

        # 2) 推演 MDP 环境 (同理)
        with torch.no_grad():
            # 同样获取 MDP 环境的室外温度 (假设和 MPC 环境时刻一致，其实就是 X_test_p 里的)
            # 这里重用 curr_tout 即可，因为时间步是一样的
            
            next_temp_mdp_real = env_mdp.step(mdp_action, curr_tout)
            
            dummy_arr_mdp = np.zeros((1, len(scaler.scale_)))
            dummy_arr_mdp[0, target_idx] = next_temp_mdp_real
            next_temp_mdp_norm = scaler.transform(dummy_arr_mdp)[0, target_idx]
            
            new_step_mdp = current_state_mdp[0, 1:, :].copy()
            next_feat_mdp = X_test_p[start_idx + t + 1][-1, :].copy()
            next_feat_mdp[target_idx] = next_temp_mdp_norm
            next_feat_mdp[0] = mdp_action[0]
            next_feat_mdp[1] = mdp_action[1]
            
            current_state_mdp = np.concatenate([new_step_mdp, next_feat_mdp.reshape(1, -1)], axis=0).reshape(1, seq_len, -1)
            history_mdp.append(next_temp_mdp_real)
    # 5. 结果可视化 (包含温度和动作对比)
    print("--> 仿真完成，正在绘图...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    time_axis = range(sim_steps)
    target_line = [25.0] * sim_steps
    
    # 子图 1: 温度曲线
    axes[0].plot(time_axis, target_line, 'k--', label='目标温度 (25°C)', alpha=0.6)
    axes[0].plot(time_axis, history_mdp, color='gray', linestyle=':', label='MDP 控制 (规则)', linewidth=1.5)
    axes[0].plot(time_axis, history_mpc, color='red', label='MPC 控制 (本文)', linewidth=2.0)
    axes[0].set_ylabel("室内温度 (°C)", fontsize=12)
    axes[0].set_title("控制效果对比: 温度保持", fontsize=14)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # 计算一些统计指标
    mae_mpc = np.mean(np.abs(np.array(history_mpc) - 25.0))
    mae_mdp = np.mean(np.abs(np.array(history_mdp) - 25.0))
    axes[0].text(sim_steps*0.02, 23, f"MDP MAE: {mae_mdp:.2f}\nMPC MAE: {mae_mpc:.2f}", bbox=dict(facecolor='white', alpha=0.8))

    # 子图 2: 加热器动作 (Heater)
    mpc_heater = [a[0] for a in actions_mpc_log]
    mdp_heater = [a[0] for a in actions_mdp_log]
    
    axes[1].step(time_axis, mdp_heater, color='gray', linestyle=':', label='MDP Heater', where='post', alpha=0.7)
    axes[1].step(time_axis, mpc_heater, color='red', label='MPC Heater', where='post', alpha=0.8)
    axes[1].set_ylabel("加热器状态 (0/1)", fontsize=12)
    axes[1].set_title("执行机构动作: 加热器 (Heater)", fontsize=14)
    axes[1].set_yticks([0, 1])
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # 子图 3: 通风动作 (Ventilation)
    mpc_vent = [a[1] for a in actions_mpc_log]
    mdp_vent = [a[1] for a in actions_mdp_log]
    
    axes[2].step(time_axis, mdp_vent, color='gray', linestyle=':', label='MDP Vent', where='post', alpha=0.7)
    axes[2].step(time_axis, mpc_vent, color='blue', label='MPC Vent', where='post', alpha=0.8)
    axes[2].set_ylabel("通风状态 (0/1)", fontsize=12)
    axes[2].set_title("执行机构动作: 通风 (Ventilation)", fontsize=14)
    axes[2].set_yticks([0, 1])
    axes[2].set_xlabel("模拟时间 (分钟)", fontsize=12)
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = "mpc_vs_mdp_simulation.png"
    plt.savefig(save_path)
    print(f"--> [Success] 结果图已保存至: {save_path}")
    
    # 如果在支持交互式环境运行，可以使用 plt.show()
    # plt.show()
    print("DONE")

if __name__ == "__main__":
    main()