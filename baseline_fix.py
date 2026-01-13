import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os

# === 辅助函数与类定义 (PyTorch) ===
# === [新版] 物理增强模块：Neural ODE 导数提取器 ===
class ODEF(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(ODEF, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(), # Softplus 比 Tanh 更适合拟合非负的物理量变化（如光照）
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, t, y):
        t_vec = torch.ones_like(y[..., :1]) * t
        cat_input = torch.cat([y, t_vec], dim=-1)
        return self.net(cat_input)

def generate_ode_derivatives(df, target_cols):
    """
    不改变原始数据的值，而是计算每分钟的'物理变化率'作为新特征
    """
    print(f"--> [ODE Pro] 正在计算物理导数特征: {target_cols}...")
    
    # 1. 准备训练数据 (仅使用非空的小时级数据)
    df_clean = df[target_cols].dropna()
    scaler = MinMaxScaler()
    data_np = scaler.fit_transform(df_clean.values)
    
    # 时间归一化 (以小时为单位)
    timestamps = (df_clean.index - df_clean.index[0]).total_seconds() / 3600.0
    t_tensor = torch.FloatTensor(timestamps).reshape(-1, 1)
    y_tensor = torch.FloatTensor(data_np)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_tensor = t_tensor.to(device)
    y_tensor = y_tensor.to(device)
    
    # 2. 训练 ODE 网络拟合变化趋势
    ode_func = ODEF(input_dim=len(target_cols)).to(device)
    optimizer = torch.optim.Adam(ode_func.parameters(), lr=0.02) # 提高学习率
    
    ode_func.train()
    print("    |-- 正在学习物理动力场 (Training Dynamics)...")
    for epoch in range(1000): # 增加迭代次数以捕捉细节
        optimizer.zero_grad()
        # 拟合目标：使得模型预测的导数 接近于 真实数据的差分
        # (y_{t+1} - y_t) / dt ~ f(t, y)
        dy_dt_target = (y_tensor[1:] - y_tensor[:-1]) / (t_tensor[1:] - t_tensor[:-1])
        t_mid = (t_tensor[1:] + t_tensor[:-1]) / 2
        y_mid = (y_tensor[1:] + y_tensor[:-1]) / 2
        
        pred_dy_dt = ode_func(t_mid, y_mid)
        loss = torch.mean((pred_dy_dt - dy_dt_target) ** 2)
        loss.backward()
        optimizer.step()
        
    # 3. 推断：生成每分钟的导数
    ode_func.eval()
    print("    |-- 正在生成分钟级导数特征...")
    
    # 创建完整的分钟级时间轴
    full_index = pd.date_range(start=df_clean.index[0], end=df_clean.index[-1], freq='1min')
    
    # 先对原始数据做简单的线性插值，作为 ODE 的输入状态 y
    # (因为我们需要知道当前大概是多少度，才能算出当前的变化率)
    df_linear_temp = df[target_cols].reindex(full_index).interpolate(method='linear').fillna(method='bfill')
    y_interp_np = scaler.transform(df_linear_temp.values)
    y_interp_tensor = torch.FloatTensor(y_interp_np).to(device)
    
    t_full_seconds = (full_index - df_clean.index[0]).total_seconds() / 3600.0
    t_full_tensor = torch.FloatTensor(t_full_seconds).reshape(-1, 1).to(device)
    
    with torch.no_grad():
        # 直接查询网络：在 t 时刻，状态为 y 时，变化率是多少？
        derivs_tensor = ode_func(t_full_tensor, y_interp_tensor)
    
    derivs_np = derivs_tensor.cpu().numpy()
    
    # 反归一化导数 (Scale back derivatives)
    # y = scaler * raw => dy = scaler * draw => draw = dy / scaler
    derivs_restored = derivs_np / (scaler.scale_ + 1e-8)
    
    # 创建 DataFrame，列名加后缀 "_Deriv"
    new_cols = [f"{c}_Deriv" for c in target_cols]
    df_derivs = pd.DataFrame(derivs_restored, index=full_index, columns=new_cols)
    
    return df_derivs

# 创建时间序列数据
def create_sequences(data, seq_length, forecast_horizon, future_indices, target_idx):
    xs_past, xs_future, ys, y_bases = [], [], [], []
    # 确保不越界：数据总长度 - (输入序列长度 + 预测距离)
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        # 输入：从 i 开始，取 seq_length 个点
        x_p = data[i:(i + seq_length)]
        # 未来控制序列：从 seq_length 结束开始，取 forecast_horizon 个点
        x_f = data[i + seq_length : i + seq_length + forecast_horizon, future_indices]
        # 任务二：输出整个预测序列 [t+1, ..., t+horizon]
        y = data[i + seq_length : i + seq_length + forecast_horizon, target_idx]
        # 基准值：输入序列的最后一个目标值，用于计算差分
        y_base = data[i + seq_length - 1, target_idx]
        xs_past.append(x_p)
        xs_future.append(x_f)
        ys.append(y)
        y_bases.append(y_base)
    return np.array(xs_past), np.array(xs_future), np.array(ys), np.array(y_bases)

# 2. 分段混合模型 (SegmentedHybridModel) - 三头专家模型 + 双分支解耦 (简化版)
class SegmentedHybridModel(nn.Module):
    def __init__(self, input_dim, control_dim, weather_dim, forecast_horizon, hidden_dim=32):
        super(SegmentedHybridModel, self).__init__()
        
        self.control_dim = control_dim
        self.weather_dim = weather_dim
        
        # --- 1. 共享特征提取器 ---
        self.past_conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.past_bigru = nn.GRU(input_size=64, hidden_size=hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.past_attention = nn.Linear(hidden_dim * 2, 1)
        
        # --- 双分支解耦 (简化版: 更小的分支) ---
        self.control_gru = nn.GRU(input_size=control_dim, hidden_size=hidden_dim // 2, num_layers=1, batch_first=True)
        self.weather_gru = nn.GRU(input_size=weather_dim, hidden_size=hidden_dim // 2, num_layers=1, batch_first=True)
        
        # feature_size = BiGRU(64) + control(16) + weather(16) = 96
        feature_size = hidden_dim * 2 + hidden_dim
        
        # 专家头 (简化版)
        self.fc_heat = nn.Sequential(
            nn.Linear(feature_size, 32),
            nn.ReLU(),
            nn.Linear(32, forecast_horizon)
        )
        self.fc_vent = nn.Sequential(
            nn.Linear(feature_size, 32),
            nn.ReLU(),
            nn.Linear(32, forecast_horizon)
        )
        self.fc_natural = nn.Sequential(
            nn.Linear(feature_size, 32),
            nn.ReLU(),
            nn.Linear(32, forecast_horizon)
        )

    def forward(self, x_past, x_future):
        # --- 1. 提取历史特征 ---
        x_p = x_past.permute(0, 2, 1)
        x_p = torch.relu(self.past_conv1(x_p))
        x_p = x_p.permute(0, 2, 1)
        gru_out_p, _ = self.past_bigru(x_p)
        weights_p = torch.softmax(self.past_attention(gru_out_p), dim=1)
        attended_p = torch.sum(weights_p * gru_out_p, dim=1)
        
        # --- 2. 双分支处理未来特征 ---
        x_ctrl = x_future[:, :, :self.control_dim]
        x_weather = x_future[:, :, self.control_dim:]
        
        # 控制分支
        _, h_ctrl = self.control_gru(x_ctrl)
        ctrl_features = h_ctrl[-1]
        
        # 天气分支
        _, h_weather = self.weather_gru(x_weather)
        weather_features = h_weather[-1]
        
        # 融合：历史 + 控制 + 天气
        combined = torch.cat([attended_p, ctrl_features, weather_features], dim=1)
        
        # --- 3. 并行计算三个专家的预测 ---
        pred_heat = self.fc_heat(combined)      # 假设全是加热
        pred_vent = self.fc_vent(combined)      # 假设全是通风
        pred_natural = self.fc_natural(combined)# 假设全是自然
        
        # --- 4. 动态门控融合 (Gated Fusion) ---
        # 从控制分支提取门控信号（Heater=第0列, Ventilation=第1列）
        heater_signal = x_ctrl[:, :, 0].mean(dim=1, keepdim=True)
        vent_signal = x_ctrl[:, :, 1].mean(dim=1, keepdim=True)
        
        w_heat = heater_signal
        w_vent = vent_signal
        w_natural = torch.clamp(1.0 - w_heat - w_vent, min=0.0)
        
        # 最终预测 = 加权求和
        final_pred = (w_heat * pred_heat) + (w_vent * pred_vent) + (w_natural * pred_natural)
        
        return final_pred

# 评估指标计算
def calculate_metrics(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"[{name}] MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    return mae, rmse, r2

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 步骤 0: 初始设置 ---
filename = 'Strawberry Greenhouse Environmental Control Dataset(version2).csv'
print(f"--- 开始处理文件: '{filename}' ---")

try:
    # --- 步骤 1-3: 数据加载, 清洗, 重采样, 填充 ---
    print("--> 正在加载和预处理数据...")
    df = pd.read_csv(filename, encoding='latin1', sep=';', decimal=',', parse_dates=['Timestamp'], dayfirst=True, index_col='Timestamp')
    
    # === [修复] 全局列名清洗 ===
    # 去除列名中的引号和多余空格，统一命名规范，防止匹配失败
    df.columns = [c.replace('"', '').strip() for c in df.columns]
    print(f"--> 列名已清理: {list(df.columns)}")

    # 修复：在转换为数值之前处理开关量列，防止被 to_numeric 转为 NaN 而丢弃
    cols_to_binary = ['Heater', 'Ventilation', 'Lighting', 'Pump 1', 'Valve 1']
    for col in cols_to_binary:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 1 if str(x).lower() in ['on', 'yes', '1'] else 0)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(axis=1, how='all', inplace=True)
    
    # === [修改点] 数据处理逻辑 ===
    print("--> 正在执行基础线性插值...")
    # 1. 基础线性插值 (保留最真实的数据骨架)
    df_resampled = df.resample('1min').mean().interpolate(method='linear').ffill().bfill()
    
    # ========================================================================
    # [新增] 加载外部天气数据 (NASA POWER)
    # ========================================================================
    weather_file = 'POWER_Point_Hourly_20250517_20250618_048d33N_025d93E_LST.csv'
    if os.path.exists(weather_file):
        print(f"--> [Weather Fix] 正在加载外部天气数据: {weather_file}")
        # 读取 NASA POWER 数据 (跳过头部注释行)
        df_weather = pd.read_csv(weather_file, skiprows=12)
        
        # 构建时间戳
        df_weather['Timestamp'] = pd.to_datetime(
            df_weather['YEAR'].astype(str) + '-' + 
            df_weather['MO'].astype(str).str.zfill(2) + '-' + 
            df_weather['DY'].astype(str).str.zfill(2) + ' ' + 
            df_weather['HR'].astype(str).str.zfill(2) + ':00:00'
        )
        df_weather = df_weather.set_index('Timestamp')
        
        # 重命名列为标准名称
        df_weather = df_weather.rename(columns={
            'T2M': 'Outdoor_Temp',
            'ALLSKY_SFC_SW_DWN': 'Outdoor_Solar',
            'RH2M': 'Outdoor_Hum',
            'WS2M': 'Outdoor_Wind'
        })
        
        # 重采样到分钟级并插值 (尝试使用三次样条插值以获得平滑曲线)
        df_weather = df_weather[['Outdoor_Temp', 'Outdoor_Solar', 'Outdoor_Hum', 'Outdoor_Wind']]
        
        # 创建一个完整的分钟级索引
        full_idx = pd.date_range(start=df_weather.index[0], end=df_weather.index[-1], freq='1min')
        df_weather = df_weather.reindex(full_idx)
        
        try:
            # 尝试三次样条插值
            df_weather = df_weather.interpolate(method='cubic').ffill().bfill()
        except:
            print("    [Warning] Cubic interpolation failed, falling back to linear.")
            df_weather = df_weather.interpolate(method='linear').ffill().bfill()
        
        # ====================================================================
        # [智能对齐] 使用温度波峰对齐两个数据集
        # 原理：室内和室外温度都会在中午达到日峰值，可用此特征对齐时间
        # ====================================================================
        print("    [Peak Align] 正在进行温度波峰对齐...")
        
        from scipy.signal import find_peaks
        
        # 1. 找室内温度的日峰值（每天中午左右）
        indoor_temp = df_resampled['Temperature, °C'].values
        indoor_peaks, _ = find_peaks(indoor_temp, distance=60*12, prominence=2)  # 至少间隔12小时
        
        # 2. 找室外温度的日峰值
        outdoor_temp = df_weather['Outdoor_Temp'].values
        outdoor_peaks, _ = find_peaks(outdoor_temp, distance=60*12, prominence=2)
        
        # 3. 找到时间范围重叠的部分
        indoor_times = df_resampled.index
        outdoor_times = df_weather.index
        
        # 确定重叠区间
        overlap_start = max(indoor_times[0], outdoor_times[0])
        overlap_end = min(indoor_times[-1], outdoor_times[-1])
        
        print(f"    [Peak Align] 室内数据范围: {indoor_times[0]} ~ {indoor_times[-1]}")
        print(f"    [Peak Align] 室外数据范围: {outdoor_times[0]} ~ {outdoor_times[-1]}")
        print(f"    [Peak Align] 重叠区间: {overlap_start} ~ {overlap_end}")
        
        # 4. 计算时间偏移（使用第一个可用的峰值对）
        time_offset = pd.Timedelta(0)
        if len(indoor_peaks) > 0 and len(outdoor_peaks) > 0:
            # 取重叠区间内的第一个峰值
            indoor_peak_times = indoor_times[indoor_peaks]
            outdoor_peak_times = outdoor_times[outdoor_peaks]
            
            # 找到在重叠区间内的峰值
            indoor_valid = indoor_peak_times[(indoor_peak_times >= overlap_start) & (indoor_peak_times <= overlap_end)]
            outdoor_valid = outdoor_peak_times[(outdoor_peak_times >= overlap_start) & (outdoor_peak_times <= overlap_end)]
            
            if len(indoor_valid) > 0 and len(outdoor_valid) > 0:
                # 使用第一对峰值计算偏移
                first_indoor_peak = indoor_valid[0]
                first_outdoor_peak = outdoor_valid[0]
                time_offset = first_indoor_peak - first_outdoor_peak
                print(f"    [Peak Align] 室内首峰: {first_indoor_peak}, 室外首峰: {first_outdoor_peak}")
                print(f"    [Peak Align] 计算时间偏移: {time_offset}")
        
        # 5. 应用时间偏移并合并
        df_weather_aligned = df_weather.copy()
        df_weather_aligned.index = df_weather_aligned.index + time_offset
        
        # 重新索引到室内数据的时间轴
        df_weather_aligned = df_weather_aligned.reindex(df_resampled.index, method='nearest')
        
        # 合并
        for col in ['Outdoor_Temp', 'Outdoor_Solar', 'Outdoor_Hum', 'Outdoor_Wind']:
            if col in df_weather_aligned.columns:
                df_resampled[col] = df_weather_aligned[col]
        
        df_resampled = df_resampled.ffill().bfill()
        
        print(f"    [Peak Align] 成功合并 4 个天气特征（已对齐）")
    else:
        print(f"--> [Weather Fix] 未找到外部天气文件 {weather_file}，使用原始数据")
    
    # 2. [ODE Boost] 生成天气特征的物理导数
    # 利用 cubic spline 插值后的平滑曲线，通过 Neural ODE 学习并推断分钟级变化率
    ode_target_cols = ['Outdoor_Temp', 'Outdoor_Solar']
    print(f"--> [ODE Boost] 正在为天气特征生成物理导数: {ode_target_cols}")
    try:
        # 注意：此时 df_resampled 已经包含了对齐且 Cubic 插值后的天气数据
        df_derivs = generate_ode_derivatives(df_resampled, ode_target_cols)
        
        # 合并导数特征
        df_resampled = pd.concat([df_resampled, df_derivs], axis=1)
        print(f"    [ODE Boost] 成功注入 {df_derivs.shape[1]} 个导数特征")
        
    except Exception as e:
        print(f"    [ODE Boost] 导数生成失败: {e}")
        import traceback
        traceback.print_exc()
        
    outdoor_cols = [c for c in df_resampled.columns if 'Outdoor' in c or 'Solar' in c]
    print(f"--> [Weather] 当前天气相关特征: {outdoor_cols}")
    
    # === [新增] 能量累积特征 (Energy Accumulation) ===
    # 目的: 让模型感知"加热器开了多久"，解决起步慢/滞后问题
    control_cols = ['Heater', 'Ventilation', 'Lighting', 'Pump 1', 'Valve 1']
    existing_control_cols = [c for c in control_cols if c in df_resampled.columns]
    
    if existing_control_cols:
        print(f"--> [Energy] 正在构建能量累积特征 (Window=60min): {existing_control_cols}...")
        for col in existing_control_cols:
            # 计算过去60分钟的累积开启时长 (即能量输入)
            df_resampled[f'{col}_Energy_60m'] = df_resampled[col].rolling(window=60, min_periods=1).sum()
        df_resampled.fillna(0, inplace=True)

    print(f"--> 最终数据维度: {df_resampled.shape}")
except Exception as e:
    print(f"处理数据时发生错误: {e}")
    sys.exit(1)

# --- 步骤 4: 物理基准模型构建 ---
print(f"\n--- 正在构建物理基准模型 ---")
# !!! 关键修改：将预测步长定义提前，确保物理模型和混合模型一致 !!!
# 1小时 = 60分钟 = 60 个 1分钟点
forecast_horizon = 120
print(f"--> 物理模型预测步长设定为: {forecast_horizon} 分钟")

physics_features = ['Temperature, °C', 'Humidity, %', 'Illumination, lx', 'CO?, ppm', 'Heater', 'Ventilation', 'Lighting', 'Outdoor_Temp', 'Outdoor_Hum', 'Outdoor_Wind', 'Outdoor_Solar']
target_col = 'Temperature, °C'
available_physics_features = [col for col in physics_features if col in df_resampled.columns]

# 准备数据
data = df_resampled[available_physics_features].dropna()
if len(data) < forecast_horizon + 1: sys.exit("错误: 物理模型数据不足。")

# !!! 修正数据切片逻辑 !!!
# 输入 X: 从 0 到 总长度 - horizon
# 输出 y: 从 horizon 到 结尾
# 这样建立了 X_t -> y_{t+60} 的关系
X_physics = data.iloc[:-forecast_horizon].copy()
y_physics = data[target_col].iloc[forecast_horizon:].values

control_features = [col for col in ['Heater', 'Ventilation', 'Lighting'] if col in available_physics_features and col != target_col]

# 使用所有可用的物理特征（包含室外天气等）
feature_columns = available_physics_features

# --- 物理模型增强：加入未来控制量的均值 ---
# 计算未来 forecast_horizon 时间段内的控制变量均值
X_physics_augmented = X_physics.copy()
for col in control_features:
    # rolling(window).mean() 计算的是过去窗口的均值，shift(-window) 将其对齐到未来
    # 这样 X_physics_augmented[t] 就包含了 t 到 t+horizon 期间 heater 的平均开启率
    X_physics_augmented[f'Future_Mean_{col}'] = data[col].rolling(window=forecast_horizon).mean().shift(-forecast_horizon).iloc[:-forecast_horizon]

feature_columns = feature_columns + [f'Future_Mean_{col}' for col in control_features]
X_physics_features = X_physics_augmented[feature_columns].fillna(0).values

train_size = int(len(X_physics) * 0.8)
if train_size == 0: sys.exit("错误: 训练数据不足。")

X_train_phy, X_test_phy = X_physics_features[:train_size], X_physics_features[train_size:]
y_train_phy, y_test_phy = y_physics[:train_size], y_physics[train_size:]

physics_model = LinearRegression()
physics_model.fit(X_train_phy, y_train_phy)
y_pred_phy = physics_model.predict(X_test_phy)
print("---> 物理基准模型训练完成。")

# --- 步骤 5: 混合深度学习模型构建 (PyTorch) ---
print("\n--- 正在构建与训练混合深度学习模型 (PyTorch) ---")

df_hybrid = df_resampled.copy()

# 引入时间位置特征 (Time-of-Day Encoding)
hour_float = df_hybrid.index.hour + df_hybrid.index.minute / 60.0
df_hybrid['Hour_Sin'] = np.sin(2 * np.pi * hour_float / 24.0)
df_hybrid['Hour_Cos'] = np.cos(2 * np.pi * hour_float / 24.0)

input_features = ['Temperature, °C', 'Humidity, %', 'Illumination, lx', 'CO?, ppm', 'Heater', 'Ventilation', 'Lighting', 'Outdoor_Temp', 'Outdoor_Hum', 'Outdoor_Wind', 'Outdoor_Solar', 'Hour_Sin', 'Hour_Cos']
# 自动添加导数特征
input_features += [c for c in df_hybrid.columns if '_Deriv' in c]
# 自动添加能量累积特征
input_features += [c for c in df_hybrid.columns if '_Energy' in c]
available_input_features = [f for f in input_features if f in df_hybrid.columns]
target_index = available_input_features.index(target_col)

# --- [修改] 强制指定未来特征顺序 (Heater, Ventilation在前) ---
# 确保 Heater 是第0个，Ventilation 是第1个，以便模型正确提取控制信号
priority_cols = ['Heater', 'Ventilation']
priority_indices = [available_input_features.index(c) for c in priority_cols if c in available_input_features]

# 其他控制变量 (Lighting 等)
other_controls = [c for c in control_features if c not in priority_cols and c in available_input_features]
other_control_indices = [available_input_features.index(c) for c in other_controls]

# 将时间特征也视为“已知未来”的控制变量
time_indices = [available_input_features.index(col) for col in ['Hour_Sin', 'Hour_Cos'] if col in available_input_features]

# 1. 定义天气列 (精简版: 只保留最相关的两个)
weather_cols = ['Outdoor_Temp', 'Outdoor_Solar']
weather_indices = [available_input_features.index(col) for col in weather_cols if col in available_input_features]

# 2. 加入未来索引
# 逻辑：未来预测 = f(未来控制 + 未来时间 + 未来天气预报) 
# 强制顺序: [控制特征] + [天气特征]
# 控制特征: Heater, Ventilation, Lighting, Hour_Sin, Hour_Cos
control_indices = priority_indices + other_control_indices + time_indices
# 天气特征: Outdoor_Temp, Outdoor_Hum, Outdoor_Wind, Outdoor_Solar, Illumination
future_indices = control_indices + weather_indices

# 计算各分支维度
control_dim = len(control_indices)  # 5: Heater, Vent, Lighting, Hour_Sin, Hour_Cos
weather_dim = len(weather_indices)  # 5: Outdoor_Temp/Hum/Wind/Solar + Illumination

print(f"---> [双分支架构] 控制特征 ({control_dim}): {[available_input_features[i] for i in control_indices]}")
print(f"---> [双分支架构] 天气特征 ({weather_dim}): {[available_input_features[i] for i in weather_indices]}")
print(f"---> [Segmented Check] 未来特征前两列: {[available_input_features[i] for i in future_indices[:2]]}")

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df_hybrid[available_input_features])

# --- 数据集划分 (训练、验证、测试) ---
# --- 参数设置区域 ---
# 假设数据间隔是 5分钟 (df_resampled 是 5min 一行)
# 1. 设置输入窗口：比如看过去 2小时的数据来预测
# 2小时 = 120分钟 = 120 个 1分钟点
sequence_length = 120
print(f"--- 正在构建数据集: 输入过去 {sequence_length} 分钟，预测未来 {forecast_horizon} 分钟 ---")

# 原始训练数据
train_data_full = scaled_features[:train_size]
# 原始测试数据
test_data = scaled_features[train_size:]

# 将原始训练数据进一步划分为训练集和验证集 (80/20)
val_split_index = int(len(train_data_full) * 0.8)
train_data = train_data_full[:val_split_index]
val_data = train_data_full[val_split_index:]

# 创建序列
X_train_past, X_train_future, y_train, y_train_bases = create_sequences(train_data, sequence_length, forecast_horizon, future_indices, target_index)
X_val_past, X_val_future, y_val, y_val_bases = create_sequences(val_data, sequence_length, forecast_horizon, future_indices, target_index)
X_test_past, X_test_future, y_test_scaled, y_test_bases = create_sequences(test_data, sequence_length, forecast_horizon, future_indices, target_index)

if len(X_train_past) == 0 or len(X_val_past) == 0 or len(X_test_past) == 0:
    sys.exit("错误: 创建序列后数据不足以划分训练/验证/测试集。")

# ========================================================================
# [诊断] 数据质量检查
# ========================================================================
print("\n" + "="*60)
print("[诊断] 数据质量检查")
print("="*60)

# 1. 特征维度
print(f"\n[维度] x_past 形状: {X_train_past.shape}  (样本数, 历史步数, 特征数)")
print(f"[维度] x_future 形状: {X_train_future.shape}  (样本数, 预测步数, 未来特征数)")
print(f"[维度] y 形状: {y_train.shape}")

# 2. 列出 x_past 的特征
print(f"\n[x_past 特征] 共 {len(available_input_features)} 个:")
for i, feat in enumerate(available_input_features):
    print(f"    [{i:2d}] {feat}")

# 3. 列出 x_future 的特征
print(f"\n[x_future 特征] 共 {len(future_indices)} 个:")
for i, idx in enumerate(future_indices):
    print(f"    [{i:2d}] future_idx={idx} -> {available_input_features[idx]}")

# 4. 检查 NaN 值
nan_past = np.isnan(X_train_past).sum()
nan_future = np.isnan(X_train_future).sum()
nan_y = np.isnan(y_train).sum()
print(f"\n[NaN 检查] x_past NaN数: {nan_past}, x_future NaN数: {nan_future}, y NaN数: {nan_y}")

# 5. 检查天气特征的分布
print(f"\n[天气特征分布] (检查是否有异常值)")
for col in ['Outdoor_Temp', 'Outdoor_Solar', 'Outdoor_Hum', 'Outdoor_Wind']:
    if col in available_input_features:
        idx = available_input_features.index(col)
        vals = X_train_past[:, :, idx].flatten()
        print(f"    {col}: min={vals.min():.4f}, max={vals.max():.4f}, mean={vals.mean():.4f}, std={vals.std():.4f}")

# 6. 检查目标变量分布
print(f"\n[目标变量分布] Temperature")
print(f"    y_train: min={y_train.min():.4f}, max={y_train.max():.4f}, mean={y_train.mean():.4f}")

print("="*60 + "\n")

# 转换为PyTorch张量
X_train_past_tensor = torch.FloatTensor(X_train_past)
X_train_future_tensor = torch.FloatTensor(X_train_future)
y_train_tensor = torch.FloatTensor(y_train)
y_train_bases_tensor = torch.FloatTensor(y_train_bases).unsqueeze(1)

X_val_past_tensor = torch.FloatTensor(X_val_past)
X_val_future_tensor = torch.FloatTensor(X_val_future)
y_val_tensor = torch.FloatTensor(y_val)
y_val_bases_tensor = torch.FloatTensor(y_val_bases).unsqueeze(1)

X_test_past_tensor = torch.FloatTensor(X_test_past)
X_test_future_tensor = torch.FloatTensor(X_test_future)
y_test_bases_tensor = torch.FloatTensor(y_test_bases).unsqueeze(1)

# 创建DataLoader
train_dataset = TensorDataset(X_train_past_tensor, X_train_future_tensor, y_train_tensor, y_train_bases_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

val_dataset = TensorDataset(X_val_past_tensor, X_val_future_tensor, y_val_tensor, y_val_bases_tensor)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# --- 模型训练与早停机制 ---
# 新增: 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--> 使用设备: {device}")

# --- 训练函数 ---
def train_and_predict(model, model_name, train_loader, val_loader, X_test_p, X_test_f, y_test_bases, scaler, target_idx, feat_cols, available_physics_features, predict_diff=False, lambda_trend=0.5):
    print(f"--> 正在训练 {model_name}...")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    num_epochs, patience = 200, 10
    best_val_loss, epochs_no_improve = float('inf'), 0
    model_path = f"best_{model_name}.pth"
    
    for epoch in range(num_epochs):
        model.train()
        for b_Xp, b_Xf, b_y, b_base in train_loader:
            b_Xp, b_Xf, b_y, b_base = b_Xp.to(device), b_Xf.to(device), b_y.to(device), b_base.to(device)
            optimizer.zero_grad()
           
            pred = model(b_Xp, b_Xf)
            # 如果是差分模式，目标是 (未来值 - 当前基准值)
            target = (b_y - b_base) if predict_diff else b_y
           
            # 基础 MSE 损失
            loss_mse = criterion(pred, target)
           
            # 任务三：趋势惩罚 (Gradient Penalty)
            # 计算预测序列和目标序列的变化率（一阶差分）
            pred_diff = pred[:, 1:] - pred[:, :-1]
            target_diff = target[:, 1:] - target[:, :-1]
            loss_trend = criterion(pred_diff, target_diff)
           
            loss = loss_mse + lambda_trend * loss_trend
            loss.backward()
            optimizer.step()
       
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for b_Xp, b_Xf, b_y, b_base in val_loader:
                b_Xp, b_Xf, b_y, b_base = b_Xp.to(device), b_Xf.to(device), b_y.to(device), b_base.to(device)
                target = (b_y - b_base) if predict_diff else b_y
                val_loss += criterion(model(b_Xp, b_Xf), target).item()  # 验证集仅观察 MSE
       
        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience: break
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        pred_tensor = model(X_test_p.to(device), X_test_f.to(device))
        # 仅取序列的最后一个点进行最终评估（对应 t+horizon）
        pred_final = pred_tensor[:, -1].unsqueeze(1)
       
        # 如果是差分模式，还原绝对值：预测的变化量 + 测试集基准值
        if predict_diff:
            pred_final = pred_final + torch.FloatTensor(y_test_bases).to(device).unsqueeze(1)
   
    pred_np = pred_final.cpu().numpy()
    dummy = np.zeros((len(pred_np), len(feat_cols)))
    dummy[:, target_idx] = pred_np.ravel()
    os.remove(model_path)
    return scaler.inverse_transform(dummy)[:, target_idx]

input_dim = X_train_past_tensor.shape[2]

# 实例化两个模型 (使用双分支架构)
model_abs = SegmentedHybridModel(input_dim, control_dim, weather_dim, forecast_horizon).to(device)
model_diff = SegmentedHybridModel(input_dim, control_dim, weather_dim, forecast_horizon).to(device)

# 1. 训练原方法（预测绝对值）
y_pred_abs = train_and_predict(model_abs, "混合模型(绝对值)", train_loader, val_loader,
                               X_test_past_tensor, X_test_future_tensor, y_test_bases, scaler, target_index, available_input_features, available_physics_features, predict_diff=False)

# 2. 训练新方法（预测一阶差分）
y_pred_diff = train_and_predict(model_diff, "混合模型(一阶差分)", train_loader, val_loader,
                                X_test_past_tensor, X_test_future_tensor, y_test_bases, scaler, target_index, available_input_features, available_physics_features, predict_diff=True)

# 获取真实值
dummy_true = np.zeros((len(y_test_scaled), len(available_input_features)))
dummy_true[:, target_index] = y_test_scaled[:, -1].ravel()  # 取序列最后一个点
y_test_hybrid = scaler.inverse_transform(dummy_true)[:, target_index]

# --- 步骤 6: 评估与可视化 ---
print("\n--- 性能对比 ---")

start_offset = sequence_length
end_offset = start_offset + len(y_test_hybrid)
y_test_phy_aligned = y_test_phy[start_offset:end_offset]
y_pred_phy_aligned = y_pred_phy[start_offset:end_offset]

min_len = min(len(y_test_hybrid), len(y_test_phy_aligned))

m1 = calculate_metrics(y_test_phy_aligned[:min_len], y_pred_phy_aligned[:min_len], "基准: 物理微分方程模型")
m2 = calculate_metrics(y_test_hybrid[:min_len], y_pred_abs[:min_len], "本文: 混合模型(绝对值)")
m3 = calculate_metrics(y_test_hybrid[:min_len], y_pred_diff[:min_len], "本文: 混合模型(一阶差分)")

plt.figure(figsize=(14, 7))
plot_len = min(300, min_len)
plt.plot(y_test_phy_aligned[:plot_len], label='真实值 (Ground Truth)', color='black', linewidth=2)
plt.plot(y_pred_phy_aligned[:plot_len], label=f'物理模型预测 (R2={m1[2]:.2f})', color='blue', linestyle='--')
plt.plot(y_pred_abs[:plot_len], label=f'混合模型-绝对值 (R2={m2[2]:.2f})', color='red', linewidth=1.5, alpha=0.7)
plt.plot(y_pred_diff[:plot_len], label=f'混合模型-一阶差分 (R2={m3[2]:.2f})', color='green', linewidth=1.5)
plt.title(f"温室温度预测对比: 物理模型 vs 混合模型 (含天气数据)", fontsize=16)
plt.xlabel("时间步 (1分钟/步)", fontsize=12)
plt.ylabel("温度 (°C)", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.5)
plt.tight_layout()

# 保存到 results/ 文件夹（带时间戳，不覆盖）
from datetime import datetime
os.makedirs('results', exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
result_path = f'results/baseline_fix_{timestamp}.png'
plt.savefig(result_path, dpi=150)
print(f"--> [Success] 结果图已保存至: {result_path}")
plt.close()

# --- 步骤 7: 输出结论文本 ---
print("\n--- 结论分析 ---")
print(f"1. 物理模型基于简化的线性假设，其 R2 分数为 {m1[2]:.4f}。")
print(f"2. 混合模型(绝对值方法) R2: {m2[2]:.4f}, MAE: {m2[0]:.4f}")
print(f"3. 混合模型(一阶差分方法) R2: {m3[2]:.4f}, MAE: {m3[0]:.4f}")
print("DONE")