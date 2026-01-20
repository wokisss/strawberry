# -*- coding: utf-8 -*-
"""
train_counterfactual.py
-----------------------
A2 方案实现：使用物理反事实数据扩增训练 MPC 预测模型。
本脚本独立运行，不修改 mpc_system.py 的核心逻辑，但会复用其模型定义。
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import os
import sys
import matplotlib.pyplot as plt

# 导入原有模块 (确保 mpc_system.py 在当前目录)
from mpc_system import SegmentedHybridModel, create_sequences, device
from counterfactual_augmentation import PhysicsBasedCounterfactualGenerator, CounterfactualDataset

# 设定随机种子
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

def load_and_preprocess_data():
    """ 
    复用 mpc_system.py 的数据处理逻辑 
    """
    print("--> [Data] Loading data...")
    filename = 'Strawberry Greenhouse Environmental Control Dataset(version2).csv'
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Dataset not found: {filename}")

    df = pd.read_csv(filename, encoding='latin1', sep=';', decimal=',', parse_dates=['Timestamp'], dayfirst=True, index_col='Timestamp')
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
    
    # 合并天气数据
    weather_file = 'POWER_Point_Hourly_20250517_20250618_048d33N_025d93E_LST.csv'
    if os.path.exists(weather_file):
        print(f"--> [Data] Merging weather data: {weather_file}")
        df_weather = pd.read_csv(weather_file, skiprows=12)
        df_weather['Timestamp'] = pd.to_datetime(
            df_weather['YEAR'].astype(str) + '-' + 
            df_weather['MO'].astype(str).str.zfill(2) + '-' + 
            df_weather['DY'].astype(str).str.zfill(2) + ' ' + 
            df_weather['HR'].astype(str).str.zfill(2) + ':00:00'
        )
        df_weather = df_weather.set_index('Timestamp')
        df_weather = df_weather.rename(columns={
            'T2M': 'Outdoor_Temp',
            'ALLSKY_SFC_SW_DWN': 'Outdoor_Solar',
            'RH2M': 'Outdoor_Hum',
            'WS2M': 'Outdoor_Wind'
        })
        df_weather = df_weather[['Outdoor_Temp', 'Outdoor_Solar', 'Outdoor_Hum', 'Outdoor_Wind']]
        df_weather = df_weather.resample('1min').interpolate(method='linear').ffill().bfill()
        df_weather = df_weather.reindex(df.index, method='ffill')
        
        for col in df_weather.columns:
            df[col] = df_weather[col]
        df = df.ffill().bfill()
        
    outdoor_cols = [c for c in df.columns if 'Outdoor' in c or 'Solar' in c]
    
    # 选取特征列
    target_col = 'Temperature, °C'
    feature_order = ['Heater', 'Ventilation', 'Lighting', 'Temperature, °C', 'Humidity, %', 'CO2, ppm'] + outdoor_cols
    feature_order = [f for f in feature_order if f in df.columns]
    
    df = df[feature_order]
    
    return df, feature_order, target_col, outdoor_cols

def train_augmented_model():
    print("--- A2 Counterfactual Training ---")
    
    # 1. 加载数据
    df, feature_order, target_col, outdoor_cols = load_and_preprocess_data()
    
    # 2. 准备特征索引 (从 mpc_system.py 复制逻辑)
    # Control vars (Heater, Vent) must be first two if present
    control_cols = ['Heater', 'Ventilation']
    control_indices = [feature_order.index(c) for c in control_cols if c in feature_order]
    weather_indices = [feature_order.index(c) for c in outdoor_cols if c in feature_order]
    future_indices = control_indices + weather_indices
    
    target_idx = feature_order.index(target_col)
    
    print(f"--> Future Indices: {future_indices}")
    
    # 3. 归一化
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)
    
    seq_len = 60
    horizon = 10
    
    X_past, X_future, y = create_sequences(data_scaled, seq_len, horizon, future_indices, target_idx)
    
    # 划分数据集
    train_size = int(len(X_past) * 0.8)
    X_train_p, X_test_p = X_past[:train_size], X_past[train_size:]
    X_train_f, X_test_f = X_future[:train_size], X_future[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 4. 初始化反事实生成器
    generator = PhysicsBasedCounterfactualGenerator(
        feature_order=feature_order,
        target_col=target_col,
        outdoor_temp_col='Outdoor_Temp', # 确保与合并后的列名一致
        solar_col='Outdoor_Solar',
        scaler=scaler
    )
    
    # 5. 创建混合数据集
    # cf_ratio=0.5 表示我们希望反事实数据占扩增部分的 50% (即 1:1 混合，或按需调整)
    train_dataset = CounterfactualDataset(
        X_train_p, X_train_f, y_train, 
        generator, future_indices, cf_ratio=0.5
    )
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    # 测试集保持纯净，不需要扩增
    test_dataset = TensorDataset(torch.FloatTensor(X_test_p), torch.FloatTensor(X_test_f), torch.FloatTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # 6. 初始化模型
    print("\n--> Initializing SegmentedHybridModel...")
    model = SegmentedHybridModel(
        input_dim=len(feature_order), 
        future_dim=len(future_indices), 
        forecast_horizon=1
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 7. 训练循环
    epochs = 30
    best_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    print(f"--> Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_p, batch_f, batch_y in train_loader:
            batch_p = batch_p.to(device)
            batch_f = batch_f.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_p, batch_f)
            # outputs shape: (batch, 1) usually due to forecast_horizon=1 logic in model
            # batch_y shape: (batch,)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_p, batch_f, batch_y in test_loader:
                batch_p = batch_p.to(device)
                batch_f = batch_f.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_p, batch_f)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(test_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f"    Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model_A2.pth')
            
    print(f"--> Training Complete. Best Val Loss: {best_loss:.6f}")
    print("--> Model saved to best_model_A2.pth")
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss (Augmented)')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('A2 Counterfactual Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig('results/training_curve_A2.png')
    print("--> Training curve saved to results/training_curve_A2.png")

if __name__ == "__main__":
    # 确保 results 目录存在
    os.makedirs("results", exist_ok=True)
    train_augmented_model()
