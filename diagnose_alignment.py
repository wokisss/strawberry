"""
诊断脚本：检查室内外温度数据的时间对齐问题
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import pearsonr
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("="*60)
print("时间对齐诊断")
print("="*60)

# 1. 加载原始室内数据
filename = 'Strawberry Greenhouse Environmental Control Dataset(version2).csv'
df = pd.read_csv(filename, encoding='latin1', sep=';', decimal=',', 
                 parse_dates=['Timestamp'], dayfirst=True, index_col='Timestamp')
df.columns = [c.replace('"', '').strip() for c in df.columns]

# 检查原始数据中是否已有室外温度
print("\n[1] 原始数据列检查:")
outdoor_cols_in_raw = [c for c in df.columns if 'Outdoor' in c or 'outdoor' in c.lower()]
print(f"    室外相关列: {outdoor_cols_in_raw}")

# 转换数值并重采样
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df_resampled = df.resample('1min').mean().interpolate(method='linear').ffill().bfill()

print(f"    室内数据时间范围: {df_resampled.index[0]} ~ {df_resampled.index[-1]}")
print(f"    室内数据点数: {len(df_resampled)}")

# 检查室内已有的室外数据质量
if 'Outdoor_Temp' in df_resampled.columns:
    outdoor_temp_raw = df_resampled['Outdoor_Temp']
    nan_ratio = outdoor_temp_raw.isna().sum() / len(outdoor_temp_raw)
    print(f"\n[2] 原始数据中的 Outdoor_Temp:")
    print(f"    NaN比例: {nan_ratio*100:.1f}%")
    print(f"    有效值范围: {outdoor_temp_raw.min():.1f} ~ {outdoor_temp_raw.max():.1f}°C")
else:
    print("\n[2] 原始数据中无 Outdoor_Temp 列")

# 2. 加载NASA POWER数据
weather_file = 'POWER_Point_Hourly_20250517_20250618_048d33N_025d93E_LST.csv'
if os.path.exists(weather_file):
    print(f"\n[3] NASA POWER 数据检查:")
    df_weather = pd.read_csv(weather_file, skiprows=12)
    
    # 构建时间戳
    df_weather['Timestamp'] = pd.to_datetime(
        df_weather['YEAR'].astype(str) + '-' + 
        df_weather['MO'].astype(str).str.zfill(2) + '-' + 
        df_weather['DY'].astype(str).str.zfill(2) + ' ' + 
        df_weather['HR'].astype(str).str.zfill(2) + ':00:00'
    )
    df_weather = df_weather.set_index('Timestamp')
    
    print(f"    NASA数据时间范围: {df_weather.index[0]} ~ {df_weather.index[-1]}")
    print(f"    NASA数据点数: {len(df_weather)} (小时级)")
    print(f"    T2M (室外温度) 范围: {df_weather['T2M'].min():.1f} ~ {df_weather['T2M'].max():.1f}°C")
    
    # 3. 检查时间重叠
    indoor_start = df_resampled.index[0]
    indoor_end = df_resampled.index[-1]
    outdoor_start = df_weather.index[0]
    outdoor_end = df_weather.index[-1]
    
    overlap_start = max(indoor_start, outdoor_start)
    overlap_end = min(indoor_end, outdoor_end)
    
    print(f"\n[4] 时间重叠分析:")
    print(f"    重叠区间: {overlap_start} ~ {overlap_end}")
    
    if overlap_start < overlap_end:
        overlap_duration = (overlap_end - overlap_start).total_seconds() / 3600
        print(f"    重叠时长: {overlap_duration:.1f} 小时")
    else:
        print("    ⚠️ 警告: 数据完全不重叠！这是问题的根源！")
    
    # 4. 使用滞后相关性分析最佳对齐
    print(f"\n[5] 滞后相关性分析:")
    
    # 将室外数据重采样到分钟级
    full_idx = pd.date_range(start=df_weather.index[0], end=df_weather.index[-1], freq='1min')
    df_weather_min = df_weather[['T2M']].reindex(full_idx).interpolate(method='linear').ffill().bfill()
    
    indoor_temp = df_resampled['Temperature, °C'].values
    outdoor_temp = df_weather_min['T2M'].values
    
    # 找共同时间范围的索引
    common_start = max(df_resampled.index[0], df_weather_min.index[0])
    common_end = min(df_resampled.index[-1], df_weather_min.index[-1])
    
    if common_start < common_end:
        indoor_mask = (df_resampled.index >= common_start) & (df_resampled.index <= common_end)
        outdoor_mask = (df_weather_min.index >= common_start) & (df_weather_min.index <= common_end)
        
        indoor_common = df_resampled.loc[indoor_mask, 'Temperature, °C'].values
        outdoor_common = df_weather_min.loc[outdoor_mask, 'T2M'].values
        
        min_len = min(len(indoor_common), len(outdoor_common))
        indoor_common = indoor_common[:min_len]
        outdoor_common = outdoor_common[:min_len]
        
        print(f"    共同时间段数据点: {min_len}")
        
        # 计算不同滞后的相关系数
        lags = range(-120, 121, 10)  # -2小时 到 +2小时，每10分钟
        correlations = []
        
        for lag in lags:
            if lag < 0:
                # 室外领先
                corr, _ = pearsonr(indoor_common[-lag:], outdoor_common[:lag])
            elif lag > 0:
                # 室内领先
                corr, _ = pearsonr(indoor_common[:-lag], outdoor_common[lag:])
            else:
                corr, _ = pearsonr(indoor_common, outdoor_common)
            correlations.append(corr)
        
        best_lag_idx = np.argmax(correlations)
        best_lag = list(lags)[best_lag_idx]
        best_corr = correlations[best_lag_idx]
        
        print(f"    最佳滞后: {best_lag} 分钟 (相关系数: {best_corr:.4f})")
        print(f"    零滞后相关: {correlations[len(lags)//2]:.4f}")
        
        # 绘制相关性曲线
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(lags, correlations, 'b-o', markersize=3)
        plt.axvline(x=best_lag, color='r', linestyle='--', label=f'最佳滞后={best_lag}min')
        plt.axvline(x=0, color='g', linestyle=':', label='零滞后')
        plt.xlabel('滞后 (分钟)')
        plt.ylabel('Pearson相关系数')
        plt.title('室内外温度滞后相关性')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 绘制时间序列对比
        plt.subplot(1, 2, 2)
        x_range = range(min(1440, min_len))  # 最多显示1天
        plt.plot(x_range, indoor_common[:len(x_range)], label='室内温度', color='red')
        plt.plot(x_range, outdoor_common[:len(x_range)], label='NASA室外温度', color='blue')
        plt.xlabel('时间 (分钟)')
        plt.ylabel('温度 (°C)')
        plt.title('室内外温度时间序列对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('diagnose_alignment.png', dpi=150)
        print(f"\n[6] 诊断图已保存: diagnose_alignment.png")
        plt.close()
        
    else:
        print("    ⚠️ 无法进行相关性分析：数据不重叠")
        
else:
    print(f"\n[3] 未找到NASA数据文件: {weather_file}")

print("\n" + "="*60)
print("诊断完成")
print("="*60)
