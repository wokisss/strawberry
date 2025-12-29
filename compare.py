# === 导入所有需要的库 ===
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# --- [新增] PyTorch 库 ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 【新增】设置中文字体 (Windows 专用)
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False   # 让负号正常显示

# --- 步骤 0: 初始设置 ---
filename = 'Strawberry Greenhouse Environmental Control Dataset(version2).csv'
print(f"--- 开始处理文件: '{filename}' ---")

try:
    # --- 步骤 1: 数据加载与初步清洗 ---
    print("--> 正在加载数据...")
    df = pd.read_csv(
        filename,
        encoding='latin1',
        sep=';',
        decimal=',',
        parse_dates=['Timestamp'],
        dayfirst=True,
        index_col='Timestamp'
    )
    print(f"--> 数据加载成功。初始维度: {df.shape}")

    # --- 步骤 2: 数据类型转换与清洗 ---
    print("--> 正在转换数据类型并将非数值列转换为 NaN...")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(axis=1, how='all', inplace=True)
    print(f"--> 清洗后维度 (仅保留数值相关列): {df.shape}")

except FileNotFoundError:
    print(f"错误: 文件 '{filename}' 未找到。请检查文件路径是否正确。")
    sys.exit(1)
except Exception as e:
    print(f"加载或处理数据时发生未知错误: {e}")
    sys.exit(1)

if df.empty:
    print("\n错误: 清洗后没有可用的数值数据进行重采样。")
    sys.exit(1)

# --- 步骤 3: 时间序列重采样 ---
print(f"\n--- 正在将数据按 5 分钟间隔重采样 ---")
df_resampled = df.resample('5min').mean()
df_resampled = df_resampled.ffill().bfill()

print(f"--> 数据重采样完成。")
print(f"--> 清洗后的原始维度: {df.shape}, 重采样后的维度: {df_resampled.shape}")

# ==========================================
# 1. 构建物理基准模型 (Physics-based Baseline)
# ==========================================
print(f"\n--- [阶段1] 构建物理基准模型 (Linear Regression) ---")

# 转换开关状态
cols_to_binary = [' "Heater"', ' "Ventilation"', ' "Lighting"', ' "Pump 1"', ' "Valve 1"']
for col in cols_to_binary:
    if col in df_resampled.columns:
        df_resampled[col] = df_resampled[col].apply(lambda x: 1 if str(x).lower() in ['on', 'yes', '1'] else 0)

# 定义物理相关列
physics_features = [' "Temperature, °C"', ' "Humidity, %"', ' "CO?, ppm"', ' "Heater"', ' "Ventilation"', ' "Lighting"']
target_col = ' "Temperature, °C"' 

available_physics_features = [col for col in physics_features if col in df_resampled.columns]
data_phy = df_resampled[available_physics_features].dropna()

if len(data_phy) == 0:
    print("错误: 没有足够的数据用于物理模型训练。")
    sys.exit(1)

# 构造 T(t) 和 T(t+1)
X_phy_raw = data_phy.iloc[:-1]
y_phy_raw = data_phy[target_col].iloc[1:].values

# 简单的物理特征选择
control_features = [col for col in [' "Heater"', ' "Ventilation"', ' "Lighting"'] if col in available_physics_features and col != target_col]
feature_columns = [target_col] + control_features
if not feature_columns:
    feature_columns = [target_col]

X_phy_features = X_phy_raw[feature_columns].values

# 划分训练/测试集 (80% 训练)
train_size = int(len(X_phy_features) * 0.8)
X_train_phy, X_test_phy = X_phy_features[:train_size], X_phy_features[train_size:]
y_train_phy, y_test_phy = y_phy_raw[:train_size], y_phy_raw[train_size:]

# 训练物理模型
print("---> 正在训练物理基准模型 (基于文献1公式的线性近似)...")
physics_model = LinearRegression()
physics_model.fit(X_train_phy, y_train_phy)
y_pred_phy = physics_model.predict(X_test_phy)
print(f"--> 物理模型训练完成。测试集样本数: {len(y_test_phy)}")


# ==========================================
# 2. 构建混合深度学习模型 (GCBS - PyTorch)
# ==========================================
print(f"\n--- [阶段2] 构建并训练混合深度学习模型 (GCBS) ---")

# A. 数据归一化 (DL模型必须步骤)
scaler = MinMaxScaler()
# 注意：我们对整个 df_resampled 进行归一化
normalized_data = scaler.fit_transform(df_resampled)
normalized_df = pd.DataFrame(normalized_data, columns=df_resampled.columns, index=df_resampled.index)

# 找到目标列 ' "Temperature, °C"' 的索引
target_col_idx = df_resampled.columns.get_loc(target_col)

# B. 构建序列函数 (Seq2Seq)
def create_sequences(data, target_idx, seq_length=24):
    xs, ys = [], []
    data_array = data.values
    for i in range(len(data_array) - seq_length):
        x = data_array[i:(i + seq_length)]
        y = data_array[i + seq_length, target_idx] # 只预测温度
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LEN = 72 # 过去12小时
print(f"--> 生成时间序列样本 (Seq_Len={SEQ_LEN},覆盖过去6小时)...")
X_dl, y_dl = create_sequences(normalized_df, target_col_idx, SEQ_LEN)

# C. 划分数据集 (必须保持与物理模型相同的比例)
split_idx = int(len(X_dl) * 0.8)
X_train_dl, X_test_dl = X_dl[:split_idx], X_dl[split_idx:]
y_train_dl, y_test_dl = y_dl[:split_idx], y_dl[split_idx:]

# 转为 Tensor
train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_dl), torch.FloatTensor(y_train_dl)), batch_size=64, shuffle=True)
test_tensor_X = torch.FloatTensor(X_test_dl)

# D. 定义 GCBS 模型
class GCBS_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=1):
        super(GCBS_Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.bigru = nn.GRU(input_size=64, hidden_size=hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1) # (Batch, Feat, Seq)
        x = torch.relu(self.conv1(x))
        x = x.permute(0, 2, 1) # (Batch, Seq, Feat)
        gru_out, _ = self.bigru(x)
        att_weights = torch.softmax(self.attention(gru_out), dim=1)
        attended = torch.sum(att_weights * gru_out, dim=1)
        return self.fc(attended)

# E. 训练模型
input_dim = X_train_dl.shape[2]
model = GCBS_Model(input_dim=input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("--> 开始训练 PyTorch 模型 (这可能需要几秒钟)...")
num_epochs = 100 
model.train()
for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X).squeeze() 
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 5 == 0:
        print(f"    Epoch {epoch+1}/{num_epochs} Loss: {loss.item():.5f}")

# F. 预测
model.eval()
with torch.no_grad():
    y_pred_dl_norm = model(test_tensor_X).squeeze().numpy()

# G. 反归一化 (将预测的 0-1 值转换回 °C)
def inverse_transform_target(pred_array, scaler, target_col_idx):
    dummy = np.zeros((len(pred_array), scaler.n_features_in_))
    dummy[:, target_col_idx] = pred_array
    res = scaler.inverse_transform(dummy)
    return res[:, target_col_idx]

# 得到真实的混合模型预测值
y_pred_hybrid = inverse_transform_target(y_pred_dl_norm, scaler, target_col_idx)

# ==========================================
# 3. 数据对齐与评估 (修复维度不匹配问题)
# ==========================================
print("\n--- [阶段3] 结果对齐与性能对比 ---")

# 1. 获取两个结果的长度
len_phy = len(y_test_phy)        # 物理模型结果长度 (例如 212)
len_dl = len(y_pred_hybrid)      # 深度学习结果长度 (例如 207)

print(f"--> 原始数据长度: 物理模型={len_phy}, 混合模型={len_dl}")

# 2. 计算最小长度 (取交集)
min_len = min(len_phy, len_dl)

# 3. 【关键步骤】截取最后 min_len 个数据，确保两者对应的时刻完全一致
# [-min_len:] 表示取数组的最后 N 个元素
y_true_final = y_test_phy[-min_len:]           # 真实的温度值 (基准)
y_pred_phy_final = y_pred_phy[-min_len:]       # 物理模型的预测 (截取后)
y_pred_hybrid_final = y_pred_hybrid[-min_len:] # 混合模型的预测 (截取后)

print(f"--> 对齐后数据长度: {len(y_true_final)}")

# 4. 定义评估函数
def calculate_metrics(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"[{name}] MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    return mae, rmse, r2

# 5. 计算指标 (现在传入的数组长度是一致的，不会报错了)
m1 = calculate_metrics(y_true_final, y_pred_phy_final, "基准: 物理微分方程模型")
m2 = calculate_metrics(y_true_final, y_pred_hybrid_final, "本文: 混合深度学习模型")

# ==========================================
# 4. 可视化
# ==========================================
plt.figure(figsize=(12, 6))

# 设置可视化的区间 (防止数据太少报错)
view_len = 200
# 如果数据总长度小于 200，就显示全部
real_view_len = min(view_len, min_len) 

# 只显示最后一段数据
start_idx = 0 
end_idx = real_view_len

# 生成 X 轴坐标
x_axis = np.arange(start_idx, end_idx)

# 截取用于画图的数据片段
y_true_plot = y_true_final[start_idx:end_idx]
y_phy_plot = y_pred_phy_final[start_idx:end_idx]
y_hybrid_plot = y_pred_hybrid_final[start_idx:end_idx]

plt.plot(x_axis, y_true_plot, label='真实值 (Ground Truth)', color='black', linewidth=1.5, alpha=0.7)
plt.plot(x_axis, y_phy_plot, label='物理模型预测 (线性基准)', color='blue', linestyle='--', linewidth=1.5)
plt.plot(x_axis, y_hybrid_plot, label='混合模型预测 (本文方法)', color='red', linewidth=1.5)

plt.title(f"温室温度预测对比: 物理基准 vs 混合模型\n(基于真实训练数据)", fontsize=14)
plt.xlabel("时间步 (Time Step)", fontsize=12)
plt.ylabel("温度 (°C)", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# 添加性能提升说明
if m1[0] != 0:
    improvement = (m1[0] - m2[0]) / m1[0] * 100
    plt.text(start_idx, np.min(y_true_plot), 
             f"物理模型 MAE: {m1[0]:.2f}\n混合模型 MAE: {m2[0]:.2f}\n精度提升: {improvement:.1f}%", 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='red'))

plt.tight_layout()
plt.show()

# ==========================================
# 5. 输出结论
# ==========================================
print("\n--- 结论分析 ---")
print(f"1. 物理模型 (Linear Regression) R2={m1[2]:.4f}。")
print(f"   由于数据中缺失 'Heater' 等控制变量，物理模型退化为仅依靠历史温度的自回归，精度有限。")
print(f"2. 混合模型 (PyTorch GCBS) MAE 降低了 {((m1[0]-m2[0])/m1[0]*100):.1f}%。")
print(f"   通过 CNN 提取特征和 BiGRU 捕捉时间依赖，能更好地拟合非线性变化。")