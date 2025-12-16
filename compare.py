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
# 定义要处理的数据文件名
filename = 'Strawberry Greenhouse Environmental Control Dataset(version2).csv'

# 打印程序开始的提示信息
print(f"--- 开始处理文件: '{filename}' ---")

try:
    # --- 步骤 1: 数据加载与初步清洗 ---
    # 使用 try-except 块来捕获可能发生的错误，例如文件不存在或格式问题
    print("--> 正在加载数据...")
    # 使用 pandas 的 read_csv 函数读取数据，并根据文件的特性进行参数配置
    df = pd.read_csv(
        filename,
        encoding='latin1', # 指定 'latin1' 编码来避免 'utf-8' 解码错误
        sep=';',  # 指定分号为列分隔符
        decimal=',',  # 指定逗号为小数点的分隔符
        parse_dates=['Timestamp'],  # 指定 'Timestamp' 列为日期时间类型进行解析
        dayfirst=True,  # 解析日期时，将天放在前面 (例如 DD.MM.YYYY)
        index_col='Timestamp'  # 将 'Timestamp' 列设置成 DataFrame 的索引，便于时间序列分析
    )
    print(f"--> 数据加载成功。初始维度: {df.shape}")

    # --- 步骤 2: 数据类型转换与清洗 ---
    print("--> 正在转换数据类型并将非数值列转换为 NaN...")
    # 遍历所有列，尝试将它们转换为数值类型
    # errors='coerce' 参数会把无法转换的文本（如 'On'/'Off'）强制替换为 NaN (Not a Number)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 删除在转换后所有值都变成 NaN 的列 (这些通常是纯文本列，无法用于数学计算)
    df.dropna(axis=1, how='all', inplace=True)
    print(f"--> 清洗后维度 (仅保留数值相关列): {df.shape}")


except FileNotFoundError:
    # 如果文件不存在，打印错误信息并退出程序
    print(f"错误: 文件 '{filename}' 未找到。请检查文件路径是否正确。")
    sys.exit(1) # 状态码 1 表示异常退出
except Exception as e:
    # 捕获其他所有在加载和处理过程中可能发生的异常
    print(f"加载或处理数据时发生未知错误: {e}")
    sys.exit(1) # 异常退出


# 确认经过清洗后，DataFrame 中是否还有可用的数据
if df.empty:
    print("\n错误: 清洗后没有可用的数值数据进行重采样。")
    sys.exit(1)

# --- 步骤 3: 时间序列重采样 ---
print(f"\n--- 正在将数据按 5 分钟间隔重采样 ---")
# 使用 '5min' 作为时间频率
# .mean() 表示使用均值来聚合这 5 分钟内的所有数据点
df_resampled = df.resample('5min').mean()

# --- 步骤 4: 填补缺失值 ---
# 重采样后可能会产生因为某个时间段内没有数据而导致的 NaN 值
# 使用 .ffill() (forward-fill) 方法，用前一个有效值来填充 NaN
# 接着链式调用 .bfill() (backward-fill) 方法，用后一个有效值来填充剩余的 NaN (这替代了旧的 .fillna(method=...) 写法)
df_resampled = df_resampled.ffill().bfill()

print(f"--> 数据重采样完成。")
print(f"--> 清洗后的原始维度: {df.shape}, 重采样后的维度: {df_resampled.shape}")

# ==========================================
# 1. 数据加载与预处理 (复用你现有的逻辑)
# ==========================================
print(f"\n--- 正在加载并处理数据: {filename} ---")

# 显示实际的列名
print(f"实际列名: {df_resampled.columns.tolist()}")

# 转换 Yes/No/On/Off 为 0/1
# 根据实际列名调整
cols_to_binary = [' "Heater"', ' "Ventilation"', ' "Lighting"', ' "Pump 1"', ' "Valve 1"']
for col in cols_to_binary:
    if col in df_resampled.columns:
        df_resampled[col] = df_resampled[col].apply(lambda x: 1 if str(x).lower() in ['on', 'yes', '1'] else 0)

# 选择物理方程中涉及的关键变量 (根据文献1公式 6-8)
# T_next = T_curr + f(Heater, Ventilation, Lighting, ...)
# 根据实际列名调整
physics_features = [' "Temperature, °C"', ' "Humidity, %"', ' "CO?, ppm"', ' "Heater"', ' "Ventilation"', ' "Lighting"']
target_col = ' "Temperature, °C"' # 这里以温度为例进行对比

# 检查所需列是否存在
missing_cols = [col for col in physics_features if col not in df_resampled.columns]
if missing_cols:
    print(f"警告: 以下列在数据中未找到: {missing_cols}")

# 只保留在数据中存在的列
available_physics_features = [col for col in physics_features if col in df_resampled.columns]
data = df_resampled[available_physics_features].dropna()

print(f"物理模型使用的特征列: {available_physics_features}")
print(f"数据形状: {data.shape}")

# ==========================================
# 2. 构建物理基准模型 (Physics-based Baseline)
# ==========================================
# 原理：根据文献公式(9) Euler Discretization
# T(t+1) = T(t) + delta_t * (coeff_heater * Heater + coeff_ventilation * Ventilation + ...)
# 因此，我们需要预测的是 delta_T，或者直接回归 T(t+1)

if len(data) == 0:
    print("错误: 没有足够的数据用于物理模型训练。")
    sys.exit(1)

# 构造 T(t) 和 T(t+1)
X_physics = data.iloc[:-1].copy() # t 时刻的状态
y_physics = data[target_col].iloc[1:].values # t+1 时刻的真实温度

# 物理模型的特征不仅包含状态，还应包含这一时刻的"控制动作"
# 比如：温度的变化是由当前的加热器、通风设备状态决定的
control_features = [col for col in [' "Heater"', ' "Ventilation"', ' "Lighting"'] if col in available_physics_features and col != target_col]
feature_columns = [target_col] + control_features

# 确保至少有一些特征列
if not feature_columns:
    feature_columns = [target_col]

X_physics_features = X_physics[feature_columns].values

print(f"物理模型特征列: {feature_columns}")
print(f"X_physics_features 形状: {X_physics_features.shape}")
print(f"y_physics 形状: {y_physics.shape}")

# 检查是否有足够的数据
if len(X_physics_features) == 0 or len(y_physics) == 0:
    print("错误: 没有足够的数据用于物理模型训练。")
    sys.exit(1)

# 划分训练集和测试集 (保持和你深度学习模型一致的比例，这里假设是 80/20)
train_size = int(len(X_physics) * 0.8)
if train_size == 0:
    print("错误: 训练数据不足。")
    sys.exit(1)

X_train_phy, X_test_phy = X_physics_features[:train_size], X_physics_features[train_size:]
y_train_phy, y_test_phy = y_physics[:train_size], y_physics[train_size:]

print("---> 正在训练物理基准模型 (基于文献1公式的线性近似)...")
# 使用线性回归来拟合物理方程的系数
physics_model = LinearRegression()
physics_model.fit(X_train_phy, y_train_phy)

# 预测
y_pred_phy = physics_model.predict(X_test_phy)

# ==========================================
# 3. 构建混合深度学习模型 (GCBS - PyTorch)
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

SEQ_LEN = 24 # 过去12小时
print(f"--> 生成时间序列样本 (Seq_Len={SEQ_LEN})...")
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
num_epochs = 20 
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

y_pred_hybrid = inverse_transform_target(y_pred_dl_norm, scaler, target_col_idx)

# ==========================================
# 4. 评估与可视化 (Evaluation & Visualization)
# ==========================================
print("\n--- [阶段3] 性能评估与可视化 ---")

# A. 数据对齐 (关键步骤)
# 两个模型的测试集长度不同，必须对齐才能比较。我们截取各自预测结果的最后部分，长度以较短的为准。
print(f"--> 正在对齐数据... 物理模型原始测试样本数: {len(y_test_phy)}, 混合模型原始测试样本数: {len(y_pred_hybrid)}")
min_len = min(len(y_test_phy), len(y_pred_hybrid))

# y_true_final 是两个模型共同的真实值参考
y_true_final = y_test_phy[-min_len:]
y_pred_phy_final = y_pred_phy[-min_len:]
y_pred_hybrid_final = y_pred_hybrid[-min_len:]
print(f"--> 对齐后用于比较的样本数: {min_len}")

# B. 性能指标计算
def calculate_metrics(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"[{name}] MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    return mae, rmse, r2

print("\n--- 对齐后性能对比 ---")
m1 = calculate_metrics(y_true_final, y_pred_phy_final, "基准: 物理微分方程模型")
m2 = calculate_metrics(y_true_final, y_pred_hybrid_final, "本文: 混合深度学习模型")

# C. 绘图
plt.figure(figsize=(12, 6))
# 可视化的数据点数量
view_len = 200
start_idx = 0 
if min_len > view_len:
    start_idx = min_len - view_len # 从最后 view_len 个点开始显示

# 截取一段用于显示
x_axis = np.arange(len(y_true_final)) # 创建一个与长度匹配的x轴
y_true_plot = y_true_final[start_idx:]
y_phy_plot = y_pred_phy_final[start_idx:]
y_hybrid_plot = y_pred_hybrid_final[start_idx:]
x_axis_plot = x_axis[start_idx:]


plt.plot(x_axis_plot, y_true_plot, label='真实值 (Ground Truth)', color='black', linewidth=1.5)
plt.plot(x_axis_plot, y_phy_plot, label='物理模型预测 (文献1基准)', color='blue', linestyle='--')
plt.plot(x_axis_plot, y_hybrid_plot, label='混合模型预测 (本文方法)', color='red', linewidth=1.5)

plt.title(f"温室温度预测对比: 物理模型 vs 混合模型 (真实训练结果)", fontsize=14)
plt.xlabel("对齐后的时间步 (Aligned Time Step)", fontsize=12)
plt.ylabel("温度 (°C)", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# 添加说明文本
if len(y_true_plot) > 0:
    improvement = (m1[0]-m2[0])/m1[0]*100 if m1[0] != 0 else 0
    text_x_pos = x_axis_plot[0] + len(x_axis_plot) * 0.05 # 放置在绘图区域的左侧
    text_y_pos = np.min(y_true_plot) + (np.max(y_true_plot) - np.min(y_true_plot)) * 0.05
    plt.text(text_x_pos, text_y_pos, 
             f"物理模型 MAE: {m1[0]:.2f}\n混合模型 MAE: {m2[0]:.2f}\n提升: {improvement:.1f}%", 
             bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

# ==========================================
# 5. 输出结论文本
# ==========================================
print("\n--- 结论分析 ---")
print(f"1. 物理模型基于文献1的公式，假设温度变化与加热、通风等控制动作呈线性关系。")
print(f"   得到的 R2 为 {m1[2]:.4f}，说明它能捕捉大概趋势，但忽略了复杂的非线性特征。")
if m2[0] < m1[0]:
    improvement_conclusion = (m1[0]-m2[0])/m1[0]*100
    print(f"2. 混合模型能够记忆历史序列特征并关注关键时刻。")
    print(f"   其预测精度显著优于物理基准，MAE 降低了约 {improvement_conclusion:.1f}% (从 {m1[0]:.2f} 降至 {m2[0]:.2f})。")
else:
    print(f"2. 在本次训练中，混合模型性能(MAE: {m2[0]:.2f})并未超越物理模型(MAE: {m1[0]:.2f})。这可能需要对模型进行进一步调优（如增加训练轮数、调整学习率等）。")
print(f"3. 这证明了用数据驱动方法替代纯物理计算模块，为 MDP 决策提供更精准状态输入的潜力。")