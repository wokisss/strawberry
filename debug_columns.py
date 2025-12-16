
import pandas as pd
import warnings

# The original script showed a UserWarning, so we'll ignore it here for cleaner output.
warnings.filterwarnings("ignore", category=UserWarning)

# We are using the same parameters as in the original script to ensure we see the same result.
filename = 'Strawberry Greenhouse Environmental Control Dataset(version2).csv'
print(f"--- 正在加载文件: {filename} ---")
try:
    df = pd.read_csv(filename, encoding='latin1', sep=';', decimal=',')
    print("--- 文件加载成功. 以下是检测到的列名: ---")
    print(list(df.columns))
except FileNotFoundError:
    print(f"错误: 文件 '{filename}' 未找到。请确保它在正确的路径下。")
except Exception as e:
    print(f"加载文件时发生错误: {e}")
