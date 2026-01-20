# -*- coding: utf-8 -*-
"""
counterfactual_augmentation.py
------------------------------
基于物理模型的反事实数据扩增 (A2方案)
该模块用于生成符合物理规律的反事实数据，以扩展训练分布，解决观测数据中的虚假相关性问题。
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import copy

class PhysicsBasedCounterfactualGenerator:
    """
    基于物理常识的反事实数据生成器
    
    原理:
    对于每条观测记录 (State_t, Action_t, NextState_t+1)，
    我们尝试所有可能的替代动作 Action' (Action' != Action_t)，
    并利用物理模型推演如果当时采取 Action'，NextState' 会是多少。
    """
    def __init__(self, feature_order, target_col='Temperature, °C', 
                 outdoor_temp_col='Outdoor_Temp', solar_col='Outdoor_Solar',
                 scaler=None):
        """
        :param feature_order: 特征列表，需与训练时一致
        :param target_col: 目标变量列名 (通常是室内温度)
        :param outdoor_temp_col: 室外温度列名
        :param solar_col: 太阳辐射列名
        :param scaler: 已拟合的 sklearn MinMaxScaler (用于反归一化物理计算)
        """
        self.feature_order = feature_order
        self.target_col = target_col
        self.outdoor_temp_col = outdoor_temp_col
        self.solar_col = solar_col
        self.scaler = scaler
        
        # 寻找关键列的索引
        self._find_indices()
        
        # 预定义物理参数 (与 mpc_system.py 中的 PhysicsGreenhouseEnv 保持一致)
        self.params = {
            'k_insulation': 0.05,
            'power_heater': 0.5,
            'eff_vent': 0.1,
            'k_solar': 0.01,
            'noise_std': 0.05
        }
        
    def _find_indices(self):
        try:
            self.idx_target = self.feature_order.index(self.target_col)
            # 假设 Heater, Ventilation 是前两列 (mpc_system.py 约定)
            self.idx_heater = 0
            self.idx_vent = 1
            
            # 查找室外环境列索引 (模糊匹配)
            self.idx_tout = -1
            self.idx_solar = -1
            
            for i, col in enumerate(self.feature_order):
                c_lower = col.lower()
                if self.outdoor_temp_col.lower() in c_lower or 'outdoor_temp' in c_lower:
                    self.idx_tout = i
                if self.solar_col.lower() in c_lower or 'solar' in c_lower:
                    self.idx_solar = i
            
            if self.idx_tout == -1: 
                print(f"[Warning] 未找到室外温度列 '{self.outdoor_temp_col}'，将使用默认值 15.0")
            if self.idx_solar == -1:
                print(f"[Warning] 未找到太阳辐射列 '{self.solar_col}'，将使用默认值 0.0")
                
        except ValueError as e:
            print(f"[Error] 特征索引查找失败: {e}")
            raise

    def physics_step(self, current_temp, action, outside_temp, solar_radiation):
        """ 物理方程推演 (核心逻辑) """
        # dT = -k*(Tin - Tout) + P_heat*u_heat - k_vent*u_vent*(Tin - Tout) + k_solar*Solar
        delta_loss = -self.params['k_insulation'] * (current_temp - outside_temp)
        delta_heat = self.params['power_heater'] * action[0]
        delta_vent = -self.params['eff_vent'] * action[1] * (current_temp - outside_temp)
        delta_solar = self.params['k_solar'] * solar_radiation
        
        # 加入随机噪声，模拟真实世界的不确定性
        noise = np.random.normal(0, self.params['noise_std'])
        
        next_temp = current_temp + delta_loss + delta_heat + delta_vent + delta_solar + noise
        return next_temp

    def _inverse_transform_val(self, val_norm, col_idx):
        """ 辅助: 单值反归一化 """
        if self.scaler is None: return val_norm
        dummy = np.zeros((1, len(self.scaler.scale_)))
        dummy[0, col_idx] = val_norm
        return self.scaler.inverse_transform(dummy)[0, col_idx]

    def _transform_val(self, val_real, col_idx):
        """ 辅助: 单值归一化 """
        if self.scaler is None: return val_real
        dummy = np.zeros((1, len(self.scaler.scale_)))
        dummy[0, col_idx] = val_real
        return self.scaler.transform(dummy)[0, col_idx]

    def generate_counterfactuals(self, X_past, X_future, y_real, future_indices):
        """
        为给定的 batch 数据生成反事实样本
        
        :param X_past: (batch, seq_len, feat_dim) 历史数据
        :param X_future: (batch, horizon, future_dim) 未来输入 (含动作)
        :param y_real: (batch, ) 真实目标值 (归一化后)
        :param future_indices: 指示 X_future 中哪些列对应 feature_order 的哪些索引
        :return: 扩增后的 X_past, X_future, y
        """
        
        # 结果容器
        aug_X_past = []
        aug_X_future = []
        aug_y = []
        
        # 定义可能的动作空间 [Heater, Vent]
        # 0:自然(0,0), 1:加热(1,0), 2:通风(0,1)
        possible_actions = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ]
        
        batch_size = len(X_past)
        
        # 遍历 batch 中的每个样本
        for i in range(batch_size):
            # 1. 提取当前状态信息
            # 历史序列的最后一帧作为 "Current State"
            curr_state_vec = X_past[i, -1, :] # (feat_dim, )
            
            # 获取反归一化的物理量
            T_in_norm = curr_state_vec[self.idx_target]
            T_in = self._inverse_transform_val(T_in_norm, self.idx_target)
            
            # 获取室外环境 (假设在历史最后时刻)
            if self.idx_tout != -1:
                T_out = self._inverse_transform_val(curr_state_vec[self.idx_tout], self.idx_tout)
            else:
                T_out = 15.0 # default
                
            if self.idx_solar != -1:
                Solar = self._inverse_transform_val(curr_state_vec[self.idx_solar], self.idx_solar)
            else:
                Solar = 0.0 # default
            
            # 2. 识别此时发生的真实动作 (从 X_future 的第一步获取，或 X_past 最后一步?)
            # 通常 MPC 训练中，y 对应的是 X_future 第一步之后的输出，
            # 所以动作应该在 X_future[i, 0, control_indices]
            # 这里的 future_indices[0], future_indices[1] 应该是 Heater, Vent
            
            real_action_heat = X_future[i, 0, 0] # 假设 future 的第0列是 Heater
            real_action_vent = X_future[i, 0, 1] # 假设 future 的第1列是 Vent
            
            # 简单判断当前动作类型 (处理浮点误差)
            current_action_code = -1
            if real_action_heat > 0.5: current_action_code = 1 # Heating
            elif real_action_vent > 0.5: current_action_code = 2 # Venting
            else: current_action_code = 0 # Natural
            
            # 3. 生成反事实 (Counterfactuals)
            for act_code, action_vec in enumerate(possible_actions):
                if act_code == current_action_code:
                    continue # 跳过真实发生的情况
                
                # 推演反事实结果
                T_next_cf = self.physics_step(T_in, action_vec, T_out, Solar)
                
                # 物理一致性校验 (简单的边界检查)
                if not (0.0 <= T_next_cf <= 50.0):
                    continue # 丢弃物理上不合理的数据
                
                # 归一化结果
                y_cf_norm = self._transform_val(T_next_cf, self.idx_target)
                
                # 构造反事实样本
                # X_past 保持不变 (历史是一样的)
                aug_X_past.append(X_past[i])
                
                # X_future 需要修改动作部分
                # 注意：假设 Horizon=1 或者我们在整个 Horizon 都应用此动作?
                # 简单起见，仅修改第一步，或者全部修改。这里修改全部以保持一致性。
                x_future_new = X_future[i].copy() 
                x_future_new[:, 0] = action_vec[0]
                x_future_new[:, 1] = action_vec[1]
                
                aug_X_future.append(x_future_new)
                aug_y.append(y_cf_norm)
                
        # 转换为 numpy/tensor
        if len(aug_X_past) > 0:
            return np.array(aug_X_past), np.array(aug_X_future), np.array(aug_y)
        else:
            return np.array([]), np.array([]), np.array([])

class CounterfactualDataset(Dataset):
    """ 混合数据集: 原始数据 + 反事实数据 """
    def __init__(self, X_past, X_future, y, generator, future_indices, cf_ratio=0.5):
        """
        :param cf_ratio: 反事实数据占比 (0.0 - 1.0)
        """
        self.X_past_real = X_past
        self.X_future_real = X_future
        self.y_real = y
        
        print("--> [A2] 正在生成物理反事实数据...")
        X_p_cf, X_f_cf, y_cf = generator.generate_counterfactuals(
            X_past, X_future, y, future_indices
        )
        
        self.num_real = len(X_past)
        self.num_cf = len(X_p_cf) if len(X_p_cf) > 0 else 0
        
        print(f"    真实样本数: {self.num_real}")
        print(f"    生成反事实样本数: {self.num_cf}")
        
        # 数据合并
        if self.num_cf > 0:
            # 根据 ratio 进行采样，如果要控制比例的话
            # 这里简单做全量合并，或者截断
            target_cf_count = int(self.num_real * cf_ratio)
            if self.num_cf > target_cf_count:
                indices = np.random.choice(self.num_cf, target_cf_count, replace=False)
                X_p_cf = X_p_cf[indices]
                X_f_cf = X_f_cf[indices]
                y_cf = y_cf[indices]
                self.num_cf = len(X_p_cf)
            
            self.X_past = np.concatenate([self.X_past_real, X_p_cf], axis=0)
            self.X_future = np.concatenate([self.X_future_real, X_f_cf], axis=0)
            self.y = np.concatenate([self.y_real, y_cf], axis=0)
        else:
            self.X_past = self.X_past_real
            self.X_future = self.X_future_real
            self.y = self.y_real
            
        # 转换为 Tensor
        self.X_past = torch.FloatTensor(self.X_past)
        self.X_future = torch.FloatTensor(self.X_future)
        self.y = torch.FloatTensor(self.y)
        
        print(f"    最终训练集大小: {len(self.y)} (反事实占比: {self.num_cf / len(self.y):.1%})")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_past[idx], self.X_future[idx], self.y[idx]

# ==============================================================================
# 单元测试 / 演示代码
# ==============================================================================
if __name__ == '__main__':
    print("Running Counterfactual Augmentation Demo...")
    
    # 1. 模拟一些数据
    batch_size = 10
    seq_len = 60
    feat_dim = 10
    horizon = 1
    future_dim = 6 # [Heater, Vent, Tout, Solar, ...]
    
    # 模拟特征列名
    feature_order = ['Heater', 'Ventilation', 'Lighting', 'Temperature, °C', 
                     'Humidity, %', 'CO2, ppm', 'Outdoor_Temp', 'Outdoor_Solar', 
                     'Outdoor_Hum', 'Outdoor_Wind']
    
    # 模拟数据
    X_past_mock = np.random.rand(batch_size, seq_len, feat_dim)
    X_future_mock = np.zeros((batch_size, horizon, future_dim)) 
    # 让 Heater=1 (Heating)
    X_future_mock[:, :, 0] = 1.0 
    
    y_mock = np.random.rand(batch_size)
    
    # 模拟 Scaler
    scaler_mock = MinMaxScaler()
    scaler_mock.fit(np.random.rand(100, feat_dim) * 30) # fit with some range
    
    # 2. 初始化生成器
    generator = PhysicsBasedCounterfactualGenerator(
        feature_order=feature_order,
        target_col='Temperature, °C',
        outdoor_temp_col='Outdoor_Temp',
        solar_col='Outdoor_Solar',
        scaler=scaler_mock
    )
    
    # 3. 生成反事实数据
    # future_indices 对应 future_mock 中的列在 feature_order 中的位置
    # 假设 future_mock 的列就是 feature_order 的前 future_dim 个（简化）
    # 在真实 mpc_system.py 中这个 indices 是跳跃的
    future_indices = [0, 1, 6, 7, 8, 9] 
    
    # 注意：generate_counterfactuals 内部并没有用到 future_indices 来反查列，
    # 而是假设 X_future 的前两列 是 Heater, Vent。
    # 为了严谨，建议在 Generator 里明确这一点，或者传入 indices。
    # 目前代码实现假设前两列是控制量。
    
    X_p_aug, X_f_aug, y_aug = generator.generate_counterfactuals(
        X_past_mock, X_future_mock, y_mock, future_indices
    )
    
    # 4. 验证结果
    if len(X_p_aug) > 0:
        print("\n[Validation Pass]")
        print(f"Output Shapes: {X_p_aug.shape}, {X_f_aug.shape}, {y_aug.shape}")
        
        # 检查动作是否改变
        # 原来全是 Heater=1 (Heating)
        # 反事实应该包含 Heater=0, Vent=0 (Natural) 和 Heater=0, Vent=1 (Vent)
        print("Sample Augmentation Action (Should be different from [1, 0]):")
        print(X_f_aug[0, 0, :2])
    else:
        print("\n[Validation Fail] No counterfactuals generated.")
