# -*- coding: utf-8 -*-
"""
V42: 滑动窗口 + 多步预测版本
- 使用前N个时间步预测未来K步
- 支持可配置的窗口大小和预测步长
- 滚动预测机制
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============== 配置参数 ==============
# 实验配置列表：自动运行多组实验
EXPERIMENT_CONFIGS = [
    {
        'name': '单步预测_jump模式',
        'window_sizes': [3, 5, 7, 10],
        'pred_steps_list': [1],
        'mode': 'jump'
    },
    {
        'name': '多步预测_weighted模式',
        'window_sizes': [3, 5, 7, 10],
        'pred_steps_list': [2, 3],
        'mode': 'weighted'
    }
]

# 目标变量（用于评估）
TARGET_VARIABLES = ['空气侧压降', '平均风速', '单位时间结霜量', '空气侧换热量', '水侧换热量']

# ============== 数据加载 ==============
def load_all_data():
    """加载所有工况数据"""
    base_path = r'c:\Users\19396\Desktop\frost'
    all_conditions = []
    
    condition_folders = [
        '插排-5mm-1', '插排-7.5mm-0.5', '插排-7.5mm-1', '插排-10mm-0.5', '插排-10mm-1',
        '插排-15mm-0.5', '插排-15mm-1', '插排-20mm-0.5', '插排-20mm-1',
        '插排-30mm-0.5', '插排-30mm-1'
    ]
    
    for folder in condition_folders:
        xlsx_path = os.path.join(base_path, folder, f"{folder}.xlsx")
        if os.path.exists(xlsx_path):
            df = pd.read_excel(xlsx_path)
            
            features = ['空气侧压降', '平均风速', '单位时间结霜量', '空气侧换热量', '水侧换热量',
                       '进口温度', '进口湿度', '进口热电偶温度', '换热器进口温度',
                       '出口平均温度', '出口平均湿度', '换热器出口温度']
            
            data = df[features].values
            
            parts = folder.replace('插排-', '').replace('mm', '').split('-')
            fin_spacing = float(parts[0])
            flow_rate = float(parts[1])
            
            all_conditions.append({
                'name': folder,
                'data': data,
                'fin_spacing': fin_spacing,
                'flow_rate': flow_rate
            })
    
    return all_conditions

# ============== 滑动窗口数据集类 ==============
class SlidingWindowDataset(Dataset):
    def __init__(self, conditions, window_size=5, pred_steps=1, 
                 state_scaler=None, condition_scaler=None):
        self.samples = []
        self.window_size = window_size
        self.pred_steps = pred_steps
        self.state_scaler = state_scaler if state_scaler else StandardScaler()
        self.condition_scaler = condition_scaler if condition_scaler else StandardScaler()
        
        # 收集所有状态数据用于标准化
        all_states = []
        for cond in conditions:
            data = cond['data']
            all_states.append(data)
            
        all_states = np.vstack(all_states)
        
        if state_scaler is None:
            self.state_scaler.fit(all_states)
        
        # 构建滑动窗口样本
        for cond in conditions:
            data = cond['data']
            normalized_data = self.state_scaler.transform(data)
            
            n_steps = len(normalized_data)
            t_normalized = np.linspace(0, 1, n_steps)
            
            fin_spacing = cond['fin_spacing']
            flow_rate = cond['flow_rate']
            initial_wind_speed = data[0, 1]
            
            # 滑动窗口：需要至少window_size + pred_steps个样本
            for i in range(n_steps - window_size - pred_steps + 1):
                # 输入：过去window_size个时间步的状态
                input_sequence = normalized_data[i:i+window_size]
                
                # 目标：未来pred_steps个时间步的状态
                target_sequence = normalized_data[i+window_size:i+window_size+pred_steps]
                
                # 工况特征（使用窗口末尾的时间）
                time_idx = i + window_size - 1
                condition_features = np.array([
                    fin_spacing, 
                    initial_wind_speed, 
                    flow_rate, 
                    t_normalized[time_idx]
                ])
                
                self.samples.append({
                    'input_sequence': input_sequence,
                    'target_sequence': target_sequence,
                    'condition': condition_features,
                })
        
        # 标准化工况特征
        all_cond_features = np.array([s['condition'] for s in self.samples])
        if condition_scaler is None:
            self.condition_scaler.fit(all_cond_features)
        
        for sample in self.samples:
            sample['condition'] = self.condition_scaler.transform(
                sample['condition'].reshape(1, -1)
            )[0]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.FloatTensor(sample['input_sequence']),
            torch.FloatTensor(sample['condition']),
            torch.FloatTensor(sample['target_sequence'])
        )

# ============== 滑动窗口LSTM模型 ==============
class SlidingWindowLSTM(nn.Module):
    def __init__(self, state_dim=12, condition_dim=4, window_size=5, pred_steps=1,
                 hidden_size=192, num_layers=3, dropout=0.2):
        super().__init__()
        
        self.state_dim = state_dim
        self.condition_dim = condition_dim
        self.window_size = window_size
        self.pred_steps = pred_steps
        self.hidden_size = hidden_size
        
        # 状态序列编码器（直接处理序列）
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 工况编码器
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM处理时间序列
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 解码器：预测未来pred_steps个时间步
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size + hidden_size // 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, state_dim * pred_steps)
        )
    
    def forward(self, input_sequence, condition):
        """
        Args:
            input_sequence: [batch, window_size, state_dim]
            condition: [batch, condition_dim]
        Returns:
            output: [batch, pred_steps, state_dim]
        """
        batch_size = input_sequence.size(0)
        
        # 编码每个时间步的状态
        # [batch, window_size, state_dim] -> [batch, window_size, hidden_size]
        encoded_sequence = self.state_encoder(input_sequence)
        
        # LSTM处理序列
        lstm_out, (hidden, cell) = self.lstm(encoded_sequence)
        
        # 使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # [batch, hidden_size]
        
        # 编码工况特征
        cond_encoded = self.condition_encoder(condition)  # [batch, hidden_size//2]
        
        # 融合
        combined = torch.cat([last_output, cond_encoded], dim=-1)
        
        # 解码：预测未来多步
        output = self.decoder(combined)  # [batch, state_dim * pred_steps]
        
        # 重塑为 [batch, pred_steps, state_dim]
        output = output.view(batch_size, self.pred_steps, self.state_dim)
        
        return output

# ============== V42损失函数：MSE + 物理约束 ==============
class V42Loss(nn.Module):
    """V42: 支持多步预测的物理约束损失"""
    def __init__(self, state_scaler, mse_weight=1.0, physics_weight=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.state_scaler = state_scaler
        self.mse_weight = mse_weight
        self.physics_weight = physics_weight
    
    def forward(self, pred, target, input_sequence):
        """
        Args:
            pred: [batch, pred_steps, state_dim]
            target: [batch, pred_steps, state_dim]
            input_sequence: [batch, window_size, state_dim]
        """
        batch_size = pred.size(0)
        pred_steps = pred.size(1)
        state_dim = pred.size(2)

        # 目标变量权重（与加权R²一致）
        var_weights = {
            2: 0.50,  # 单位时间结霜量
            1: 0.15,  # 平均风速
            0: 0.15,  # 空气侧压降
            3: 0.10,  # 空气侧换热量
            4: 0.10   # 水侧换热量
        }
        # 步权重（多步时，距离越近权重越大，归一化）
        step_weights = np.array([1.0 / (k + 1) for k in range(pred_steps)])
        step_weights = step_weights / step_weights.sum()

        # 单步预测：直接MSE+物理约束
        if pred_steps == 1:
            mse_loss = self.mse(pred, target)
            # 物理约束同原实现
            pred_step = pred[:, 0, :]
            pred_denorm = torch.FloatTensor(
                self.state_scaler.inverse_transform(pred_step.detach().cpu().numpy())
            ).to(pred.device)
            prev_step = input_sequence[:, -1, :]
            prev_denorm = torch.FloatTensor(
                self.state_scaler.inverse_transform(prev_step.detach().cpu().numpy())
            ).to(pred.device)
            # 能量守恒
            air_heat = pred_denorm[:, 3]
            water_heat = pred_denorm[:, 4]
            heat_balance = torch.mean((air_heat - water_heat) ** 2) / (torch.mean(air_heat ** 2) + 1e-6)
            # 单调性约束（结霜量递增）
            frost_prev = prev_denorm[:, 2]
            frost_pred = pred_denorm[:, 2]
            frost_decrease = torch.relu(frost_prev - frost_pred)
            frost_monotonicity = torch.mean(frost_decrease ** 2)
            # 压力增长约束
            pressure_prev = prev_denorm[:, 0]
            pressure_pred = pred_denorm[:, 0]
            pressure_decrease = torch.relu(pressure_prev - pressure_pred) * 0.5
            pressure_growth = torch.mean(pressure_decrease ** 2)
            # 温度合理性
            temp_penalty = 0.0
            temp_vars = [(5, 0.1), (7, 0.1), (8, 0.1), (9, 0.1), (11, 0.1)]
            for idx, weight in temp_vars:
                temp_diff = torch.abs(pred_denorm[:, idx] - prev_denorm[:, idx])
                temp_penalty += weight * torch.mean(torch.relu(temp_diff - 5.0) ** 2)
            physics_loss = heat_balance + frost_monotonicity + pressure_growth + temp_penalty
            total_loss = self.mse_weight * mse_loss + self.physics_weight * physics_loss
            return total_loss

        # 多步预测：目标变量加权MSE + 步加权 + 物理约束步加权
        mse_losses = []
        physics_losses = []
        for step in range(pred_steps):
            pred_step = pred[:, step, :]  # [batch, state_dim]
            target_step = target[:, step, :]
            # 目标变量加权MSE
            mse_step = 0.0
            for idx, w in var_weights.items():
                mse_step += w * self.mse(pred_step[:, idx], target_step[:, idx])
            mse_losses.append(mse_step)

            # 物理约束
            pred_denorm = torch.FloatTensor(
                self.state_scaler.inverse_transform(pred_step.detach().cpu().numpy())
            ).to(pred.device)
            if step == 0:
                prev_step = input_sequence[:, -1, :]
            else:
                prev_step = pred[:, step-1, :]
            prev_denorm = torch.FloatTensor(
                self.state_scaler.inverse_transform(prev_step.detach().cpu().numpy())
            ).to(pred.device)
            air_heat = pred_denorm[:, 3]
            water_heat = pred_denorm[:, 4]
            heat_balance = torch.mean((air_heat - water_heat) ** 2) / (torch.mean(air_heat ** 2) + 1e-6)
            frost_prev = prev_denorm[:, 2]
            frost_pred = pred_denorm[:, 2]
            frost_decrease = torch.relu(frost_prev - frost_pred)
            frost_monotonicity = torch.mean(frost_decrease ** 2)
            pressure_prev = prev_denorm[:, 0]
            pressure_pred = pred_denorm[:, 0]
            pressure_decrease = torch.relu(pressure_prev - pressure_pred) * 0.5
            pressure_growth = torch.mean(pressure_decrease ** 2)
            temp_penalty = 0.0
            temp_vars = [(5, 0.1), (7, 0.1), (8, 0.1), (9, 0.1), (11, 0.1)]
            for idx2, weight in temp_vars:
                temp_diff = torch.abs(pred_denorm[:, idx2] - prev_denorm[:, idx2])
                temp_penalty += weight * torch.mean(torch.relu(temp_diff - 5.0) ** 2)
            physics_loss = heat_balance + frost_monotonicity + pressure_growth + temp_penalty
            physics_losses.append(physics_loss)

        # 步加权平均
        mse_loss = sum([w * l for w, l in zip(step_weights, mse_losses)])
        avg_physics_loss = sum([w * l for w, l in zip(step_weights, physics_losses)])
        total_loss = self.mse_weight * mse_loss + self.physics_weight * avg_physics_loss
        return total_loss

# ============== 训练函数 ==============
def train_model(model, train_loader, criterion, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (input_seq, condition, target_seq) in enumerate(train_loader):
        input_seq = input_seq.to(device)
        condition = condition.to(device)
        target_seq = target_seq.to(device)
        
        optimizer.zero_grad()
        
        pred = model(input_seq, condition)
        loss = criterion(pred, target_seq, input_seq)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    
    return avg_loss

# ============== 评估指标计算 ==============
def calculate_target_metrics(data, predictions, target_vars=TARGET_VARIABLES):
    """
    计算目标变量的综合评估指标
    
    Returns:
        dict: 包含各种评估指标
    """
    feature_names = ['空气侧压降', '平均风速', '单位时间结霜量', '空气侧换热量', '水侧换热量',
                   '进口温度', '进口湿度', '进口热电偶温度', '换热器进口温度',
                   '出口平均温度', '出口平均湿度', '换热器出口温度']
    
    metrics = {}
    target_r2_scores = []
    
    # 计算每个变量的R²
    for var in feature_names:
        idx = feature_names.index(var)
        r2 = r2_score(data[:, idx], predictions[:, idx])
        metrics[var] = r2
        
        # 如果是目标变量，记录下来
        if var in target_vars:
            target_r2_scores.append(r2)
    
    # 综合指标
    metrics['目标变量平均R²'] = np.mean(target_r2_scores)
    metrics['目标变量最小R²'] = np.min(target_r2_scores)
    metrics['目标变量几何平均R²'] = np.exp(np.mean(np.log(np.maximum(target_r2_scores, 1e-10))))
    
    # 加权平均（根据重要性调整权重）
    weights = {
        '单位时间结霜量': 0.50,
        '平均风速': 0.15,
        '空气侧压降': 0.15,
        '空气侧换热量': 0.10,
        '水侧换热量': 0.10
    }
    weighted_r2 = sum(metrics[var] * weights[var] for var in target_vars)
    metrics['目标变量加权R²'] = weighted_r2
    
    # 综合评分（考虑平均和最小值，防止某个变量过差）
    metrics['综合评分'] = 0.7 * metrics['目标变量平均R²'] + 0.3 * metrics['目标变量最小R²']
    
    return metrics

# ============== 滚动预测函数（支持多种融合策略）==============
def rolling_prediction(model, data, state_scaler, condition_scaler, 
                       fin_spacing, flow_rate, initial_wind_speed,
                       window_size, pred_steps, mode='jump'):
    """
    滚动预测：使用滑动窗口
    
    Args:
        mode: 预测模式
            'jump': 窗口每次移动pred_steps步（无重叠，效率高）
            'weighted': 窗口每次移动1步，加权平均多次预测（权重归一化，距离越近权重越大）
    
    注: weighted模式使用归一化权重（权重和=1），避免数值偏差
    """
    model.eval()
    
    n_steps = len(data)
    predictions = np.zeros_like(data)
    
    # 初始窗口使用真实值
    predictions[:window_size] = data[:window_size]
    
    t_normalized = np.linspace(0, 1, n_steps)
    
    if mode == 'jump':
        # ========== 跳跃式预测（原实现）==========
        with torch.no_grad():
            current_pos = window_size
            
            while current_pos < n_steps:
                window_start = current_pos - window_size
                current_window = predictions[window_start:current_pos]
                
                window_normalized = state_scaler.transform(current_window)
                window_tensor = torch.FloatTensor(window_normalized).unsqueeze(0).to(device)
                
                time_idx = current_pos - 1
                condition_features = np.array([[
                    fin_spacing, initial_wind_speed, flow_rate, t_normalized[time_idx]
                ]])
                condition_features = condition_scaler.transform(condition_features)
                condition_tensor = torch.FloatTensor(condition_features).to(device)
                
                pred = model(window_tensor, condition_tensor)
                pred_np = pred.cpu().numpy()[0]
                pred_denorm = state_scaler.inverse_transform(pred_np)
                
                steps_to_fill = min(pred_steps, n_steps - current_pos)
                predictions[current_pos:current_pos+steps_to_fill] = pred_denorm[:steps_to_fill]
                
                # 跳跃式：每次移动pred_steps步
                current_pos += pred_steps
    
    elif mode == 'weighted':
        # ========== 加权滑动式预测（窗口每次移动1步，权重归一化）==========
        # 用于累积多次预测
        prediction_accumulator = np.zeros((n_steps, data.shape[1]))
        prediction_weights = np.zeros(n_steps)  # 累积权重和
        
        with torch.no_grad():
            current_pos = window_size
            
            while current_pos < n_steps:
                window_start = current_pos - window_size
                current_window = predictions[window_start:current_pos]
                
                window_normalized = state_scaler.transform(current_window)
                window_tensor = torch.FloatTensor(window_normalized).unsqueeze(0).to(device)
                
                time_idx = current_pos - 1
                condition_features = np.array([[
                    fin_spacing, initial_wind_speed, flow_rate, t_normalized[time_idx]
                ]])
                condition_features = condition_scaler.transform(condition_features)
                condition_tensor = torch.FloatTensor(condition_features).to(device)
                
                pred = model(window_tensor, condition_tensor)
                pred_np = pred.cpu().numpy()[0]
                pred_denorm = state_scaler.inverse_transform(pred_np)
                
                # 计算归一化权重（关键修正！）
                # 原始权重：1/(step+1)，即 [1.0, 0.5, 0.33, ...]
                raw_weights = np.array([1.0 / (step + 1) for step in range(pred_steps)])
                # 归一化：使权重和=1
                normalized_weights = raw_weights / np.sum(raw_weights)
                
                # 填充预测结果（使用归一化权重）
                for step_offset in range(pred_steps):
                    target_pos = current_pos + step_offset
                    if target_pos >= n_steps:
                        break
                    
                    weight = normalized_weights[step_offset]
                    prediction_accumulator[target_pos] += pred_denorm[step_offset] * weight
                    prediction_weights[target_pos] += weight
                
                # 滑动式：每次移动1步
                current_pos += 1
        
        # 加权平均：计算最终预测
        for i in range(window_size, n_steps):
            if prediction_weights[i] > 0:
                predictions[i] = prediction_accumulator[i] / prediction_weights[i]
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'jump' or 'weighted'.")
    
    return predictions

# ============== 评估函数（返回指标）==============
def evaluate_model_with_metrics(model, data_conditions, state_scaler, condition_scaler, 
                                window_size, pred_steps, output_dir, prediction_mode, prefix=''):
    """评估模型并返回详细指标"""
    model.eval()
    
    feature_names = ['空气侧压降', '平均风速', '单位时间结霜量', '空气侧换热量', '水侧换热量',
                   '进口温度', '进口湿度', '进口热电偶温度', '换热器进口温度',
                   '出口平均温度', '出口平均湿度', '换热器出口温度']
    
    all_results = {}
    
    for cond in data_conditions:
        data = cond['data']
        name = cond['name']
        
        predictions = rolling_prediction(
            model, data, state_scaler, condition_scaler,
            cond['fin_spacing'], cond['flow_rate'], data[0, 1],
            window_size, pred_steps, mode=prediction_mode
        )
        
        # 计算指标
        metrics = calculate_target_metrics(data, predictions)
        all_results[name] = metrics
        
        print(f"\n--- {prefix}{name} ---")
        print(f"  综合评分: {metrics['综合评分']:.4f}")
        print(f"  目标变量平均R²: {metrics['目标变量平均R²']:.4f}")
        print(f"  目标变量最小R²: {metrics['目标变量最小R²']:.4f}")
        for var in TARGET_VARIABLES:
            print(f"  {var}: R² = {metrics[var]:.4f}")
        
        # 绘图
        os.makedirs(output_dir, exist_ok=True)
        
        key_vars = TARGET_VARIABLES
        other_vars = ['进口温度', '进口湿度', '进口热电偶温度', '换热器进口温度',
                     '出口平均温度', '出口平均湿度', '换热器出口温度']
        
        n_rows = max(len(key_vars), len(other_vars))
        fig, axes = plt.subplots(n_rows, 2, figsize=(16, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, var_name in enumerate(key_vars):
            ax = axes[i, 0]
            col_idx = feature_names.index(var_name)
            
            ax.plot(data[:, col_idx], 'blue', label='真实值', linewidth=2)
            ax.plot(predictions[:, col_idx], 'red', label='V42预测', linewidth=2, linestyle='--')
            ax.axvline(x=window_size, color='green', linestyle=':', 
                      label=f'预测起点(窗口={window_size})', alpha=0.7)
            
            ax.set_xlabel('时间步')
            ax.set_ylabel(var_name)
            ax.set_title(f'{var_name} (R² = {metrics[var_name]:.4f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        for i in range(len(key_vars), n_rows):
            axes[i, 0].set_visible(False)
        
        for i, var_name in enumerate(other_vars):
            ax = axes[i, 1]
            col_idx = feature_names.index(var_name)
            
            ax.plot(data[:, col_idx], 'blue', label='真实值', linewidth=2)
            ax.plot(predictions[:, col_idx], 'red', label='V42预测', linewidth=2, linestyle='--')
            ax.axvline(x=window_size, color='green', linestyle=':', 
                      label=f'预测起点(窗口={window_size})', alpha=0.7)
            
            ax.set_xlabel('时间步')
            ax.set_ylabel(var_name)
            ax.set_title(f'{var_name} (R² = {metrics[var_name]:.4f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        for i in range(len(other_vars), n_rows):
            axes[i, 1].set_visible(False)
        
        filename = f'{prefix}{name.replace("-", "_")}.png'
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
    
    return all_results

# ============== 生成Markdown报告 ==============
def generate_markdown_report(all_results, output_path):
    """生成超参数网格搜索的综合报告"""
    from datetime import datetime
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# V42 滑动窗口LSTM 超参数网格搜索报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 评估指标说明\n\n")
        f.write("### 目标变量\n")
        f.write("本模型重点关注以下5个目标变量：\n\n")
        for i, var in enumerate(TARGET_VARIABLES, 1):
            f.write(f"{i}. {var}\n")
        
        f.write("\n### 综合评估指标\n\n")
        f.write("| 指标名称 | 计算方法 | 说明 |\n")
        f.write("|---------|---------|------|\n")
        f.write("| 目标变量平均R² | 算术平均 | 5个目标变量R²的平均值 |\n")
        f.write("| 目标变量最小R² | 最小值 | 防止某个变量表现过差 |\n")
        f.write("| 目标变量加权R² | 加权平均 | 结霜0.50, 风速0.15, 压降0.15, 空气换热0.10, 水侧换热0.10 |\n")
        f.write("| 综合评分 | 0.7×平均R² + 0.3×最小R² | 平衡整体和最差情况 |\n\n")
        
        f.write("---\n\n")
        
        # 按超参数配置组织结果
        for config_key in sorted(all_results.keys()):
            window_size, pred_steps = config_key
            config_results = all_results[config_key]
            
            f.write(f"## 配置: 窗口大小={window_size}, 预测步长={pred_steps}\n\n")
            
            # 汇总所有测试集的综合评分
            f.write("### 各测试集综合评分\n\n")
            f.write("| 测试集 | 综合评分 | 平均R² | 最小R² | 加权R² |\n")
            f.write("|--------|---------|--------|--------|--------|\n")
            
            test_scores = []
            for test_name in sorted(config_results.keys()):
                test_metrics = config_results[test_name]['test']
                # test_metrics是一个字典，key是工况名，value是指标
                for cond_name, metrics in test_metrics.items():
                    score = metrics['综合评分']
                    avg_r2 = metrics['目标变量平均R²']
                    min_r2 = metrics['目标变量最小R²']
                    weighted_r2 = metrics['目标变量加权R²']
                    
                    f.write(f"| {cond_name} | {score:.4f} | {avg_r2:.4f} | {min_r2:.4f} | {weighted_r2:.4f} |\n")
                    test_scores.append(score)
            
            # 整体平均
            overall_score = np.mean(test_scores)
            f.write(f"| **平均** | **{overall_score:.4f}** | - | - | - |\n\n")
            
            # 详细的目标变量R²表
            f.write("### 各测试集目标变量R²详情\n\n")
            f.write("| 测试集 |")
            for var in TARGET_VARIABLES:
                f.write(f" {var} |")
            f.write("\n|--------|")
            for _ in TARGET_VARIABLES:
                f.write("--------|")
            f.write("\n")
            
            for test_name in sorted(config_results.keys()):
                test_metrics = config_results[test_name]['test']
                for cond_name, metrics in test_metrics.items():
                    f.write(f"| {cond_name} |")
                    for var in TARGET_VARIABLES:
                        r2 = metrics[var]
                        f.write(f" {r2:.4f} |")
                    f.write("\n")
            
            f.write("\n---\n\n")
        
        # 超参数对比总结
        f.write("## 超参数配置对比总结\n\n")
        f.write("| 窗口大小 | 预测步长 | 平均综合评分 | 最佳测试集评分 | 最差测试集评分 |\n")
        f.write("|---------|---------|-------------|---------------|---------------|\n")
        
        summary_data = []
        for config_key in sorted(all_results.keys()):
            window_size, pred_steps = config_key
            config_results = all_results[config_key]
            
            all_scores = []
            for test_name in config_results.keys():
                test_metrics = config_results[test_name]['test']
                for cond_name, metrics in test_metrics.items():
                    all_scores.append(metrics['综合评分'])
            
            avg_score = np.mean(all_scores)
            best_score = np.max(all_scores)
            worst_score = np.min(all_scores)
            
            summary_data.append({
                'window': window_size,
                'pred_steps': pred_steps,
                'avg': avg_score,
                'best': best_score,
                'worst': worst_score
            })
            
            f.write(f"| {window_size} | {pred_steps} | {avg_score:.4f} | {best_score:.4f} | {worst_score:.4f} |\n")
        
        # 找出最佳配置
        best_config = max(summary_data, key=lambda x: x['avg'])
        f.write(f"\n**最佳配置**: 窗口大小={best_config['window']}, 预测步长={best_config['pred_steps']}, ")
        f.write(f"平均综合评分={best_config['avg']:.4f}\n\n")
        
        f.write("---\n\n")
        f.write("**注**: 综合评分越高越好，R²值范围为[-∞, 1]，1表示完美预测\n")

# ============== 评估函数（旧版，保留兼容）==============
def evaluate_model(model, data_conditions, state_scaler, condition_scaler, 
                   window_size, pred_steps, output_dir, prefix=''):
    model.eval()
    
    feature_names = ['空气侧压降', '平均风速', '单位时间结霜量', '空气侧换热量', '水侧换热量',
                   '进口温度', '进口湿度', '进口热电偶温度', '换热器进口温度',
                   '出口平均温度', '出口平均湿度', '换热器出口温度']
    
    for cond in data_conditions:
        data = cond['data']
        name = cond['name']
        
        predictions = rolling_prediction(
            model, data, state_scaler, condition_scaler,
            cond['fin_spacing'], cond['flow_rate'], data[0, 1],
            window_size, pred_steps
        )
        
        print(f"\n--- {prefix}{name} ---")
        r2_scores = {}
        for i, var_name in enumerate(feature_names):
            r2 = r2_score(data[:, i], predictions[:, i])
            r2_scores[var_name] = r2
            print(f"  {var_name}: R² = {r2:.4f}")
        
        # 绘图
        os.makedirs(output_dir, exist_ok=True)
        
        key_vars = ['空气侧压降', '平均风速', '单位时间结霜量', '空气侧换热量', '水侧换热量']
        other_vars = ['进口温度', '进口湿度', '进口热电偶温度', '换热器进口温度',
                     '出口平均温度', '出口平均湿度', '换热器出口温度']
        
        n_rows = max(len(key_vars), len(other_vars))
        fig, axes = plt.subplots(n_rows, 2, figsize=(16, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, var_name in enumerate(key_vars):
            ax = axes[i, 0]
            col_idx = feature_names.index(var_name)
            
            ax.plot(data[:, col_idx], 'blue', label='真实值', linewidth=2)
            ax.plot(predictions[:, col_idx], 'red', label='V42预测', linewidth=2, linestyle='--')
            ax.axvline(x=window_size, color='green', linestyle=':', 
                      label=f'预测起点(窗口={window_size})', alpha=0.7)
            
            ax.set_xlabel('时间步')
            ax.set_ylabel(var_name)
            ax.set_title(f'{var_name} (R² = {r2_scores[var_name]:.4f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        for i in range(len(key_vars), n_rows):
            axes[i, 0].set_visible(False)
        
        for i, var_name in enumerate(other_vars):
            ax = axes[i, 1]
            col_idx = feature_names.index(var_name)
            
            ax.plot(data[:, col_idx], 'blue', label='真实值', linewidth=2)
            ax.plot(predictions[:, col_idx], 'red', label='V42预测', linewidth=2, linestyle='--')
            ax.axvline(x=window_size, color='green', linestyle=':', 
                      label=f'预测起点(窗口={window_size})', alpha=0.7)
            
            ax.set_xlabel('时间步')
            ax.set_ylabel(var_name)
            ax.set_title(f'{var_name} (R² = {r2_scores[var_name]:.4f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        for i in range(len(other_vars), n_rows):
            axes[i, 1].set_visible(False)
        
        filename = f'{prefix}{name.replace("-", "_")}.png'
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

# ============== 主函数（超参数网格搜索）==============
def main():
    print("="*80)
    print("V42: 滑动窗口LSTM - 自动实验运行")
    print("="*80)
    
    print("\n加载数据...")
    all_conditions = load_all_data()
    print(f"共加载 {len(all_conditions)} 个工况")
    
    # 存储所有实验配置的结果
    all_experiments_results = {}
    
    # 循环运行每组实验配置
    for exp_idx, exp_config in enumerate(EXPERIMENT_CONFIGS, 1):
        exp_name = exp_config['name']
        window_sizes = exp_config['window_sizes']
        pred_steps_list = exp_config['pred_steps_list']
        prediction_mode = exp_config['mode']
        
        print(f"\n{'#'*80}")
        print(f"实验 [{exp_idx}/{len(EXPERIMENT_CONFIGS)}]: {exp_name}")
        print(f"{'#'*80}")
        print(f"  窗口大小: {window_sizes}")
        print(f"  预测步长: {pred_steps_list}")
        print(f"  预测模式: {prediction_mode}")
        print(f"  总配置数: {len(window_sizes) * len(pred_steps_list)}")
        
        # 存储当前实验的所有配置结果
        exp_results = {}
        
        # 网格搜索
        for window_size in window_sizes:
            for pred_steps in pred_steps_list:
                config_key = (window_size, pred_steps)
                print(f"\n{'='*80}")
                print(f"配置: 窗口大小={window_size}, 预测步长={pred_steps}, 模式={prediction_mode}")
                print(f"{'='*80}")
                
                output_base_dir = rf'C:\Users\19396\Desktop\frost\plots_v42_window{window_size}_pred{pred_steps}'
                config_results = {}
                
                # Leave-One-Out交叉验证
                for test_idx in range(len(all_conditions)):
                    test_condition = [all_conditions[test_idx]]
                    train_conditions = [c for i, c in enumerate(all_conditions) if i != test_idx]
                    
                    test_name = test_condition[0]['name']
                    print(f"\n{'='*60}")
                    print(f"留一法交叉验证 [{test_idx+1}/{len(all_conditions)}]")
                    print(f"测试工况: {test_name}")
                    print(f"训练工况数: {len(train_conditions)}")
                    print(f"{'='*60}")
                    
                    test_output_dir = os.path.join(output_base_dir, test_name)
                    os.makedirs(test_output_dir, exist_ok=True)
                    
                    try:
                        train_dataset = SlidingWindowDataset(
                            train_conditions, 
                            window_size=window_size, 
                            pred_steps=pred_steps
                        )
                        state_scaler = train_dataset.state_scaler
                        condition_scaler = train_dataset.condition_scaler
                        
                        print(f"训练样本数: {len(train_dataset)}")
                        print(f"预测模式: {prediction_mode}")
                        
                        if len(train_dataset) == 0:
                            print("警告: 训练样本数为0，跳过此配置")
                            continue
                        
                        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
                        
                        model = SlidingWindowLSTM(
                            state_dim=12,
                            condition_dim=4,
                            window_size=window_size,
                            pred_steps=pred_steps,
                            hidden_size=192,
                            num_layers=3,
                            dropout=0.2
                        ).to(device)
                        
                        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
                        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
                        criterion = V42Loss(state_scaler=state_scaler, mse_weight=1.0, physics_weight=0.1)
                        
                        print("\n开始训练...")
                        best_loss = float('inf')
                        patience = 20
                        patience_counter = 0
                        
                        for epoch in range(100):
                            avg_loss = train_model(model, train_loader, criterion, optimizer, scheduler)
                            
                            if epoch % 10 == 0:
                                print(f"Epoch {epoch:3d} | Loss: {avg_loss:.6f}")
                            
                            if avg_loss < best_loss:
                                best_loss = avg_loss
                                patience_counter = 0
                            else:
                                patience_counter += 1
                                if patience_counter >= patience:
                                    print(f"Early stopping at epoch {epoch}")
                                    break
                        
                        print("\n评估训练集...")
                        train_metrics = evaluate_model_with_metrics(
                            model, train_conditions, state_scaler, condition_scaler, 
                            window_size, pred_steps, test_output_dir, prediction_mode, prefix='train_'
                        )
                        
                        print("\n评估测试集...")
                        test_metrics = evaluate_model_with_metrics(
                            model, test_condition, state_scaler, condition_scaler, 
                            window_size, pred_steps, test_output_dir, prediction_mode, prefix='test_'
                        )
                        
                        config_results[test_name] = {
                            'train': train_metrics,
                            'test': test_metrics
                        }
                    
                    except Exception as e:
                        print(f"错误: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                exp_results[config_key] = config_results
        
        # 保存当前实验的结果
        all_experiments_results[exp_name] = exp_results
        
        # 生成当前实验的报告
        print(f"\n{'='*80}")
        print(f"生成实验报告: {exp_name}")
        print(f"{'='*80}")
        
        report_filename = f'v42_{exp_name.replace(" ", "_").replace("模式", "")}_report.md'
        output_path = rf'C:\Users\19396\Desktop\frost\{report_filename}'
        generate_markdown_report(exp_results, output_path)
        
        print(f"报告已保存至: {output_path}")
    
    # 所有实验完成
    print("\n" + "="*80)
    print("所有实验完成！")
    print("="*80)
    print(f"\n共完成 {len(EXPERIMENT_CONFIGS)} 组实验：")
    for exp_name in all_experiments_results.keys():
        print(f"  - {exp_name}")
    print("\n实验报告已分别保存。")

if __name__ == '__main__':
    main()
