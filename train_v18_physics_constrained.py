import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============== 数据加载 ==============
def load_single_condition(folder_path):
    """加载单个工况的数据"""
    file_path = os.path.join(folder_path, '有用数据.xlsx')
    if not os.path.exists(file_path):
        return None
    
    df = pd.read_excel(file_path)
    features = ['空气侧压降', '平均风速', '单位时间结霜量', '空气侧换热量', '水侧换热量',
                '进口温度', '进口湿度', '进口热电偶温度', '换热器进口温度',
                '出口平均温度', '出口平均湿度', '换热器出口温度']
    
    data = df[features].values
    return data

def extract_condition_from_folder(folder_name):
    """从文件夹名提取工况参数"""
    parts = folder_name.replace('插排-', '').split('-')
    fin_spacing = float(parts[0].replace('mm', ''))
    flow_rate = float(parts[1])
    return fin_spacing, flow_rate

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
            
            # 提取特征
            features = ['空气侧压降', '平均风速', '单位时间结霜量', '空气侧换热量', '水侧换热量',
                       '进口温度', '进口湿度', '进口热电偶温度', '换热器进口温度',
                       '出口平均温度', '出口平均湿度', '换热器出口温度']
            
            data = df[features].values
            
            # 提取工况参数
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

# ============== 数据集类 ==============
class FrostDataset(Dataset):
    def __init__(self, conditions, state_scaler=None, condition_scaler=None):
        self.samples = []
        self.state_scaler = state_scaler if state_scaler else StandardScaler()
        self.condition_scaler = condition_scaler if condition_scaler else StandardScaler()
        
        all_states = []
        all_conditions = []
        
        # 收集所有数据用于归一化
        for cond in conditions:
            data = cond['data']
            all_states.append(data)
            
        all_states = np.vstack(all_states)
        
        # 拟合scaler
        if state_scaler is None:
            self.state_scaler.fit(all_states)
        
        # 归一化并创建序列
        for cond in conditions:
            data = cond['data']
            normalized_data = self.state_scaler.transform(data)
            
            # 归一化时间步：每个工况独立归一化到[0,1]
            n_steps = len(normalized_data)
            t_normalized = np.linspace(0, 1, n_steps).reshape(-1, 1)
            
            # 提取工况参数
            fin_spacing = cond['fin_spacing']
            flow_rate = cond['flow_rate']
            
            # 初始风速（从第一个时刻获取，避免数据泄露）
            initial_wind_speed = data[0, 1]  # 第一个时刻的风速
            
            # 工况特征：[翅片间距, 初始风速, 流量, 归一化时间]
            for i in range(len(normalized_data) - 1):
                condition_features = np.array([fin_spacing, initial_wind_speed, flow_rate, t_normalized[i, 0]])
                
                self.samples.append({
                    'current_state': normalized_data[i],
                    'next_state': normalized_data[i + 1],
                    'condition': condition_features,
                    'time_step': i,
                    'total_steps': n_steps - 1,
                    'raw_current': data[i],
                    'raw_next': data[i + 1],
                })
        
        # 归一化工况特征
        all_cond_features = np.array([s['condition'] for s in self.samples])
        if condition_scaler is None:
            self.condition_scaler.fit(all_cond_features)
        
        for sample in self.samples:
            sample['condition'] = self.condition_scaler.transform(sample['condition'].reshape(1, -1))[0]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.FloatTensor(sample['current_state']),
            torch.FloatTensor(sample['condition']),
            torch.FloatTensor(sample['next_state']),
            torch.FloatTensor(sample['raw_current']),
            torch.FloatTensor(sample['raw_next'])
        )

# ============== 物理约束损失 ==============
class PhysicsConstrainedLoss(nn.Module):
    """物理一致性约束损失"""
    def __init__(self, mse_weight=1.0, physics_weight=0.1):
        super().__init__()
        self.mse_weight = mse_weight
        self.physics_weight = physics_weight
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target, raw_current, raw_pred):
        """
        pred: 预测的归一化值 [batch, 12]
        target: 真实的归一化值 [batch, 12]
        raw_current: 当前时刻原始值 [batch, 12]
        raw_pred: 预测的原始值（反归一化后）[batch, 12]
        """
        # 基础MSE损失
        mse_loss = self.mse(pred, target)
        
        # 物理约束损失
        physics_loss = 0.0
        
        # 1. 能量守恒：空气侧换热量 ≈ 水侧换热量
        air_heat = raw_pred[:, 3]  # 空气侧换热量
        water_heat = raw_pred[:, 4]  # 水侧换热量
        heat_balance = torch.mean((air_heat - water_heat) ** 2) / (torch.mean(air_heat ** 2) + 1e-6)
        physics_loss += heat_balance
        
        # 2. 单调性约束：累积结霜量应该增加
        frost_current = raw_current[:, 2]  # 当前时刻结霜量
        frost_pred = raw_pred[:, 2]  # 预测时刻结霜量
        frost_decrease = torch.relu(frost_current - frost_pred)  # 惩罚结霜量减少
        physics_loss += torch.mean(frost_decrease)
        
        # 3. 压降增长约束：结霜导致压降增加
        pressure_current = raw_current[:, 0]
        pressure_pred = raw_pred[:, 0]
        pressure_decrease = torch.relu(pressure_current - pressure_pred)  # 惩罚压降减少
        physics_loss += torch.mean(pressure_decrease) * 0.5
        
        # 4. 温度湿度合理性：温度变化不应过大
        temp_vars = [5, 7, 8, 9, 11]  # 温度相关变量索引
        for idx in temp_vars:
            temp_diff = torch.abs(raw_pred[:, idx] - raw_current[:, idx])
            temp_penalty = torch.relu(temp_diff - 5.0)  # 惩罚超过5度的变化
            physics_loss += torch.mean(temp_penalty) * 0.1
        
        # 总损失
        total_loss = self.mse_weight * mse_loss + self.physics_weight * physics_loss
        
        return total_loss, mse_loss, physics_loss

# ============== 增强LSTM模型 ==============
class EnhancedLSTM(nn.Module):
    def __init__(self, state_dim=12, condition_dim=4, hidden_size=192, num_layers=3, dropout=0.2):
        super().__init__()
        
        self.state_dim = state_dim
        self.condition_dim = condition_dim
        self.hidden_size = hidden_size
        
        # 状态编码器
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
        
        # LSTM层（输入：状态编码 + 工况编码）
        self.lstm = nn.LSTM(
            input_size=hidden_size + hidden_size // 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, state_dim)
        )
    
    def forward(self, state, condition):
        batch_size = state.size(0)
        
        # 编码
        state_encoded = self.state_encoder(state).unsqueeze(1)  # [batch, 1, hidden]
        condition_encoded = self.condition_encoder(condition).unsqueeze(1)  # [batch, 1, hidden/2]
        
        # 拼接
        lstm_input = torch.cat([state_encoded, condition_encoded], dim=2)  # [batch, 1, hidden*1.5]
        
        # LSTM
        lstm_out, _ = self.lstm(lstm_input)  # [batch, 1, hidden]
        
        # 输出
        output = self.output_layer(lstm_out.squeeze(1))  # [batch, state_dim]
        
        return output

# ============== 训练函数 ==============
def train_model(model, train_loader, criterion, optimizer, epoch, scheduler, state_scaler):
    model.train()
    total_loss = 0
    total_mse = 0
    total_physics = 0
    
    for batch_idx, (current_state, condition, next_state, raw_current, raw_next) in enumerate(train_loader):
        current_state = current_state.to(device)
        condition = condition.to(device)
        next_state = next_state.to(device)
        raw_current = raw_current.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        pred = model(current_state, condition)
        
        # 反归一化预测值用于物理约束
        pred_np = pred.detach().cpu().numpy()
        raw_pred = state_scaler.inverse_transform(pred_np)
        raw_pred = torch.FloatTensor(raw_pred).to(device)
        
        # 计算损失
        loss, mse_loss, physics_loss = criterion(pred, next_state, raw_current, raw_pred)
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_mse += mse_loss.item()
        total_physics += physics_loss.item()
    
    avg_loss = total_loss / len(train_loader)
    avg_mse = total_mse / len(train_loader)
    avg_physics = total_physics / len(train_loader)
    
    if scheduler:
        scheduler.step()
    
    return avg_loss, avg_mse, avg_physics

# ============== 评估函数 ==============
def evaluate_model(model, data_conditions, state_scaler, condition_scaler, output_dir, prefix=''):
    """评估模型性能
    
    Args:
        model: 训练好的模型
        data_conditions: 工况列表
        state_scaler: 状态归一化器
        condition_scaler: 工况归一化器
        output_dir: 输出目录
        prefix: 文件名前缀（如'train_'或'test_'）
    """
    model.eval()
    
    with torch.no_grad():
        for cond in data_conditions:
            data = cond['data']
            name = cond['name']
            
            n_steps = len(data)
            predictions = np.zeros_like(data)
            predictions[0] = data[0]
            
            # 归一化时间
            t_normalized = np.linspace(0, 1, n_steps)
            
            # 提取工况参数
            fin_spacing = cond['fin_spacing']
            flow_rate = cond['flow_rate']
            initial_wind_speed = data[0, 1]  # 使用初始风速，避免数据泄露
            
            # 滚动预测
            current_state = state_scaler.transform(data[0].reshape(1, -1))
            
            for t in range(1, n_steps):
                current_state_tensor = torch.FloatTensor(current_state).to(device)
                
                # 工况特征
                condition_features = np.array([[fin_spacing, initial_wind_speed, flow_rate, t_normalized[t]]])
                condition_features = condition_scaler.transform(condition_features)
                condition_tensor = torch.FloatTensor(condition_features).to(device)
                
                # 预测
                pred = model(current_state_tensor, condition_tensor)
                pred_np = pred.cpu().numpy()
                
                # 反归一化
                pred_denorm = state_scaler.inverse_transform(pred_np)
                predictions[t] = pred_denorm[0]
                
                # 更新状态
                current_state = pred_np
            
            # 计算R²
            print(f"\n--- {prefix}{name} ---")
            feature_names = ['空气侧压降', '平均风速', '单位时间结霜量', '空气侧换热量', '水侧换热量',
                           '进口温度', '进口湿度', '进口热电偶温度', '换热器进口温度',
                           '出口平均温度', '出口平均湿度', '换热器出口温度']
            
            r2_scores = {}
            for i, name_var in enumerate(feature_names):
                r2 = r2_score(data[:, i], predictions[:, i])
                r2_scores[name_var] = r2
                print(f"  {name_var}: R² = {r2:.4f}")
            
            # 绘图 - V9风格：两列子图布局
            os.makedirs(output_dir, exist_ok=True)
            
            # 分成关键变量和其他变量
            key_vars = ['空气侧压降', '平均风速', '单位时间结霜量', '空气侧换热量', '水侧换热量']
            other_vars = ['进口温度', '进口湿度', '进口热电偶温度', '换热器进口温度',
                         '出口平均温度', '出口平均湿度', '换热器出口温度']
            
            n_rows = max(len(key_vars), len(other_vars))
            
            fig, axes = plt.subplots(n_rows, 2, figsize=(16, 4*n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            # 左列：关键变量
            for i, var_name in enumerate(key_vars):
                ax = axes[i, 0]
                col_idx = feature_names.index(var_name)
                
                ax.plot(data[:, col_idx], 'blue', label='真实值', linewidth=2)
                ax.plot(predictions[:, col_idx], 'red', label='V18物理约束', linewidth=2, linestyle='--')
                
                ax.set_xlabel('时间步')
                ax.set_ylabel(var_name)
                ax.set_title(f'{var_name} (R² = {r2_scores[var_name]:.4f})')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # 隐藏左列多余的子图
            for i in range(len(key_vars), n_rows):
                axes[i, 0].set_visible(False)
            
            # 右列：其他变量
            for i, var_name in enumerate(other_vars):
                ax = axes[i, 1]
                col_idx = feature_names.index(var_name)
                
                ax.plot(data[:, col_idx], 'blue', label='真实值', linewidth=2)
                ax.plot(predictions[:, col_idx], 'red', label='V18物理约束', linewidth=2, linestyle='--')
                
                ax.set_xlabel('时间步')
                ax.set_ylabel(var_name)
                ax.set_title(f'{var_name} (R² = {r2_scores[var_name]:.4f})')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # 隐藏右列多余的子图
            for i in range(len(other_vars), n_rows):
                axes[i, 1].set_visible(False)
            
            # 添加前缀到文件名
            filename = f'{prefix}{name.replace("-", "_")}.png'
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
            plt.close()

# ============== 主函数 ==============
def main():
    print("加载数据...")
    all_conditions = load_all_data()
    print(f"共加载 {len(all_conditions)} 个工况")
    
    output_base_dir = r'C:\Users\19396\Desktop\frost\plots_v18_loo_cv'
    
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
        
        # 为当前测试工况创建独立文件夹
        test_output_dir = os.path.join(output_base_dir, test_name)
        os.makedirs(test_output_dir, exist_ok=True)
        
        # 创建数据集
        train_dataset = FrostDataset(train_conditions)
        state_scaler = train_dataset.state_scaler
        condition_scaler = train_dataset.condition_scaler
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
        
        # 创建模型
        model = EnhancedLSTM(
            state_dim=12,
            condition_dim=4,
            hidden_size=192,
            num_layers=3,
            dropout=0.2
        ).to(device)
        
        # 优化器和损失
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        criterion = PhysicsConstrainedLoss(mse_weight=1.0, physics_weight=0.1)
        
        # 训练
        print("\n开始训练...")
        best_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(100):
            avg_loss, avg_mse, avg_physics = train_model(
                model, train_loader, criterion, optimizer, epoch, scheduler, state_scaler
            )
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | Loss: {avg_loss:.6f} | MSE: {avg_mse:.6f} | Physics: {avg_physics:.6f}")
            
            # Early Stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # 评估训练集（添加train_前缀）
        print("\n评估训练集...")
        evaluate_model(model, train_conditions, state_scaler, condition_scaler, test_output_dir, prefix='train_')
        
        # 评估测试集（添加test_前缀）
        print("\n评估测试集...")
        evaluate_model(model, test_condition, state_scaler, condition_scaler, test_output_dir, prefix='test_')

if __name__ == '__main__':
    main()
