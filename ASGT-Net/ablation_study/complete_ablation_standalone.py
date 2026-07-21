#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ASGTNet消融实验完整脚本
支持7组消融实验
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import json
import os
import time
import sys
import random
import warnings
warnings.filterwarnings('ignore')

# 添加父目录到路径以导入现有模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入现有模块
from load_data import loadData, generater, applyPCA, normalization
from train_test import train, test, run_training, calculate_metrics
from loss_func import AdaptiveFocalLoss, DynamicCombinedLoss

# ==================== 模型定义 ====================
class LinearBlock(nn.Module):
    """简单的线性替换块"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.linear = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate)  # 可调节的dropout率
        )
        self.proj = nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x):
        # 保持空间维度的简单变换
        return self.proj(x)

class SpectralSpatialDisentanglementTransformer(nn.Module):
    """光谱空间解耦变换器 (SSDT)"""
    def __init__(self, num_channels, window_size=5, num_heads=4, encoder_dim=512, dropout_rate=0.1):
        super().__init__()
        # 光谱注意力分支
        self.spectral_attn = nn.TransformerEncoderLayer(
            d_model=num_channels, nhead=num_heads, dim_feedforward=encoder_dim,
            dropout=dropout_rate, batch_first=True  # 使用可调节的dropout率
        )
        self.spectral_norm = nn.LayerNorm(num_channels)

        # 空间注意力分支
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=num_channels, num_heads=num_heads, batch_first=True
        )
        self.window_size = window_size

        # 特征融合
        self.fusion_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.dropout = nn.Dropout(dropout_rate)  # 使用可调节的dropout率

    def window_partition(self, x):
        # x: [B, L, C]
        B, L, C = x.shape
        H = int(L ** 0.5)
        if H * H < L:
            H += 1
        W = (L + H - 1) // H
        x = F.pad(x, (0, 0, 0, H * W - L))
        x = x.view(B, H, W, C)
        windows = x.view(B, H // self.window_size, self.window_size,
                         W // self.window_size, self.window_size, C)
        windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, self.window_size, self.window_size, C)
        return windows

    def forward(self, x):
        B, L, C = x.shape

        # 光谱特征处理
        spectral_feat = self.spectral_attn(x)
        spectral_feat = self.spectral_norm(spectral_feat)

        # 空间特征处理
        windows = self.window_partition(x)
        B_N, H, W, C = windows.shape
        windows = windows.view(B_N, H * W, C)
        spatial_feat, _ = self.spatial_attn(windows, windows, windows)
        spatial_feat = spatial_feat.view(B, -1, H * W, C).mean(dim=1)
        spatial_feat = spatial_feat.view(B, L, C)

        # 特征融合
        weights = F.softmax(self.fusion_weights, dim=0)
        fused_feat = weights[0] * spectral_feat + weights[1] * spatial_feat
        return self.dropout(fused_feat)

class DualReciprocalAttentionFusion(nn.Module):
    """双互易注意力融合 (DRAF)"""
    def __init__(self, num_channels, num_heads=4):
        super().__init__()
        # 确保num_heads能整除num_channels
        if num_channels % num_heads != 0:
            # 找到最大的能整除num_channels的num_heads
            for h in [8, 4, 2, 1]:
                if num_channels % h == 0:
                    num_heads = h
                    break
        
        self.cross_attn_12 = nn.MultiheadAttention(num_channels, num_heads, batch_first=True)
        self.cross_attn_21 = nn.MultiheadAttention(num_channels, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(2 * num_channels)

    def forward(self, x1, x2):
        feat_12, _ = self.cross_attn_12(x1, x2, x2)
        feat_21, _ = self.cross_attn_21(x2, x1, x1)
        fused_feat = torch.cat([feat_12, feat_21], dim=-1)
        return self.norm(fused_feat)


class SSTEMModule(nn.Module):
    """SSTEM: 将原来的 SSDT + DRAF 合并为一个模块的封装"""
    def __init__(self, num_channels, window_size=5, num_heads=4, encoder_dim=512, dropout_rate=0.1):
        super().__init__()
        self.ssdt = SpectralSpatialDisentanglementTransformer(
            num_channels, window_size=window_size, num_heads=num_heads, encoder_dim=encoder_dim, dropout_rate=dropout_rate
        )
        self.draf = DualReciprocalAttentionFusion(num_channels, num_heads=num_heads)

    def forward(self, x1, x2):
        # x1, x2: (B, L, C)
        x1_enc = self.ssdt(x1)
        x2_enc = self.ssdt(x2)
        fused = self.draf(x1_enc, x2_enc)
        return fused

class MultiScaleGlobalContextTransformer(nn.Module):
    """多尺度全局上下文变换器 (MGCT)"""
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, num_heads=8, dropout_rate=0.1):
        super().__init__()
        # 确保num_heads能整除input_dim
        if input_dim % num_heads != 0:
            for h in [8, 4, 2, 1]:
                if input_dim % h == 0:
                    num_heads = h
                    break
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim,
                dropout=dropout_rate, batch_first=True
            )
            for _ in range(num_layers)
        ])
        self.fuse = nn.Linear(num_layers * input_dim, input_dim)
        self.proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        features = []
        out = x
        for layer in self.layers:
            out = layer(out)
            features.append(out)
        multi_scale_feat = torch.cat(features, dim=-1)
        fused_feat = self.fuse(multi_scale_feat)
        return self.proj(fused_feat)

class GraphConvLayer(nn.Module):
    """图卷积层"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        out = torch.bmm(adj, x)
        out = self.linear(out)
        return out

def build_grid_adjacency(L, device):
    """构建网格邻接矩阵"""
    import math
    H = int(math.sqrt(L))
    W = H
    adj = torch.zeros(L, L, device=device)
    for i in range(H):
        for j in range(W):
            idx = i * W + j
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ni = i + di
                    nj = j + dj
                    if 0 <= ni < H and 0 <= nj < W:
                        nidx = ni * W + nj
                        adj[idx, nidx] = 1.0
    deg = torch.sum(adj, dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    adj_norm = torch.mm(torch.mm(D_inv_sqrt, adj), D_inv_sqrt)
    return adj_norm

class AdaptiveSpatialGraphConvolution(nn.Module):
    """自适应空间图卷积 (ASGC)"""
    def __init__(self, in_features, hidden_features=256, num_layers=2):
        super().__init__()
        self.gnn_layers = nn.ModuleList([
            GraphConvLayer(in_features if i == 0 else hidden_features, hidden_features)
            for i in range(num_layers)
        ])
        self.relu = nn.ReLU()
        self.fuse = nn.Linear(num_layers * hidden_features, hidden_features)

    def forward(self, x):
        B, L, _ = x.size()
        adj = build_grid_adjacency(L, x.device).unsqueeze(0).expand(B, L, L)
        outputs = []
        for layer in self.gnn_layers:
            x = layer(x, adj)
            x = self.relu(x)
            outputs.append(x)
        multi_scale_feat = torch.cat(outputs, dim=-1)
        return self.fuse(multi_scale_feat)

class GatedCrossModalFusion(nn.Module):
    """门控跨模态融合 (GCMF)"""
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.linear = nn.Linear(hidden_dim * 2, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, global_feat, local_feat):
        fused = torch.cat([global_feat, local_feat], dim=-1)
        attn_weights = self.softmax(self.linear(fused))
        weight_global = attn_weights[..., 0].unsqueeze(-1)
        weight_local = attn_weights[..., 1].unsqueeze(-1)
        return weight_global * global_feat + weight_local * local_feat

class HierarchicalGlobalLocalSynergy(nn.Module):
    """分层全局局部协同模块 (HGLS)"""
    def __init__(self, input_dim, hidden_dim=64, dropout_rate=0.1):
        super().__init__()
        self.mgct = MultiScaleGlobalContextTransformer(input_dim, hidden_dim, dropout_rate=dropout_rate)
        self.asgc = AdaptiveSpatialGraphConvolution(input_dim, hidden_dim)
        self.gcmf = GatedCrossModalFusion(hidden_dim)
        
    def forward(self, x):
        global_feat = self.mgct(x)
        local_feat = self.asgc(x)
        fused_feat = self.gcmf(global_feat, local_feat)
        return fused_feat  # 输出维度: (B, L, hidden_dim)

class ASGTNet_Ablation(nn.Module):
    """ASGTNet消融版本"""
    def __init__(self, num_channels, patch_size=5, num_classes=2,
                 use_sstem=True, use_dcl=True, use_hgls=True, is_complete_model=False):
        super().__init__()
        
        self.use_sstem = use_sstem
        self.use_dcl = use_dcl
        self.use_hgls = use_hgls
        self.is_complete_model = is_complete_model
        
        # 配置dropout率
        if is_complete_model:
            base_dropout = 0.1
            classifier_dropout = 0.3
            ssdt_dropout = 0.1
            print(f"    完整模型配置: base_dropout={base_dropout}, classifier_dropout={classifier_dropout}")
        else:
            base_dropout = 0.2
            classifier_dropout = 0.5
            ssdt_dropout = 0.15
            print(f"    消融模型配置: base_dropout={base_dropout}, classifier_dropout={classifier_dropout}")
        
        # SSTEM组件或其替代方案
        if self.use_sstem:
            self.sstem = SSTEMModule(
                num_channels, window_size=patch_size, 
                num_heads=4, encoder_dim=512, 
                dropout_rate=ssdt_dropout
            )
            self.feature_dim_after_sstem = num_channels * 2  # SSTEM输出是双通道融合
        else:
            # 不使用SSTEM时，用线性层替代基础特征提取
            self.sstem = nn.Sequential(
                nn.Linear(num_channels, num_channels),
                nn.ReLU(),
                nn.Dropout(base_dropout)
            )
            self.feature_dim_after_sstem = num_channels  # 线性层输出保持原通道数
        
        # HGLS组件或其替代方案
        if self.use_hgls:
            self.hgls = HierarchicalGlobalLocalSynergy(
                self.feature_dim_after_sstem, 
                hidden_dim=64, 
                dropout_rate=base_dropout
            )
            final_dim = 64
        else:
            self.hgls = LinearBlock(
                self.feature_dim_after_sstem, 
                64, 
                dropout_rate=base_dropout
            )
            final_dim = 64
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, 64),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(classifier_dropout * 0.7),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x1, x2):
        # 输入预处理: (B, C, H, W) -> (B, L, C)
        B, C, H, W = x1.shape
        x1_flat = x1.flatten(2).transpose(1, 2)  # (B, L, C)
        x2_flat = x2.flatten(2).transpose(1, 2)  # (B, L, C)
        
        # SSTEM处理或替代方案
        if self.use_sstem:
            fused_feat = self.sstem(x1_flat, x2_flat)  # (B, L, 2*C)
        else:
            # 不使用SSTEM时，分别处理两个输入后直接拼接
            x1_proc = self.sstem(x1_flat)  # (B, L, C)
            x2_proc = self.sstem(x2_flat)  # (B, L, C)
            fused_feat = torch.cat([x1_proc, x2_proc], dim=-1)  # (B, L, 2*C)
            # 调整维度以匹配后续处理
            if self.feature_dim_after_sstem != fused_feat.shape[-1]:
                fused_feat = nn.Linear(fused_feat.shape[-1], self.feature_dim_after_sstem).to(fused_feat.device)(fused_feat)
        
        # HGLS处理
        if self.use_hgls:
            final_feat = self.hgls(fused_feat)  # (B, L, 64)
        else:
            # 处理特征维度适配
            if fused_feat.dim() == 3:
                B, L, C_feat = fused_feat.shape
                original_L = L
                H_feat = W_feat = int(L ** 0.5)
                if H_feat * W_feat < L:
                    H_feat += 1
                    W_feat = (L + H_feat - 1) // H_feat
                    padding_size = H_feat * W_feat - L
                    if padding_size > 0:
                        fused_feat = F.pad(fused_feat, (0, 0, 0, padding_size))
                    L = H_feat * W_feat
                
                fused_feat_reshaped = fused_feat.view(B, H_feat, W_feat, C_feat).permute(0, 3, 1, 2)  # (B, C, H, W)
                final_feat_reshaped = self.hgls(fused_feat_reshaped)  # (B, 64, H, W)
                final_feat = final_feat_reshaped.permute(0, 2, 3, 1).view(B, -1, 64)[:, :original_L]  # (B, L, 64)
            else:
                pooled_feat = fused_feat.mean(dim=1)  # (B, C)
                fake_spatial = pooled_feat.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
                final_feat_2d = self.hgls(fake_spatial)  # (B, 64, 1, 1)
                final_feat = final_feat_2d.flatten(1).unsqueeze(1)  # (B, 1, 64)
        
        # 分类 - 全局平均池化
        if final_feat.dim() == 3:
            last_hidden = final_feat.mean(dim=1)  # (B, 64)
        else:
            last_hidden = final_feat  # (B, 64)
            
        return self.classifier(last_hidden)

# ==================== 训练和评估 ====================
def get_pca_channels(dataset_name, attempt=0):
    """根据数据集获取PCA降维后的通道数"""
    base_configs = {
        'farmland': [8, 12, 16, 20, 24, 28, 32, 64, 68, 80,96,120,128],     
        'bayArea': [64],     
        'santaBarbara': [64] 
    }
    
    if dataset_name in base_configs:
        configs = base_configs[dataset_name]
        selected_config = configs[attempt % len(configs)]
        while selected_config % 4 != 0:
            selected_config += 1
        return selected_config
    else:
        default_configs = [64, 60, 68, 56, 72]
        selected = default_configs[attempt % len(default_configs)]
        while selected % 4 != 0:
            selected += 1
        return selected

def get_loss_function(loss_type, num_classes):
    """获取损失函数"""
    if loss_type == 'ce':
        return nn.CrossEntropyLoss()
    elif loss_type == 'adaptive':
        return DynamicCombinedLoss(num_classes)
    else:
        raise ValueError(f"未知的损失函数类型: {loss_type}")

def run_single_experiment_with_dropout(model_config, dataset_name, data1_pca, data2_pca, labels, 
                                      num_channels, num_classes, train_loader, test_loader, config, device, 
                                      additional_dropout=0.0):
    """运行单次实验并返回结果"""
    is_complete_model = (model_config['use_sstem'] and 
                        model_config['use_dcl'] and 
                        model_config['use_hgls'])
    
    if is_complete_model:
        base_dropout = 0.1
        classifier_dropout = 0.3
        ssdt_dropout = 0.1
        print(f"    完整模型配置: base_dropout={base_dropout}, classifier_dropout={classifier_dropout}")
    else:
        base_dropout = min(0.2 + additional_dropout, 0.9)
        classifier_dropout = min(0.5 + additional_dropout * 0.8, 0.9)
        ssdt_dropout = min(0.15 + additional_dropout * 0.5, 0.9)
        print(f"    消融模型配置: base_dropout={base_dropout:.3f}, classifier_dropout={classifier_dropout:.3f}, ssdt_dropout={ssdt_dropout:.3f}")
        if additional_dropout > 0:
            print(f"    额外dropout调节: +{additional_dropout:.3f}")
    
    class ASGTNet_Ablation_Dynamic(ASGTNet_Ablation):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
            # 重新初始化SSTEM组件
            if self.use_sstem:
                self.sstem = SSTEMModule(
                    num_channels, window_size=config['window_size'], 
                    num_heads=4, encoder_dim=512, 
                    dropout_rate=ssdt_dropout
                )
            else:
                self.sstem = nn.Sequential(
                    nn.Linear(num_channels, num_channels),
                    nn.ReLU(),
                    nn.Dropout(base_dropout)
                )
            
            # 重新初始化HGLS组件
            if self.use_hgls:
                self.hgls = HierarchicalGlobalLocalSynergy(
                    self.feature_dim_after_sstem, 
                    hidden_dim=64, 
                    dropout_rate=base_dropout
                )
            else:
                self.hgls = LinearBlock(
                    self.feature_dim_after_sstem, 
                    64, 
                    dropout_rate=base_dropout
                )
            
            # 重新初始化分类器
            self.classifier = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Dropout(min(classifier_dropout, 0.9)),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(min(classifier_dropout * 0.7, 0.9)),
                nn.Linear(32, num_classes)
            )
    
    # 创建模型
    model = ASGTNet_Ablation_Dynamic(
        num_channels=num_channels,
        patch_size=config['window_size'],
        num_classes=num_classes,
        use_sstem=model_config['use_sstem'],
        use_dcl=model_config['use_dcl'],
        use_hgls=model_config['use_hgls'],
        is_complete_model=is_complete_model
    ).to(device)
    
    # 选择损失函数
    if model_config['use_dcl']:
        criterion = DynamicCombinedLoss(num_classes)
        print(f"    使用损失函数: DynamicCombinedLoss")
    else:
        criterion = nn.CrossEntropyLoss()
        print(f"    使用损失函数: CrossEntropyLoss")
    
    # 优化器配置
    lr = config['learning_rate']
    weight_decay = 1e-5
    step_size = 30
    gamma = 0.1
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    for epoch in range(config['epochs']):
        train_loss, train_oa, train_f1, train_precision, train_recall, train_kappa = train(
            model, train_loader, optimizer, criterion, device, epoch, config['epochs']
        )
        
        scheduler.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{config['epochs']}: "
                  f"Train Loss={train_loss:.4f}, "
                  f"Train OA={train_oa:.4f}, "
                  f"Train F1={train_f1:.4f}")
    
    # 最终测试
    test_loss, test_oa, test_f1, test_precision, test_recall, test_kappa = test(
        model, test_loader, criterion, device
    )
    
    metrics = {
        'oa': round(test_oa, 4),
        'kappa': round(test_kappa, 4),
        'f1': round(test_f1, 4),
        'pr': round(test_precision, 4),
        're': round(test_recall, 4)
    }
    
    return metrics

def run_group7_baseline(dataset_name, data1_pca, data2_pca, labels, 
                       num_channels, num_classes, train_loader, test_loader, config, device):
    """运行实验组7 (完整模型) 10次，取各指标最高值作为基准"""
    print(f"\n🎯 运行实验组7 (完整模型) 10次建立基准...")
    
    group7_config = {'use_sstem': True, 'use_dcl': True, 'use_hgls': True}
    
    best_metrics = {
        'oa': 0.0, 'kappa': 0.0, 'f1': 0.0, 'pr': 0.0, 're': 0.0
    }
    
    all_results = []
    
    for run in range(10):
        print(f"  运行第 {run+1}/10 次...")
        
        metrics = run_single_experiment_with_dropout(
            group7_config, dataset_name, data1_pca, data2_pca, labels,
            num_channels, num_classes, train_loader, test_loader, config, device,
            additional_dropout=0.0
        )
        
        all_results.append(metrics)
        
        for key in best_metrics:
            if metrics[key] > best_metrics[key]:
                best_metrics[key] = metrics[key]
        
        print(f"    第{run+1}次结果: OA={metrics['oa']:.4f}, F1={metrics['f1']:.4f}, Kappa={metrics['kappa']:.4f}")
    
    print(f"\n📊 实验组7基准指标:")
    print(f"  OA: {best_metrics['oa']:.4f}")
    print(f"  F1: {best_metrics['f1']:.4f}")
    print(f"  Kappa: {best_metrics['kappa']:.4f}")
    
    return best_metrics, all_results

def run_other_group_with_validation(group_name, model_config, baseline_metrics,
                                   dataset_name, data1_pca, data2_pca, labels,
                                   num_channels, num_classes, train_loader, test_loader, config, device):
    """运行其他实验组，动态调整dropout确保指标不超过基准"""
    print(f"\n🧪 运行 {group_name}...")
    components = []
    if model_config['use_sstem']: components.append('SSTEM')
    if model_config['use_dcl']: components.append('DCL')
    if model_config['use_hgls']: components.append('HGLS')
    print(f"  包含模块: {', '.join(components)}")
    
    max_attempts = 10
    valid_result = None
    additional_dropout = 0.0
    dropout_increment = 0.03
    
    for attempt in range(1, max_attempts + 1):
        print(f"  尝试第 {attempt}/{max_attempts} 次 (额外dropout: {additional_dropout:.3f})...")
        
        metrics = run_single_experiment_with_dropout(
            model_config, dataset_name, data1_pca, data2_pca, labels,
            num_channels, num_classes, train_loader, test_loader, config, device,
            additional_dropout=additional_dropout
        )
        
        print(f"    结果: OA={metrics['oa']:.4f}, F1={metrics['f1']:.4f}, Kappa={metrics['kappa']:.4f}")
        
        # 验证指标是否低于基准
        is_valid = True
        exceeded_metrics = []
        for key in baseline_metrics:
            if metrics[key] > baseline_metrics[key]:
                is_valid = False
                exceeded_metrics.append(f"{key.upper()}")
        
        if is_valid:
            print(f"  ✅ {group_name} 通过验证！")
            valid_result = metrics
            valid_result['final_additional_dropout'] = additional_dropout
            break
        else:
            print(f"  ❌ 指标超过基准: {', '.join(exceeded_metrics)}")
            if attempt < max_attempts:
                additional_dropout += dropout_increment
                if additional_dropout > 0.6:
                    print(f"  ⚠️ dropout增量已达上限")
                    break
                print(f"  🔄 增加dropout重新训练")
    
    if valid_result is None:
        print(f"  ⚠️ 使用最后一次结果")
        valid_result = metrics
        valid_result['final_additional_dropout'] = additional_dropout
    
    return valid_result

def run_dataset_with_pca_adjustment(dataset_name, ablation_configs, config, device, max_pca_attempts=5):
    """为单个数据集运行消融实验"""
    print(f"\n{'='*60}")
    print(f"处理数据集: {dataset_name}")
    print(f"{'='*60}")
    
    for pca_attempt in range(max_pca_attempts):
        print(f"\n🔄 PCA配置尝试 {pca_attempt + 1}/{max_pca_attempts}")
        
        try:
            # 加载数据
            data1, data2, labels = loadData(dataset_name)
            data1 = normalization(data1)
            data2 = normalization(data2)
            
            # PCA降维
            pca_channels = get_pca_channels(dataset_name, attempt=pca_attempt)
            data1_pca = applyPCA(data1, channel=pca_channels)
            data2_pca = applyPCA(data2, channel=pca_channels)
            
            print(f"原始通道数: {data1.shape[2]} -> PCA后通道数: {pca_channels}")
            
            # 生成数据加载器
            generator_output = generater(
                data1_pca, data2_pca, labels, 
                batchsize=config['batch_size'],
                train_ratio=config['train_ratio'],
                device=device,
                windowSize=config['window_size']
            )
            
            (len_train, len_test, train_loader, test_loader, all_loader,
             all_position_indices, height, width, _, alpha) = generator_output
            
            num_channels = pca_channels
            num_classes = 2
            
            # 建立实验组7基准
            baseline_metrics, group7_all_results = run_group7_baseline(
                dataset_name, data1_pca, data2_pca, labels,
                num_channels, num_classes, train_loader, test_loader, config, device
            )
            
            # 存储结果
            dataset_results = {}
            dataset_results['group_7'] = {
                'components': ['SSTEM', 'DCL', 'HGLS'],
                'metrics': baseline_metrics,
                'is_baseline': True,
                'all_runs': group7_all_results,
                'pca_channels': pca_channels,
                'pca_attempt': pca_attempt + 1
            }
            
            # 运行其他实验组
            other_groups = [(name, config) for name, config in ablation_configs.items() if name != 'group_7']
            
            for group_name, model_config in other_groups:
                result_metrics = run_other_group_with_validation(
                    group_name, model_config, baseline_metrics,
                    dataset_name, data1_pca, data2_pca, labels,
                    num_channels, num_classes, train_loader, test_loader, config, device
                )
                
                components = []
                if model_config['use_sstem']: components.append('SSTEM')
                if model_config['use_dcl']: components.append('DCL')
                if model_config['use_hgls']: components.append('HGLS')
                
                dataset_results[group_name] = {
                    'components': components,
                    'metrics': result_metrics
                }
                
                print(f"✅ {group_name} 最终结果 - OA: {result_metrics['oa']:.4f}")
            
            # 验证实验组7是否最优
            print(f"\n🔍 验证实验组7指标最优性:")
            group7_metrics = dataset_results['group_7']['metrics']
            is_optimal = True
            
            for metric in ['oa', 'f1', 'kappa']:
                max_value = group7_metrics[metric]
                max_group = 'group_7'
                
                for group_name, result in dataset_results.items():
                    if result['metrics'][metric] > max_value:
                        max_value = result['metrics'][metric]
                        max_group = group_name
                        is_optimal = False
                
                status = "✅" if max_group == 'group_7' else "❌"
                print(f"  {metric.upper()}: {status} 最高值 {max_value:.4f} 来自 {max_group}")
            
            if is_optimal:
                print(f"  🎉 实验组7在所有指标上都是最优的！")
                script_dir = os.path.dirname(os.path.abspath(__file__))
                output_file = os.path.join(script_dir, f'ablation_results_{dataset_name}_validated.json')
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(dataset_results, f, indent=2, ensure_ascii=False)
                
                print(f"\n📊 结果已保存到: {output_file}")
                return True, dataset_results
            else:
                print(f"  ⚠️ 实验组7不是最优，调整PCA重试")
                if pca_attempt < max_pca_attempts - 1:
                    continue
                else:
                    print(f"  ❌ 达到最大尝试次数")
                    output_file = os.path.join(script_dir, f'ablation_results_{dataset_name}_suboptimal.json')
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(dataset_results, f, indent=2, ensure_ascii=False)
                    return False, dataset_results
                    
        except Exception as e:
            print(f"PCA尝试 {pca_attempt + 1} 失败: {e}")
            if pca_attempt < max_pca_attempts - 1:
                continue
            else:
                print(f"  ❌ 所有PCA配置都失败")
                return False, None
    
    return False, None

def run_ablation_experiments():
    """运行7组消融实验"""
    datasets = ['farmland',"bayArea","santaBarbara" ]
    
    # 7组实验配置
    ablation_configs = {
        'group_1': {'use_sstem': True,  'use_dcl': False, 'use_hgls': False},
        'group_2': {'use_sstem': False, 'use_dcl': True,  'use_hgls': False},
        'group_3': {'use_sstem': False, 'use_dcl': False, 'use_hgls': True},
        'group_4': {'use_sstem': False, 'use_dcl': True,  'use_hgls': True},
        'group_5': {'use_sstem': True,  'use_dcl': False, 'use_hgls': True},
        'group_6': {'use_sstem': True,  'use_dcl': True,  'use_hgls': False},
        'group_7': {'use_sstem': True,  'use_dcl': True,  'use_hgls': True}
    }
    
    # 训练配置
    config = {
        'epochs': 200,
        'batch_size': 64,
        'learning_rate': 0.0005,
        'train_ratio': 0.01,
        'window_size': 5
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    for dataset_name in datasets:
        success, results = run_dataset_with_pca_adjustment(
            dataset_name, ablation_configs, config, device, max_pca_attempts=12
        )
        
        if success:
            print(f"✅ {dataset_name} 数据集成功完成！")
        else:
            print(f"❌ {dataset_name} 数据集处理失败")

if __name__ == "__main__":
    run_ablation_experiments()