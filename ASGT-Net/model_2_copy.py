import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from thop import profile
from torchsummary import summary
import json

class SSTEM(nn.Module):
    """
    Spectral-Spatial Temporal Encoding Module (SSTEM)
    """

    def __init__(self, num_channels, window_size, num_heads, encoder_dim):
        super().__init__()
        
        # 光谱注意力分支
        self.spectral_attn = nn.TransformerEncoderLayer(
            d_model=num_channels, nhead=num_heads, dim_feedforward=encoder_dim,
            dropout=0.5, batch_first=True
        )
        self.spectral_norm = nn.LayerNorm(num_channels)

        # 空间注意力分支
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=num_channels, num_heads=num_heads, batch_first=True
        )
        self.window_size = window_size

        # 光谱-空间特征融合
        self.fusion_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.dropout = nn.Dropout(0.5)
        
        self.cross_attn_12 = nn.MultiheadAttention(num_channels, num_heads, batch_first=True)
        self.cross_attn_21 = nn.MultiheadAttention(num_channels, num_heads, batch_first=True)
        self.temporal_norm = nn.LayerNorm(2 * num_channels)

    def window_partition(self, x):
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

    def ssdt_forward(self, x):
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

    def draf_forward(self, x1, x2):
        feat_12, _ = self.cross_attn_12(x1, x2, x2)
        feat_21, _ = self.cross_attn_21(x2, x1, x1)
        fused_feat = torch.cat([feat_12, feat_21], dim=-1)
        return self.temporal_norm(fused_feat)

    def forward(self, x1, x2):
        # 光谱-空间解耦
        x1_enc = self.ssdt_forward(x1)
        x2_enc = self.ssdt_forward(x2)
        
        # 时序特征融合
        fused_feat = self.draf_forward(x1_enc, x2_enc)
        return fused_feat


class HGLS(nn.Module):
    """
    Hierarchical Global-Local Synergy (HGLS)
    """

    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, gnn_hidden_features):
        super().__init__()
        
        # 多尺度全局上下文Transformer
        self.global_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim,
                dropout=0.5, batch_first=True
            )
            for _ in range(num_layers)
        ])
        self.global_fuse = nn.Linear(num_layers * input_dim, input_dim)
        self.global_proj = nn.Linear(input_dim, hidden_dim)
        
        #  自适应空间图卷积
        self.gnn_layers = nn.ModuleList([
            nn.Linear(input_dim if i == 0 else gnn_hidden_features, gnn_hidden_features)
            for i in range(num_layers)
        ])
        self.relu = nn.ReLU()
        self.gnn_fuse = nn.Linear(num_layers * gnn_hidden_features, gnn_hidden_features)
        self.local_proj = nn.Linear(gnn_hidden_features, hidden_dim)
        
        # 门控跨模态融合
        self.fusion_linear = nn.Linear(hidden_dim * 2, 2)
        self.softmax = nn.Softmax(dim=-1)

    def build_grid_adjacency(self, L, device):
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

    def mgct_forward(self, x):
        features = []
        out = x
        for layer in self.global_layers:
            out = layer(out)
            features.append(out)
        multi_scale_feat = torch.cat(features, dim=-1)
        fused_feat = self.global_fuse(multi_scale_feat)
        return self.global_proj(fused_feat)

    def asgc_forward(self, x):
        B, L, _ = x.size()
        adj = self.build_grid_adjacency(L, x.device).unsqueeze(0).expand(B, L, L)
        outputs = []
        out = x
        for linear in self.gnn_layers:
            out = torch.bmm(adj, out)
            out = linear(out)
            out = self.relu(out)
            outputs.append(out)
        multi_scale_feat = torch.cat(outputs, dim=-1)
        fused_feat = self.gnn_fuse(multi_scale_feat)
        return self.local_proj(fused_feat)

    def gcmf_forward(self, global_feat, local_feat):
        fused = torch.cat([global_feat, local_feat], dim=-1)
        attn_weights = self.softmax(self.fusion_linear(fused))
        weight_global = attn_weights[..., 0].unsqueeze(-1)
        weight_local = attn_weights[..., 1].unsqueeze(-1)
        return weight_global * global_feat + weight_local * local_feat

    def forward(self, x):
        global_feat = self.mgct_forward(x)
        local_feat = self.asgc_forward(x)
        fused_output = self.gcmf_forward(global_feat, local_feat)
        return fused_output

class ChangeClassifier(nn.Module):
    """变化检测分类头 (Change Detection Classifier)"""

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        return self.fc(x)

class ASGTNet(nn.Module):
    """
    Adaptive Spectral-Spatial Graph Transformer Network (ASGT-Net)
    Architecture:
    1. Spectral-Spatial Temporal Encoding Module (SSTEM)
    2. Hierarchical Global-Local Synergy (HGLS)
    """

    def __init__(self, num_channels=128, patch_size=5, num_classes=2,
                 global_hidden_dim=64, gnn_hidden_dim=6,
                 global_layers=2, gnn_layers=2, num_heads=4, encoder_dim=512):
        super().__init__()
        # Spectral-Spatial Temporal Encoding Module (SSTEM)
        self.sstem = SSTEM(
            num_channels=num_channels, window_size=patch_size,
            num_heads=num_heads, encoder_dim=encoder_dim
        )

        # Hierarchical Global-Local Synergy (HGLS)
        self.hgls = HGLS(
            input_dim=num_channels * 2, hidden_dim=global_hidden_dim,
            num_layers=global_layers, num_heads=num_heads,
            gnn_hidden_features=gnn_hidden_dim
        )

        # 分类器
        self.classifier = ChangeClassifier(global_hidden_dim, num_classes)

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.xavier_uniform_(module.in_proj_weight)

    def forward(self, x1, x2):
        # 输入预处理
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)

        # Spectral-Spatial Temporal Encoding Module (SSTEM)
        fused_feat = self.sstem(x1, x2)

        # Hierarchical Global-Local Synergy (HGLS)
        fused_output = self.hgls(fused_feat)

        # 分类
        last_hidden = fused_output[:, -1, :]
        return self.classifier(last_hidden)


# 计算模型参数量
def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params


# 计算FLOPs
def compute_flops(model, input_size):
    x1 = torch.randn(1, *input_size)
    x2 = torch.randn(1, *input_size)
    flops, _ = profile(model, inputs=(x1, x2))
    return flops


# 计算平均推理时间
def measure_inference_time(model, input_size, num_trials=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x1 = torch.randn(1, *input_size).to(device)
    x2 = torch.randn(1, *input_size).to(device)

    # 预热
    with torch.no_grad():
        for _ in range(10):
            model(x1, x2)

    # 测量时间
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_trials):
            model(x1, x2)
    end_time = time.time()

    # 计算平均推理时间（毫秒）
    avg_time = (end_time - start_time) * 1000 / num_trials
    return avg_time


# 主函数
def main():
    # 模型配置
    num_channels = 120
    patch_size = 5
    input_size = (num_channels, patch_size, patch_size)

    # 初始化模型
    model = ASGTNet(num_channels=num_channels)
    model.eval()

    # 计算参数量
    total_params = count_parameters(model)

    # 计算FLOPs
    total_flops = compute_flops(model, input_size)

    # 计算平均推理时间
    avg_inference_time = measure_inference_time(model, input_size)

    # 格式化结果并添加单位（修改部分）
    def format_with_unit(value, unit_type):
        """根据类型格式化数值和单位"""
        if unit_type == "params":
            if value >= 1e6:
                return f"{value/1e6:.2f}M"  # 百万参数
            elif value >= 1e3:
                return f"{value/1e3:.2f}K"  # 千参数
            else:
                return f"{value}个"
        elif unit_type == "flops":
            return f"{value/1e9:.3f}G"  # 十亿次运算
        elif unit_type == "time":
            return f"{value:.2f}ms"  # 毫秒
        return str(value)

    # 计算后生成结果字典（修改部分）
    results = {
    "参数量": f"{total_params / 1e6:.2f} M",
    "计算量(FLOPs)": f"{total_flops / 1e9:.3f} G",
    "平均推理时间": f"{avg_inference_time:.2f} ms"
}

    # 保存为JSON文件
    with open("model_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    print("模型性能指标已保存至 model_metrics.json")
    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
