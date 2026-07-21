import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from thop import profile
from torchsummary import summary
import json


class FactorizedSpatialAttention(nn.Module):
    """
    Factorized 2D Spatial Attention (FSA)

    Parallel decomposition of 2D window attention into two independent 1D attentions:
        1) Horizontal Attention: row-wise (captures horizontal patterns)
        2) Vertical Attention:   column-wise (captures vertical patterns)
    Both paths share the same input and are merged via concatenation + projection.

    Complexity: O(2 * ws^2 * C) per window, vs O(ws^4 * C) for standard 2D attention.
    Each direction has independent QKV projections for direction-specific patterns.
    """

    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        # 方向独立 QKV 投影
        self.h_qkv = nn.Linear(dim, dim * 3)
        self.v_qkv = nn.Linear(dim, dim * 3)
        # 拼接后投影回原维度
        self.proj = nn.Linear(dim * 2, dim)

    def _attn_1d(self, x, qkv_layer):
        """1D multi-head self-attention for sequences of length ws"""
        B, N, C = x.shape
        qkv = qkv_layer(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return out

    def forward(self, x):
        # x: [B*num_windows, ws, ws, C]
        B_n, ws_h, ws_w, C = x.shape

        # === Parallel: Horizontal (row-wise) ===
        x_h = x.reshape(B_n * ws_h, ws_w, C)
        h_out = self._attn_1d(x_h, self.h_qkv)          # [B_n*ws_h, ws_w, C]
        h_out = h_out.reshape(B_n, ws_h, ws_w, C)

        # === Parallel: Vertical (column-wise) ===
        x_v = x.permute(0, 2, 1, 3).reshape(B_n * ws_w, ws_h, C)
        v_out = self._attn_1d(x_v, self.v_qkv)          # [B_n*ws_w, ws_h, C]
        v_out = v_out.reshape(B_n, ws_w, ws_h, C).permute(0, 2, 1, 3)

        # Concat + Project
        merged = torch.cat([h_out, v_out], dim=-1)       # [B_n, ws, ws, 2C]
        return self.proj(merged)                           # [B_n, ws, ws, C]


class SpectralSpatialDisentanglementTransformer(nn.Module):
    """
    Spectral-Spatial Disentanglement Transformer (SSDT)
    Implements dual-path feature decoupling with:
    1) Spectral Attention Path: Captures band-wise interactions
       - Full TransformerEncoderLayer (Attn + FFN), global sequence attention
    2) Spatial Window Attention: Models local spatial dependencies
       - FactorizedSpatialAttention: horizontal + vertical decomposition
    Features fused via learnable spectral-spatial weighting.
    """

    def __init__(self, num_channels, window_size, num_heads, encoder_dim):
        super().__init__()
        # 光谱注意力分支 — 全局序列 attention + FFN
        self.spectral_attn = nn.TransformerEncoderLayer(
            d_model=num_channels, nhead=num_heads, dim_feedforward=encoder_dim,
            dropout=0.5, batch_first=True
        )
        self.spectral_norm = nn.LayerNorm(num_channels)

        # 空间注意力分支 — 因式分解注意力 + LN + 残差
        self.spatial_attn = FactorizedSpatialAttention(
            dim=num_channels, num_heads=num_heads, window_size=window_size
        )
        self.spatial_norm1 = nn.LayerNorm(num_channels)
        self.spatial_norm2 = nn.LayerNorm(num_channels)
        self.spatial_dropout = nn.Dropout(0.5)
        self.window_size = window_size

        # 特征融合
        self.fusion_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.dropout = nn.Dropout(0.5)

    def window_partition(self, x):
        # x: [B, L, C]
        B, L, C = x.shape
        H = int(L ** 0.5)
        if H * H < L:
            H += 1
        W = (L + H - 1) // H
        pad_len = H * W - L
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        x = x.view(B, H, W, C)
        windows = x.view(B, H // self.window_size, self.window_size,
                         W // self.window_size, self.window_size, C)
        windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, self.window_size, self.window_size, C)
        return windows, B, H, W, L

    def forward(self, x):
        B, L, C = x.shape

        # 光谱特征处理
        spectral_feat = self.spectral_attn(x)
        spectral_feat = self.spectral_norm(spectral_feat)

        # 空间特征处理 (因式分解 + 残差 + LN)
        windows, B_orig, H, W, L_orig = self.window_partition(x)
        normed = self.spatial_norm1(windows)
        spatial_out = self.spatial_attn(normed)                # [B_n, ws, ws, C]
        spatial_out = self.spatial_dropout(spatial_out) + windows
        # 每个 window 内 pool → [B_n, C]
        spatial_feat = spatial_out.mean(dim=(1, 2))
        # 恢复到 [B, L, C]: 每个 token 对应其所在 window 的聚合特征
        num_windows = B_orig * H * W  # 不对，应该是 (H/ws)*(W/ws)
        nwh = H // self.window_size
        nww = W // self.window_size
        spatial_feat = spatial_feat.view(B_orig, nwh, nww, C)
        spatial_feat = spatial_feat.view(B_orig, -1, C)       # [B, nwh*nww, C]
        spatial_feat = spatial_feat[:, :L_orig, :]             # 去除 padding
        spatial_feat = self.spatial_norm2(spatial_feat)

        # 特征融合
        weights = F.softmax(self.fusion_weights, dim=0)
        fused_feat = weights[0] * spectral_feat + weights[1] * spatial_feat
        return self.dropout(fused_feat)

class DualReciprocalAttentionFusion(nn.Module):
    """
    Dual Reciprocal Attention Fusion (DRAF)
    Features:
    - Bidirectional cross-attention streams
    - Mutual feature recalibration
    - Symmetric temporal feature fusion
    - Residual feature concatenation
    """

    def __init__(self, num_channels, num_heads):
        super().__init__()
        self.cross_attn_12 = nn.MultiheadAttention(num_channels, num_heads, batch_first=True)
        self.cross_attn_21 = nn.MultiheadAttention(num_channels, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(2 * num_channels)

    def forward(self, x1, x2):
        feat_12, _ = self.cross_attn_12(x1, x2, x2)
        feat_21, _ = self.cross_attn_21(x2, x1, x1)
        fused_feat = torch.cat([feat_12, feat_21], dim=-1)
        return self.norm(fused_feat)


class MultiScaleGlobalContextTransformer(nn.Module):
    """
    Multi-scale Global Context Transformer (MGCT)
    Features:
    - Deep feature pyramid extraction
    - Multi-layer feature aggregation
    - Global dependency modeling
    """

    def __init__(self, input_dim, hidden_dim, num_layers, num_heads):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim,
                dropout=0.5, batch_first=True
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
    """Graph Convolution Layer with adaptive normalization"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        out = torch.bmm(adj, x)
        out = self.linear(out)
        return out


def build_grid_adjacency(L, device):
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
    """
    Adaptive Spatial Graph Convolution (ASGC)
    Features:
    - Topology-aware graph learning
    - Multi-hop neighborhood aggregation
    - Dynamic adjacency normalization
    """

    def __init__(self, in_features, hidden_features, num_layers):
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
    """
    Gated Cross-Modal Fusion (GCMF)
    Features:
    - Attention-aware feature gating
    - Learnable fusion weights
    - Feature recalibration
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(hidden_dim * 2, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, global_feat, local_feat):
        fused = torch.cat([global_feat, local_feat], dim=-1)
        attn_weights = self.softmax(self.linear(fused))
        weight_global = attn_weights[..., 0].unsqueeze(-1)
        weight_local = attn_weights[..., 1].unsqueeze(-1)
        return weight_global * global_feat + weight_local * local_feat

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
    1. Spectral-Spatial Temporal Encoding Module (SSTEM):
        - SSDT: SpectralSpatialDisentanglementTransformer
        - DRAF：DualReciprocalAttentionFusion
    2. Hierarchical Global-Local Synergy (HGLS):
        - MGCT: Multi-scale Global Context Transformer
        - ASGC: Adaptive Spatial Graph Convolution
        - GCMF: Gated Cross-Modal Fusion
    """

    def __init__(self, num_channels=128, patch_size=5, num_classes=2,
                 global_hidden_dim=64, gnn_hidden_dim=64,
                 global_layers=2, gnn_layers=2, num_heads=4,encoder_dim = 512):
        super().__init__()
        # Spectral-Spatial Temporal Encoding Module (SSTEM)
        # 融合原"特征编码器"与"时序特征融合"模块
        self.spectral_spatial_encoder = SpectralSpatialDisentanglementTransformer(
            num_channels, patch_size, num_heads=num_heads,encoder_dim = encoder_dim
        )
        self.temporal_fuser = DualReciprocalAttentionFusion(num_channels, num_heads=num_heads)

        # Hierarchical Global-Local Synergy (HGLS)
        self.global_branch = MultiScaleGlobalContextTransformer(
            input_dim=num_channels * 2, hidden_dim=global_hidden_dim,
            num_layers=global_layers, num_heads=num_heads
        )
        self.local_branch = AdaptiveSpatialGraphConvolution(
            in_features=num_channels * 2, hidden_features=gnn_hidden_dim,
            num_layers=gnn_layers
        )

        # 特征融合与分类
        self.stream_fusion = GatedCrossModalFusion(hidden_dim=global_hidden_dim)
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
        # 融合特征编码器与时序特征融合
        x1_enc = self.spectral_spatial_encoder(x1)
        x2_enc = self.spectral_spatial_encoder(x2)
        fused_feat = self.temporal_fuser(x1_enc, x2_enc)

        # Hierarchical Global-Local Synergy (HGLS)
        global_feat = self.global_branch(fused_feat)
        local_feat = self.local_branch(fused_feat)

        # 特征融合与分类
        fused_output = self.stream_fusion(global_feat, local_feat)
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
