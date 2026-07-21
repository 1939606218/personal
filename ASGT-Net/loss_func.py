"""
损失函数模块
包含自适应Focal Loss和动态组合损失
"""

import torch
from torch import nn
import torch.nn.functional as F


class AdaptiveFocalLoss(nn.Module):
    """
    自适应Focal Loss损失函数

    特点：
    1. 动态调整类别权重，基于当前batch的类别分布
    2. 自适应调整gamma值，困难样本获得更大的gamma
    3. 添加分类边界增强损失

    Args:
        num_classes: 类别数量
        gamma: Focal Loss的gamma参数（动态调整基准值）
        margin: 分类边界增强的边际值
        margin_weight: 边界损失的权重
        alpha: 类别平衡因子，用于平滑类别权重
    """

    def __init__(self, num_classes, gamma=2.0, margin=0.3, margin_weight=0.5, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.margin = margin
        self.margin_weight = margin_weight
        self.alpha = alpha
        self.num_classes = num_classes
        self.eps = 1e-8

    def _compute_class_weights(self, targets):
        """
        计算类别权重

        Args:
            targets: 标签张量 [B]

        Returns:
            class_weights: 类别权重 [num_classes]
        """
        batch_counts = torch.bincount(targets, minlength=self.num_classes).float()

        # 反频率权重 + 平滑因子
        class_weights = 1.0 / (batch_counts + self.alpha)
        class_weights = class_weights / class_weights.sum() * self.num_classes

        return class_weights

    def _compute_margin_loss(self, inputs, targets, class_weights):
        """
        计算分类边界增强损失

        Args:
            inputs: 预测logits [B, num_classes]
            targets: 标签 [B]
            class_weights: 类别权重 [num_classes]

        Returns:
            margin_loss: 边界损失 [B]
        """
        device = inputs.device
        num_classes = inputs.size(1)

        # 获取真实类别的logits [B]
        true_logits = inputs[torch.arange(inputs.size(0)), targets]

        # 获取其他类别的最大logits [B]
        mask = torch.ones_like(inputs, dtype=torch.bool)
        mask[torch.arange(inputs.size(0)), targets] = False
        masked_logits = inputs.masked_fill(~mask, float('-inf'))
        max_other_logits = masked_logits.max(dim=1)[0]

        # 计算边界损失
        margin_loss = F.relu(max_other_logits - (true_logits + self.margin))

        return margin_loss

    def forward(self, inputs, targets):
        """
        前向传播

        Args:
            inputs: 预测logits [B, num_classes]
            targets: 标签 [B]

        Returns:
            total_loss: 总损失
        """
        # 计算类别权重
        class_weights = self._compute_class_weights(targets)

        # 计算基础交叉熵损失 [B]
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=class_weights)

        # 计算预测置信度pt [B]
        pt = torch.exp(-ce_loss.detach())

        # 自适应gamma：困难样本gamma更大
        adaptive_gamma = self.gamma * (1.0 - pt + self.eps)

        # Focal Loss
        focal_loss = (class_weights[targets] * (1 - pt) ** adaptive_gamma * ce_loss).mean()

        # 分类边界增强损失
        margin_loss = self._compute_margin_loss(inputs, targets, class_weights).mean()

        # 组合损失
        total_loss = focal_loss + self.margin_weight * margin_loss

        return total_loss


class DynamicCombinedLoss(nn.Module):
    """
    动态组合损失函数

    结合交叉熵损失（可选Label Smoothing）和自适应Focal Loss，
    并动态调整两个损失的权重。

    Args:
        num_classes: 类别数量
        lambda_: 初始的CE权重（FL权重为1-lambda_）
        smoothing: Label Smoothing系数（0表示不使用）
        lambda_range: lambda的动态调整范围 [min, max]
        use_dynamic: 是否使用动态权重调整
    """

    def __init__(self, num_classes, lambda_=0.7, smoothing=0.1,
                 lambda_range=(0.3, 0.9), use_dynamic=False):
        super().__init__()
        self.fl_loss = AdaptiveFocalLoss(num_classes)
        self.lambda_ = lambda_
        self.smoothing = smoothing
        self.lambda_min = lambda_range[0]
        self.lambda_max = lambda_range[1]
        self.use_dynamic = use_dynamic

        # EMA平滑的lambda（用于动态调整）
        self.register_buffer('ema_loss_ce', torch.tensor(0.0))
        self.register_buffer('ema_loss_fl', torch.tensor(0.0))
        self.register_buffer('step', torch.tensor(0))
        self.ema_momentum = 0.9

    def _compute_smoothed_loss(self, inputs, targets):
        """
        计算带Label Smoothing的交叉熵损失

        Args:
            inputs: 预测logits [B, num_classes]
            targets: 标签 [B]

        Returns:
            smoothed_ce_loss: 平滑后的CE损失
        """
        if self.smoothing == 0:
            return F.cross_entropy(inputs, targets)

        num_classes = inputs.size(1)
        log_probs = F.log_softmax(inputs, dim=-1)

        with torch.no_grad():
            # 创建平滑标签
            smoothed_targets = torch.full_like(
                log_probs, self.smoothing / (num_classes - 1)
            )
            smoothed_targets.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)

        smoothed_ce_loss = - (smoothed_targets * log_probs).sum(dim=1).mean()

        return smoothed_ce_loss

    def _update_dynamic_lambda(self, loss_ce, loss_fl):
        """
        根据损失值动态更新lambda

        当某个损失变大时，减小其权重，实现自适应平衡

        Args:
            loss_ce: CE损失值
            loss_fl: FL损失值
        """
        # EMA更新
        self.ema_loss_ce = self.ema_momentum * self.ema_loss_ce + (1 - self.ema_momentum) * loss_ce
        self.ema_loss_fl = self.ema_momentum * self.ema_loss_fl + (1 - self.ema_momentum) * loss_fl

        # 动态调整lambda
        if self.ema_loss_ce > self.ema_loss_fl:
            # CE损失大，减小CE权重
            self.lambda_ = max(self.lambda_min, self.lambda_ * 0.95)
        else:
            # FL损失大，增加CE权重
            self.lambda_ = min(self.lambda_max, self.lambda_ * 1.05)

        self.step += 1

    def forward(self, inputs, targets):
        """
        前向传播

        Args:
            inputs: 预测logits [B, num_classes]
            targets: 标签 [B]

        Returns:
            combined_loss: 组合损失
        """
        # 计算带Label Smoothing的CE损失
        loss_ce = self._compute_smoothed_loss(inputs, targets)

        # 计算自适应Focal Loss
        loss_fl = self.fl_loss(inputs, targets)

        # 动态调整lambda
        if self.use_dynamic:
            self._update_dynamic_lambda(loss_ce, loss_fl)

        # 组合损失
        combined_loss = self.lambda_ * loss_ce + (1 - self.lambda_) * loss_fl

        return combined_loss

    def get_current_lambda(self):
        """获取当前的lambda值"""
        return self.lambda_.item()

