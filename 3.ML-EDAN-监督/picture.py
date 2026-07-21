import torch
import numpy as np
import matplotlib.pyplot as plt
from data.data_process import loadData
from data.pre_process import normalization, applyPCA, create_patches
from model.model import ML_EDAN
from Global import *
import scipy.io as sio
import torch.utils.data as Data
import torch.nn.functional as F


class ImagePredictor:
    def __init__(self, model_path, in_channel=pca_channel):
        self.model_path = model_path
        self.in_channel = in_channel
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = ML_EDAN(in_channel=self.in_channel).to(self.device)
        self.net.load_state_dict(torch.load(f"{model_path}/best_model.pth"))
        self.net.eval()

    def predict_and_visualize(self, all_iter, Y):
        all_predicted_labels = []

        with torch.no_grad():
            for x1, x2, _ in all_iter:
                x1, x2 = x1.to(self.device), x2.to(self.device)
                y_hat, _, _ = self.net(x1, x2)
                predicted_labels = y_hat.argmax(dim=1).cpu().numpy().ravel()
                all_predicted_labels.extend(predicted_labels)

        all_predicted_labels = np.array(all_predicted_labels)
        labels = Y
        valid_mask = labels != 2  # 有效区域掩码（标签≠2）

        # 校验预测数量与有效区域数量
        assert len(all_predicted_labels) == valid_mask.sum(), \
            f"预测数量({len(all_predicted_labels)})与有效区域数量({valid_mask.sum()})不匹配"

        # 初始化预测图像：无效区域(标签2)设为2，有效区域填充预测值(0/1)
        predicted_image = np.full_like(labels, 2, dtype=np.int32)  # 初始化为2（灰色）
        predicted_image[valid_mask] = all_predicted_labels  # 有效区域填充0/1

        # 计算错误掩码（仅在有效区域内计算）
        error_mask = valid_mask & (predicted_image != labels)

        # 生成可视化图形
        self._generate_visualization(labels, predicted_image, error_mask)

        return predicted_image

    def _generate_visualization(self, labels, predicted_image, error_mask):
        """生成可视化图形（0黑、1白、2灰）"""
        height, width = labels.shape

        # 原始标签处理：2映射为0.5灰度（灰色）
        labels_display = labels.astype(float)
        labels_display[labels == 2] = 0.5  # 灰色对应0.5灰度值

        # 预测结果处理：2保持为灰色，错误点叠加红色
        predicted_display = predicted_image.astype(float)
        predicted_display[error_mask] = np.nan  # 错误点在灰度图中隐藏，仅通过红色标记显示

        # 创建红色错误覆盖层（仅在有效区域显示错误）
        error_overlay = np.zeros((height, width, 4), dtype=np.uint8)
        error_overlay[error_mask] = [255, 0, 0, 255]  # 半透明红色（错误点）

        # 绘制子图
        plt.figure(figsize=(15, 6))

        # 原始标签（0黑、1白、2灰）
        plt.subplot(1, 2, 1)
        plt.imshow(labels_display, cmap='gray', vmin=0, vmax=1)
        plt.title('Ground Truth (0=Black, 1=White, 2=Gray)')
        plt.axis('off')

        # 预测结果（灰色为无效区域，红色为错误点）
        plt.subplot(1, 2, 2)
        plt.imshow(predicted_display, cmap='gray', vmin=0, vmax=1)
        plt.imshow(error_overlay, interpolation='none')
        plt.title('Prediction (Gray=Invalid, Red=Error)')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(f"{self.model_path}/predicted_with_errors.png", bbox_inches='tight')
        plt.close()


def generater(X_1, X_2, Y, batchsize, windowSize):
    """生成有效区域样本数据加载器"""
    x1_patches, y1_patches = create_patches(X_1, Y, window_size=windowSize)
    x1_patches = np.transpose(x1_patches, (0, 3, 1, 2))

    x2_patches, y2_patches = create_patches(X_2, Y, window_size=windowSize)
    x2_patches = np.transpose(x2_patches, (0, 3, 1, 2))

    # 过滤无效区域样本（标签=2为无效）
    valid_mask = y1_patches != 2
    x1_valid = x1_patches[valid_mask]
    x2_valid = x2_patches[valid_mask]
    y_valid = y1_patches[valid_mask]

    # 转换为Tensor
    x1_tensor = torch.from_numpy(x1_valid).float()
    x2_tensor = torch.from_numpy(x2_valid).float()
    y_tensor = torch.from_numpy(y_valid).long()
    y_tensor = F.one_hot(y_tensor)

    # 创建数据加载器
    dataset = Data.TensorDataset(x1_tensor, x2_tensor, y_tensor)
    data_loader = Data.DataLoader(dataset, batch_size=batchsize, shuffle=False, num_workers=0)

    # 校验数据一致性
    valid_total = (Y != 2).sum()
    filtered_total = len(y_valid)
    print(f"数据校验：有效区域总数={valid_total}, 过滤后样本数={filtered_total}")
    return filtered_total, data_loader


if __name__ == "__main__":
    X1, X2, Y = loadData(dataname)
    X1 = normalization(X1)
    X2 = normalization(X2)
    X1 = applyPCA(X1, channel=pca_channel)
    X2 = applyPCA(X2, channel=pca_channel)

    batchsize = 32
    windowSize = 5
    _, all_iter = generater(X1, X2, Y, batchsize, windowSize)

    model_path_map = {
        'hermiston': r'Z:\pycharm\pythonProject\3.ML-EDAN-main\result\hermiston',
        'farm': r'Z:\pycharm\pythonProject\3.ML-EDAN-main\result\farm',
        'river': r'Z:\pycharm\pythonProject\3.ML-EDAN-main\result\river'
    }
    model_path = model_path_map.get(dataname, None)

    if not model_path:
        raise ValueError(f"未找到数据集 {dataname} 对应的模型路径")

    predictor = ImagePredictor(model_path)
    predictor.predict_and_visualize(all_iter, Y)