import torch
import numpy as np
import math
from torchmetrics.image import StructuralSimilarityIndexMeasure


class FusionMetricsMeter:
    def __init__(self, device='cuda'):
        # 初始化 PyTorch 原生的 SSIM 计算器，并放到 GPU 上
        self.ssim_calc = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.reset()

    def reset(self):
        self.EN = 0.0
        self.SF = 0.0
        self.AG = 0.0
        self.SSIM = 0.0
        self.count = 0

    def update(self, fused_img, rgb, thermal):
        """
        fused_img, rgb, thermal: (B, C, H, W) 张量，必须在 [0, 1] 范围内
        """
        B = fused_img.size(0)
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        rgb = 0.299 * r + 0.587 * g + 0.114 * b
        thermal = thermal[:, 0:1]
        # ==========================================================
        # 1. 在 GPU 上极速计算 SSIM (直接使用 Tensor, 无需转 CPU)
        # ==========================================================
        # 分别计算融合图与 RGB、Thermal 的 SSIM
        with torch.no_grad():
            ssim_r = self.ssim_calc(fused_img, rgb)
            ssim_t = self.ssim_calc(fused_img, thermal)
            # torchmetrics 默认返回 batch 的平均值，我们需要乘回 B 以累加总分
            self.SSIM += ((ssim_r + ssim_t) / 2.0).item() * B

        # ==========================================================
        # 2. 计算 EN, SF, AG (这些指标需要特定的像素遍历，暂时转回 Numpy 计算)
        # ==========================================================
        fused_np = (fused_img.detach().cpu().numpy()*255.0).clip(0,255)
        rgb_np = rgb.detach().cpu().numpy()
        thermal_np = thermal.detach().cpu().numpy()

        for i in range(B):
            f_img = fused_np[i].transpose(1, 2, 0)

            # 转灰度图
            f_gray = self.rgb2gray(f_img)

            self.EN += self.calculate_en(f_gray)
            self.SF += self.calculate_sf(f_gray)
            self.AG += self.calculate_ag(f_gray)

        # 记录累加的样本总数
        self.count += B

    def get_scores(self):
        if self.count == 0:
            return 0, 0, 0, 0
        return self.EN / self.count, self.SF / self.count, self.AG / self.count, self.SSIM / self.count

    @staticmethod
    def rgb2gray(rgb):
        if rgb.shape[2] == 3:
            return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
        return rgb.squeeze()

    @staticmethod
    def calculate_en(image):
        histogram, _ = np.histogram(image, bins=256, range=(0, 255))
        histogram = histogram / float(np.sum(histogram))
        entropy = -np.sum(histogram * np.log2(histogram + 1e-7))
        return entropy

    @staticmethod
    def calculate_sf(image):
        RF = np.diff(image, axis=0) ** 2
        CF = np.diff(image, axis=1) ** 2
        RF = np.mean(RF)
        CF = np.mean(CF)
        return math.sqrt(RF + CF)

    @staticmethod
    def calculate_ag(image):
        dx = np.diff(image, axis=1)
        dy = np.diff(image, axis=0)
        dx = dx[:-1, :]
        dy = dy[:, :-1]
        ag = np.mean(np.sqrt((dx ** 2 + dy ** 2) / 2.0))
        return ag


