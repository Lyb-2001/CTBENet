import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm


# class FeatureDiffusionKD(nn.Module):
#     def __init__(self, feature_dim1, feature_dim2, latent_dim=64, timesteps=50,
#                  beta_schedule="linear", device='cuda'):
#         super().__init__()
#
#         # 初始化参数
#         self.timesteps = timesteps
#         self.device = device
#         self.latent_dim = latent_dim
#
#         # 修正噪声调度计算
#         if beta_schedule == "linear":
#             self.beta = torch.linspace(0.0001, 0.02, timesteps).to(device)  # 降低初始beta值
#         elif beta_schedule == "cosine":
#             steps = timesteps + 1
#             x = torch.linspace(0, timesteps, steps, device=device)
#             alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
#             alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
#             betas = 1 - (alphas_cumprod[1:] / (alphas_cumprod[:-1] + 1e-8))  # 防止除零
#             self.beta = torch.clip(betas, 0, 0.999)
#
#         # 计算alpha相关参数
#         self.alpha = 1 - self.beta
#         self.alpha_bar = torch.cumprod(self.alpha, dim=0)
#
#         # 修正sigma计算：使用DDIM正确公式并添加数值稳定性处理
#         alpha_bar_prev = torch.cat([torch.ones(1, device=device), self.alpha_bar[:-1]])
#         self.sigma = torch.sqrt(
#             ((1 - alpha_bar_prev[1:]) * (1 - self.alpha[1:])) / (1 - self.alpha_bar[1:] + 1e-8)
#         ).clamp(max=0.999)  # 确保非负并限制最大值
#
#         # 特征编码器（添加归一化层）
#         self.encoder1 = nn.Sequential(
#             nn.Conv2d(feature_dim1, latent_dim, 1),
#             nn.GroupNorm(8, latent_dim),  # 增加分组数
#             nn.ReLU(),
#             nn.Dropout2d(0.1)  # 添加dropout增强稳定性
#         )
#         self.encoder2 = nn.Sequential(
#             nn.Conv2d(feature_dim2, latent_dim, 1),
#             nn.GroupNorm(8, latent_dim),
#             nn.ReLU(),
#             nn.Dropout2d(0.1)
#         )
#
#         # 解码器（添加归一化）
#         self.decoder1 = nn.Sequential(
#             nn.Conv2d(latent_dim, feature_dim1, 1),
#             nn.GroupNorm(8, feature_dim1)
#         )
#         self.decoder2 = nn.Sequential(
#             nn.Conv2d(latent_dim, feature_dim2, 1),
#             nn.GroupNorm(8, feature_dim2)
#         )
#
#         # 噪声预测器（优化结构）
#         class DenoiserBlock(nn.Module):
#             def __init__(self, channels):
#                 super().__init__()
#                 self.conv = nn.Sequential(
#                     nn.Conv2d(channels, channels, 3, padding=1),
#                     nn.GroupNorm(4, channels),
#                     nn.SiLU(),
#                     nn.Conv2d(channels, channels, 3, padding=1),
#                 )
#                 self.norm = nn.GroupNorm(4, channels)
#
#             def forward(self, x):
#                 return self.norm(x + self.conv(x))
#
#         self.denoiser1 = nn.Sequential(
#             *[DenoiserBlock(latent_dim) for _ in range(3)],
#             nn.Conv2d(latent_dim, latent_dim, 1)
#         )
#         self.denoiser2 = nn.Sequential(
#             *[DenoiserBlock(latent_dim) for _ in range(3)],
#             nn.Conv2d(latent_dim, latent_dim, 1)
#         )
#
#         # 噪声水平预测（优化结构）
#         self.noise_level_pred = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(latent_dim, 64, 1),
#             nn.SiLU(),
#             nn.Conv2d(64, 1, 1),
#             nn.Sigmoid()
#         )
#
#     def get_noise_level(self, x):
#         """改进噪声水平预测"""
#         noise_level = self.noise_level_pred(x).view(-1)
#         return torch.clamp((noise_level * self.timesteps).long(), 1, self.timesteps - 1)
#
#     def add_noise(self, x, t=None):
#         """改进噪声添加：特征归一化"""
#         x = F.layer_norm(x, x.shape[1:])  # 层归一化
#
#         if t is None:
#             t = self.get_noise_level(x)
#         elif isinstance(t, (int, float)):
#             t = torch.full((x.size(0),), t, device=self.device, dtype=torch.long)
#
#         alpha_bar = self.alpha_bar[t].view(-1, 1, 1, 1)
#         noise = torch.randn_like(x)
#         noised = torch.sqrt(alpha_bar) * x + torch.sqrt(1 - alpha_bar) * noise
#         return noised, noise, t
#
#     def denoise_step_ddim(self, x_t, t, predicted_noise):
#         """改进去噪步骤：数值稳定性处理"""
#         alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
#         alpha_bar_prev = self.alpha_bar[t - 1].view(-1, 1, 1, 1) if t.min() > 0 else alpha_bar_t
#
#         # 改进x0预测的数值稳定性
#         sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t.clamp(min=1e-8))
#         x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / sqrt_alpha_bar_t
#
#         # 动态调整特征范围
#         x0_pred = torch.tanh(x0_pred)  # 限制到[-1,1]范围
#
#         # 计算x_{t-1}
#         sigma_t = self.sigma[t - 1].view(-1, 1, 1, 1) if t.min() > 0 else 0
#         x_prev = (torch.sqrt(alpha_bar_prev) * x0_pred +
#                   torch.sqrt(1 - alpha_bar_prev - sigma_t ** 2) * predicted_noise +
#                   sigma_t * torch.randn_like(x_t))
#         return x_prev
#
#     def denoise_feature_ddim(self, noisy_feature, start_t, denoiser):
#         """改进去噪过程：梯度裁剪"""
#         x_t = noisy_feature
#         for t_step in reversed(range(start_t)):
#             t = torch.full((x_t.size(0),), t_step, device=self.device)
#             with torch.no_grad():
#                 predicted_noise = denoiser(x_t.detach())
#             x_t = self.denoise_step_ddim(x_t, t, predicted_noise)
#             torch.nn.utils.clip_grad_norm_(x_t, 1.0)  # 梯度裁剪
#         return x_t
#
#     def diff_loss_1to2(self, feature1, feature2):
#         # 特征归一化
#         feature1 = F.layer_norm(feature1, feature1.shape[1:])
#         feature2 = F.layer_norm(feature2, feature2.shape[1:])
#
#         latent1 = self.encoder1(feature1)
#         latent2 = self.encoder2(feature2.detach())
#
#         # 训练denoiser2
#         noisy_latent2, noise2, t2 = self.add_noise(latent2.detach())
#         pred_noise2 = self.denoiser2(noisy_latent2)
#         noise_loss = F.smooth_l1_loss(pred_noise2, noise2, beta=0.1)
#
#         # 对齐过程
#         start_t = min(10, self.timesteps // 5)  # 动态起始步
#         noisy_latent1, _, _ = self.add_noise(latent1, t=start_t)
#         denoised = self.denoise_feature_ddim(noisy_latent1, start_t, self.denoiser2)
#         align_loss = F.mse_loss(denoised, latent2)
#
#         # 总损失
#         return align_loss + 0.5 * noise_loss + 0.1 * F.mse_loss(self.decoder1(latent1), feature1)
#     def diff_loss_2to1(self, feature2, feature1):
#         """模型2 (A) 特征用 模型1 (B) 的扩散模型去噪，并与模型1 (B) 特征蒸馏 (**修正Denoiser和蒸馏对象, 反向蒸馏方向** )"""
#         feature2 = F.layer_norm(feature2, feature2.shape[1:])
#         feature1 = F.layer_norm(feature1, feature1.shape[1:])
#
#         latent2 = self.encoder2(feature2)
#         latent1 = self.encoder1(feature1.detach())
#
#         # 训练denoiser2
#         noisy_latent1, noise1, t1 = self.add_noise(latent1.detach())
#         pred_noise1 = self.denoiser1(noisy_latent1)
#         noise_loss = F.smooth_l1_loss(pred_noise1, noise1, beta=0.1)
#
#         # 对齐过程
#         start_t = min(10, self.timesteps // 5)  # 动态起始步
#         noisy_latent2, _, _ = self.add_noise(latent2, t=start_t)
#         denoised = self.denoise_feature_ddim(noisy_latent2, start_t, self.denoiser1)
#         align_loss = F.mse_loss(denoised, latent1)
#
#         # 总损失
#         return align_loss + 0.5 * noise_loss + 0.1 * F.mse_loss(self.decoder2(latent2), feature2)
#
#
# class Bottleneck(nn.Module):
#     """ResNet-style bottleneck block - 轻量级实现"""
#
#     def __init__(self, in_planes, hidden_planes):
#         super().__init__()
#
#         self.conv1 = nn.Conv2d(in_planes, hidden_planes, kernel_size=1, bias=False)
#         self.bn1 = nn.GroupNorm(4, hidden_planes)
#
#         self.conv2 = nn.Conv2d(hidden_planes, hidden_planes, kernel_size=3,
#                                padding=1, bias=False)
#         self.bn2 = nn.GroupNorm(4, hidden_planes)
#
#         self.conv3 = nn.Conv2d(hidden_planes, in_planes, kernel_size=1, bias=False)
#         self.bn3 = nn.GroupNorm(4, in_planes)
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
class FeatureDiffusionKD(nn.Module):
    """修正后的双向特征扩散蒸馏模块"""

    def __init__(self,
                 feature_dim1,
                 feature_dim2,
                 latent_dim=64,
                 timesteps=50,
                 beta_schedule="cosine",
                 device='cuda'):
        super().__init__()

        self.timesteps = timesteps
        self.device = device
        self.latent_dim = latent_dim

        # 改进的噪声调度（带数值安全检查）
        if beta_schedule == "linear":
            beta = torch.linspace(0.0001, 0.02, timesteps)  # 更温和的噪声增长
        elif beta_schedule == "cosine":
            steps = timesteps + 2  # 增加边界保护
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + 0.01) / 1.01 * math.pi * 0.5) ** 2  # 调整偏移量
            alphas_cumprod = alphas_cumprod / (alphas_cumprod[0] + 1e-7)  # 防止除以零
            betas = 1 - (alphas_cumprod[1:-1] / alphas_cumprod[:-2])  # 调整索引范围
            beta = torch.clip(betas, 0, 0.02)  # 限制最大beta值

        self.beta = beta.to(device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        # 改进的sigma计算（带数值稳定性处理）
        alpha_bar_ratio = self.alpha_bar[:-1] / (self.alpha_bar[1:] + 1e-7)
        sigma_sq = (1 - alpha_bar_ratio) * (1 - self.alpha_bar[1:]) / (1 - self.alpha_bar[:-1] + 1e-7)
        self.sigma = torch.sqrt(torch.clamp(sigma_sq, min=0))

        # 增强的编码器（添加归一化层）
        self.encoder1 = nn.Sequential(
            nn.Conv2d(feature_dim1, latent_dim, kernel_size=1),
            nn.GroupNorm(4, latent_dim),
            nn.ReLU(inplace=True),

        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(feature_dim2, latent_dim, kernel_size=1),
            nn.GroupNorm(4, latent_dim),
            nn.ReLU(inplace=True),

        )

        # 简化解码器结构
        self.decoder1 = nn.Conv2d(latent_dim, feature_dim1, kernel_size=1)
        self.decoder2 = nn.Conv2d(latent_dim, feature_dim2, kernel_size=1)

        # 更稳定的Denoiser结构（添加跳跃连接）
        class DenoiserBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.GroupNorm(4, channels),
                    nn.ReLU(),
                    nn.Conv2d(channels, channels, 3, padding=1)
                )
                self.gamma = nn.Parameter(torch.zeros(1))

            def forward(self, x):
                return x + self.gamma * self.conv(x)

        self.denoiser1 = nn.Sequential(
            DenoiserBlock(latent_dim),
            DenoiserBlock(latent_dim),
            nn.Conv2d(latent_dim, latent_dim, 1)
        )

        self.denoiser2 = nn.Sequential(
            DenoiserBlock(latent_dim),
            DenoiserBlock(latent_dim),
            nn.Conv2d(latent_dim, latent_dim, 1)
        )

        # 改进的噪声水平预测器
        self.noise_level_pred = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def get_noise_level(self, x):
        """带稳定处理的噪声水平预测"""
        noise_level = self.noise_level_pred(x.detach())  # 分离梯度
        return torch.clamp(noise_level, 0.1, 0.9) * self.timesteps

    def add_noise(self, x, t=None):
        """带数值安全检查的加噪过程"""
        if t is None:
            t = self.get_noise_level(x).long()
        elif isinstance(t, (int, float)):
            t = torch.full((x.size(0),), t, device=self.device).long()

        t = torch.clamp(t, 0, self.timesteps - 1)  # 确保索引有效

        alpha_bar = self.alpha_bar[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(x) * 0.5  # 限制噪声幅度

        # 带安全限制的加噪公式
        noised_x = torch.sqrt(alpha_bar) * x + torch.sqrt(1 - alpha_bar + 1e-6) * noise
        return noised_x, noise, t

    def denoise_step_ddim(self, t_step, x_t, t, predicted_noise, denoiser):
        """修正后的稳定去噪步骤"""
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)

        # 安全处理时间索引
        prev_t = torch.clamp(t - 1, min=0)
        alpha_bar_prev = self.alpha_bar[prev_t].view(-1, 1, 1, 1)
        sigma_t = self.sigma[prev_t].view(-1, 1, 1, 1) if t_step > 0 else torch.tensor(0.0,device=self.device).view(-1,1,1,1)

        # 带截断的x0预测
        x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / (torch.sqrt(alpha_bar_t) + 1e-7)
        x0_pred = torch.clamp(x0_pred, -5.0, 5.0)  # 更严格的截断

        # DDIM更新公式
        direction = torch.sqrt(1 - alpha_bar_prev - sigma_t ** 2 + 1e-6) * predicted_noise
        x_prev = torch.sqrt(alpha_bar_prev) * x0_pred + direction
        if (sigma_t > 0).any():
            x_prev += sigma_t * torch.randn_like(x_t) * 0.5  # 减少随机性

        return x_prev

    def denoise_feature_ddim(self, noisy_feature, start_t, denoiser):
        """带进度条和数值监控的去噪过程"""
        x_t = noisy_feature
        for t_step in range(start_t - 1, -1, -1):
            current_t = torch.full((x_t.size(0),), t_step, device=self.device)

            # 监控数值范围
            if torch.isnan(x_t).any():
                raise ValueError("NaN detected in denoising steps")

            with torch.no_grad():  # 冻结denoiser梯度
                pred_noise = denoiser(x_t)

            x_t = self.denoise_step_ddim(
                t_step=current_t[0].item(),
                x_t=x_t,
                t=current_t,
                predicted_noise=pred_noise,
                denoiser=denoiser
            )

            # 定期检查数值稳定性
            if t_step % 10 == 0:
                x_t = torch.clamp(x_t, -10.0, 10.0)

        return x_t

    def diff_loss_1to2(self, feature1, feature2):
        """稳定化的特征对齐损失计算"""
        # 编码并标准化特征
        latent1 = self.encoder1(feature1)
        latent2 = self.encoder2(feature2.detach())  # 阻断梯度传播

        # 训练denoiser2的噪声预测
        noisy_latent2, noise2, t2 = self.add_noise(latent2.detach())
        pred_noise2 = self.denoiser2(noisy_latent2)
        noise_loss = F.smooth_l1_loss(pred_noise2, noise2)  # 更鲁棒的损失

        # 特征对齐过程（冻结denoiser2）
        with torch.no_grad():
            start_t = 10  # 更保守的起始步数
            noisy_latent1, _, _ = self.add_noise(latent1, t=start_t)
            denoised_latent1 = self.denoise_feature_ddim(noisy_latent1, start_t, self.denoiser2)

        # 带动态缩放的损失计算
        align_loss = F.mse_loss(denoised_latent1, latent2) * 0.5
        recon_loss = F.mse_loss(self.decoder1(latent1), feature1) * 0.1

        return align_loss + noise_loss + recon_loss

    def diff_loss_2to1(self, feature2, feature1):
        """对称的反向损失计算"""
        latent2 = self.encoder2(feature2)
        latent1 = self.encoder1(feature1.detach())

        # 训练denoiser1
        noisy_latent1, noise1, t1 = self.add_noise(latent1.detach())
        pred_noise1 = self.denoiser1(noisy_latent1)
        noise_loss = F.smooth_l1_loss(pred_noise1, noise1)

        # 特征对齐
        with torch.no_grad():
            start_t = 10
            noisy_latent2, _, _ = self.add_noise(latent2, t=start_t)
            denoised_latent2 = self.denoise_feature_ddim(noisy_latent2, start_t, self.denoiser1)

        align_loss = F.mse_loss(denoised_latent2, latent1) * 0.5
        recon_loss = F.mse_loss(self.decoder2(latent2), feature2) * 0.1

        return align_loss + noise_loss + recon_loss


class Bottleneck(nn.Module):
    """改进的残差块结构"""

    def __init__(self, in_planes, hidden_planes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, hidden_planes, 1)
        self.bn1 = nn.GroupNorm(4, hidden_planes)

        self.conv2 = nn.Conv2d(hidden_planes, hidden_planes, 3, padding=1)
        self.bn2 = nn.GroupNorm(4, hidden_planes)

        self.conv3 = nn.Conv2d(hidden_planes, in_planes, 1)
        self.bn3 = nn.GroupNorm(4, in_planes)

        self.gate = nn.Parameter(torch.tensor(0.1))  # 可学习缩放因子
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return identity + self.gate * out