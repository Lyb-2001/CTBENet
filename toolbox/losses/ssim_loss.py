import torch
import torch.nn.functional as F
from torch import nn
import math

def gaussian(window_size, sigma):
    """生成一维高斯核"""
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2 / float(2*sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel, sigma=1.5):
    """创建二维高斯窗口，归一化且可重复使用"""
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)  # (window_size, 1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)  # (1,1,window_size,window_size)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, sigma=1.5, size_average=True, full=False, val_range=1.0):
    """
    计算 SSIM，输入图像范围假定为 [0, val_range]。
    返回值在 [0,1] 之间。
    """
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same shape")

    (_, channel, height, width) = img1.size()
    window = create_window(window_size, channel, sigma)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    # 均值
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # 方差和协方差
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    # 常数
    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2
    # 添加小 epsilon 避免除零
    eps = 1e-8

    # SSIM 公式
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + eps)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, ssim_map
    return torch.clamp(ret, 0, 1)   # 确保返回值在 [0,1] 内