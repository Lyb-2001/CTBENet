# import torch
import torch.nn as nn
# import torch.nn.functional as F
import math
# from toolbox.losses.diffkd_modules import DiffusionModel,NoiseAdapter,AutoEncoder,DDIMPipeline
# from toolbox.losses.scheduling_ddim import DDIMScheduler
# from tqdm import tqdm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# def conv3x3_bn_relu(in_planes, out_planes, k=3, s=1, p=1,g=1, b=False):
#     return nn.Sequential(
#             nn.Conv2d(in_planes, out_planes, kernel_size=k, stride=s, padding=p,groups=g, bias=b),
#             nn.BatchNorm2d(out_planes),
#             nn.ReLU(),
#             )
# def conv1x1_bn_relu(in_planes, out_planes, k=1, s=1, p=0, b=False):
#     return nn.Sequential(
#             nn.Conv2d(in_planes, out_planes, kernel_size=k, stride=s, padding=p, bias=b),
#             nn.BatchNorm2d(out_planes),
#             nn.ReLU(),
#             )
# class SpectralGatingNetwork(nn.Module):
#     def __init__(self, dim, h, w):
#         super().__init__()
#         self.complex_weight = nn.Parameter(torch.randn(dim, h, w, 2, dtype=torch.float32) * 0.02)
#         self.w = w
#         self.h = h
#
#     def forward(self, x):
#         B, H,W,C = x.shape
#
#
#         x = x.permute(0, 3, 1, 2).contiguous()
#
#         x = x.to(torch.float32)
#
#         x = torch.fft.fft2(x, dim=(-2, -1), norm='ortho')
#         weight = torch.view_as_complex(self.complex_weight)
#         # print(x.shape)
#         # print(weight.shape)
#         x = x * weight
#         x = torch.fft.ifft2(x, dim=(-2, -1), norm='ortho').real
#
#         x = x.permute(0, 2, 3, 1).contiguous()
#
#         return x
#
# # class Block(nn.Module):
# #
# #     def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8):
# #         super().__init__()
# #         self.norm1 = norm_layer(dim)
# #         self.filter = SpectralGatingNetwork(dim, h=h, w=w)
# #         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
# #         self.norm2 = norm_layer(dim)
# #         mlp_hidden_dim = int(dim * mlp_ratio)
# #         # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
# #
# #         # self.conv1 = conv1x1_bn_relu(outchanel, outchanel)
# #     def forward(self, x):
# #         B,C,H,W = x.shape
# #         x = x.flatten(2).transpose(1, 2)
# #         x = x + self.drop_path(self.filter(self.norm1(x)))
# #         out = self.norm2(x).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
# #         return out
class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=2):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, s,t, H, W):
        B, N, C = s.shape
        q = self.q(s).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            t_ = t.permute(0, 2, 1).reshape(B, C, H, W)
            t_ = self.sr(t_).reshape(B, C, -1).permute(0, 2, 1)
            t_ = self.norm(t_)
            kv = self.kv(t_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(s).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
#
# class cnntotrans(nn.Module):
#
#     def __init__(self,inchanel,outchanel,h,w):
#         super().__init__()
#         self.conv=nn.Conv2d(inchanel,outchanel,1)
#         self.norm1 = nn.LayerNorm(outchanel, eps=1e-6)
#         self.norm2 = nn.Sequential(nn.BatchNorm2d(outchanel),nn.ReLU())
#         # self.attn = Attention(outchanel,
#         #     num_heads=1, qkv_bias=True, qk_scale=None,
#         #     attn_drop=0., proj_drop=0., sr_ratio=2)
#         self.attn = SpectralGatingNetwork(outchanel,h,w)
#         self.conv1 = conv1x1_bn_relu(outchanel,outchanel)
#         self.h = h
#         self.w = w
#     def forward(self, x):
#         B, _, H, W = x.shape
#         x = self.conv(x)
#         x = x.permute(0, 2, 3, 1)
#         x = x + self.attn(self.norm1(x))
#         x = self.norm2(x.permute(0, 3, 1, 2).contiguous())
#
#         return x
#
#
# class Permute(nn.Module):
#     def __init__(self, *args):
#         super().__init__()
#         self.args = args
#
#     def forward(self, x: torch.Tensor):
#         return x.permute(*self.args)
# class transtocnn(nn.Module):
#
#     def __init__(self,inchanel,outchanel):
#         super().__init__()
#         self.down = nn.Conv2d(inchanel,outchanel,1)
#         self.conv=nn.Sequential(
#             nn.Conv2d(outchanel, outchanel, kernel_size=7, padding=3, groups=outchanel),
#             Permute(0, 2, 3, 1),
#             nn.LayerNorm(outchanel, eps=1e-6),
#             nn.Linear(outchanel, outchanel),
#             Permute(0, 3, 1, 2),
#         )
#         self.conv1 = conv1x1_bn_relu(outchanel,outchanel)
#         self.norm1 = nn.LayerNorm(outchanel, eps=1e-6)
#         self.norm2 = nn.LayerNorm(outchanel, eps=1e-6)
#
#     def forward(self, x):
#         x = self.down(x)
#         B,C,H,W = x.shape
#         out = self.conv(self.norm1(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()) + x
#         out = self.norm2(out.permute(0, 2, 3, 1).contiguous())
#         return out.permute(0, 3, 1, 2).contiguous()
#
# class FeatureDiffusionKD(nn.Module):
#     """修正后的双向特征扩散蒸馏模块"""
#
#     def __init__(self,
#                  feature_dim1,
#                  feature_dim2,
#                  h,w,
#                  latent_dim=64,
#                  timesteps=1000,
#                  device='cuda'):
#         super().__init__()
#         # self.trans = cnntotrans(feature_dim1,feature_dim2//2,h,w)
#         self.trans = conv1x1_bn_relu(feature_dim1,feature_dim2)
#         # self.down = nn.Conv2d(feature_dim2,feature_dim2//2,1)
#         self.timesteps = timesteps
#         self.device = device
#         self.model = DiffusionModel(channels_in=feature_dim2,kernel_size=3)
#         self.scheduler = DDIMScheduler(num_train_timesteps=timesteps,clip_sample=False,beta_schedule="linear")
#         self.noise_adapter = NoiseAdapter(channels=feature_dim2,kernel_size=3)
#         self.pipeline = DDIMPipeline(self.model,self.scheduler,self.noise_adapter)
#         self.proj = nn.Sequential(nn.Conv2d(feature_dim2,feature_dim2,1),nn.BatchNorm2d(feature_dim2))
#
#         # self.trans1 = transtocnn(feature_dim2, feature_dim1//2)
#         self.trans1 = conv1x1_bn_relu(feature_dim2, feature_dim1)
#         # self.down1 = nn.Conv2d(feature_dim1, feature_dim1 // 2,1)
#         self.model1 = DiffusionModel(channels_in=feature_dim1,kernel_size=3)
#         self.scheduler1 = DDIMScheduler(num_train_timesteps=timesteps,clip_sample=False,beta_schedule="linear")
#         self.noise_adapter1 = NoiseAdapter(channels=feature_dim1,kernel_size=3)
#         self.pipeline1 = DDIMPipeline(self.model1,self.scheduler1,self.noise_adapter1)
#         self.proj1 = nn.Sequential(nn.Conv2d(feature_dim1,feature_dim1,1),nn.BatchNorm2d(feature_dim1))
#
#     def diff_loss_1to2(self, feature1, feature2):
#         feature1 = self.trans(feature1)
#         # feature2 = self.down(feature2)
#         refined_feat = self.pipeline(
#             batch_size=feature1.shape[0],
#             device=feature1.device,
#             dtype=feature1.dtype,
#             shape=feature1.shape[1:],
#             feat=feature1,
#             num_inference_steps=5,
#             proj=self.proj
#         )
#         refined_feat = self.proj(refined_feat)
#
#         # train diffusion model
#         ddim_loss = self.ddim_loss(feature2)
#         kd_loss = F.mse_loss(refined_feat, feature2)
#         # kd_loss = F.cross_entropy(refined_feat, feature2)
#         return kd_loss + ddim_loss
#
#     def diff_loss_2to1(self, feature2, feature1):
#         feature2 = self.trans1(feature2)
#         # feature1 = self.down1(feature1)
#         refined_feat = self.pipeline1(
#             batch_size=feature2.shape[0],
#             device=feature2.device,
#             dtype=feature2.dtype,
#             shape=feature2.shape[1:],
#             feat=feature2,
#             num_inference_steps=5,
#             proj=self.proj1
#         )
#         refined_feat = self.proj1(refined_feat)
#
#         # train diffusion model
#         ddim_loss = self.ddim_loss1(feature1)
#         kd_loss = F.mse_loss(refined_feat, feature1)
#         return kd_loss + ddim_loss
#
#
#     def ddim_loss(self, gt_feat):
#         # Sample noise to add to the images
#         noise = torch.randn(gt_feat.shape, device=gt_feat.device) #.to(gt_feat.device)
#         bs = gt_feat.shape[0]
#
#         # Sample a random timestep for each image
#         timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bs,), device=gt_feat.device).long()
#         # Add noise to the clean images according to the noise magnitude at each timestep
#         # (this is the forward diffusion process)
#         noisy_images = self.scheduler.add_noise(gt_feat, noise, timesteps)
#         noise_pred = self.model(noisy_images, timesteps)
#         loss = F.mse_loss(noise_pred, noise)
#         return loss
#
#     def ddim_loss1(self, gt_feat):
#         # Sample noise to add to the images
#         noise = torch.randn(gt_feat.shape, device=gt_feat.device) #.to(gt_feat.device)
#         bs = gt_feat.shape[0]
#
#         # Sample a random timestep for each image
#         timesteps = torch.randint(0, self.scheduler1.num_train_timesteps, (bs,), device=gt_feat.device).long()
#         # Add noise to the clean images according to the noise magnitude at each timestep
#         # (this is the forward diffusion process)
#         noisy_images = self.scheduler1.add_noise(gt_feat, noise, timesteps)
#         noise_pred = self.model1(noisy_images, timesteps)
#         loss = F.mse_loss(noise_pred, noise)
#         return loss
#
import torch
from torch import nn
import torch.nn.functional as F
from .diffkd_modules import DiffusionModel, NoiseAdapter, AutoEncoder, DDIMPipeline
from .scheduling_ddim import DDIMScheduler
class SpectralGatingNetwork(nn.Module):
    def __init__(self,dim):
        super().__init__()

        # self.w = w
        # self.h = h
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
    def _init_weight(self,dim,h,w):
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w, 2, dtype=torch.float32)).cuda()

    def forward(self, x):
        B, C,H,W = x.shape

        self._init_weight(C,H,W)
        x = x.to(torch.float32)

        x = torch.fft.fft2(x, dim=(-2, -1), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        # print(x.shape)
        # print(weight.shape)
        x = x * self.sigmoid(weight) + x
        x = torch.fft.ifft2(x, dim=(-2, -1), norm='ortho').real
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class SpatialAttentionModule(nn.Module):
    def __init__(self, dim, reduction_ratio=8):
        super().__init__()

        # Channel attention branch
        self.channel_gate = nn.Sequential(
            nn.Linear(dim, dim // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction_ratio, dim),
            nn.Sigmoid()
        )

        # Spatial attention branch
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(dim,eps=1e-6)

    def forward(self, x):
        B, C, H, W = x.shape

        # Channel attention
        avg_pool = x.mean(dim=[2, 3])  # [B, C]
        # max_pool = x.max(dim=[2, 3])  # [B, C]
        channel_attn = self.channel_gate(avg_pool)
        x_channel = x * channel_attn.view(B, C, 1, 1)


        avg_out = torch.mean(x_channel, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x_channel, dim=1, keepdim=True)  # [B, 1, H, W]
        spatial_map = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        spatial_attn = self.spatial_gate(spatial_map)  # [B, 1, H, W]
        # Apply spatial attention
        x_out = x_channel * spatial_attn + x  # [B, C, H, W]

        return self.norm(x_out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class DiffAdapter(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.proj=nn.Sequential(nn.Conv2d(in_dim,out_dim,1),nn.GroupNorm(4,out_dim),nn.GELU())
        self.attn = Attention(out_dim,
                    num_heads=1, qkv_bias=True, qk_scale=None,
                    attn_drop=0., proj_drop=0., sr_ratio=4)

    def forward(self,student_feat,teacher_feat):
        s_proj = self.proj(student_feat)
        B,C,H,W = s_proj.shape
        t_flat = teacher_feat.flatten(2).permute(0,2,1)
        attout = self.attn(
            s_proj.flatten(2).permute(0,2,1),
            t_flat,
            H,
            W
        )
        return attout.permute(0,2,1).view_as(teacher_feat)


class DiffKD1to2(nn.Module):
    def __init__(
            self,
            student_channels,
            teacher_channels,
            kernel_size=3,
            inference_steps=5,
            num_train_timesteps=1000,
            use_ae=True,
            ae_channels=None,
    ):
        super().__init__()
        self.use_ae = use_ae
        self.diffusion_inference_steps = inference_steps
        # AE for compress teacher feature
        if use_ae:
            if ae_channels is None:
                ae_channels = teacher_channels // 2
            self.ae = AutoEncoder(teacher_channels, ae_channels)
            teacher_channels = ae_channels

        # transform student feature to the same dimension as teacher
        self.trans = DiffAdapter(student_channels, teacher_channels)
        self.fa = SpectralGatingNetwork(teacher_channels)
        # diffusion model - predict noise
        self.model = DiffusionModel(channels_in=teacher_channels, kernel_size=kernel_size)
        self.scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False,
                                       beta_schedule="linear")
        self.noise_adapter = NoiseAdapter(teacher_channels, kernel_size)
        # pipeline for denoising student feature
        self.pipeline = DDIMPipeline(self.model, self.scheduler, self.noise_adapter)
        self.proj = nn.Sequential(nn.Conv2d(teacher_channels, teacher_channels, 1), nn.BatchNorm2d(teacher_channels))
        # self.norm
    def forward(self, student_feat, teacher_feat):
        # project student feature to the same dimension as teacher feature


        # use autoencoder on teacher feature
        if self.use_ae:
            hidden_t_feat, rec_t_feat = self.ae(teacher_feat)
            rec_loss = F.mse_loss(teacher_feat, rec_t_feat)
            teacher_feat = hidden_t_feat.detach()
        else:
            rec_loss = None
        student_feat = self.fa(self.trans(student_feat, teacher_feat)).to(torch.float16)
        # denoise student feature
        refined_feat = self.pipeline(
            batch_size=student_feat.shape[0],
            device=student_feat.device,
            dtype=student_feat.dtype,
            shape=student_feat.shape[1:],
            feat=student_feat,
            num_inference_steps=self.diffusion_inference_steps,
            proj=self.proj
        )
        refined_feat = self.proj(refined_feat)

        ddim_loss = self.ddim_loss(teacher_feat)
        disloss = F.mse_loss(refined_feat, teacher_feat)
        loss = disloss+ddim_loss+rec_loss
        return loss,disloss,ddim_loss

    def ddim_loss(self, gt_feat):
        # Sample noise to add to the images
        noise = torch.randn(gt_feat.shape, device=gt_feat.device)  # .to(gt_feat.device)
        bs = gt_feat.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bs,), device=gt_feat.device).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.scheduler.add_noise(gt_feat, noise, timesteps)
        noise_pred = self.model(noisy_images, timesteps)
        loss = F.mse_loss(noise_pred, noise)
        return loss
class DiffKD2to1(nn.Module):
    def __init__(
            self,
            student_channels,
            teacher_channels,
            kernel_size=3,
            inference_steps=5,
            num_train_timesteps=1000,
            use_ae=True,
            ae_channels=None,
    ):
        super().__init__()
        self.use_ae = use_ae
        self.diffusion_inference_steps = inference_steps
        # AE for compress teacher feature
        if use_ae:
            if ae_channels is None:
                ae_channels = teacher_channels // 2
            self.ae = AutoEncoder(teacher_channels, ae_channels)
            teacher_channels = ae_channels

        # transform student feature to the same dimension as teacher
        self.trans = DiffAdapter(student_channels, teacher_channels)
        self.sa = SpatialAttentionModule(teacher_channels)
        # diffusion model - predict noise
        self.model = DiffusionModel(channels_in=teacher_channels, kernel_size=kernel_size)
        self.scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False,
                                       beta_schedule="linear")
        self.noise_adapter = NoiseAdapter(teacher_channels, kernel_size)
        # pipeline for denoising student feature
        self.pipeline = DDIMPipeline(self.model, self.scheduler, self.noise_adapter)
        self.proj = nn.Sequential(nn.Conv2d(teacher_channels, teacher_channels, 1), nn.BatchNorm2d(teacher_channels))

    def forward(self, student_feat, teacher_feat):
        # project student feature to the same dimension as teacher feature


        # use autoencoder on teacher feature
        if self.use_ae:
            hidden_t_feat, rec_t_feat = self.ae(teacher_feat)
            rec_loss = F.mse_loss(teacher_feat, rec_t_feat)
            teacher_feat = hidden_t_feat.detach()
        else:
            rec_loss = None
        student_feat = self.sa(self.trans(student_feat, teacher_feat)).to(torch.float16)
        # denoise student feature
        refined_feat = self.pipeline(
            batch_size=student_feat.shape[0],
            device=student_feat.device,
            dtype=student_feat.dtype,
            shape=student_feat.shape[1:],
            feat=student_feat,
            num_inference_steps=self.diffusion_inference_steps,
            proj=self.proj
        )
        refined_feat = self.proj(refined_feat)

        ddim_loss = self.ddim_loss(teacher_feat)
        disloss = F.mse_loss(refined_feat, teacher_feat)
        loss = disloss+ddim_loss+rec_loss
        return loss,disloss,ddim_loss

    def ddim_loss(self, gt_feat):
        # Sample noise to add to the images
        noise = torch.randn(gt_feat.shape, device=gt_feat.device)  # .to(gt_feat.device)
        bs = gt_feat.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bs,), device=gt_feat.device).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.scheduler.add_noise(gt_feat, noise, timesteps)
        noise_pred = self.model(noisy_images, timesteps)
        loss = F.mse_loss(noise_pred, noise)
        return loss

