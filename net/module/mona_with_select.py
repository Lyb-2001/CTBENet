# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention as MSDeformAttn
# from mmcv.ops import DeformConv2d
# from mmengine.model import BaseModule
# INNER_DIM = 64
#
# class MonaOp(nn.Module):
#     def __init__(self, in_features):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
#         self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
#         self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)
#
#         self.projector = nn.Conv2d(in_features, in_features, kernel_size=1, )
#
#     def forward(self, x):
#         identity = x
#         conv1_x = self.conv1(x)
#         conv2_x = self.conv2(x)
#         conv3_x = self.conv3(x)
#
#         x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity
#
#         identity = x
#
#         x = self.projector(x)
#
#         return identity + x
#
#
# class Mona(BaseModule):   # 假设 BaseModule 是 nn.Module 的子类
#     def __init__(self,
#                  in_dim,
#                  factor=4):
#         super().__init__()
#         inner_dim = INNER_DIM   # 定义内部维度
#
#         self.project1 = nn.Linear(in_dim, inner_dim)
#         self.nonlinear = F.gelu
#         self.project2 = nn.Linear(inner_dim, in_dim)
#
#         self.dropout = nn.Dropout(p=0.1)
#
#         self.adapter_conv = MonaOp(inner_dim)   # MonaOp 需预先定义
#
#         self.norm = nn.LayerNorm(in_dim)
#         self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
#         self.gammax = nn.Parameter(torch.ones(in_dim))
#         nn.init.zeros_(self.project2.weight)
#         nn.init.zeros_(self.project2.bias)
#     def forward(self, x):   # x: (b, c, h, w)
#         identity = x
#         b, c, h, w = x.shape
#
#         # 转换为序列 (b, n, c)
#         x_seq = x.flatten(2).transpose(1, 2)   # (b, h*w, c)
#
#         # 层归一化与门控
#         x_seq = self.norm(x_seq) * self.gamma + x_seq * self.gammax
#
#         # 第一个线性投影
#         proj1 = self.project1(x_seq)   # (b, n, inner_dim)
#
#         # 重塑为图像格式，进行卷积
#         proj1_img = proj1.reshape(b, h, w, -1).permute(0, 3, 1, 2)   # (b, inner_dim, h, w)
#         proj1_conv = self.adapter_conv(proj1_img)                    # 保持形状
#         proj1_seq = proj1_conv.permute(0, 2, 3, 1).reshape(b, h * w, -1)   # (b, n, inner_dim)
#
#         # 非线性激活与 dropout
#         nonlinear = self.nonlinear(proj1_seq)
#         nonlinear = self.dropout(nonlinear)
#
#         # 第二个线性投影
#         proj2 = self.project2(nonlinear)   # (b, n, c)
#
#         # 转换回 BCHW
#         out = proj2.transpose(1, 2).reshape(b, c, h, w)
#
#         # 残差连接
#         return identity + out
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


# ==========================================
# 1. 基础组件: 兼容 4D 视觉特征的 LayerNorm
# ==========================================
class LayerNorm(nn.Module):
    """ 支持通道在前的 LayerNorm，避免极其耗时的维度转换 """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# ==========================================
# 2. 核心创新组件: 交叉注意力动态路由网关
# ==========================================
class CrossAttnTopKGate(nn.Module):
    def __init__(self, in_features, cond_dim, num_experts, init_t=0.5):
        super(CrossAttnTopKGate, self).__init__()
        self.num_experts = num_experts
        proj_dim = min(in_features // 2, 256)

        # Q: 局部网格的视觉特征查询
        self.q_proj = nn.Linear(in_features, proj_dim)

        # K: 由天气先验动态生成的专家身份键
        self.k_proj = nn.Linear(cond_dim, num_experts * proj_dim)

        self.temperature = nn.Parameter(torch.log(torch.full([1], 1.0 / init_t)), requires_grad=True)
        self.clamp_max = torch.log(torch.tensor(1. / 0.01)).item()

    def forward(self, x_seq, cond):
        """
        x_seq: (B, N, C) 图像像素特征序列
        cond: (B, cond_dim) VQA 或其他模态的天气/场景先验
        """
        b, n, c = x_seq.shape

        q = self.q_proj(x_seq)  # (B, N, proj_dim)
        k = self.k_proj(cond).view(b, self.num_experts, -1)  # (B, num_experts, proj_dim)

        # 余弦相似度取代点积，深层网络极其稳定
        q_norm = F.normalize(q, dim=-1)
        k_norm = F.normalize(k, dim=-1)

        # Cross-Attention: 计算局部像素与动态专家的匹配度
        attn_logits = torch.bmm(q_norm, k_norm.transpose(1, 2))  # (B, N, num_experts)

        logit_scale = torch.clamp(self.temperature, max=self.clamp_max).exp()
        attn_logits = (attn_logits * logit_scale).view(b * n, self.num_experts)

        return attn_logits


# ==========================================
# 3. 极速调度组件: 适配 AMP 的稀疏分配器
# ==========================================
class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = sorted_experts[index_sorted_experts[:, 1], 0]
        self._part_sizes = list((gates > 0).sum(0).cpu().numpy())
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            if len(stitched.shape) == 3:
                stitched = stitched.mul(self._nonzero_gates.unsqueeze(1))
            else:
                stitched = stitched.mul(self._nonzero_gates)

        # 动态捕捉数据类型，护航 fp16/bf16 混合精度训练
        target_dtype = stitched.dtype
        if len(stitched.shape) == 3:
            zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), expert_out[-1].size(-1),
                                requires_grad=True, device=stitched.device, dtype=target_dtype)
        else:
            zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1),
                                requires_grad=True, device=stitched.device, dtype=target_dtype)

        combined = zeros.index_add(0, self._batch_index, stitched)
        return combined

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


# ==========================================
# 4. 同质专家组件: 纯通道语义映射 (1x1 等效)
# ==========================================
class ExpertMLP(nn.Module):
    def __init__(self, in_features, hidden_factor=2):
        super().__init__()
        hidden_features = in_features * hidden_factor
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# ==========================================
# 5. 网格级 MoE 主体: Cross-Attention 驱动
# ==========================================
class CondGridLevelSparseMoE(nn.Module):
    def __init__(self, in_channels, cond_dim=16, num_experts=4, top_k=2, noisy_gating=True):
        super(CondGridLevelSparseMoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.input_size = in_channels
        self.k = top_k

        self.experts = nn.ModuleList([ExpertMLP(in_features=self.input_size) for _ in range(self.num_experts)])

        self.w_gate = CrossAttnTopKGate(self.input_size, cond_dim, self.num_experts)
        self.w_noise = nn.Parameter(torch.zeros(self.input_size, self.num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(-1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1: return torch.Tensor([0]).to(x.device)
        if len(x.shape) == 2: x = x.sum(dim=0)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch) * m + self.k
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in.to(top_values_flat.device)), 1)
        if len(noisy_values.shape) == 3: threshold_if_in = threshold_if_in.unsqueeze(1)

        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out.to(top_values_flat.device)), 1)
        if len(noisy_values.shape) == 3: threshold_if_out = threshold_if_out.unsqueeze(1)

        normal = Normal(self.mean.to(noise_stddev.device), self.std.to(noise_stddev.device))
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / (noise_stddev + 1e-6))
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / (noise_stddev + 1e-6))
        return torch.where(is_in, prob_if_in, prob_if_out)

    def noisy_top_k_gating(self, x_seq, x_flat, cond, train, noise_epsilon=1e-2):
        clean_logits = self.w_gate(x_seq, cond)

        if self.noisy_gating and train:
            raw_noise_stddev = x_flat @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon) * int(train))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=-1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(-1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, cond, loss_coef=1e-2):
        b, c, h, w = x.shape

        # 准备 Cross-Attention 所需的三维序列 (B, N, C)
        x_seq = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        # 展平用于加噪和 Dispatcher (B*N, C)
        x_flat = x_seq.view(-1, c)

        train = self.training

        gates, load = self.noisy_top_k_gating(x_seq, x_flat, cond, train)
        importance = gates.sum(dim=0)

        aux_loss = self.cv_squared(importance) + self.cv_squared(load)
        aux_loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x_flat)

        expert_outputs = [self.experts[i](expert_inputs[i]).view(-1, c) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)

        # 极致纯粹的恢复，移除嵌套残差，严守内存连续性
        y = y.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        return y, aux_loss


# ==========================================
# 6. 顶层架构: Mona Adapter
# ==========================================
class Mona(nn.Module):
    def __init__(self, in_dim, cond_dim=16, factor=8):
        super().__init__()
        inner_dim = in_dim // factor

        self.norm = LayerNorm(in_dim, data_format="channels_first")
        # self.gamma = nn.Parameter(torch.ones(1, in_dim, 1, 1) * 1e-6)

        # 空间无损的降维投影
        self.project1 = nn.Conv2d(in_dim, inner_dim, kernel_size=1)
        # self.drop = nn.Dropout(p=0.1)

        # 极速且智能的单阶段路由
        self.moe_layer = CondGridLevelSparseMoE(
            in_channels=inner_dim,
            cond_dim=cond_dim,
            num_experts=4,
            top_k=2
        )

        self.project2 = nn.Conv2d(inner_dim, in_dim, kernel_size=1)
        nn.init.zeros_(self.project2.weight)
        nn.init.zeros_(self.project2.bias)

    def forward(self, x, cond):
        identity = x

        # 1. 预处理
        x_norm = self.norm(x)
        # x_norm = x_norm * self.gamma + x_norm
        x_mid = self.project1(x_norm)

        # 2. 网格级自适应协同 (Synergize)
        moe_out, aux_loss = self.moe_layer(x_mid, cond)

        # 3. 输出并连接全局残
        out = self.project2(moe_out)

        return identity + out, aux_loss