from pathlib import Path
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bb.convnextv2_each import convnextv2_atto
# from models.dinov3.module.DySample import DySample


DEFAULT_SEMANTIC_PROMPTS = [
    "background in an rgb thermal semantic segmentation scene",
    "road in an rgb thermal semantic segmentation scene",
    "sidewalk in an rgb thermal semantic segmentation scene",
    "person in an rgb thermal semantic segmentation scene",
    "car in an rgb thermal semantic segmentation scene",
    "bicycle in an rgb thermal semantic segmentation scene",
]

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2=None):
        if x2 is not None:
            diff_y = x1.size(2) - x2.size(2)
            diff_x = x1.size(3) - x2.size(3)
            x2 = F.pad(x2, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
            x1 = torch.cat([x1, x2], dim=1)
        return self.conv(self.up(x1))


class ThermalBlockEncoder(nn.Module):
    def __init__(self, weight_path=None):
        super().__init__()
        self.net = convnextv2_atto()
        self.state_dims = self.net.block_dims
        self.num_states = len(self.state_dims)

        if weight_path is not None and Path(weight_path).exists():
            checkpoint = torch.load(weight_path, map_location="cpu")
            state_dict = checkpoint.get("model", checkpoint)
            self.net.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.net.forward_block_features(x)


class StateAligner(nn.Module):
    def __init__(self, state_dims, target_dim):
        super().__init__()
        self.projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(state_dim, target_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(target_dim),
                )
                for state_dim in state_dims
            ]
        )

    def forward(self, states, target_hw):
        aligned = []
        for state, proj in zip(states, self.projs):
            state = proj(state)
            if state.shape[2:] != target_hw:
                state = F.interpolate(state, size=target_hw, mode="bilinear", align_corners=False)
            aligned.append(state)
        return aligned


class SemanticPromptBank(nn.Module):
    def __init__(self, prompt_texts=None, clip_path=None, embed_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
        prompt_texts = prompt_texts or DEFAULT_SEMANTIC_PROMPTS
        self.num_prompts = len(prompt_texts)
        text_prototypes = self._load_clip_text_feature(prompt_texts, clip_path)

        if text_prototypes is None:
            self.semantic_context = nn.Parameter(torch.randn(self.num_prompts, embed_dim) * 0.02)
            self.uses_vlm = False
        else:
            self.register_buffer("semantic_context", text_prototypes)
            self.uses_vlm = True

    def _load_clip_text_feature(self, prompt_texts, clip_path):
        text_feature = self._load_openai_clip_text_feature(prompt_texts)
        if text_feature is not None:
            return text_feature

        if clip_path is None:
            clip_path = PROJECT_ROOT / "bb" / "local_clip_vit_b32"
        clip_path = Path(clip_path)
        if not clip_path.exists():
            return None

        try:
            from transformers import CLIPTextModel, CLIPTokenizer

            tokenizer = CLIPTokenizer.from_pretrained(str(clip_path), local_files_only=True)
            text_model = CLIPTextModel.from_pretrained(str(clip_path), local_files_only=True)
            text_model.eval()
            for param in text_model.parameters():
                param.requires_grad = False

            tokens = tokenizer(prompt_texts, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                text_feat = text_model(**tokens).pooler_output.float()
                text_feat = F.normalize(text_feat, dim=-1)
            return text_feat
        except Exception as exc:
            print(f"SemanticPromptBank: CLIP text feature unavailable, use learnable fallback. ({exc})")
            return None

    def _load_openai_clip_text_feature(self, prompt_texts):
        try:
            import clip

            text_model, _ = clip.load("ViT-B/32", device="cpu", jit=False)
            text_model.eval()
            for param in text_model.parameters():
                param.requires_grad = False

            tokens = clip.tokenize(prompt_texts, truncate=True)
            with torch.no_grad():
                text_feat = text_model.encode_text(tokens).float()
                text_feat = F.normalize(text_feat, dim=-1)
            return text_feat
        except Exception:
            return None

    def forward(self, batch_size=None, device=None):
        context = self.semantic_context
        if device is not None:
            context = context.to(device)
        if batch_size is None:
            return context
        return context.unsqueeze(0).expand(batch_size, -1, -1)


class VisualPrototypeBank(nn.Module):
    def __init__(self, prototype_path=None, class_names=("road", "sidewalk", "person", "car", "bicycle")):
        super().__init__()
        if prototype_path is None:
            prototype_path = PROJECT_ROOT / "bb" / "convnext_prototypes.pth"
        prototype_path = Path(prototype_path)

        prototype_dict = torch.load(prototype_path, map_location="cpu")
        key_alias = {"bicycle": "bike"}
        prototypes = []
        for class_name in class_names:
            key = class_name if class_name in prototype_dict else key_alias.get(class_name, class_name)
            prototypes.append(prototype_dict[key].float())

        prototypes = F.normalize(torch.stack(prototypes, dim=0), dim=-1)
        self.register_buffer("prototypes", prototypes)

    def forward(self):
        return F.normalize(self.prototypes, dim=-1)


class LastStagePrototypeCausalIntervention(nn.Module):
    def __init__(self, dim=1024, num_confounders=4, temperature=6.0, residual_scale=0.05):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(float(temperature)))
        self.residual_scale = residual_scale
        self.confounders = nn.Parameter(torch.randn(num_confounders, dim) * 0.02)
        self.refine = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim // 4),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1, bias=False),
        )
        # nn.init.zeros_(self.refine[-1].weight)

    def forward(self, feature, prototypes):
        feature_norm = F.normalize(feature, dim=1)
        prototypes = F.normalize(prototypes, dim=-1)

        class_affinity = torch.einsum("bchw,nc->bnhw", feature_norm, prototypes)
        class_prob = F.softmax(class_affinity * self.temperature.clamp(min=0.1, max=100.0), dim=1)
        class_context = torch.einsum("bnhw,nc->bchw", class_prob, prototypes)+feature

        entropy = -(class_prob * (class_prob + 1e-6).log()).sum(dim=1, keepdim=True)
        entropy = entropy / math.log(class_prob.shape[1])

        confounders = F.normalize(self.confounders, dim=-1)
        confounder_affinity = torch.einsum("bchw,kc->bkhw", feature_norm, confounders)
        confounder_prob = F.softmax(confounder_affinity, dim=1)
        confounder_context = torch.einsum("bkhw,kc->bchw", confounder_prob, self.confounders)+feature

        causal_context = class_context - entropy * confounder_context
        residual = self.refine(causal_context)
        return residual


class MoSAdapter(nn.Module):
    def __init__(
        self,
        dim,
        num_states,
        topk=3,
        semantic_dim=512,
        semantic_hidden=128,
        epsilon=0.05,
        thermal_drop_prob=0.05,
    ):
        super().__init__()
        self.num_states = num_states
        self.topk = min(topk, num_states)
        self.epsilon = epsilon
        self.thermal_drop_prob = thermal_drop_prob
        hidden_dim = max(dim // 4, 32)
        semantic_hidden = min(semantic_hidden, dim)

        self.router = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, num_states, kernel_size=1),
        )
        self.adapter_t = nn.Sequential(
            nn.Conv2d(2*dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False),
        )
        nn.init.zeros_(self.adapter_t[-1].weight)

        self.rgb_semantic = nn.Conv2d(dim, semantic_hidden, kernel_size=1, bias=False)
        self.text_semantic = nn.Linear(semantic_dim, semantic_hidden, bias=False)
        self.semantic_router = nn.Conv2d(semantic_hidden, num_states, kernel_size=1)
        self.semantic_gate = nn.Conv2d(semantic_hidden, dim, kernel_size=1)
        self.semantic_scale = nn.Parameter(torch.tensor(10.0))
        nn.init.zeros_(self.semantic_router.weight)
        nn.init.zeros_(self.semantic_router.bias)
        nn.init.zeros_(self.semantic_gate.weight)
        nn.init.zeros_(self.semantic_gate.bias)

    def build_semantic_token(self, rgb_x, semantic_context):
        if semantic_context.dim() == 2:
            semantic_context = semantic_context.unsqueeze(0).expand(rgb_x.shape[0], -1, -1)

        rgb_sem = F.normalize(self.rgb_semantic(rgb_x), dim=1)
        text_sem = F.normalize(self.text_semantic(semantic_context), dim=-1)

        affinity = torch.einsum("bchw,bnc->bnhw", rgb_sem, text_sem)
        affinity = F.softmax(affinity * self.semantic_scale.clamp(min=0.1, max=100.0), dim=1)
        semantic_token = torch.einsum("bnhw,bnc->bchw", affinity, text_sem)
        return semantic_token

    def forward(self, rgb_x, thermal_states, semantic_context=None, disable_thermal=False):
        logits = self.router(rgb_x)
        semantic_token = None
        if semantic_context is not None:
            semantic_token = self.build_semantic_token(rgb_x, semantic_context)
            logits = logits + self.semantic_router(semantic_token)
        weights = F.softmax(logits, dim=1)

        if self.topk < self.num_states:
            _, topk_idx = torch.topk(weights, self.topk, dim=1)
            topk_mask = torch.zeros_like(weights).scatter_(1, topk_idx, 1.0)

            if self.training and self.epsilon > 0:
                random_scores = torch.rand_like(weights)
                _, random_idx = torch.topk(random_scores, self.topk, dim=1)
                random_mask = torch.zeros_like(weights).scatter_(1, random_idx, 1.0)
                explore = torch.rand(
                    weights.shape[0],
                    1,
                    weights.shape[2],
                    weights.shape[3],
                    device=weights.device,
                    dtype=weights.dtype,
                ) < self.epsilon
                mask = torch.where(explore, random_mask, topk_mask)
            else:
                mask = topk_mask

            weights = weights * mask
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

        fused_t = torch.zeros_like(rgb_x)
        if not disable_thermal:
            for state_idx, state in enumerate(thermal_states):
                fused_t = fused_t + state * weights[:, state_idx : state_idx + 1]
            if self.training and self.thermal_drop_prob > 0:
                drop_mask = (
                    torch.rand(
                        fused_t.shape[0],
                        1,
                        1,
                        1,
                        device=fused_t.device,
                        dtype=fused_t.dtype,
                    )
                    < self.thermal_drop_prob
                )
                fused_t = torch.where(drop_mask, torch.zeros_like(fused_t), fused_t)
        residual = self.adapter_t(torch.cat([rgb_x, fused_t], dim=1))
        return rgb_x + residual

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class FPNDecoderHead(nn.Module):
    def __init__(self, num_classes=6, in_channels=[128, 256, 512, 1024], embedding_dim=256, dropout_ratio=0.1):
        super().__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.lateral_c1 = ConvBNAct(c1_in_channels, embedding_dim, kernel_size=1)
        self.lateral_c2 = ConvBNAct(c2_in_channels, embedding_dim, kernel_size=1)
        self.lateral_c3 = ConvBNAct(c3_in_channels, embedding_dim, kernel_size=1)
        self.lateral_c4 = ConvBNAct(c4_in_channels, embedding_dim, kernel_size=1)

        self.smooth_c4 = ConvBNAct(embedding_dim, embedding_dim, kernel_size=3)
        self.smooth_c3 = ConvBNAct(embedding_dim, embedding_dim, kernel_size=3)
        self.smooth_c2 = ConvBNAct(embedding_dim, embedding_dim, kernel_size=3)
        self.smooth_c1 = ConvBNAct(embedding_dim, embedding_dim, kernel_size=3)

        self.fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.GELU(),
            ConvBNAct(embedding_dim, embedding_dim, kernel_size=3),
        )
        self.dropout = nn.Dropout2d(dropout_ratio)
        self.classifier = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        p4 = self.lateral_c4(c4)
        p3 = self.lateral_c3(c3) + F.interpolate(p4, size=c3.shape[2:], mode="bilinear", align_corners=False)
        p2 = self.lateral_c2(c2) + F.interpolate(p3, size=c2.shape[2:], mode="bilinear", align_corners=False)
        p1 = self.lateral_c1(c1) + F.interpolate(p2, size=c1.shape[2:], mode="bilinear", align_corners=False)

        p4 = self.smooth_c4(p4)
        p3 = self.smooth_c3(p3)
        p2 = self.smooth_c2(p2)
        p1 = self.smooth_c1(p1)

        target_size = p1.shape[2:]
        fused = torch.cat(
            [
                p1,
                F.interpolate(p2, size=target_size, mode="bilinear", align_corners=False),
                F.interpolate(p3, size=target_size, mode="bilinear", align_corners=False),
                F.interpolate(p4, size=target_size, mode="bilinear", align_corners=False),
            ],
            dim=1,
        )
        fused = self.fuse(fused)
        fused = self.dropout(fused)
        return self.classifier(fused)
class RGBTBlock(nn.Module):
    def __init__(self, block, dim, num_states, topk=3, semantic_dim=512, epsilon=0.05, thermal_drop_prob=0.05):
        super().__init__()
        self.adapter_t = MoSAdapter(
            dim=dim,
            num_states=num_states,
            topk=topk,
            semantic_dim=semantic_dim,
            epsilon=epsilon,
            thermal_drop_prob=thermal_drop_prob,
        )
        self.block = block

    def forward(self, rgb_x, thermal_states, semantic_context=None, disable_thermal=False):
        return self.block(self.adapter_t(rgb_x, thermal_states, semantic_context, disable_thermal=disable_thermal))


class DINOv3_UNet(nn.Module):
    def __init__(
        self,
        dinov3_weight_path=None,
        dinov3_local_path=None,
        thermal_weight_path=None,
        num_classes=6,
        topk=3,
        epsilon=0.05,
        prompt_texts=None,
        clip_text_path=None,
        semantic_dim=512,
        prototype_path=None,
        num_confounders=4,
        causal_residual_scale=0.05,
        thermal_drop_prob=0.05,
    ):
        super().__init__()
        root = Path(__file__).resolve().parents[2]
        dinov3_local_path = Path(dinov3_local_path) if dinov3_local_path is not None else root / "model_others" / "dinov3"
        dinov3_weight_path = (
            Path(dinov3_weight_path)
            if dinov3_weight_path is not None
            else root / "bb" / "dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth"
        )
        thermal_weight_path = (
            Path(thermal_weight_path)
            if thermal_weight_path is not None
            else root / "bb" / "convnextv2_atto_1k_224_ema.pt"
        )

        self.dino = torch.hub.load(
            repo_or_dir=str(dinov3_local_path),
            model="dinov3_convnext_base",
            source="local",
            pretrained=False,
            trust_repo=True,
        )
        if dinov3_weight_path.exists():
            checkpoint = torch.load(dinov3_weight_path, map_location="cpu")
            state_dict = checkpoint.get("model", checkpoint)
            self.dino.load_state_dict(state_dict, strict=True)
        for param in self.dino.parameters():
            param.requires_grad = False

        self.thermal_encoder = ThermalBlockEncoder(weight_path=thermal_weight_path)
        num_states = self.thermal_encoder.num_states
        state_dims = self.thermal_encoder.state_dims
        self.semantic_bank = SemanticPromptBank(
            prompt_texts=prompt_texts,
            clip_path=clip_text_path,
            embed_dim=semantic_dim,
        )
        self.prototype_bank = VisualPrototypeBank(prototype_path=prototype_path)

        self.rgb_dims = [stage[0].dwconv.in_channels for stage in self.dino.stages]
        self.end_indices = []
        total = 0
        for stage in self.dino.stages:
            total += len(stage)
            self.end_indices.append(total - 1)

        self.state_aligners = nn.ModuleList([StateAligner(state_dims, dim) for dim in self.rgb_dims])
        self.rgb_blocks = nn.ModuleList(
            [
                RGBTBlock(
                    block=block,
                    dim=self.rgb_dims[stage_idx],
                    num_states=num_states,
                    topk=topk,
                    epsilon=epsilon,
                    semantic_dim=semantic_dim,
                    thermal_drop_prob=thermal_drop_prob,
                )
                for stage_idx, stage in enumerate(self.dino.stages)
                for block in stage
            ]
        )

        # self.reduce4 = nn.Conv2d(self.rgb_dims[3], 128, 1)
        # self.reduce3 = nn.Conv2d(self.rgb_dims[2], 128, 1)
        # self.reduce2 = nn.Conv2d(self.rgb_dims[1], 128, 1)
        # self.reduce1 = nn.Conv2d(self.rgb_dims[0], 128, 1)
        #
        # self.up4 = Up(128, 128)
        # self.up3 = Up(256, 128)
        # self.up2 = Up(256, 128)
        # self.up1 = Up(256, 128)
        # self.head = nn.Conv2d(128, num_classes, 1)

        self.prototype_causal = LastStagePrototypeCausalIntervention(
            dim=self.rgb_dims[-1],
            num_confounders=num_confounders,
            residual_scale=causal_residual_scale,
        )
        self.de = FPNDecoderHead()
    def forward(self, image, thermal, class_names=None, counterfactual=False):
        del class_names
        thermal_states = self.thermal_encoder(thermal)
        semantic_context = self.semantic_bank(batch_size=image.shape[0], device=image.device)

        rgb_x = self.dino.downsample_layers[0](image)
        rgb_features = []
        stage_idx = 0
        aligned_states = self.state_aligners[stage_idx](thermal_states, rgb_x.shape[2:])

        for block_idx, rgb_block in enumerate(self.rgb_blocks):
            rgb_x = rgb_block(rgb_x, aligned_states, semantic_context, disable_thermal=counterfactual)

            if block_idx in self.end_indices:
                rgb_features.append(rgb_x)
                if stage_idx < len(self.dino.downsample_layers) - 1:
                    stage_idx += 1
                    rgb_x = self.dino.downsample_layers[stage_idx](rgb_x)
                    aligned_states = self.state_aligners[stage_idx](thermal_states, rgb_x.shape[2:])

        # f4 = self.reduce4(rgb_features[3])
        # x4 = self.up4(f4)
        # f3 = self.reduce3(rgb_features[2])
        # x3 = self.up3(x4, f3)
        # f2 = self.reduce2(rgb_features[1])
        # x2 = self.up2(x3, f2)
        # f1 = self.reduce1(rgb_features[0])
        # feat = self.up1(x2, f1)
        rgb_features[3] = self.prototype_causal(rgb_features[3], self.prototype_bank())
        logits = self.de(rgb_features)
        logits = F.interpolate(logits, size=image.shape[2:], mode="bilinear", align_corners=False)
        return logits


if __name__ == "__main__":
    model = DINOv3_UNet().eval()
    print(model)
    with torch.no_grad():
        rgb = torch.randn(1, 3, 480, 640)
        thermal = torch.randn(1, 3, 480, 640)
        pred = model(rgb, thermal)
        print(pred.shape)