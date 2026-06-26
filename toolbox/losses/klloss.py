import torch.nn as nn
import torch.nn.functional as F
from toolbox.losses.lovasz_losses import lovasz_softmax
import torch


class MscCrossEntropyLoss(nn.Module):

    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super(MscCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        # print(input.shape)
        # print(target.shape)
        if not isinstance(input, tuple):
            input = (input,)

        loss = 0
        for item in input:
            h, w = item.size(2), item.size(3)
            item_target = F.interpolate(target.unsqueeze(1).float(), size=(h, w))
            loss += F.cross_entropy(item, item_target.squeeze(1).long(), weight=self.weight,
                        ignore_index=self.ignore_index, reduction=self.reduction)
        return loss / len(input)


# class MscLovaszSoftmaxLoss(nn.Module):
#     def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
#         super(MscLovaszSoftmaxLoss, self).__init__()
#         self.weight = weight
#         self.ignore_index = ignore_index
#         self.reduction = reduction
#
#     def forward(self, input, target):
#         if not isinstance(input, tuple):
#             input = (input,)
#
#         loss = 0
#         for item in input:
#             h, w = item.size(2), item.size(3)
#             item_target = F.interpolate(target.unsqueeze(1).float(), size=(h, w))
#             loss += LovaszSoftmax(item, item_target.squeeze(1).long())
#         return loss / len(input)


class FocalLossbyothers(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, weight=None, ignore_index=-100):
        super(FocalLossbyothers, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=ignore_index)

    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1-pt) ** self.gamma) * self.alpha * logpt
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, reduction='mean'):
        """
        多分类Dice Loss
        Args:
            smooth (float): 平滑系数，防止分母为0
            reduction (str): 损失计算方式，'mean' 或 'sum'
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, input, target):
        """
        Args:
            input (torch.Tensor): 模型预测输出，shape为(N, C, H, W)
            target (torch.Tensor): 真实标签，shape为(N, H, W)
        """
        # 将输入转换为概率分布
        # input = F.softmax(input, dim=1)

        # 将target转换为one-hot编码
        # n_classes = input.size(1)
        # target_one_hot = F.one_hot(target, n_classes).permute(0, 3, 1, 2).float()

        # 计算每个类别的Dice系数
        dims = (0, 2, 3)  # 在batch、height和width维度上计算
        intersection = torch.sum(input * target, dims)
        cardinality = torch.sum(input + target, dims)
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)

        # 计算损失
        dice_loss = 1. - dice_score

        # 应用reduction
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss

# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1.0):
#         """
#         多分类Dice Loss
#         Args:
#             smooth (float): 平滑系数，防止分母为0
#             reduction (str): 损失计算方式，'mean' 或 'sum'
#         """
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth
#
#
#     def forward(self, input, target):
#
#
#         # 计算每个类别的Dice系数
#         dims = (0, 2, 3)  # 在batch、height和width维度上计算
#         intersection = torch.sum(input * target, dims)
#         cardinality = torch.sum(input + target, dims)
#         dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
#
#         # 计算损失
#         dice_loss = 1. - dice_score

        # return dice_loss

def ofa_loss(logits1, logits2, target_mask, temperature=1.):


    ce1 = F.cross_entropy(
        logits1.detach(),
        target_mask,
        reduction='none'
    ).unsqueeze(1)  # [B, 1, H, W]

    ce2 = F.cross_entropy(
        logits2.detach(),
        target_mask,
        reduction='none'
    ).unsqueeze(1)  # [B, 1, H, W]
    # Step 3: 计算权重因子（公式6）
    # total_ce = ce1 + ce2 + 1e-8  # 防止除零
    # weight1 = 1 - (ce1 / total_ce)  # S^t_hw
    # weight2 = 1 - weight1  # 1 - S^t_hw
    w1 = (ce1>ce2).float()
    w2 = (ce2>ce1).float()
    # w2 = torch.ones_like(ce2)
    # w2[ce2<=ce1] = 0
    # # Step 4: 生成混合logits（公式7）
    # mixed_logits = (
    #         weight1 * F.softmax(logits1,dim=1) +
    #         weight2 * F.softmax(logits2,dim=1)
    # ).detach()
    logits1= logits1 / temperature
    logits2 = logits2 / temperature
    # dice1 = DiceLoss()
    # dice2 = DiceLoss()
    loss1 = -(F.softmax(logits2.detach(),dim=1)*F.log_softmax(logits1,dim=1)).sum(dim=1,keepdim=True)
    loss2 = -(F.softmax(logits1.detach(),dim=1)*F.log_softmax(logits2,dim=1)).sum(dim=1,keepdim=True)
    demon1 = w1.sum()+1e-8
    demon2 = w2.sum()+1e-8
    loss1 = (loss1*w1).sum() / demon1
    loss2 = (loss2*w2).sum() / demon2

    return loss1,loss2
class KLDLoss(nn.Module):
    def __init__(self, alpha=1, tau=1):
        super().__init__()
        self.alpha_0 = alpha
        self.alpha = alpha
        self.tau = tau



        self.KLD = torch.nn.KLDivLoss(reduction='none')


    def forward(self, x_student, x_teacher):
        x_student = F.log_softmax(x_student / self.tau, dim=1)
        x_teacher = F.softmax(x_teacher / self.tau, dim=1)
        # print(x_student.shape)
        b,c,h,w = x_student.shape
        # numpix = b*h*w
        # loss = self.KLD(x_student, x_teacher) / numpix
        # print("self.alpha", self.alpha)
        loss = self.KLD(x_student,x_teacher).sum(1).mean()
        loss = self.alpha * loss
        return loss*(self.tau**2)


class CosLoss(nn.Module):
    def __init__(self):
        super(CosLoss, self).__init__()

    def forward(self, stu_map, tea_map):
        similiar = 1 - F.cosine_similarity(stu_map, tea_map, dim=1)
        loss = similiar.sum()
        return loss

if __name__ == '__main__':
    kld = KLDLoss()
    x = torch.randn(4, 4, 256, 256)
    y = torch.randn(4, 4, 256, 256)
    print(kld(x, y))