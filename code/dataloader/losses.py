import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class SparseLoss(nn.Module):
    def __init__(self, sparsity_target):
        super(SparseLoss, self).__init__()
        self.sparsity_target = sparsity_target

    def forward(self, model):
        total_loss = 0.0
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm3d):
                scale = module.weight.abs()  # 获取缩放因子的绝对值
                sparsity_loss = torch.mean(scale)  # 平均缩放因子作为稀疏性损失
                total_loss += sparsity_loss

        total_loss *= self.sparsity_target
        return total_loss


class LogBarrierLoss():
    def __init__(self, t=5):
        self.t = t

    def penalty(self, z):
        if z <= - 1 / self.t**2:
            return - torch.log(-z) / self.t
        else:
            return self.t * z - np.log(1 / (self.t**2)) / self.t + 1 / self.t


class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, p):
        p_sm = torch.softmax(p, dim=1)
        H = - p_sm * torch.log(p_sm)
        H_sum = torch.sum(H, dim=1)
        H_mean = torch.mean(H_sum)
        return H_mean


class BCE(nn.Module):
    def __init__(self):
        super(BCE, self).__init__()

    def forward(self, p, q):
        return -1 * torch.mean(q * torch.log(p+1e-6) + (1-q) * torch.log(1-p+1e-6))


class CosLoss(nn.Module):
    def __init__(self):
        super(CosLoss, self).__init__()
        self.ce = BCE()

    def forward(self, p_bg, p_fg):
        cos_sim = F.cosine_similarity(p_bg.view(p_bg.size(0), 1024, -1), p_fg.view(p_fg.size(0), 1024, -1))
        cos_sim_sm = torch.sigmoid(cos_sim)
        l = self.ce(cos_sim_sm, torch.LongTensor(np.zeros(cos_sim_sm.shape[0])).cuda())
        return l

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    cos = CosLoss()
    p1 = torch.randn([1, 1024, 1, 1, 1]).cuda()
    p2 = torch.randn([1, 1024, 1, 1, 1]).cuda()
    loss = cos(p1, p2)
    print(loss)
