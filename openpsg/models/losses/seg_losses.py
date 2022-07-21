import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weighted_loss


#@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def dice_loss(input, target, mask=None, eps=0.001):
    N, H, W = input.shape

    input = input.contiguous().view(N, H * W)
    target = target.contiguous().view(N, H * W).float()
    if mask is not None:
        mask = mask.contiguous().view(N, H * W).float()
        input = input * mask
        target = target * mask
    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + eps
    c = torch.sum(target * target, 1) + eps
    d = (2 * a) / (b + c)
    #print('1-d max',(1-d).max())
    return 1 - d


@LOSSES.register_module()
class psgtrDiceLoss(nn.Module):
    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(psgtrDiceLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.count = 0

    def forward(self, inputs, targets, num_matches):
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return self.loss_weight * loss.sum() / num_matches


@LOSSES.register_module()
class MultilabelCrossEntropy(torch.nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, inputs, targets):
        assert (targets.sum(1) != 0).all()
        loss = -(F.log_softmax(inputs, dim=1) *
                 targets).sum(1) / targets.sum(1)
        loss = loss.mean()
        return self.loss_weight * loss


@LOSSES.register_module()
class MultilabelLogRegression(torch.nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, inputs, targets):
        assert (targets.sum(1) != 0).all()
        loss_1 = -(torch.log((inputs + 1) / 2 + 1e-14) * targets).sum()
        loss_0 = -(torch.log(1 - (inputs + 1) / 2 + 1e-14) *
                   (1 - targets)).sum()
        # loss = loss.mean()
        return self.loss_weight * (loss_1 + loss_0) / (targets.sum() +
                                                       (1 - targets).sum())


@LOSSES.register_module()
class LogRegression(torch.nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, inputs, targets):
        positive_rate = 50
        loss_1 = -(torch.log(
            (inputs + 1) / 2 + 1e-14) * targets).sum() * positive_rate
        loss_0 = -(torch.log(1 - (inputs + 1) / 2 + 1e-14) *
                   (1 - targets)).sum()
        return self.loss_weight * (loss_1 + loss_0) / (targets.sum() +
                                                       (1 - targets).sum())

    # def forward(self, inputs, targets):
    #     loss_1 = -(torch.log((inputs + 1) / 2 + 1e-14) * targets).sum()
    #     return self.loss_weight * loss_1

    # def forward(self, inputs, targets):
    #     inputs  = (inputs + 1) / 2 + 1e-14
    #     loss = F.mse_loss(inputs, targets.float(), reduction='mean')
    #     return self.loss_weight * loss


@LOSSES.register_module()
class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='sum', loss_weight=1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, inputs, targets, num_matches):

        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs,
                                                     targets,
                                                     reduction='none')
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t)**self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        return self.loss_weight * loss.mean(1).sum() / num_matches

        # pt = torch.sigmoid(_input)
        # bs = len(pt)
        # target = target.type(torch.long)
        # # print(pt.shape, target.shape)
        # alpha = self.alpha
        # loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
        #     (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        # # print('loss_shape',loss.shape)
        # if self.reduction == 'elementwise_mean':
        #   loss = torch.mean(loss)
        # elif self.reduction == 'sum':
        #   loss = torch.sum(loss)

        # return loss*self.loss_weight/bs
