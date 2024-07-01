import torch
import torch.nn as nn

class MaskCrossEnrtopyLoss(nn.Module):
    def __init__(self, weight=None):
        super(MaskCrossEnrtopyLoss, self).__init__()
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction='none', )

    def forward(self, pred, target, mask=None):
        loss = self.ce(pred, target)
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.sum() / loss.numel()
        return loss

class MaskBCELoss(nn.Module):
    def __init__(self):
        super(MaskBCELoss, self).__init__()
        self.bce = torch.nn.BCELoss(reduction='none')

    def forward(self, pred, target, mask=None):

        loss = self.bce(pred, target)
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.sum() / loss.numel()
        return loss

class BalanceBCELoss(nn.Module):
    def __init__(self, negative_ratio=5.0, eps=1e-8):
        super(BalanceBCELoss, self).__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(self, pred, target, mask=None):
        loss = nn.functional.binary_cross_entropy(pred, target, reduction='none')
        loss = loss * mask
        positive_index = (target == 1.0).float()
        negative_index = (target == 0.0).float()
        positive_count = int(positive_index.float().sum())
        negative_count = min(int(negative_index.float().sum()), int(positive_count * self.negative_ratio))
        positive_loss = loss * positive_index
        negative_loss = loss * negative_index
        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)

        balance_loss = (positive_loss.sum() + negative_loss.sum()) /\
            (positive_count + negative_count + self.eps)

        return balance_loss
    