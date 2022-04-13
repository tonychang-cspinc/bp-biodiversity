import sys

import torch
import torch.nn.functional as F
from torch import nn


class FDLoss(nn.Module):
    """Combines class loss and regression loss
    """
    def __init__(self, class_loss, reg_loss, weights):
        super(FDLoss, self).__init__()
        self.class_loss = getattr(sys.modules[__name__], class_loss)(weights)
        self.regression_loss = getattr(sys.modules[__name__], reg_loss)(weights)

    def forward(self, labels, pred_labels, regressions, pred_regressions):
        if pred_labels is not None and pred_regressions is not None:
            cel_loss = self.class_loss(labels, pred_labels)
            wmse_loss = self.regression_loss(labels, regressions, pred_regressions)
            return cel_loss + wmse_loss
        elif pred_labels is not None:
            return self.class_loss(labels, pred_labels)
        else:
            return self.regression_loss(labels, regressions, pred_regressions)


class CustomRegressionLoss(nn.Module):
    def __init__(self, weights):
        super(CustomRegressionLoss, self).__init__()

    def forward(self, labels, regressions, pred_regressions):
        raise NotImplementedError


class CustomClassLoss(nn.Module):
    def __init__(self, weights):
        super(CustomClassLoss, self).__init__()

    def forward(self, labels, pred_labels):
        raise NotImplementedError


class WeightedCrossEntropyLoss(CustomClassLoss):
    def __init__(self, weights):
        super(WeightedCrossEntropyLoss, self).__init__(weights)
        self.cel = nn.CrossEntropyLoss(weights)

    def forward(self, labels, pred_labels):
        return self.cel(pred_labels, labels)


class CrossEntropyLoss(CustomClassLoss):
    def __init__(self, weights):
        super(CrossEntropyLoss, self).__init__(weights)
        self.cel = nn.CrossEntropyLoss()

    def forward(self, labels, pred_labels):
        return self.cel(pred_labels, labels)


class WeightedMSELoss(CustomClassLoss):
    """Computes weighted MSE loss given a set of weights and labels for multiple regression variables at once

    If there is more than one regression variable, then output is the sum of the wMSE for each variable
    """
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__(weights)
        self.weights = weights
        self.num_classes = len(self.weights)

    def forward(self, labels, regressions, pred_regressions):
        batch_weights = (F.one_hot(labels, num_classes=self.num_classes) * self.weights).sum(dim=1).unsqueeze(1)
        per_reg_wmse = (((regressions - pred_regressions) ** 2) * batch_weights).mean(dim=0)
        return per_reg_wmse.sum()


class MSELoss(CustomClassLoss):
    def __init__(self, weights):
        super(MSELoss, self).__init__(weights)
        self.mse = nn.MSELoss()

    def forward(self, labels, regressions, pred_regressions):
        return self.mse(pred_regressions, regressions)
