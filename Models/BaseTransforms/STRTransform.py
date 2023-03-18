import torch
import torch.nn as nn
from Models.BaseTransforms.LinearTransform import LinearTransform
from Models.BaseTransforms.L0Regularization import L0Regularization
from torch import abs


# [B,F,D] *[F,D]->[B,F,D]
class STRTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.Sigmoid()
        self.outPut = None

    def forward(self, feature: torch.Tensor, indicator: torch.Tensor):
        print(feature.device, indicator.device)
        indicator = torch.sigmoid(indicator)
        self.outPut = torch.sign(feature) * (torch.relu(abs(feature) - indicator))
        return self.outPut
