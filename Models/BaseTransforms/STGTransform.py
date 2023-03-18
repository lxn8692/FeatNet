from typing import List
import torch
from torch import nn
from math import log2


class STGTransform(nn.Module):
    def __init__(self, mu):
        super(STGTransform, self).__init__()
        # self.sigmoid = nn.Sigmoid()
        self.mu = mu
        self.L0 = None
        self.output = None

    # concatHead:[B,H,F,D], [B,F,D] ->[B,F,D*N]
    def forward(self, inputFeature: torch.Tensor,indicator):
        return 
        # return output

    def calL0(self, logAlpha):
        L0 = torch.sigmoid(logAlpha - self.beta * log2((-self.gamma) / self.zeta))
        return L0
