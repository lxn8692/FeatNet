from typing import List
import torch
from torch import nn
from math import log2


class L0Regularization(nn.Module):
    def __init__(self, beta, zeta, gamma):
        super(L0Regularization, self).__init__()
        self.gamma = gamma
        self.zeta = zeta
        self.beta = beta
        # self.sigmoid = nn.Sigmoid()
        self.L0 = None
        self.output = None

    # concatHead:[B,H,F,D], [B,F,D] ->[B,F,D*N]
    def forward(self, inputFeature: torch.Tensor):
        logAlpha = inputFeature
        if self.train():
            u = torch.rand_like(logAlpha)
            logU = torch.log2(u)
            logMu = torch.log2(1 - u)
            s = torch.sigmoid((logAlpha + logU - logMu) / self.beta)
            sHat = s * (self.zeta - self.gamma) + self.gamma
        else:
            sHat = self.sigmoid(logAlpha) * (self.zeta - self.gamma) + self.gamma
        zHat = torch.clamp(sHat, min=0, max=1)
        output = zHat
        # self.L0 = self.calL0(logAlpha).mean()
        return output,self.calL0(logAlpha).mean()
        # return output

    def calL0(self, logAlpha):
        L0 = torch.sigmoid(logAlpha - self.beta * log2((-self.gamma) / self.zeta))
        return L0
