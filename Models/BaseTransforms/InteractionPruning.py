import torch
import torch.nn as nn
from Models.BaseTransforms.LinearTransform import LinearTransform
from Models.BaseTransforms.L0Regularization import L0Regularization


# [B,F,D]->[B,F,F]
class InteractionPruning(nn.Module):
    def __init__(self, dim, featureNumb, beta, zeta=1.1, gamma=-0.1):
        super().__init__()
        self.featureNumb = featureNumb
        self.L0Reg = L0Regularization(beta, zeta, gamma)
        self.matrix = nn.Parameter(torch.zeros(size=(featureNumb, featureNumb, dim, dim)))
        torch.nn.init.normal_(self.matrix.data, mean=0, std=0.001)
        self.L0Out = None
        self.L0 = None
        self.output = None

    def forward(self, feature: torch.Tensor):
        mask = torch.triu(torch.ones(self.featureNumb, self.featureNumb, device=feature.device, requires_grad=False),
                          1)[
               :, :, None, None]

        matrix,_ = self.L0Reg(self.matrix)
        self.L0Out = torch.mul(mask, matrix)
        L0 = self.L0Reg.calL0(self.matrix)
        maskL0 = torch.mul(mask, L0)
        self.L0 = torch.mean(maskL0)
        interaction = torch.matmul(feature[:, :, None, :, None], feature[:, None, :, None, :])
        pruning = interaction * self.L0Out[None, :, :, :, :]
        # print('interaction shape', interaction.shape)
        self.output = torch.sum(pruning, dim=(3, 4))
        return self.output
