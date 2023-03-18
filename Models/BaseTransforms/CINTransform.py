import torch
import torch.nn as nn


class CINTransform(nn.Module):
    def __init__(self, headNumb, dim1, dim2):
        super(CINTransform, self).__init__()
        self.headNumb = headNumb
        self.dim1 = dim1
        self.dim2 = dim2
        self.weight = nn.Parameter(torch.ones(headNumb, dim1, dim2))
        nn.init.normal(self.weight, mean=0, std=0.01)
        self.output = None

    # [B, 1,F, D]
    def forward(self, input1, input2):
        interaction = torch.unsqueeze(input1, dim=2) * torch.unsqueeze(input2, dim=1)
        reWeight = torch.unsqueeze(interaction, dim=1) * self.weight[None, :, :, :, None]
        self.output = torch.sum(reWeight, dim=(2, 3))
        return self.output
