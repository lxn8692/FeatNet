import torch
import torch.nn as nn


class DCNCell(nn.Module):
    def __init__(self, featureNum):
        super(DCNCell, self).__init__()
        self.weight = nn.Linear(featureNum, featureNum, bias=True)

    def forward(self, feature, x0):
        inter = self.weight(feature)
        out = x0 * inter + feature
        return out


class DCNV2Layer(nn.Module):
    def __init__(self, featureDim, contextDim, depth):
        super(DCNV2Layer, self).__init__()
        self.contextDim = contextDim
        self.featureDim = featureDim
        self.depth = depth
        self.weight = nn.ModuleList([DCNCell(self.featureDim + self.contextDim) for i in range(depth)])

    def forward(self, feature, context):
        cat = torch.cat((torch.mean(feature, dim=2), torch.mean(context, dim=2)), dim=1)
        mid = cat
        # print(f"cat:{cat.shape}")
        for i in range(self.depth):
            mid = self.weight[i](cat, mid)
        return mid

