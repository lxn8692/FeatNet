import torch
import torch.nn as nn


class AggregationLayer(nn.Module):
    def __init__(self, featureNumb, embedSize):
        super(AggregationLayer, self).__init__()
        self.embedSize = embedSize
        self.featureNumb = featureNumb
        self.weightA = nn.Parameter(torch.zeros(featureNumb, featureNumb))
        self.weightB = nn.Parameter(torch.zeros(featureNumb, featureNumb))
        nn.init.normal(self.weightA, mean=0, std=0.01)
        nn.init.normal(self.weightB, mean=0, std=0.01)

    ##[B,F,F,D] [B,F,D]
    def forward(self, interaction, feature):
        leftWeight = self.weightA[None, None, :, :]
        rightWeight = self.weightB[None, None, :, :]
        left = torch.matmul(leftWeight, interaction)
        midInter = left.transpose(1, 2)
        right = torch.matmul(rightWeight, midInter)

        # aggregate:
        result = right * feature[:, None, :, :]
        return result
