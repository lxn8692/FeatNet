from typing import List
import torch
import numpy as np
from torch import nn

from Models.BaseTransforms.LinearTransform import LinearTransform


# [B,F.D]
class ConcatMlpLayerV2(nn.Module):
    def __init__(self, layerDim: list, dropLast=True):
        super(ConcatMlpLayerV2, self).__init__()
        self.Mlp = LinearTransform(layerDim, dropLast=dropLast)
        self.output = None
        self.concat = None

    def forward(self, featureVec: torch.Tensor):
        # reshape
        shape = featureVec.shape
        self.concat = featureVec.reshape((shape[0], -1))
        batch = self.concat.shape[0]
        self.concat = self.concat.reshape((batch, -1))
        self.output = self.Mlp(self.concat)
        return self.output


class ConcatMlpLayer(nn.Module):
    def __init__(self, layerDim: list, nameList):
        super(ConcatMlpLayer, self).__init__()
        self.Mlp = LinearTransform(layerDim, dropLast=True)
        self.output = None
        self.nameList: List[str] = nameList
        self.concat = None

    def forward(self, featureVec: dict):
        if self.nameList != None:
            featureList = [featureVec[i] for i in self.nameList]
        else:
            featureList = list(featureVec.values())
        return self.matching(featureList)

    def matching(self, temp, ):
        # concat
        self.concat = torch.cat(temp, 1)
        self.output = self.Mlp(self.concat)
        return self.output
