from typing import List

import numpy as np
from torch import nn
import torch

from Models.BaseTransforms.GRUTransform import GruTransform


class ConcatGruLayer(nn.Module):
    def __init__(self, inputSize, outSize, nameList, resultLenKey):
        super(ConcatGruLayer, self).__init__()
        self.nameList: List[str] = nameList
        self.Gru = GruTransform(inputSize, outSize)
        self.resultLenKey = resultLenKey
        self.output = None
        self.concat = None

    def forward(self, featureVec):
        resultLen = featureVec[self.resultLenKey].data
        nameFeatureVec = [featureVec[i].data for i in self.nameList]
        return self.matching(nameFeatureVec, resultLen)

    def matching(self, temp, resultLen):
        # concat
        temp = [i.unsqueeze(0) if (i.ndim == 2) else i for i in temp]
        self.concat = torch.cat(temp, 2)
        self.output = self.Gru(self.concat, resultLen)
        return self.output
