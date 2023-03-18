from typing import List

import torch.nn as nn

from Models.BaseTransforms.GaussianTransform import GaussianTransform


class GaussianLayers(nn.Module):
    def __init__(self, featureNumb, inputDim: list, outputDim: list, nameList: List, ):
        super(GaussianLayers, self).__init__()
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.nameList = nameList
        self.gaussianKernel: nn.ModuleList[GaussianTransform] = nn.ModuleList(
            [GaussianTransform(inputDim[i], outputDim[i]) for i in
             range(featureNumb)])
        self.output = None

    def forward(self, featureVec, matchVec):
        self.output = []
        for i in range(len(featureVec)):
            self.output.append(self.gaussianKernel[i](featureVec[i], matchVec[i])[None, :, :])
        return self.output
