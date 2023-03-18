from typing import Any, List

import numpy
from Models.BaseTransforms.LinearTransform import LinearTransform
from Models.Layers.ConcatMlpLayers import ConcatMlpLayer
from DataSource.BaseDataFormat import *
import torch.nn as nn


class DefaultItemTransModule(nn.Module):
    def __init__(self):
        super(DefaultItemTransModule, self).__init__()

    def forward(self, feature: list):
        return None


class LinearDimsTransV2Module(nn.Module):

    def __init__(self, inputDim: dict, outDim: int, ):
        super(LinearDimsTransV2Module, self).__init__()
        self.outDim = outDim
        self.inputDim = inputDim
        self.linear = nn.ModuleDict(
            {i: LinearTransform([inputDim[i], self.outDim], dropLast=False) for i in
             inputDim.keys()})
        self.out = None

    def forward(self, feature: dict, ):
        self.out = []
        for i in feature.keys():
            if feature[i].fieldType == FIELDTYPE.DIMS2Dims:
                self.out.append(self.linear[i](feature[i].data))
        return DataPack(data=self.out, name="DimsTrans", fieldType=FIELDTYPE.SIDEINFO)


class LinearItemTrans(nn.Module):
    def __init__(self, featureNum, laydersDim: list, nameList=None):
        super(LinearItemTrans, self).__init__()
        self.featureNum = featureNum
        self.linearTrans = ConcatMlpLayer(laydersDim, nameList)
        self.output = None

    def forward(self, feature: list):
        self.output = self.linearTrans(feature)
        return self.output


class LinearDimsTransModule(nn.Module):

    def __init__(self, inputDim: dict, outDim: int, ):
        super(LinearDimsTransModule, self).__init__()
        self.outDim = outDim
        self.inputDim = inputDim
        self.linear = None
        self.out = None

    def forward(self, dimsFeature: dict, ) -> List[numpy.ndarray]:
        self.out = []
        if self.linear is None:
            device = list(dimsFeature.values())[0].data.device
            self.linear = nn.ModuleDict(
                {i: LinearTransform([dimsFeature[i].data.shape[1], self.outDim], dropLast=False) for i in
                 dimsFeature.keys()})
            self.linear.to(device)
        for i in dimsFeature.keys():
            self.out.append(self.linear[i](dimsFeature[i].data))
        return self.out
