from typing import Any
from Models.BaseTransforms.LinearTransform import LinearTransform
from torch import nn
from DataSource.BaseDataFormat import *
import torch
from Data.WxBiz import DataPack


class CatTrans(nn.Module):
    def __init__(self, inputDim: list, outDim: int):
        super(CatTrans, self).__init__()
        self.outDim = outDim
        self.inputDim = inputDim
        self.out = None

    def forward(self, feature: dict):
        self.out = []
        for k, v in feature.items():
            if v.fieldType == FIELDTYPE.CAT:
                self.out.append(v.data)
        return DataPack(data=self.out, name="CatTrans", fieldType=FIELDTYPE.SIDEINFO)
        # torch.sum(a,dim=())


class LinearContextTrans(nn.Module):
    def __init__(self, inputDim: list, outDim: int):
        super(LinearContextTrans, self).__init__()
        self.outDim = outDim
        self.inputDim = inputDim
        self.out = None

    def forward(self, catFeature: dict):
        self.out = []
        feature = list(catFeature.values())
        for i in range(len(catFeature)):
            self.out.append(feature[i].data)
        return self.out
        # torch.sum(a,dim=())
