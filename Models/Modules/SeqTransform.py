from typing import List
from DataSource.BaseDataFormat import *
from Models.BaseTransforms.LinearTransform import LinearTransform
from torch import nn
from Models.Layers.ConcatGruLayers import ConcatGruLayer


class UserDefaultTrans(nn.Module):
    def __init__(self):
        super(UserDefaultTrans, self).__init__()

    def forward(self, feature):
        return feature


class UserConcatMlpGruTrans(nn.Module):
    def __init__(self, inputDim, outDim):
        super(UserConcatMlpGruTrans, self).__init__()
        self.concatGru = ConcatGruLayer(inputDim, outDim)
        self.out = None

    def forward(self, feature):
        self.out = self.concatGru(feature)
        return self.out


class GRULinearDimsTransModule(nn.Module):

    def __init__(self, transInputDim: dict, outDim: int, gruInputDim: int, gruNameList: List,
                 gruResultLenKey: str):
        super(GRULinearDimsTransModule, self).__init__()
        self.GRUInputDim = gruInputDim
        self.gruResultLenKey = gruResultLenKey
        self.gruNameList = gruNameList
        self.outDim = outDim
        self.transInputDim: dict = transInputDim
        self.concatGru = ConcatGruLayer(gruInputDim, outDim, gruNameList, gruResultLenKey)
        # self.linear = nn.ModuleDict(
        #     {i: LinearTransform([transInputDim[i], outDim], dropLast=False) for i in transInputDim.keys()})
        self.out = None

    def forward(self, userFeature: dict):
        self.out = [userFeature[i] for i in userFeature.keys() if
                    (i not in self.gruNameList and i not in self.transInputDim.keys() \
                     and i not in self.gruResultLenKey)]
        gruOut = self.concatGru(userFeature)
        self.out.append(gruOut)
        return self.out


class GRULinearDimsTransModuleV2(nn.Module):

    def __init__(self, transInputDim: dict, outDim: int, gruInputDim: int, gruNameList: List,
                 gruResultLenKey: str):
        super(GRULinearDimsTransModuleV2, self).__init__()
        self.GRUInputDim = gruInputDim
        self.gruResultLenKey = gruResultLenKey
        self.gruNameList = gruNameList
        self.outDim = outDim
        self.transInputDim: dict = transInputDim
        self.concatGru = ConcatGruLayer(gruInputDim, outDim, gruNameList, gruResultLenKey)
        # self.linear = nn.ModuleDict(
        #     {i: LinearTransform([transInputDim[i], outDim], dropLast=False) for i in transInputDim.keys()})
        self.out = None

    def forward(self, userFeature: dict):
        seqFeature = {k: v
                      for k, v in userFeature.items() if v.fieldType == FIELDTYPE.SEQUENCE}
        self.out = [seqFeature[i] for i in seqFeature.keys() if
                    (i not in self.gruNameList and i not in self.transInputDim.keys() \
                     and i not in self.gruResultLenKey)]
        gruOut = self.concatGru(seqFeature)
        self.out.append(gruOut)
        return DataPack(data=self.out, fieldType=FIELDTYPE.SIDEINFO, name='SeqOut')
