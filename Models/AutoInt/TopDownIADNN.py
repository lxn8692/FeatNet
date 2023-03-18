from collections import OrderedDict
from typing import List

from Models.BaseTransforms.LinearTransform import LinearTransform
from Models.BaseModelV2 import BaseModelV2, BaseEmbedFormat
from torch import nn
import torch
from Models.BaseTransforms.STRTransform import STRTransform
from Utils.HyperParamLoadModule import FeatureInfo, HyperParam, FEATURETYPE


class headDNN(nn.Module):
    def __init__(self, head, inDim, outDim):
        super(headDNN, self).__init__()
        self.outDim = outDim
        self.inDim = inDim
        self.head = head
        self.para = nn.Parameter(torch.zeros(1, head, self.inDim, self.outDim))
        nn.init.normal(self.para.data, mean=0, std=0.01)

    # [128,10,92,16]->[128,10,1,92*16] [1,10,92*16,128]->[128,10,1,128]->[128,10,1,1]->[128,10]->[10,1]->[128,1]
    def forward(self, feature):
        # print(feature.shape)
        re = torch.matmul(feature, self.para)
        act = torch.relu(re)
        return act


class STRTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.Sigmoid()
        self.outPut = None

    def forward(self, feature: torch.Tensor, indicator: torch.Tensor):
        indicator = torch.sigmoid(indicator)
        self.outPut = torch.sign(feature) * (torch.relu(abs(feature) - indicator))
        return self.outPut

class TopDownIADNN(BaseModelV2):
    def __init__(self, featureInfo: List[FeatureInfo], embedFormat: BaseEmbedFormat):
        self.nameList = [i.featureName for i in featureInfo if (i.enable == True and i.featureType == FEATURETYPE.USER)]
        super().__init__(embedFormat)
        self.featureNum = HyperParam.AutoIntFeatureNum
        self.featureDim = HyperParam.AutoIntFeatureDim
        self.layerNorm = nn.LayerNorm([self.featureNum, self.featureDim])

        self.poolingDim = self.featureNum * 4
        self.init = -23

        """"""
        self.filterLayer = nn.Linear(HyperParam.AutoIntFeatureNum * HyperParam.AutoIntFeatureDim, HyperParam.AutoIntFeatureNum)
        self.filterReLu = nn.LeakyReLU()
        self.filterNorm = nn.LayerNorm(HyperParam.AutoIntFeatureNum)

        self.STR = STRTransform()
        # self.topk = [30, 20, 20, 10, 10]
        # self.headNum = 50
        self.headNum = 10
        self.pruningWeight = nn.Parameter(self.init * torch.ones(1, self.featureNum, self.headNum))
        nn.init.normal(self.pruningWeight.data, std=0.01, mean=self.init)
        # self.headNum = 30
        # self.headNum = 40
        self.layerNormPool = nn.LayerNorm([self.headNum, self.poolingDim, self.featureDim])
        # self.classDNN = LinearTransform(
        #     layerDimension=[self.featureDim, self.headNum * 4, self.headNum * 2, self.headNum], dropLast=False)
        self.classDNN = LinearTransform(layerDimension=[self.featureDim, self.headNum], dropLast=False)
        self.query = nn.Parameter(torch.zeros(size=(1, self.headNum, self.poolingDim, self.featureDim)))
        """"""
        # self.KV = nn.Parameter(torch.zeros(size=(self.headNum, self.featureNum, self.featureDim, self.featureDim)))
        self.KV = nn.Parameter(torch.zeros(size=(self.headNum, self.featureDim, self.featureDim)))
        nn.init.normal(self.query.data, std=0.01, mean=0)
        nn.init.normal(self.KV.data, std=0.01, mean=0)
        # self.layerDims = [self.poolingDim * self.featureDim, 256, 1]
        # self.layerDims = [self.poolingDim * self.featureDim, 128, 1]橙色
        self.layerDims = [self.poolingDim * self.featureDim, 128, 128, 128, 1]  # 1
        layer = []
        for i in range(len(self.layerDims) - 1):
            layer.append((f"headDNN:{i}", headDNN(self.headNum, self.layerDims[i], self.layerDims[i + 1])))
        self.featureDNN = nn.Sequential(OrderedDict(layer))
        self.out = LinearTransform([self.headNum, 1], True)
        self.act = nn.Sigmoid()

    # [b,f,d]
    def mainForward(self, feature):

        feature_reshaped = torch.reshape(feature, [-1, HyperParam.AutoIntFeatureNum * HyperParam.AutoIntFeatureDim])
        filter_weight = self.filterLayer(feature_reshaped)
        filter_weight = self.filterReLu(filter_weight)
        normed_filter_weight = self.filterNorm(filter_weight)

        feature = torch.multiply(feature, normed_filter_weight.unsqueeze(-1))

        feature = self.layerNorm(feature)
        batch = feature.shape[0]
        # [B,F,C]
        # [b,1,f,d] *[B,c,f]->[B,c,f,d]
        cluster: torch.Tensor = self.classDNN(feature)
        pruning = self.STR(cluster, self.pruningWeight)

        # count = 0
        # for i in range(batch):
        #     if cluster[i].equal(pruning[i]):
        #         count += 1
        # print(f"{count}/{batch}")

        # print(torch.count_nonzero(cluster.transpose(1, 2).detach(), dim=2))
        # print(f"cluster dim {cluster.shape}")
        # return pruning
        clusterInput = torch.transpose(pruning, 1, 2).unsqueeze(3) * feature.unsqueeze(1)
        # [b.c.f.d]
        # [128,10,23,1,16] [1,10,23,16,16]->[128,10,23,1,16]
        """"""
        # key: torch.Tensor = torch.matmul(clusterInput.unsqueeze(3), self.KV).squeeze(3)
        # key: torch.Tensor = torch.matmul(clusterInput.unsqueeze(3), self.KV.squeeze(0)).squeeze(3)
        key: torch.Tensor = torch.matmul(clusterInput, self.KV)
        # q:[1,10,92,16] [128,10,23,16]->[128,10,92,23]
        score = torch.relu(torch.matmul(self.query, key.transpose(2, 3)))
        # return score
        # [128,10,92,23] [128,10,23,16]->[128,10,92,16]
        represent = self.layerNormPool(torch.matmul(score, clusterInput))
        # []
        # print(f"cclusterInput:size{clusterInput.shape}")
        cluster = represent.reshape(-1, self.headNum, 1, self.poolingDim * self.featureDim)
        fuse = self.featureDNN(cluster).squeeze()
        # print(fuse.shape)
        setOut = self.out(fuse.reshape(batch, -1))
        out = self.act(setOut)
        return out
