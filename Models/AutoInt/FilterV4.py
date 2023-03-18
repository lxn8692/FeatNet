from collections import OrderedDict
from typing import List
from torch import nn
import torch
import numpy as np

from Models.Modules.MatchModule import AllConcatMlpV2
from Utils.HyperParamLoadModule import HyperParam, FeatureInfo, FEATURETYPE
from Models.BaseModelV2 import BaseModelV2, BaseEmbedFormat
from torch.nn import LayerNorm

'''
Add feature cross layer
Input embedding doesn't split in filter layer.
'''


class mlp(nn.Module):
    def __init__(self, inputSize, outputSize):
        super(mlp, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(inputSize, inputSize // 2),
            nn.ReLU(inplace=True),
            nn.Linear(inputSize // 2, outputSize),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.linear(x)


class crossLayer(nn.Module):
    def __init__(self, bucketNum):
        super(crossLayer, self).__init__()
        self.bucketNum = bucketNum
        # self.pruningWeight = nn.Parameter(torch.randn(self.bucketNum, 1, HyperParam.AutoIntFeatureNum, 1, dtype=torch.float32))
        # # self.tem1 = self.pruningWeight.clone()
        # self.param = nn.Parameter(torch.randn(self.bucketNum, 1, HyperParam.AutoIntFeatureNum, 1, dtype=torch.float32))
        # # self.tem2 = self.param.clone()

        # ''' 方案一：仿照FeatNet，直接利用pruningWeight进行特征选择 '''
        # self.init = -50
        # self.pruningWeight = nn.Parameter(torch.randn(self.bucketNum, 1, HyperParam.AutoIntFeatureNum, 1, dtype=torch.float32) * self.init)

        # ''' 方案二：将self.param初始化为[0,1]的参数 '''
        # self.init = -50
        # self.pruningWeight = nn.Parameter(torch.randn(self.bucketNum, 1, HyperParam.AutoIntFeatureNum, 1, dtype=torch.float32) * self.init)
        # self.param = nn.Parameter(torch.ones(self.bucketNum, 1, HyperParam.AutoIntFeatureNum, 1, dtype=torch.float32))
        # nn.init.uniform_(self.param, 0, 1)

        ''' 方案三：不再把pruningWeight置为一个较大的负数，并对其进行RuLU激活，保证其为正数 '''
        # self.pruningWeight = nn.Parameter(torch.ones(self.bucketNum, 1, HyperParam.AutoIntFeatureNum, 1, dtype=torch.float32))
        # self.param = nn.Parameter(torch.ones(self.bucketNum, 1, HyperParam.AutoIntFeatureNum, 1, dtype=torch.float32))

        self.pruningWeight = nn.Parameter(torch.ones(self.bucketNum, HyperParam.AutoIntFeatureNum, dtype=torch.float32))
        self.param = nn.Parameter(torch.ones(self.bucketNum, HyperParam.AutoIntFeatureNum, dtype=torch.float32))
        # self.softmax = nn.Softmax(dim=1)

        # self.layernorm = LayerNorm([self.bucketNum, 1, HyperParam.AutoIntFeatureNum, 1])
        nn.init.uniform_(self.pruningWeight, -4, 4)
        # nn.init.normal_(self.pruningWeight, -5, 0)
        nn.init.uniform_(self.param, 0.4, 1.2)
        # nn.init.normal_(self.pruningWeight, mean=0.2, std=0.01)


    def forward(self, embeddings):

        # ''' 方案一 '''
        # indicator = torch.sigmoid(self.pruningWeight)
        # embeddings = embeddings.unsqueeze(0)
        # embeddings = torch.sign(embeddings) * (torch.relu(abs(embeddings) - indicator))

        ''' 方案二/三 '''
        indicator = torch.sigmoid(self.pruningWeight)
        # weight = torch.sign(self.param) * (torch.relu(abs(self.param) - indicator))
        weight = torch.relu(self.param - indicator)
        weight = weight/weight.sum(dim=1, keepdim=True)
        # weight = self.softmax(weight)

        weight = weight.unsqueeze(2).unsqueeze(1)

        # weight = self.layernorm(weight)
        # print(weight)
        embeddings = embeddings.unsqueeze(0) * weight


        # indicator = torch.sigmoid(self.pruningWeight)
        # weight = torch.sign(self.param) * (torch.relu(abs(self.param) - indicator))
        # embeddings = embeddings.unsqueeze(0) * weight.gt(0)
        # # print(weight)
        # for i in range(self.bucketNum):
        #     tem = weight[i].gt(0).squeeze(0).squeeze(0).squeeze(-1)
        #     lst = []
        #     cnt = 0
        #     for j in range(HyperParam.AutoIntFeatureNum):
        #         if tem[j]==0:
        #             lst.append(j)
        #             cnt += 1
        #     print(f"{i:} {cnt}/{HyperParam.AutoIntFeatureNum} {cnt/HyperParam.AutoIntFeatureNum}")
        #     print(lst)
        # # print(self.tem1.data.equal(self.pruningWeight.data), self.tem2.data.equal(self.param.data))
        # print()
        # print(self.param)
        # print(self.pruningWeight)
        return embeddings


class FilterLayer(nn.Module):
    def __init__(self, bucketNum, headNum):
        super(FilterLayer, self).__init__()
        self.headNum = headNum
        self.bucketNum = bucketNum
        self.filterWeight = nn.Parameter(torch.randn(self.bucketNum, self.headNum, 1, HyperParam.AutoIntFeatureNum, HyperParam.AutoIntFeatureDim // 2 + 1, 2, dtype=torch.float32) * 0.02)
        # self.filterWeight = np.ones([self.bucketNum, self.headNum, 1, HyperParam.AutoIntFeatureNum, HyperParam.AutoIntFeatureDim // 2 + 1, 2])
        # self.tem = self.filterWeight.clone()
        self.dropout = nn.Dropout(0.5)
        self.mlp = mlp(headNum * HyperParam.AutoIntFeatureDim, HyperParam.AutoIntFeatureDim)
        self.LayerNorm = nn.LayerNorm([HyperParam.AutoIntFeatureNum, HyperParam.AutoIntFeatureDim])

    def forward(self, embeddings):
        _, batch, featureNum, featureDim = embeddings.shape
        embeddings = embeddings.unsqueeze(1)
        x = torch.fft.rfft(embeddings, dim=4, norm='ortho')
        weight = torch.view_as_complex(self.filterWeight)
        x = x * weight
        emb_fft = torch.fft.irfft(x, n=featureDim, dim=4, norm='ortho')
        hidden_states = self.dropout(emb_fft)
        hidden_states = hidden_states + embeddings
        # hidden_states = embeddings.unsqueeze(1)
        hidden_states = torch.cat(torch.split(hidden_states, 1, dim=1), dim=4).squeeze(1)
        # hidden_states = torch.cat((embeddings, embeddings),dim=3)
        # hidden_states = torch.cat((hidden_states, embeddings), dim=3)
        hidden_states = self.mlp(hidden_states)
        # hidden_states = self.relu(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        # print(self.tem.data.equal(self.filterWeight.data))
        # self.tem = self.filterWeight.clone()
        return hidden_states


class FilterV4(BaseModelV2):
    def __init__(self, featureInfo: List[FeatureInfo],embedFormat: BaseEmbedFormat):
        self.nameList = [i.featureName for i in featureInfo if (
                i.enable == True and i.featureType == FEATURETYPE.USER)]
        super().__init__(embedFormat)

        self.bucketNum = 6

        # cross layer
        self.crossLayer = crossLayer(self.bucketNum)

        # filter layer
        self.filterHeadNum = [2,2]
        self.featureNum = HyperParam.AutoIntFeatureNum
        self.featureDim = HyperParam.AutoIntFeatureDim

        self.filter = nn.Sequential(
            OrderedDict([
                (f'filter{i}',FilterLayer(self.bucketNum, self.filterHeadNum[i])) for i in range(len(self.filterHeadNum))
            ])
        )
        # self.filter = FilterLayer(self.bucketNum, self.filterHeadNum)

        # predict layer
        self.mlp = AllConcatMlpV2([self.bucketNum*self.featureNum*self.featureDim, 512, 128, 64, 1])

    def mainForward(self, feature):
        # print(feature.shape)
        embeddings = self.crossLayer(feature)
        # embeddings = feature.unsqueeze(0)
        # output = embeddings
        output = self.filter(embeddings)
        output = torch.cat(torch.split(output, 1, dim=0), dim=3).squeeze(0)
        result = self.mlp(None, None, None, None, None, None, output)
        return result
