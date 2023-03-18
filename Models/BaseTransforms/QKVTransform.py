from typing import List

import torch
from collections import OrderedDict
import torch.nn as nn
import numpy as np


##[B*1*F*D] * 【3*H，D，K 】->[B,H,3，F.K]
# share the same Transform Matrix
class QKVTransform(nn.Module):
    def __init__(self, headNum, inDim, outDim):
        super(QKVTransform, self).__init__()
        self.headNum = headNum
        self.inDim = inDim
        self.outDim = outDim
        self.weight = None
        self.weight = nn.Parameter(torch.zeros(size=(3 * headNum, inDim, outDim)))
        nn.init.xavier_uniform(self.weight.data, gain=1.414)
        self.output = None

    def forward(self, featureVec) -> torch.Tensor:
        shape = featureVec.shape
        # print(shape)
        cat = featureVec[:, None, :, :]
        # print(self.inDim,self.outDim,cat.shape)
        self.output = cat.matmul(self.weight)
        return self.output.reshape((-1, self.headNum, 3, cat.shape[-2], self.outDim))


## no head:
# [F,D]->[3,F,D]
class QKVTransformV2(nn.Module):
    def __init__(self, shapeNumb, inDim, outDim):
        super(QKVTransformV2, self).__init__()
        self.shapeNumb = shapeNumb
        self.inDim = inDim
        self.outDim = outDim
        self.weight = nn.Parameter(torch.zeros(size=(self.shapeNumb, inDim, outDim)))
        nn.init.xavier_uniform(self.weight.data, gain=1.414)
        self.output = None

    def forward(self, featureVec) -> torch.Tensor:
        # print(self.inDim,self.outDim,cat.shape)
        weight = self.weight
        if self.shapeNumb != 1:
            weight = weight[None, :, :, :]
            featureVec = featureVec[:, None, :, :]
        self.output = featureVec.matmul(weight)
        return self.output


##[B*1*F*1*1*D] * 【1*H*F*F*3*D】->[B*H*F*F*3*D]
class QKVFieldTransform(nn.Module):
    def __init__(self, headNum, inDim, featureNumb):
        super(QKVFieldTransform, self).__init__()
        self.featureNumb = featureNumb
        self.headNum = headNum
        self.inDim = inDim
        self.weight = nn.Parameter(torch.zeros(size=(headNum, featureNumb, featureNumb, 3, inDim)))
        nn.init.xavier_uniform(self.weight.data, gain=1.414)
        self.output = None

    def forward(self, featureVec) -> torch.Tensor:
        cat = featureVec[:, None, :, None, None, :]
        # print(self.inDim,self.outDim,cat.shape)
        weight = self.weight[None, :, :, :, :]
        self.output = cat * weight
        # print('qkv shape', self.output.shape)
        return self.output


class QKVFieldTransformV2(nn.Module):
    def __init__(self, headNum, inDim, featureNumb):
        super(QKVFieldTransformV2, self).__init__()
        self.featureNumb = featureNumb
        self.headNum = headNum
        self.inDim = inDim
        self.weight = nn.Parameter(torch.zeros(size=(headNum, featureNumb, featureNumb, 3, inDim)))
        nn.init.xavier_uniform(self.weight.data, gain=1.414)
        self.output = None

    def forward(self, featureVec) -> torch.Tensor:
        cat = featureVec[:, None, :, None, None, :]
        # print(self.inDim,self.outDim,cat.shape)
        weight = self.weight[None, :, :, :, :]
        self.output = cat * weight
        # print('qkv shape', self.output.shape)
        return self.output
