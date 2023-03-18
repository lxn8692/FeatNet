from typing import List
import torch
from torch import nn

from Models.BaseTransforms.LinearTransform import LinearTransform


class ConcatResnetTransform(nn.Module):
    def __init__(self, concatDim: list, originDim: int):
        super(ConcatResnetTransform, self).__init__()
        self.concatDim = concatDim
        self.originDim = originDim
        self.weight = nn.Linear(originDim, sum(concatDim), bias=False)
        nn.init.xavier_normal(self.weight.weight, gain=1.414)
        self.output = None
        self.concat = None
        self.activation = nn.ReLU()

    # concatHead:[B,H,F,D], [B,F,D] ->[B,F,D*N]
    def forward(self, concatFeature: torch.Tensor, originFeature):
        shape = concatFeature.shape
        self.concat = concatFeature.reshape((shape[0], shape[2], -1))
        # print(self.concat.shape,sum(self.concatDim),self.originDim)
        self.output = self.activation(self.weight(originFeature) + self.concat)
        return self.output


class ConcatResnetTransformV2(nn.Module):
    def __init__(self, outDim: int, originDim: int):
        super(ConcatResnetTransformV2, self).__init__()
        self.outDim = outDim
        self.originDim = originDim
        self.weight = nn.Linear(originDim, outDim, bias=False)
        nn.init.xavier_normal(self.weight.weight, gain=1.414)
        self.output = None
        self.concat = None
        self.activation = nn.ReLU()

    # concatHead:[B,H,F,D], [B,F,D] ->[B,F,D*N]
    def forward(self, feature: torch.Tensor, originFeature):
        # print(self.concat.shape,sum(self.concatDim),self.originDim)
        self.output = self.activation(self.weight(originFeature) + feature)
        return self.output


class ConcatResnetTransformV3(nn.Module):
    def __init__(self, headNumb, originDim: int):
        super(ConcatResnetTransformV3, self).__init__()
        self.originDim = originDim
        self.weight = nn.Linear(headNumb * originDim, originDim, bias=False)
        nn.init.xavier_normal(self.weight.weight, gain=1.414)
        self.output = None
        self.concat = None
        # self.activation = nn.ReLU()

    # concatHead:[B,H,F,D], [B,F,D] ->[B,F,D*N]
    def forward(self, concatFeature: torch.Tensor):
        shape = concatFeature.shape
        self.concat = concatFeature.reshape((shape[0], shape[2], -1))
        # print(self.concat.shape,sum(self.concatDim),self.originDim)
        self.output = self.weight(self.concat)
        return self.output
