from typing import List

import torch.nn as nn
import torch
from Models.Layers.ConcatMlpLayers import ConcatMlpLayer
from Models.BaseTransforms.ConcatResnetTransform import ConcatResnetTransform
from Models.BaseTransforms.QKVTransform import QKVTransform
from Models.BaseTransforms.Attention import DotAttention, GaussianAttention
from enum import Enum


class MlpGaussianLayer(nn.Module):
    def __init__(self, featureNum, inDim, outDim,layerDims:list):
        super(MlpGaussianLayer, self).__init__()
        self.layerDims = layerDims
        self.featureNum = featureNum
        self.inputDim = inDim
        self.outputDim = outDim
        self.conCatMlp=ConcatMlpLayer()
        self.output = None

    # QKV:[B,H,3,F,D]
    def forward(self, featureVec):
        # QKV :
        self.QKV = self.QKVKernel(featureVec)
        # attention:
        # score: [B,H,F,D] *[B,H,F,D] -> [B,H,F,F]
        # sum: 【B,H,F,F】 * [B,H,F,D] -> [B,H,F,D]
        self.headOut = self.dotAttention(self.QKV[:, :, 0, :, :], self.QKV[:, :, 1, :, :], self.QKV[:, :, 2, :, :])
        # concatHead:[B,H,F,D], [B,F,D] ->[B,F,D*N]
        self.output = self.concatRes(self.headOut, featureVec)
        # print(f'out:{self.output.shape}')
        return self.output

