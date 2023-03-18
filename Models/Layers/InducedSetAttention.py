from typing import List
import torch
import torch.nn as nn
from Models.BaseTransforms.LinearTransform import LinearTransform
from torch.nn import LayerNorm
from Models.BaseTransforms.ConcatResnetTransform import ConcatResnetTransform, ConcatResnetTransformV3
from Models.BaseTransforms.QKVTransform import QKVTransform, QKVFieldTransform, QKVTransformV2
from Models.BaseTransforms.Attention import DotAttention, GaussianAttention, DotWeight, DotAttentionV3, DotAttentionV2


class InducedSetAttention(nn.Module):
    def __init__(self, featureDim, inducedDim):
        super(InducedSetAttention, self).__init__()
        self.inducedDim = inducedDim
        self.KV: QKVTransformV2 = QKVTransformV2(2, featureDim, featureDim)
        self.query = nn.Parameter(torch.zeros(size=(inducedDim, featureDim)))
        nn.init.normal(self.query.data, mean=0, std=0.01)
        self.output = None
        self.headOut = None
        self.layerNorm1 = LayerNorm(featureDim)
        self.layerNorm2 = LayerNorm(featureDim)
        self.dotAttention = DotAttention()
        self.DNN = LinearTransform([featureDim, featureDim * 4, featureDim], True)

    # QKV:[B,H,3,F,D]
    def forward(self, featureVec: torch.Tensor):
        # QKV :
        kv = self.KV(featureVec)
        # attention:
        # score: [B,H,F,D] *[B,H,F,D] -> [B,H,F,F]
        # sum: 【B,H,F,F】 * [B,H,F,D] -> [B,H,F,D]
        self.headOut = self.dotAttention(self.query.unsqueeze(0), kv[:, 0, :, :], kv[:, 1, :, :])
        # concatHead:[B,H,F,D], [B,F,D] ->[B,F,D*N]
        res = self.layerNorm1(self.headOut)
        DNNOut = self.DNN(res)
        res2 = self.layerNorm2(res + DNNOut)
        self.output = res2
        # print(f'out:{self.output.shape}')
        return self.output
