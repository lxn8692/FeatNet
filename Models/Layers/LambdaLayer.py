import torch
import torch.nn as nn
from Models.BaseTransforms.ConcatResnetTransform import ConcatResnetTransform
from Models.BaseTransforms.QKVTransform import QKVTransformV2


class LambdaLayer(nn.Module):
    def __init__(self, featureDim, outDim=None, BN=True):
        super(LambdaLayer, self).__init__()
        self.BN = BN
        self.featureDim = featureDim
        self.outDim = outDim if outDim is not None else featureDim
        self.QK = QKVTransformV2(2, featureDim, featureDim)
        self.V = QKVTransformV2(1, featureDim, self.outDim)
        self.bN = nn.BatchNorm1d(featureDim)

    def forward(self, feature):
        QK = self.QK(feature)
        value = self.V(feature)
        query = QK[:, 0]
        key = QK[:, 1]
        softmax = torch.softmax(key, dim=1)
        _lambda = softmax.transpose(1, 2) @ value
        if self.BN is True:
            BN = self.bN(_lambda)
        else:
            BN = _lambda
        result = query @ BN
        return result


class LambdaLayerV2(nn.Module):
    def __init__(self, featureDim, midDim, contextDim=None,outDim=None, BN=True):
        super(LambdaLayerV2, self).__init__()
        self.BN = BN
        self.featureDim = featureDim
        self.midDim = midDim
        self.outDim = outDim if outDim is not None else featureDim
        self.contextDim=contextDim if contextDim is not None else featureDim
        self.Q = QKVTransformV2(1, featureDim, self.midDim)
        self.K = QKVTransformV2(1, self.contextDim, midDim)
        self.V = QKVTransformV2(1, self.contextDim, self.outDim)
        self.bN = nn.BatchNorm1d(midDim)

    def forward(self, feature, context):
        # print(feature.shape, context.shape)
        value = self.V(context)
        query = self.Q(feature)
        key = self.K(context)
        softmax = torch.softmax(key, dim=1)
        _lambda = softmax.transpose(1, 2) @ value
        if self.BN is True:
            BN = self.bN(_lambda)
        else:
            BN = _lambda
        result = torch.matmul(query, BN)
        return result
