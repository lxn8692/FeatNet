from Models.BaseTransforms.ConcatResnetTransform import ConcatResnetTransform, ConcatResnetTransformV2
from Models.Layers.LambdaLayer import LambdaLayer
from Models.BaseTransforms.LTE import LTETransform
import torch
import torch.nn as nn


class LambdaLTELayers(nn.Module):
    def __init__(self, featureDim, outDim=None):
        super(LambdaLTELayers, self).__init__()
        self.featureDim = featureDim
        self.outDim = outDim if outDim is not None else featureDim
        self.Lambda = LambdaLayer(featureDim, outDim)
        self.Prune = LambdaLayer(featureDim, outDim)
        # self.resNet = ConcatResnetTransformV2(outDim=outDim, originDim=featureDim)

    def forward(self, feature):
        gate = torch.sigmoid(self.Prune(feature))
        out = torch.tanh(self.Lambda(feature))
        output = gate * out
        # res = self.resNet(output, feature)
        return output


class LambdaGateLayers(nn.Module):
    def __init__(self, featureDim, outDim=None):
        super(LambdaGateLayers, self).__init__()
        self.featureDim = featureDim
        self.outDim = outDim if outDim is not None else featureDim
        self.Lambda = LambdaLayer(featureDim, outDim)
        self.LTE=LTETransform()
        self.Prune = LambdaLayer(featureDim, outDim)

    def forward(self, feature):
        gate = self.LTE.apply(torch.tanh(self.Prune(feature)))
        out = self.Lambda(feature)
        output = gate * out
        return output


class LambdaGateLayersV2(nn.Module):
    def __init__(self, featureDim, outDim=None):
        super(LambdaGateLayersV2, self).__init__()
        self.featureDim = featureDim
        self.outDim = outDim if outDim is not None else featureDim
        self.Lambda = LambdaLayer(featureDim, outDim)
        self.Prune = LambdaLayer(featureDim, outDim)

    def forward(self, feature):
        gate = torch.tanh(self.Prune(feature))
        out = self.Lambda(feature)
        output = gate * out
        return output
