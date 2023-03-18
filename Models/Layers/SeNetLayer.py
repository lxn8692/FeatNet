import torch
import torch.nn as nn
from Models.BaseTransforms.LTE import LTETransform
from Models.BaseTransforms.LinearTransform import LinearTransform


class SeNetLayer(nn.Module):
    def __init__(self, featureNumb):
        super(SeNetLayer, self).__init__()
        self.featureNumb = featureNumb
        self.linear = LinearTransform([self.featureNumb, self.featureNumb // 2, self.featureNumb], False)
        self.output = None

    def forward(self, feature: torch.Tensor):
        # squeeze [B,F,D]->[B,F]
        squeeze = torch.mean(feature, dim=-1)
        # excitation
        excitation = self.linear(squeeze)
        self.output = feature * torch.unsqueeze(excitation, -1)
        return self.output


class SeNetNormLayer(nn.Module):
    def __init__(self, featureNumb):
        super(SeNetNormLayer, self).__init__()
        self.featureNumb = featureNumb
        self.linear = LinearTransform([self.featureNumb, self.featureNumb // 8, self.featureNumb], False)
        self.output = None

    def forward(self, feature: torch.Tensor):
        # squeeze [B,F,D]->[B,F]
        squeeze = torch.mean(feature, dim=-1)
        # excitation
        # excitation = self.linear(squeeze)
        self.output = feature * torch.sigmoid(torch.unsqueeze(squeeze, -1))
        return self.output


class SeNetGateLayer(nn.Module):
    def __init__(self, featureNumb):
        super(SeNetGateLayer, self).__init__()
        self.featureNumb = featureNumb
        self.linear = LinearTransform([self.featureNumb, self.featureNumb // 8, self.featureNumb], True)
        self.LTE = LTETransform()
        self.output = None

    def forward(self, feature: torch.Tensor):
        # squeeze [B,F,D]->[B,F]
        squeeze = torch.mean(feature, dim=-1)
        # excitation
        excitation = self.linear(squeeze)
        # indicator = torch.tanh(excitation)
        # gate = self.LTE.apply(indicator)
        gate = excitation
        self.output = feature * torch.unsqueeze(gate, -1)
        return self.output


class SeNetLayerNoRelu(nn.Module):
    def __init__(self, featureNumb):
        super(SeNetLayerNoRelu, self).__init__()
        self.featureNumb = featureNumb
        self.linear = LinearTransform([self.featureNumb, self.featureNumb // 8, self.featureNumb], True)
        self.output = None

    def forward(self, feature: torch.Tensor):
        # squeeze [B,F,D]->[B,F]
        squeeze = torch.mean(feature, dim=-1)
        # excitation
        excitation = self.linear(squeeze)
        self.output = feature * torch.unsqueeze(excitation, -1)
        return self.output, excitation


class SeNetLayerLTE(nn.Module):
    def __init__(self, featureNumb):
        super(SeNetLayerLTE, self).__init__()
        self.featureNumb = featureNumb
        self.linear = LinearTransform([self.featureNumb, self.featureNumb // 8, self.featureNumb], True)
        self.output = None
        self.LTE = LTETransform()

    def forward(self, feature: torch.Tensor):
        # squeeze [B,F,D]->[B,F]
        squeeze = torch.mean(feature, dim=-1)
        # excitation
        excitation = self.LTE.apply(self.linear(squeeze))
        self.output = feature * torch.unsqueeze(excitation, -1)
        return self.output
