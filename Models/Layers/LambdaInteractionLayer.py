import torch
import torch.nn as nn
from Models.Layers.BiLinearLayer import BiLinearLayer
from Models.BaseTransforms.LTE import LTETransform
from Models.Layers.LambdaLayer import LambdaLayerV2


class LambdaInteractionLayer(nn.Module):
    def __init__(self, featureNumb, embedSize, headNum=3, depth=4):
        super(LambdaInteractionLayer, self).__init__()
        self.embedSize = embedSize
        self.featureNumb = featureNumb
        self.headNum = headNum
        self.depth = depth
        lambdaLayer = []
        for i in range(depth):
            lambdaLayer.append(LambdaLayerV2(embedSize, headNum, embedSize))
        self.Lambda = nn.ModuleList(lambdaLayer)

    def forward(self, feature, fieldWeight=None):
        # [B,F,1,D][1,F,D,D]-> [b,f,d]
        if type(feature) is tuple:
            fieldWeight = feature[1]
            feature = feature[0]
        result = []
        result.append(feature)
        input1 = feature
        input2 = fieldWeight
        for i in range(self.depth):
            input1 = self.Lambda[i](input1, input2)
            if i == 0:
                input2 = input1
            result.append(input1)
        result = torch.cat(result, dim=1)
        return result


class LambdaInteractionLayerV2(nn.Module):
    def __init__(self, featureNumb, embedSize, headNum=3, depth=4):
        super(LambdaInteractionLayerV2, self).__init__()
        self.embedSize = embedSize
        self.featureNumb = featureNumb
        self.headNum = headNum
        self.depth = depth
        lambdaLayer = []
        for i in range(depth):
            lambdaLayer.append(LambdaLayerV2(embedSize, headNum, embedSize))
        self.Lambda = nn.ModuleList(lambdaLayer)

    def forward(self, feature, fieldWeight=None):
        # [B,F,1,D][1,F,D,D]-> [b,f,d]
        if type(feature) is tuple:
            fieldWeight = feature[1]
            feature = feature[0]
        result = []
        result.append(feature)
        input1 = feature
        input2 = fieldWeight
        for i in range(self.depth):
            input2 = self.Lambda[i](input1, input2)
            if i == 0:
                input1 = input2
            result.append(input2)
        result = torch.cat(result, dim=1)
        return result


class LambdaInteractionLayerV3(nn.Module):
    def __init__(self, featureNumb, embedSize, headNum=16, ):
        super(LambdaInteractionLayerV3, self).__init__()
        self.embedSize = embedSize
        self.featureNumb = featureNumb
        self.headNum = headNum
        self.gate = LambdaLayerV2(embedSize, headNum, 1, False)
        self.featureField = LambdaLayerV2(embedSize, headNum, embedSize)
        self.interaction = LambdaLayerV2(embedSize, headNum, embedSize)

    def forward(self, feature, fieldWeight=None):
        # [B,F,1,D][1,F,D,D]-> [b,f,d]
        gate = torch.sigmoid(self.gate(feature, fieldWeight))
        fieldFeature = self.featureField(feature, fieldWeight)
        filter = fieldFeature * gate
        interaction = self.interaction(filter, filter)
        return interaction


class LambdaInteractionLayerV4(nn.Module):
    def __init__(self, featureNumb, embedSize, headNum=16, ):
        super(LambdaInteractionLayerV4, self).__init__()
        self.embedSize = embedSize
        self.featureNumb = featureNumb
        self.headNum = headNum
        self.LTE = LTETransform()
        self.gate = LambdaLayerV2(embedSize, headNum, 1, False)
        self.featureField = LambdaLayerV2(embedSize, headNum, embedSize)
        self.interaction = BiLinearLayer(featureNumb, embedSize)

    def forward(self, feature, fieldWeight=None, fieldGate=None):
        # [B,F,1,D][1,F,D,D]-> [b,f,d]
        # gate = torch.sigmoid(self.gate(feature, fieldGate))
        gate = self.LTE.apply(torch.tanh(self.gate(feature, fieldGate)))
        # fieldFeature = self.featureField(feature, fieldWeight)
        filter = feature * gate
        interaction = self.interaction(filter)
        return interaction
