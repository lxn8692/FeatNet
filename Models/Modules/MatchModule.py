from itertools import chain
from typing import List

import torch.nn as nn
import torch
from abc import abstractmethod, ABCMeta
from Models.BaseTransforms.GaussianTransform import GaussianTransform
from Models.Layers.GaussianLayers import GaussianLayers
from Models.Layers.ConcatMlpLayers import ConcatMlpLayer, ConcatMlpLayerV2
from Models.BaseTransforms.FmFMTransform import FM2LinearTransform


# class MatchInterface(ABCMeta):
#     @abstractmethod
#     def matchInput(self, userTrans, userFeature, itemFeature, itemTrans, fusionFeature):
#         pass


def defaultHooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
    return list(chain(userTrans, itemTrans, contextTrans))


class MuItemAttention(nn.Module):
    def __init__(self, hooker):
        super(MuItemAttention, self).__init__()
        self.hooker = hooker
        self.activation = nn.Sigmoid()
        self.out = None

    def forward(self, userFeature, userTrans, itemFeature, itemTrans: torch.tensor, contextFeature,
                contextTrans, fusionFeature):
        mu, itemTrans, fusionFeature = self.hooker(userFeature, userTrans, itemFeature, itemTrans, contextFeature,
                                                   contextTrans, fusionFeature)
        dot = torch.matmul(itemTrans, mu).squeeze()
        weight = torch.softmax(dot, dim=1)
        matchValue = torch.sum(fusionFeature, dim=2)
        score = torch.matmul(weight[:, None, :], matchValue[:, :, None]).squeeze()
        self.out = self.activation(score)
        return self.out


class ItemGaussian(nn.Module):
    class ItemGaussian(nn.Module):
        def __init__(self, featureNumb, itemInputDim: list, outputDim: list):
            super(ItemGaussian, self).__init__()
            self.featureNumb = featureNumb
            self.Gaussian = GaussianLayers(featureNumb, itemInputDim, outputDim)

        def forward(self, userFeature, userTrans, itemFeature, itemTrans, contextFeature,
                    contextTrans, fusionFeature):
            GaussianVec = self.Gaussian([itemFeature for i in range(self.featureNumb)], userFeature)
            weight = torch.sum(GaussianVec)
            matchValue = torch.sum(fusionFeature)
            score = torch.dot(weight, matchValue)
            return score


class UserConcatMlp(nn.Module):
    def __init__(self, featureNum, layersDim: list):
        super(UserConcatMlp, self).__init__()
        self.featureNum = featureNum,
        self.concatMLP = ConcatMlpLayer(layersDim)

    def forward(self, userFeature, userTrans, itemFeature, itemTrans, contextFeature,
                contextTrans, fusionFeature):
        weight = self.concatMLP(userFeature)
        matchValue = torch.sum(fusionFeature)
        score = torch.dot(matchValue, weight)
        return score


class AllConcatMlp(nn.Module):
    def __init__(self, layersDim: list, nameList: list, hooker):
        super(AllConcatMlp, self).__init__()
        self.concatMLP = ConcatMlpLayer(layersDim, nameList)
        self.hooker = hooker
        self.activation = nn.Sigmoid()

    def forward(self, userFeature, userTrans, itemFeature, itemTrans, contextFeature,
                contextTrans, fusionFeature):
        concatVec: dict = self.hooker(userFeature, userTrans, itemFeature, itemTrans, contextFeature,
                                      contextTrans, fusionFeature)
        score = self.concatMLP(concatVec)
        return self.activation(score)


class AllConcatMlpV2(nn.Module):
    def __init__(self, layersDim: list, hooker=None, ):
        super(AllConcatMlpV2, self).__init__()
        self.concatMLP = ConcatMlpLayerV2(layersDim)
        self.hooker = hooker
        self.activation = nn.Sigmoid()

    def forward(self, userFeature, userTrans, itemFeature, itemTrans, contextFeature,
                contextTrans, fusionFeature):
        if self.hooker is not None:
            concatVec: torch.Tensor = self.hooker(userFeature, userTrans, itemFeature, itemTrans, contextFeature,
                                                  contextTrans, fusionFeature)
        else:
            concatVec = fusionFeature
        score = self.concatMLP(concatVec)

        return self.activation(score)


class AllConcatMlpWithLinear(nn.Module):
    def __init__(self, layersDim: list, featureNumb, featureDim, hooker=None):
        super(AllConcatMlpWithLinear, self).__init__()
        self.concatMLP = ConcatMlpLayerV2(layersDim)
        self.hooker = hooker
        self.activation = nn.Sigmoid()
        self.linear = FM2LinearTransform(featureNumb, featureDim)
        self.linearHooker = defaultHooker
        self.act = nn.LeakyReLU()

    def forward(self, userFeature, userTrans, itemFeature, itemTrans, contextFeature,
                contextTrans, fusionFeature):
        if self.hooker is not None:
            concatVec = self.hooker(userFeature, userTrans, itemFeature, itemTrans, contextFeature,
                                    contextTrans, fusionFeature)
        else:
            concatVec = fusionFeature
        score = self.concatMLP(concatVec)
        feature = self.linearHooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i[:, None, :] for i in feature]
        cat = torch.cat(featureVec, dim=1)
        cat = self.act(cat)
        linear = self.linear(cat)
        if len(linear.shape) == 1:
            linear = linear[None, :]
        linearScore = torch.sum(linear, dim=1, keepdim=True)
        result = linearScore + score
        return self.activation(result)


class AllConcatMlpWithWeightLinear(nn.Module):
    def __init__(self, layersDim: list, featureNumb, featureDim, fieldWeight=None, hooker=None):
        super(AllConcatMlpWithWeightLinear, self).__init__()
        self.fieldWeight = fieldWeight
        self.concatMLP = ConcatMlpLayerV2(layersDim)
        self.hooker = hooker
        self.activation = nn.Sigmoid()
        self.linear = FM2LinearTransform(featureNumb, featureDim)
        self.linearHooker = hooker
        self.act = nn.LeakyReLU()

    def forward(self, userFeature, userTrans, itemFeature, itemTrans, contextFeature,
                contextTrans, fusionFeature):
        if self.hooker is not None:
            concatVec = self.hooker(userFeature, userTrans, itemFeature, itemTrans, contextFeature,
                                    contextTrans, fusionFeature)
        else:
            concatVec = fusionFeature
        score = self.activation(self.concatMLP(concatVec))
        feature = self.linearHooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i[:, None, :] for i in feature]
        cat = torch.cat(featureVec, dim=1)
        cat = self.act(cat)
        if self.fieldWeight is None:
            linear = self.linear(cat)
        else:
            linear = self.linear(cat, self.fieldWeight)
        if len(linear.shape) == 1:
            linear = linear[None, :]
        linearScore = self.activation(torch.sum(linear, dim=1, keepdim=True))
        result = 0.4 * linearScore + 0.6 * score
        return result


class DefaultMatchModule(nn.Module):
    def __init__(self):
        super(DefaultMatchModule, self).__init__()

    def forward(self, userFeature, userTrans, itemFeature, itemTrans, contextFeature,
                contextTrans, fusionFeature):
        return fusionFeature


class FmMatchModule(nn.Module):
    def __init__(self):
        super(FmMatchModule, self).__init__()
        self.activation = nn.Sigmoid()
        self.out = None

    def forward(self, userFeature, userTrans, itemFeature, itemTrans, contextFeature,
                contextTrans, fusionFeature):
        # print("fussionFeature\n",fusionFeature)
        sum = torch.sum(fusionFeature, dim=1)
        # print('FmMatchModule\n',sum)
        self.out = self.activation(sum)
        return self.out
