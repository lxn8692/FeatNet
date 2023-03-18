from typing import List
from datetime import datetime
import torch
from Models.Layers.LambdaInteractionLayer import LambdaInteractionLayer, LambdaInteractionLayerV2, \
    LambdaInteractionLayerV3, LambdaInteractionLayerV4
from Models.Layers.LambdaLayer import LambdaLayer
from Models.Layers.LambdaLTELayers import LambdaLTELayers, LambdaGateLayers
from Models.Layers.FestureLTELayer import FeatureLTELayer
from Models.Layers.InteractionPruningLayer import InteractionPruningLayer, InteractionPruningEluLayer, \
    InteractionPruningEluLayerV2, InteractionPruningEluLayerV3, InteractionXPruneLayer
from Models.BaseTransforms.InteractionPruning import InteractionPruning
from Models.BaseTransforms.featureFiledAwaredPruning import FeaturePruningLayer, StructPruningLayer, \
    FeaturePruningLayerV2
from Models.Layers.AggregationLayer import AggregationLayer
from Models.Layers.CINLayers import CINLayer
from torch import nn
from collections import OrderedDict
from Models.Layers.GaussianLayers import GaussianLayers
from Models.Layers.MultiHeadTransformerLayer import MultiHeadTransformerLayer, MultiHeadTransformerGauLayer, \
    SelfAttentionLayer, MultiHeadTransformerWithFieldLayer, MultiHeadTransformerWithFieldLayerV3, \
    MultiHeadTransformerWithFieldLayerV4, MultiHeadTransformerXLayer, MultiHeadTransformerELUXLayer, \
    MHTestTransformerLayer, MultiHeadTransformerSparseLayer
from Models.Layers.ConcatMlpLayers import ConcatMlpLayer, ConcatMlpLayerV2
from Models.Layers.SeNetLayer import SeNetLayer, SeNetGateLayer, SeNetNormLayer, SeNetLayerNoRelu, SeNetLayerLTE
from Models.Layers.BiLinearLayer import BiLinearLayer, BiLinearFieldLayer, BiLinearLayerV2, BiLinearDotLayer
from Models.BaseTransforms.FmFMTransform import FmFMTransform, FM2LinearTransform, FvFMTransform
from Models.BaseTransforms.FMTransform import FMTransform, FmLinearTransform
from itertools import chain
from Models.BaseTransforms.LinearTransform import LinearTransform
from math import sqrt
from Models.Layers.FeatureSTRLayer import FeatureSTRLayer


class DefaultFusionModule(nn.Module):
    def __init__(self):
        super(DefaultFusionModule, self).__init__()

    def forward(self, userTrans, userFeature, itemFeature, itemTrans):
        return None


class GaussianFusionModule(nn.Module):
    def __init__(self, featureNumb, inputDim: list, outputDim: list, nameList=None, backupHooker=None):
        super(GaussianFusionModule, self).__init__()
        assert len(inputDim) == len(outputDim) == featureNumb
        self.backupHooker = backupHooker
        self.featureNumb = featureNumb
        self.gaussian = GaussianLayers(featureNumb, inputDim, outputDim, nameList)
        self.output = None

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        keyVal, matchVal = self.backupHooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature,
                                             contextTrans)
        self.output = self.gaussian(keyVal, matchVal)
        return self.output


def defaultHooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
    return list(chain(userTrans, itemTrans, contextTrans))


class MultiLayerTransformerSparse(nn.Module):
    def __init__(self, featureNum, layerDims: List, headNum: List, hooker=None):
        super(MultiLayerTransformerSparse, self).__init__()
        self.headNum = headNum
        self.layerDims = layerDims
        self.transformer = nn.Sequential(OrderedDict([
            (f'transformer{i}',
             MultiHeadTransformerSparseLayer(featureNum, layerDims[i] * headNum[i], layerDims[i + 1], headNum[i])) for
            i in
            range(len(layerDims) - 1)])
        )
        self.hooker = defaultHooker if hooker is None else hooker
        self.output = None

        # 所有feature转换成【B，F，D】

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        # print(cat.shape)
        self.output = self.transformer(cat)
        return self.output
        # return cat

class MultiLayerTransformer(nn.Module):
    def __init__(self, featureNum, layerDims: List, headNum: List, hooker=None):
        super(MultiLayerTransformer, self).__init__()
        self.headNum = headNum
        self.layerDims = layerDims
        self.transformer = nn.Sequential(OrderedDict([
            (f'transformer{i}',
             MultiHeadTransformerLayer(featureNum, layerDims[i] * headNum[i], layerDims[i + 1], headNum[i])) for
            i in
            range(len(layerDims) - 1)])
        )
        self.hooker = defaultHooker if hooker is None else hooker
        self.output = None

        # 所有feature转换成【B，F，D】

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        # print(cat.shape)
        self.output = self.transformer(cat)
        return self.output
        # return cat


class MLTestTransformer(nn.Module):
    def __init__(self, featureNum, featureDim, headNumb, depth=3, hooker=None):
        super(MLTestTransformer, self).__init__()
        self.headNum = headNumb
        self.position = nn.Parameter(torch.zeros(size=(featureNum, featureDim)))
        nn.init.xavier_normal(self.position.data, gain=1.414)
        self.transformer = nn.Sequential(OrderedDict([
            (f'transformer{i}',
             MHTestTransformerLayer(featureNum, featureDim, headNumb=headNumb)) for
            i in range(depth)])
        )
        self.hooker = defaultHooker if hooker is None else hooker
        self.output = None

        # 所有feature转换成【B，F，D】

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        cat = cat + self.position[None, :, :]
        # print(cat.shape)
        self.output = self.transformer(cat)
        return self.output
        # return cat


class MLTestTransformerV2(nn.Module):
    def __init__(self, featureNum, featureDim, headNumb, depth=3, position=None, hooker=None):
        super(MLTestTransformerV2, self).__init__()
        self.headNum = headNumb
        self.position = position
        self.transformer = nn.Sequential(OrderedDict([
            (f'transformer{i}',
             MHTestTransformerLayer(featureNum, featureDim, headNumb=headNumb)) for
            i in range(depth)])
        )
        self.hooker = defaultHooker if hooker is None else hooker
        self.output = None

        # 所有feature转换成【B，F，D】

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        cat = cat + self.position[None, :, :]
        # print(cat.shape)
        self.output = self.transformer(cat)
        return self.output
        # return cat


class MultiLayerTransformerRaw(nn.Module):
    def __init__(self, featureNum, layerDims: List, headNum: List, DNNLayerDims, hooker=None):
        super(MultiLayerTransformerRaw, self).__init__()
        self.DNNLayerDims = DNNLayerDims
        self.headNum = headNum
        self.layerDims = layerDims
        self.transformer = nn.Sequential(OrderedDict([
            (f'transformer{i}',
             MultiHeadTransformerLayer(featureNum, layerDims[i] * headNum[i], layerDims[i + 1], headNum[i])) for
            i in
            range(len(layerDims) - 1)])
        )
        self.DNN = ConcatMlpLayerV2(DNNLayerDims)
        self.AutoDNN = ConcatMlpLayerV2(DNNLayerDims)
        self.hooker = defaultHooker if hooker is None else hooker
        self.output = None

        # 所有feature转换成【B，F，D】

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        # print(cat.shape)
        dnn = self.DNN(cat)
        output = self.AutoDNN(self.transformer(cat))
        self.output = torch.sigmoid(dnn + output)
        return self.output
        # return cat


class MultiLayerTransformerX(nn.Module):
    def __init__(self, featureNum, layerDims: List, headNum: List, hooker=None):
        super(MultiLayerTransformerX, self).__init__()
        self.headNum = headNum
        self.layerDims = layerDims
        self.transformer = nn.Sequential(OrderedDict([
            (f'transformer{i}',
             MultiHeadTransformerXLayer(featureNum, layerDims[i] * headNum[i], layerDims[i + 1], headNum[i])) for
            i in
            range(len(layerDims) - 1)])
        )
        self.hooker = defaultHooker if hooker is None else hooker
        self.output = None

        # 所有feature转换成【B，F，D】

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        # print(cat.shape)
        self.output = self.transformer(cat)
        return self.output
        # return cat


class MultiLayerTransformerGau(nn.Module):
    def __init__(self, featureNum, layerDims: List, headNum: List, hooker=None):
        super(MultiLayerTransformerGau, self).__init__()
        self.headNum = headNum
        self.layerDims = layerDims
        self.transformer = nn.Sequential(OrderedDict([
            (f'transformer{i}',
             MultiHeadTransformerGauLayer(featureNum, layerDims[i] * headNum[i], layerDims[i + 1], headNum[i])) for
            i in
            range(len(layerDims) - 1)])
        )
        self.hooker = defaultHooker if hooker is None else hooker
        self.output = None

        # 所有feature转换成【B，F，D】

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i[:, None, :] for i in feature]
        cat = torch.cat(featureVec, dim=1)
        # print(cat.shape)
        # self.output = self.transformer(cat)
        # return self.output

        return self.output


class MultiLayerTransformerWithField(nn.Module):
    def __init__(self, featureNum, layerDims: List, headNum: List, hooker=None):
        super(MultiLayerTransformerWithField, self).__init__()
        self.headNum = headNum
        self.layerDims = layerDims
        self.transformer = nn.Sequential(OrderedDict([
            (f'transformer{i}',
             MultiHeadTransformerWithFieldLayer(featureNum, layerDims[i] * headNum[i], headNum[i]))
            for
            i in
            range(len(layerDims))])
        )
        self.hooker = defaultHooker if hooker is None else hooker
        self.output = None

        # 所有feature转换成【B，F，D】

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i[:, None, :] for i in feature]
        cat = torch.cat(featureVec, dim=1)
        # print(cat.shape)
        self.output = self.transformer(cat)
        return self.output


class FmFMFusionModule(nn.Module):
    def __init__(self, featureNum, featureDim, hooker=None):
        super(FmFMFusionModule, self).__init__()
        self.output = None
        self.featureDim = featureDim
        self.act = torch.nn.LeakyReLU()
        self.featureNum = featureNum
        self.FmFMTrans = FmFMTransform(featureNum, featureDim)
        self.FmTrans = FM2LinearTransform(featureNum, featureDim)
        self.hooker = defaultHooker if hooker is None else hooker

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        # def forward(self, cat):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i[:, None, :] for i in feature]
        cat = torch.cat(featureVec, dim=1)
        cat = self.act(cat)
        # print(cat.shape)
        secondTerm = self.FmFMTrans(cat)
        firstTerm = self.FmTrans(cat)
        if len(firstTerm.shape) == 1:
            secondTerm = secondTerm.reshape(1, -1)
            firstTerm = firstTerm.reshape(1, -1)
        self.output = torch.cat([firstTerm, secondTerm], dim=1)
        # print('FmFMFusion\n',self.output)
        return self.output


class FvFMFusionModule(nn.Module):
    def __init__(self, featureNum, featureDim, hooker=None):
        super(FvFMFusionModule, self).__init__()
        self.output = None
        self.featureDim = featureDim
        self.act = torch.nn.LeakyReLU()
        self.featureNum = featureNum
        self.FvFMTrans = FvFMTransform(featureNum, featureDim)
        self.FmTrans = FM2LinearTransform(featureNum, featureDim)
        self.hooker = defaultHooker if hooker is None else hooker

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        # def forward(self, cat):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i[:, None, :] for i in feature]
        cat = torch.cat(featureVec, dim=1)
        cat = self.act(cat)
        # print(cat.shape)
        secondTerm = self.FvFMTrans(cat)
        firstTerm = self.FmTrans(cat)
        if len(firstTerm.shape) == 1:
            secondTerm = secondTerm.reshape(1, -1)
            firstTerm = firstTerm.reshape(1, -1)
        self.output = torch.cat([firstTerm, secondTerm], dim=1)
        # print('FmFMFusion\n',self.output)
        return self.output


class FBDFMFusionModule(nn.Module):
    def __init__(self, featureNum, featureDim, hooker=None):
        super(FBDFMFusionModule, self).__init__()
        self.featureDim = featureDim
        self.featureNum = featureNum
        self.fieldEmbedding = torch.zeros(size=(featureNum, featureDim))[None, :, :]
        nn.init.normal(mean=0, std=0.01)
        self.fieldTrans = SelfAttentionLayer(featureNum, featureDim, featureDim)
        self.featureTrans = SelfAttentionLayer(featureNum, featureDim, featureDim)
        self.concatMlp = ConcatMlpLayerV2()
        self.hooker = defaultHooker if hooker is None else hooker

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        # def forward(self, cat):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i[:, None, :] for i in feature]
        cat = torch.cat(featureVec, dim=1)
        cat = self.act(cat)
        # print(cat.shape)
        fieldGraph = self.fieldTrans(self.fieldEmbedding)
        featureGraph = self.featureTrans(cat)

        # print('FmFMFusion\n',self.output)
        return self.output


class FMFusionModule(nn.Module, ):
    def __init__(self, featureNumb, hooker=None):
        super(FMFusionModule, self).__init__()
        self.featureNumb = featureNumb
        self.FMTrans = FMTransform()
        self.FMLinear = FmLinearTransform(featureNumb)
        self.act = torch.nn.LeakyReLU()
        self.hooker = defaultHooker if hooker is None else hooker
        self.output = None

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = []
        for i in feature:
            norm = torch.norm(i[:, None, :], p=2) * sqrt(self.featureNumb)
            if norm != 0:
                featureVec.append(i[:, None, :] / norm)
            else:
                featureVec.append(i[:, None, :])
        cat = torch.cat(featureVec, dim=1)
        cat = self.act(cat)
        # cat: [B,F,D]
        trans = self.FMTrans(cat)
        print('trrans\n', trans)
        linear = self.FMLinear()
        batch = trans.shape[0]
        shape = linear.shape[1]
        linear = linear.expand(batch, shape)
        self.output = torch.cat((trans, linear), dim=1)
        print('output\n', self.output)
        return self.output


class FieldAwarePruning(nn.Module):
    def __init__(self, featureNumb, featureDim, fieldDim, beta: list, zeta: list, gamma: list, hooker=None):
        super(FieldAwarePruning, self).__init__()
        self.gamma = gamma
        self.zeta = zeta
        self.beta = beta
        self.FieldDim = fieldDim
        self.FeatureDim = featureDim
        self.featureNumb = featureNumb
        self.field = nn.Parameter(torch.zeros(size=(featureNumb, fieldDim)))
        self.valueTransform = nn.Parameter(torch.zeros(size=(featureDim, featureDim)))
        torch.nn.init.normal_(self.valueTransform, mean=0, std=0.001)
        torch.nn.init.normal_(self.field.data, mean=0, std=0.001)
        self.featurePruning = FeaturePruningLayer(featureNumb, fieldDim, featureDim, beta[0], zeta[0], gamma[0])
        self.interactionPruning = InteractionPruning(featureDim, featureNumb, beta[1], zeta[1], gamma[1])
        self.structPruning: StructPruningLayer = StructPruningLayer(fieldDim, fieldDim, beta[2], zeta[2], gamma[2])
        self.hooker = defaultHooker if hooker is None else hooker
        self.featureL0 = None
        self.interactionL0 = None
        self.structureL0 = None
        self.output = None

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = []
        for i in feature:
            norm = torch.norm(i[:, None, :], p=2) * sqrt(self.featureNumb)
            if norm != 0:
                featureVec.append(i[:, None, :] / norm)
            else:
                featureVec.append(i[:, None, :])
        cat = torch.cat(featureVec, dim=1)
        featurePruning, _ = self.featurePruning(cat, self.field)
        self.featureL0 = self.featurePruning.L0
        structPruning = self.structPruning(self.field)
        self.structureL0 = self.structPruning.L0
        interactionPruning = self.interactionPruning(featurePruning)
        self.interactionL0 = self.interactionPruning.L0
        ### 看效果修改
        weight = interactionPruning * structPruning[None, :, :]
        value = torch.matmul(featurePruning, self.valueTransform)
        self.output = torch.matmul(weight, value)
        return self.output


class FieldAwarePruningV2(nn.Module):
    def __init__(self, featureNumb, featureDim, fieldDim, beta: list, zeta: list, gamma: list, hooker=None):
        super(FieldAwarePruningV2, self).__init__()
        self.gamma = gamma
        self.zeta = zeta
        self.beta = beta
        self.FieldDim = fieldDim
        self.FeatureDim = featureDim
        self.featureNumb = featureNumb
        self.field = nn.Parameter(torch.zeros(size=(featureNumb, fieldDim)))
        self.valueTransform = nn.Parameter(torch.zeros(size=(featureDim, featureDim)))
        torch.nn.init.normal_(self.valueTransform, mean=0, std=0.001)
        torch.nn.init.normal_(self.field.data, mean=0, std=0.001)
        self.featurePruning = FeaturePruningLayer(featureNumb, fieldDim, featureDim, beta[0], zeta[0], gamma[0])
        self.interactionPruning = InteractionPruning(featureDim, featureNumb, beta[1], zeta[1], gamma[1])
        # self.structPruning: StructPruningLayer = StructPruningLayer(fieldDim, fieldDim, beta[2], zeta[2], gamma[2])
        self.hooker = defaultHooker if hooker is None else hooker
        self.featureL0 = None
        self.interactionL0 = None
        self.structureL0 = None
        self.output = None

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = []
        for i in feature:
            norm = torch.norm(i[:, None, :], p=2) * sqrt(self.featureNumb)
            if norm != 0:
                featureVec.append(i[:, None, :] / norm)
            else:
                featureVec.append(i[:, None, :])
        cat = torch.cat(featureVec, dim=1)
        featurePruning, _ = self.featurePruning(cat, self.field)
        self.featureL0 = self.featurePruning.L0
        # structPruning = self.structPruning(self.field)
        # self.structureL0 = self.structPruning.L0
        interactionPruning = self.interactionPruning(featurePruning)
        self.interactionL0 = self.interactionPruning.L0
        ### 看效果修改
        weight = interactionPruning
        value = torch.matmul(featurePruning, self.valueTransform)
        self.output = torch.matmul(weight, value)
        return self.output


# 由于代码bug..这个就是等于只有featurePruning之后直接进mlp
class MultiLayerTransformerWithFieldFeaturePruningV1(nn.Module):
    def __init__(self, featureNum, layerDims: List, headNum: List, featureNumb, featureDim, fieldDim, beta: list,
                 zeta: list, gamma: list, hooker=None):
        super(MultiLayerTransformerWithFieldFeaturePruningV1, self).__init__()
        self.gamma = gamma
        self.zeta = zeta
        self.beta = beta
        self.featureDim = featureDim
        self.headNum = headNum
        self.layerDims = layerDims
        self.transformer = nn.Sequential(OrderedDict([
            (f'transformer{i}',
             MultiHeadTransformerWithFieldLayer(featureNum, layerDims[i] * headNum[i], layerDims[i + 1],
                                                headNum[i]))
            for
            i in
            range(len(layerDims))])
        )

        self.field = nn.Parameter(torch.Tensor(featureNumb, fieldDim))
        torch.nn.init.normal(self.field.data, mean=0, std=0.001)
        self.featurePruning = FeaturePruningLayer(featureNumb, fieldDim, featureDim, beta[0], zeta[0], gamma[0])
        self.hooker = defaultHooker if hooker is None else hooker
        self.output = None
        self.featureL0 = None

        # 所有feature转换成【B，F，D】

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i[:, None, :] for i in feature]
        cat = torch.cat(featureVec, dim=1)
        featurePruning, self.featureL0 = self.featurePruning(cat, self.field)
        # print(cat.shape)
        self.output = self.transformer(featurePruning)
        return self.output


## featurePruning接fieldAware +dot attetion
class MultiLayerTransformerWithFieldFeaturePruningV2(nn.Module):
    def __init__(self, featureNum, layerDims: List, headNum: List, featureNumb, featureDim, fieldDim, beta: list,
                 zeta: list, gamma: list, hooker=None):
        super(MultiLayerTransformerWithFieldFeaturePruningV2, self).__init__()
        self.gamma = gamma
        self.zeta = zeta
        self.beta = beta
        self.featureDim = featureDim
        self.headNum = headNum
        self.layerDims = layerDims
        self.transformer = nn.Sequential(OrderedDict([
            (f'transformer{i}',
             MultiHeadTransformerWithFieldLayer(featureNum, layerDims[i] * headNum[i], headNum[i]))
            for
            i in
            range(len(layerDims))])
        )

        self.field = nn.Parameter(torch.Tensor(featureNumb, fieldDim))
        torch.nn.init.normal(self.field.data, mean=0, std=0.001)
        self.featurePruning = FeaturePruningLayer(featureNumb, fieldDim, featureDim, beta[0], zeta[0], gamma[0])
        self.hooker = defaultHooker if hooker is None else hooker
        self.output = None
        self.featureL0 = None

        # 所有feature转换成【B，F，D】

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i[:, None, :] for i in feature]
        cat = torch.cat(featureVec, dim=1)
        featurePruning, self.featureL0 = self.featurePruning(cat, self.field)
        # print(cat.shape)
        self.output = self.transformer(featurePruning)
        return self.output


class MultiLayerTransformerWithFieldFeaturePruningV3(nn.Module):
    def __init__(self, featureNum, layerDims: List, headNum: List, featureNumb, featureDim, fieldDim, beta: list,
                 zeta: list, gamma: list, hooker=None):
        super(MultiLayerTransformerWithFieldFeaturePruningV3, self).__init__()
        self.gamma = gamma
        self.zeta = zeta
        self.beta = beta
        self.featureDim = featureDim
        self.headNum = headNum
        self.layerDims = layerDims
        self.transformer = nn.Sequential(OrderedDict([
            (f'transformer{i}',
             MultiHeadTransformerWithFieldLayerV3(featureNum, layerDims[i] * headNum[i], headNum[i]))
            for
            i in
            range(len(layerDims))])
        )

        self.field = nn.Parameter(torch.Tensor(featureNumb, fieldDim))
        torch.nn.init.normal(self.field.data, mean=0, std=0.001)
        self.featurePruning = FeaturePruningLayer(featureNumb, fieldDim, featureDim, beta[0], zeta[0], gamma[0])
        self.hooker = defaultHooker if hooker is None else hooker
        self.output = None
        self.featureL0 = None

        # 所有feature转换成【B，F，D】

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i[:, None, :] for i in feature]
        cat = torch.cat(featureVec, dim=1)
        featurePruning, self.featureL0 = self.featurePruning(cat, self.field)
        # print(cat.shape)
        self.output = self.transformer(featurePruning)
        return self.output


class MultiLayerTransformerWithFieldFeaturePruningV4(nn.Module):
    def __init__(self, featureNum, layerDims: List, headNum: List, featureNumb, featureDim, fieldDim, beta: list,
                 zeta: list, gamma: list, hooker=None):
        super(MultiLayerTransformerWithFieldFeaturePruningV4, self).__init__()
        self.gamma = gamma
        self.zeta = zeta
        self.beta = beta
        self.featureDim = featureDim
        self.headNum = headNum
        self.layerDims = layerDims
        self.transformer = nn.Sequential(OrderedDict([
            (f'transformer{i}',
             MultiHeadTransformerWithFieldLayerV4(featureNum, layerDims[i] * headNum[i], headNum[i], featureDim,
                                                  beta[1], zeta[1],
                                                  gamma[1]))
            for
            i in
            range(len(layerDims))])
        )

        self.field = nn.Parameter(torch.Tensor(featureNumb, fieldDim))
        torch.nn.init.normal(self.field.data, mean=0, std=0.001)
        self.featurePruning = FeaturePruningLayer(featureNumb, fieldDim, featureDim, beta[0], zeta[0], gamma[0])
        self.hooker = defaultHooker if hooker is None else hooker
        self.output = None
        self.featureL0 = None
        self.interactionL0 = None
        # 所有feature转换成【B，F，D】

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i[:, None, :] for i in feature]
        cat = torch.cat(featureVec, dim=1)
        featurePruning, self.featureL0 = self.featurePruning(cat, self.field)
        # print(cat.shape)
        self.output, self.interactionL0 = self.transformer(featurePruning)
        return self.output


# 剪枝加叉乘
class MultiLayerTransformerPruningX(nn.Module):
    def __init__(self, featureNum, layerDims: List, headNum: List, featureDim, fieldDim, beta: list,
                 zeta: list, gamma: list, hooker=None):
        super(MultiLayerTransformerPruningX, self).__init__()
        self.gamma = gamma
        self.zeta = zeta
        self.beta = beta
        self.featureDim = featureDim
        self.fieldDim = fieldDim
        self.featureNumb = featureNum
        self.headNum = headNum
        self.layerDims = layerDims
        self.transformer = nn.Sequential(OrderedDict([
            (f'transformer{i}',
             MultiHeadTransformerXLayer(featureNum, layerDims[i] * headNum[i], layerDims[i + 1], headNum[i])) for
            i in
            range(len(layerDims) - 1)])
        )
        self.field = nn.Parameter(torch.Tensor(self.featureNumb, fieldDim))
        torch.nn.init.normal(self.field.data, mean=0, std=0.001)
        self.featurePruning = FeaturePruningLayerV2(self.featureNumb, fieldDim, featureDim, beta[0], zeta[0], gamma[0])
        self.hooker = defaultHooker if hooker is None else hooker
        self.output = None
        self.featureL0 = None

        # 所有feature转换成【B，F，D】

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        featurePruning, self.featureL0 = self.featurePruning(cat, self.field)
        # print(cat.shape)
        self.output = self.transformer(cat)
        return self.output
        # return cat


class MultiLayerTransformerSTR(nn.Module):
    def __init__(self, featureNum, layerDims: List, headNum: List, featureDim, hooker=None):
        super(MultiLayerTransformerSTR, self).__init__()
        self.headNum = headNum
        self.layerDims = layerDims
        self.STR = FeatureSTRLayer(featureNum, featureDim)
        self.transformer = nn.Sequential(OrderedDict([
            (f'transformer{i}',
             MultiHeadTransformerLayer(featureNum, layerDims[i] * headNum[i], layerDims[i + 1], headNum[i])) for
            i in
            range(len(layerDims) - 1)])
        )
        self.hooker = defaultHooker if hooker is None else hooker
        self.output = None

        # 所有feature转换成【B，F，D】

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        prune = self.STR(cat)
        # print(cat.shape)
        self.output = self.transformer(prune)
        return self.output
        # return cat


class MultiLayerTransformerSTRX(nn.Module):
    def __init__(self, featureNum, layerDims: List, headNum: List, featureDim, hooker=None):
        super(MultiLayerTransformerSTRX, self).__init__()
        self.headNum = headNum
        self.layerDims = layerDims
        # self.STR = FeatureSTRLayer(featureNum, featureDim)
        self.transformer = nn.Sequential(OrderedDict([
            (f'transformer{i}',
             MultiHeadTransformerELUXLayer(featureNum, layerDims[i] * headNum[i], layerDims[i + 1], headNum[i])) for
            i in
            range(len(layerDims) - 1)])
        )
        self.hooker = defaultHooker if hooker is None else hooker
        self.output = None

        # 所有feature转换成【B，F，D】

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        # prune = self.STR(cat)
        # print(cat.shape)
        self.output = self.transformer(cat)
        return self.output
        # return cat

    def setMaskTraining(self, isTrain: bool):
        self.STR.field.requires_grad = isTrain
        self.transformer[0].interactionPrune.requires_grad = isTrain


class MultiLayerTransformerSTRELUX(nn.Module):
    def __init__(self, featureNum, layerDims: List, headNum: List, featureDim, trainMask=False, hooker=None):
        super(MultiLayerTransformerSTRELUX, self).__init__()
        self.headNum = headNum
        self.trainMask = trainMask
        self.layerDims = layerDims
        self.STR = FeatureSTRLayer(featureNum, featureDim)
        self.transformer: nn.Sequential = nn.Sequential(OrderedDict([
            (f'transformer{i}',
             MultiHeadTransformerELUXLayer(featureNum, layerDims[i] * headNum[i], layerDims[i + 1], headNum[i])) for
            i in
            range(len(layerDims) - 1)])
        )
        self.hooker = defaultHooker if hooker is None else hooker
        self.output = None

        # 所有feature转换成【B，F，D】

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        prune = self.STR(cat)
        # print(cat.shape)
        self.output = self.transformer(prune)
        return self.output
        # return cat

    def setMaskTraining(self, isTrain: bool):
        self.STR.field.requires_grad = isTrain
        self.transformer[0].interactionPrune.requires_grad = isTrain


class MultiLayerTransformerBlank(nn.Module):
    def __init__(self, featureNum, layerDims: List, headNum: List, hooker=None):
        super(MultiLayerTransformerBlank, self).__init__()
        self.headNum = headNum
        self.layerDims = layerDims
        self.transformer = nn.Sequential(OrderedDict([
            (f'transformer{i}',
             MultiHeadTransformerLayer(featureNum, layerDims[i] * headNum[i], layerDims[i + 1], headNum[i])) for
            i in
            range(len(layerDims) - 1)])
        )
        self.hooker = defaultHooker if hooker is None else hooker
        self.output = None

        # 所有feature转换成【B，F，D】

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        # cat = torch.mean(cat, dim=-1)
        return cat
        # print(cat.shape)
        # self.output = self.transformer(cat)
        # return self.output
        # return cat


class MultiTransformerLTELeakyReluDot(nn.Module):
    def __init__(self, featureNum, layerDims: List, headNum: List, featureDim, trainMask=False, hooker=None):
        super(MultiTransformerLTELeakyReluDot, self).__init__()
        self.headNum = headNum
        self.layerDims = layerDims
        self.STR = FeatureSTRLayer(featureNum, featureDim)
        self.transformer: nn.Sequential = nn.Sequential(OrderedDict([
            (f'transformer{i}',
             MultiHeadTransformerELUXLayer(featureNum, layerDims[i] * headNum[i], layerDims[i + 1], headNum[i])) for
            i in
            range(len(layerDims) - 1)])
        )
        self.hooker = defaultHooker if hooker is None else hooker
        self.output = None

        # 所有feature转换成【B，F，D】

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        prune = self.STR(cat)
        # print(cat.shape)
        self.output = self.transformer(prune)
        return self.output
        # return cat

    def setMaskTraining(self, isTrain: bool):
        self.STR.field.requires_grad = isTrain
        self.transformer[0].interactionPrune.requires_grad = isTrain


class TransformerLTE(nn.Module):
    def __init__(self, featureNum, layerDims: List, headNum: List, featureDim, trainMask=False, hooker=None):
        super(TransformerLTE, self).__init__()
        self.headNum = headNum
        self.field = nn.Parameter(torch.ones(featureNum, featureDim))
        nn.init.normal(self.field, mean=0, std=0.001)
        self.layerDims = layerDims
        self.LTE = FeatureLTELayer(featureNum, featureDim)
        self.transformer = InteractionPruningLayer(featureNum, layerDims[0] * headNum[0])
        self.hooker = defaultHooker if hooker is None else hooker
        self.output = None

        # 所有feature转换成【B，F，D】

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        prune = self.LTE(cat)
        # print(cat.shape)
        self.output = self.transformer(prune, self.field)
        return self.output
        # return cat


class TransformerLTEElu(nn.Module):
    def __init__(self, featureNum, layerDims: List, headNum: List, featureDim, init=3, trainMask=False, hooker=None):
        super(TransformerLTEElu, self).__init__()
        self.headNum = headNum
        self.field = nn.Parameter(torch.ones(featureNum, featureDim))
        nn.init.normal(self.field, mean=0, std=0.001)
        self.layerDims = layerDims
        self.LTE = FeatureLTELayer(featureNum, featureDim, init)
        self.transformer = InteractionPruningEluLayer(featureNum, layerDims[0] * headNum[0])
        self.hooker = defaultHooker if hooker is None else hooker
        self.output = None

        # 所有feature转换成【B，F，D】

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        prune = self.LTE(cat)
        # print(cat.shape)
        self.output = self.transformer(prune, self.field)
        return self.output

    # def setMaskTraining(self, isTrain: bool):
    #     self.STR.field.requires_grad = isTrain
    #     self.transformer[0].interactionPrune.requires_grad = isTrain


class TransformerLTEEluV2(nn.Module):
    def __init__(self, featureNum, layerDims: List, headNum: List, featureDim, init=3, trainMask=False, hooker=None):
        super(TransformerLTEEluV2, self).__init__()
        self.headNum = headNum
        self.field = nn.Parameter(torch.ones(featureNum, featureDim))
        nn.init.normal(self.field, mean=0, std=0.001)
        self.layerDims = layerDims
        self.LTE = FeatureLTELayer(featureNum, featureDim, init)
        self.transformer = InteractionPruningEluLayerV2(featureNum, layerDims[0] * headNum[0])
        self.hooker = defaultHooker if hooker is None else hooker
        self.output = None

        # 所有feature转换成【B，F，D】

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        prune = self.LTE(cat)
        # print(cat.shape)
        self.output = self.transformer(prune, self.field)
        return self.output


class TransformerLTEEluV3(nn.Module):
    def __init__(self, featureNum, layerDims: List, headNum: List, featureDim, init=3, trainMask=False, hooker=None):
        super(TransformerLTEEluV3, self).__init__()
        self.headNum = headNum
        self.field = nn.Parameter(torch.ones(featureNum, featureDim))
        nn.init.normal(self.field, mean=0, std=0.001)
        self.layerDims = layerDims
        # self.LTE = FeatureLTELayer(featureNum, featureDim, init)
        self.transformer = InteractionPruningEluLayer(featureNum, layerDims[0] * headNum[0])
        self.hooker = defaultHooker if hooker is None else hooker
        self.output = None

        # 所有feature转换成【B，F，D】

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        # prune = self.LTE(cat)
        # print(cat.shape)
        self.output = self.transformer(cat, self.field)
        return self.output


class TransformerLTEEluV4(nn.Module):
    def __init__(self, featureNum, layerDims: List, headNum: List, featureDim, init=3, trainMask=False, hooker=None):
        super(TransformerLTEEluV4, self).__init__()
        self.headNum = headNum
        self.field = nn.Parameter(torch.ones(3, featureNum, featureDim))
        nn.init.normal(self.field, mean=0, std=0.01)
        self.layerDims = layerDims
        self.LTE = FeatureLTELayer(featureNum, featureDim, init)
        self.transformer = InteractionPruningEluLayerV3(featureNum, layerDims[0] * headNum[0])
        self.hooker = defaultHooker if hooker is None else hooker
        self.output = None

        # 所有feature转换成【B，F，D】

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        prune = self.LTE(cat)
        # print(cat.shape)
        self.output = self.transformer(prune, self.field)
        return self.output


class SeNetBiLinearDotFusion(nn.Module):
    def __init__(self, featureNumb, featureDim, hooker=None):
        super(SeNetBiLinearDotFusion, self).__init__()
        self.featureNumb = featureNumb
        self.featureDim = featureDim
        self.hooker = defaultHooker if hooker is None else hooker
        self.SeNet = SeNetLayer(featureNumb)
        self.BiLinear = BiLinearDotLayer(featureNumb, featureDim)
        self.output = None

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        SeNet = self.SeNet(cat)
        # SeNet = cat
        # print(cat.shape)
        self.output = self.BiLinear(SeNet)
        return self.output


class SeNetBiLinearLNFusion(nn.Module):
    def __init__(self, featureNumb, featureDim, hooker=None):
        super(SeNetBiLinearLNFusion, self).__init__()
        self.featureNumb = featureNumb
        self.featureDim = featureDim
        self.hooker = defaultHooker if hooker is None else hooker
        self.LN = torch.nn.LayerNorm([featureNumb, featureDim])
        self.BiLinear = BiLinearLayer(featureNumb, featureDim)
        self.output = None

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        SeNet = self.LN(cat)
        # SeNet = cat
        # print(cat.shape)
        self.output = self.BiLinear(SeNet)
        return self.output


class SeNetBiLinearBNFusion(nn.Module):
    def __init__(self, featureNumb, featureDim, hooker=None):
        super(SeNetBiLinearBNFusion, self).__init__()
        self.featureNumb = featureNumb
        self.featureDim = featureDim
        self.hooker = defaultHooker if hooker is None else hooker
        self.BN = torch.nn.BatchNorm1d(featureNumb)
        self.BiLinear = BiLinearLayer(featureNumb, featureDim)
        self.output = None

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        SeNet = self.BN(cat)
        # SeNet = cat
        # print(cat.shape)
        self.output = self.BiLinear(SeNet)
        return self.output


class SeNetBiLinearNoneReluFusion(nn.Module):
    def __init__(self, featureNumb, featureDim, hooker=None):
        super(SeNetBiLinearNoneReluFusion, self).__init__()
        self.featureNumb = featureNumb
        self.featureDim = featureDim
        self.hooker = defaultHooker if hooker is None else hooker
        self.SeNet = SeNetLayerNoRelu(featureNumb)
        self.BiLinear = BiLinearLayer(featureNumb, featureDim)
        self.output = None

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        SeNet = self.SeNet(cat)
        # SeNet = cat
        # print(cat.shape)
        self.output = self.BiLinear(SeNet)
        return self.output


class SeNetBiLinearLTEFusion(nn.Module):
    def __init__(self, featureNumb, featureDim, hooker=None):
        super(SeNetBiLinearLTEFusion, self).__init__()
        self.featureNumb = featureNumb
        self.featureDim = featureDim
        self.hooker = defaultHooker if hooker is None else hooker
        self.SeNet = SeNetLayerLTE(featureNumb)
        self.BiLinear = BiLinearLayer(featureNumb, featureDim)
        self.output = None

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        SeNet = self.SeNet(cat)
        # SeNet = cat
        # print(cat.shape)
        self.output = self.BiLinear(SeNet)
        return self.output


class SeNetBiLinearFusion(nn.Module):
    def __init__(self, featureNumb, featureDim, hooker=None):
        super(SeNetBiLinearFusion, self).__init__()
        self.featureNumb = featureNumb
        self.featureDim = featureDim
        self.hooker = defaultHooker if hooker is None else hooker
        self.SeNet = SeNetLayer(featureNumb)
        self.BiLinear = BiLinearLayer(featureNumb, featureDim)
        self.output = None

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        SeNet = self.SeNet(cat)
        # SeNet = cat
        # print(cat.shape)
        self.output = self.BiLinear(SeNet)
        return self.output


class SeNetBiLinearNormFusion(nn.Module):
    def __init__(self, featureNumb, featureDim, hooker=None):
        super(SeNetBiLinearNormFusion, self).__init__()
        self.featureNumb = featureNumb
        self.featureDim = featureDim
        self.hooker = defaultHooker if hooker is None else hooker
        self.SeNet = SeNetNormLayer(featureNumb)
        self.BiLinear = BiLinearLayer(featureNumb, featureDim)
        self.output = None

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        SeNet = self.SeNet(cat)
        # SeNet = cat
        # print(cat.shape)
        self.output = self.BiLinear(SeNet)
        return self.output


class SeNetBiLinearFusionV2(nn.Module):
    def __init__(self, featureNumb, featureDim, hooker=None):
        super(SeNetBiLinearFusionV2, self).__init__()
        self.featureNumb = featureNumb
        self.featureDim = featureDim
        self.hooker = defaultHooker if hooker is None else hooker
        # self.SeNet = SeNetLayer(featureNumb)
        self.BiLinear = BiLinearLayer(featureNumb, featureDim)
        self.output = None

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        # SeNet = self.SeNet(cat)
        SeNet = cat
        # print(cat.shape)
        self.output = self.BiLinear(SeNet)
        return self.output


class AttentionLinearFusion(nn.Module):
    def __init__(self, featureNumb, featureDim, hooker=None):
        super(AttentionLinearFusion, self).__init__()
        self.featureNumb = featureNumb
        self.featureDim = featureDim
        self.hooker = defaultHooker if hooker is None else hooker
        self.transformer = MultiHeadTransformerLayer(self.featureNumb, featureDim, int(featureDim / 2), 2)
        self.BiLinear = BiLinearLayer(featureNumb, featureDim)
        self.output = None

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        # SeNet = self.SeNet(cat)
        SeNet = self.transformer(cat)
        # print(cat.shape)
        self.output = self.BiLinear(SeNet)
        return self.output


class XDeepFMFusion(nn.Module):
    def __init__(self, featureNumb, featureDim, layersDim, hooker=None):
        super(XDeepFMFusion, self).__init__()
        self.layersDim = layersDim
        self.featureNumb = featureNumb
        self.featureDim = featureDim
        self.hooker = defaultHooker if hooker is None else hooker
        self.CIN = CINLayer(featureNumb, featureDim)
        self.DNN = ConcatMlpLayerV2(layersDim)
        self.output = None

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        cin = self.CIN(cat)
        dnn = self.DNN(cat)
        # print(cat.shape)
        self.output = torch.sigmoid(dnn + cin)
        return self.output


class XDeepFMFusionV2(nn.Module):
    def __init__(self, featureNumb, featureDim, layersDim, hooker=None):
        super(XDeepFMFusionV2, self).__init__()
        self.layersDim = layersDim
        self.featureNumb = featureNumb
        self.featureDim = featureDim
        self.hooker = defaultHooker if hooker is None else hooker
        self.CIN = CINLayer(featureNumb, featureDim)
        self.SeNet = SeNetLayer(featureNumb)
        self.BiLinear = BiLinearLayer(featureNumb, featureDim)
        self.DNN = ConcatMlpLayerV2(layersDim)
        self.output = None
        self.buildMask(featureNumb)

    def buildMask(self, featureNum):
        xList = []
        yList = []
        for i in range(featureNum):
            for j in range(i + 1, featureNum):
                xList.append(i)
                yList.append(j)
        self.x = torch.Tensor(xList).type(torch.long)
        self.y = torch.Tensor(yList).type(torch.long)

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        SeNet = self.SeNet(cat)
        # print(cat.shape)
        Bilinear = self.BiLinear(SeNet)
        cin = self.CIN(SeNet)
        dnn = self.DNN(Bilinear[:, self.x, self.y, :])
        # print(cat.shape)
        self.output = torch.sigmoid(dnn + cin)
        return self.output


class LinearBiLinearResFusion(nn.Module):
    def __init__(self, featureNumb, featureDim, BiLayersDim, layerDim, hooker=None):
        super(LinearBiLinearResFusion, self).__init__()
        self.BiLayersDim = BiLayersDim
        self.featureNumb = featureNumb
        self.featureDim = featureDim
        self.SeNet = SeNetLayer(featureNumb)
        self.hooker = defaultHooker if hooker is None else hooker
        self.BiLinear = BiLinearLayer(featureNumb, featureDim)
        BiLayersDim[0] = BiLayersDim[0] + layerDim[0]
        self.BiDNN = ConcatMlpLayerV2(BiLayersDim)
        # self.DNN = ConcatMlpLayerV2(layerDim)
        # self.weight = LinearTransform([2, 1], True)
        self.output = None
        self.buildMask(featureNumb)

    def buildMask(self, featureNum):
        xList = []
        yList = []
        for i in range(featureNum):
            for j in range(i + 1, featureNum):
                xList.append(i)
                yList.append(j)
        self.x = torch.Tensor(xList).type(torch.long)
        self.y = torch.Tensor(yList).type(torch.long)

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        SeNet = self.SeNet(cat)
        # print(cat.shape)
        Bilinear = self.BiLinear(SeNet)
        BiDnn = self.BiDNN(torch.cat([Bilinear[:, self.x, self.y, :], cat], dim=1))

        self.output = torch.sigmoid(BiDnn)
        return self.output


class LinearBiLinearFusionV3(nn.Module):
    def __init__(self, featureNumb, featureDim, BiLayersDim, layerDim, hooker=None):
        super(LinearBiLinearFusionV3, self).__init__()
        self.BiLayersDim = BiLayersDim
        self.featureNumb = featureNumb
        self.featureDim = featureDim
        self.SeNet = SeNetLayer(featureNumb)
        self.hooker = defaultHooker if hooker is None else hooker
        self.BiLinear = BiLinearLayer(featureNumb, featureDim)
        BiLayersDim[0] = BiLayersDim[0] + layerDim[0]
        self.BiDNN = ConcatMlpLayerV2(BiLayersDim)
        # self.DNN = ConcatMlpLayerV2(layerDim)
        # self.weight = LinearTransform([2, 1], True)
        self.output = None
        self.buildMask(featureNumb)

    def buildMask(self, featureNum):
        xList = []
        yList = []
        for i in range(featureNum):
            for j in range(i + 1, featureNum):
                xList.append(i)
                yList.append(j)
        self.x = torch.Tensor(xList).type(torch.long)
        self.y = torch.Tensor(yList).type(torch.long)

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        SeNet = self.SeNet(cat)
        # print(cat.shape)
        Bilinear = self.BiLinear(SeNet)
        BiDnn = self.BiDNN(torch.cat([Bilinear[:, self.x, self.y, :], SeNet], dim=1))

        self.output = torch.sigmoid(BiDnn)
        return self.output


class LinearBiLinearFusionV2(nn.Module):
    def __init__(self, featureNumb, featureDim, BiLayersDim, layerDim, hooker=None):
        super(LinearBiLinearFusionV2, self).__init__()
        self.BiLayersDim = BiLayersDim
        self.featureNumb = featureNumb
        self.featureDim = featureDim
        self.hooker = defaultHooker if hooker is None else hooker
        self.BiLinear = BiLinearLayer(featureNumb, featureDim)
        BiLayersDim[0] = BiLayersDim[0] + layerDim[0]
        self.BiDNN = ConcatMlpLayerV2(BiLayersDim)
        # self.DNN = ConcatMlpLayerV2(layerDim)
        # self.weight = LinearTransform([2, 1], True)
        self.output = None
        self.buildMask(featureNumb)

    def buildMask(self, featureNum):
        xList = []
        yList = []
        for i in range(featureNum):
            for j in range(i + 1, featureNum):
                xList.append(i)
                yList.append(j)
        self.x = torch.Tensor(xList).type(torch.long)
        self.y = torch.Tensor(yList).type(torch.long)

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        # print(cat.shape)
        Bilinear = self.BiLinear(cat)
        BiDnn = self.BiDNN(torch.cat([Bilinear[:, self.x, self.y, :], cat], dim=1))

        self.output = torch.sigmoid(BiDnn)
        return self.output


class LinearBiLinearFusion(nn.Module):
    def __init__(self, featureNumb, featureDim, BiLayersDim: list, layerDim: list, hooker=None):
        super(LinearBiLinearFusion, self).__init__()
        self.BiLayersDim = BiLayersDim
        self.featureNumb = featureNumb
        self.featureDim = featureDim
        self.hooker = defaultHooker if hooker is None else hooker
        self.BiLinear = BiLinearLayer(featureNumb, featureDim)
        # print(BiLayersDim,layerDim)
        self.BiDNN = ConcatMlpLayerV2(BiLayersDim)
        self.DNN = ConcatMlpLayerV2(layerDim)
        self.weight = LinearTransform([BiLayersDim[-1] + layerDim[-1], 16, 1], True)
        self.output = None
        self.buildMask(featureNumb)

    def buildMask(self, featureNum):
        xList = []
        yList = []
        for i in range(featureNum):
            for j in range(i + 1, featureNum):
                xList.append(i)
                yList.append(j)
        self.x = torch.Tensor(xList).type(torch.long)
        self.y = torch.Tensor(yList).type(torch.long)

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        # print(cat.shape)
        Bilinear = self.BiLinear(cat)
        BiDnn = self.BiDNN(Bilinear[:, self.x, self.y, :])
        dnn = self.DNN(cat)
        # print(cat.shape)
        self.output = torch.sigmoid(self.weight(torch.cat([BiDnn, dnn], dim=1)))
        return self.output


class LTEBilinearFusion(nn.Module):
    def __init__(self, featureNum, layerDims: List, headNum: List, featureDim, init=3, trainMask=False, hooker=None):
        super(LTEBilinearFusion, self).__init__()
        self.headNum = headNum
        self.field = nn.Parameter(torch.ones(featureNum, featureDim))
        nn.init.normal(self.field, mean=0, std=0.001)
        self.layerDims = layerDims
        self.LTE = FeatureLTELayer(featureNum, featureDim, init)
        self.biLinear = BiLinearFieldLayer(featureNum)
        self.gate = InteractionXPruneLayer(featureNum, featureDim)
        self.hooker = defaultHooker if hooker is None else hooker
        self.output = None

        # 所有feature转换成【B，F，D】

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        prune = self.LTE(cat)
        # print(cat.shape)
        biLinear = self.biLinear(prune, self.field)
        self.output = self.gate(biLinear, self.field)
        return self.output


class LambdaNetFusion(nn.Module):
    def __init__(self, featureNum, layerDims: List, headNum: List, hooker=None):
        super(LambdaNetFusion, self).__init__()
        layerDims = [16, 16, 16]
        self.transformer = nn.Sequential(OrderedDict([
            (f'transformer{i}',
             LambdaLayer(layerDims[i], layerDims[i + 1])) for
            i in
            range(len(layerDims) - 1)])
        )
        self.hooker = defaultHooker if hooker is None else hooker
        self.output = None

        # 所有feature转换成【B，F，D】

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        # print(cat.shape)
        self.output = self.transformer(cat)
        return self.output
        # return cat


class LTEMixFusion(nn.Module):
    def __init__(self, featureNum, layerDims: List, headNum: List, featureDim, init=3, trainMask=False, hooker=None):
        super(LTEMixFusion, self).__init__()
        self.headNum = headNum
        self.layerDims = layerDims
        self.LTE = FeatureLTELayer(featureNum, featureDim, init)
        self.seNet = SeNetGateLayer(featureNum)
        self.biLinear = BiLinearLayerV2(featureNum, featureDim)
        self.aggregation = AggregationLayer(featureNum, featureDim)
        self.hooker = defaultHooker if hooker is None else hooker
        self.output = None

        # 所有feature转换成【B，F，D】

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        prune = self.LTE(cat)
        # print(cat.shape)
        SeNet = self.seNet(prune)
        biLinear = self.biLinear(SeNet)
        self.output = self.aggregation(biLinear, SeNet)
        return self.output


class LambdaGateFusion(nn.Module):
    def __init__(self, featureNum, layerDims: List, headNum: List, hooker=None):
        super(LambdaGateFusion, self).__init__()
        layerDims = [16, 16, 16]
        self.transformer = nn.Sequential(OrderedDict([
            (f'transformer{i}',
             LambdaLTELayers(layerDims[i], layerDims[i + 1])) for i in range(len(layerDims) - 1)])
        )
        self.hooker = defaultHooker if hooker is None else hooker
        self.output = None

        # 所有feature转换成【B，F，D】

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        # print(cat.shape)
        self.output = self.transformer(cat)
        return self.output
        # return cat


class LambdaGateFusionV2(nn.Module):
    def __init__(self, featureNum, layerDims: List, headNum: List, hooker=None):
        super(LambdaGateFusionV2, self).__init__()
        layerDims = [16, 8, 1]
        self.transformer = nn.Sequential(OrderedDict([
            (f'transformer{i}',
             LambdaGateLayers(layerDims[i], layerDims[i + 1])) for i in range(len(layerDims) - 1)])
        )
        self.hooker = defaultHooker if hooker is None else hooker
        self.output = None

        # 所有feature转换成【B，F，D】

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        # print(cat.shape)
        self.output = self.transformer(cat)
        return self.output


class LambdaGateFusionV3(nn.Module):
    def __init__(self, featureNum, featureDim, headNum, fieldWeight, hooker=None):
        super(LambdaGateFusionV3, self).__init__()
        self.Lambda = LambdaInteractionLayer(featureNum, featureDim, headNum)
        self.fieldWeight = fieldWeight
        self.hooker = defaultHooker if hooker is None else hooker
        self.output = None

        # 所有feature转换成【B，F，D】

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        # print(cat.shape)
        self.output = self.Lambda((cat, self.fieldWeight[None, :, :]))
        return self.output


class LambdaGateFusionV4(nn.Module):
    def __init__(self, featureNum, featureDim, headNum, fieldWeight, depth=3, hooker=None):
        super(LambdaGateFusionV4, self).__init__()
        self.Lambda = LambdaInteractionLayerV2(featureNum, featureDim, headNum, depth)
        self.fieldWeight = fieldWeight
        self.hooker = defaultHooker if hooker is None else hooker
        self.output = None

        # 所有feature转换成【B，F，D】

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        # print(cat.shape)
        self.output = self.Lambda((cat, self.fieldWeight[None, :, :]))
        return self.output


class LambdaGateFusionV5(nn.Module):
    def __init__(self, featureNum, featureDim, headNum, fieldWeight, hooker=None):
        super(LambdaGateFusionV5, self).__init__()
        self.Lambda = LambdaInteractionLayerV3(featureNum, featureDim, headNum)
        self.fieldWeight = fieldWeight
        self.hooker = defaultHooker if hooker is None else hooker
        self.output = None

        # 所有feature转换成【B，F，D】

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        # print(cat.shape)
        self.output = self.Lambda(cat, self.fieldWeight[None, :, :])
        return self.output


class LambdaGateFusionV6(nn.Module):
    def __init__(self, featureNum, featureDim, headNum, fieldWeight, fieldGate, hooker=None):
        super(LambdaGateFusionV6, self).__init__()
        self.Lambda = LambdaInteractionLayerV4(featureNum, featureDim, headNum)
        self.fieldWeight = fieldWeight
        self.fieldGate = fieldGate
        self.hooker = defaultHooker if hooker is None else hooker
        self.output = None

        # 所有feature转换成【B，F，D】

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        # print(cat.shape)
        self.output = self.Lambda(cat, self.fieldWeight[None, :, :], self.fieldGate[None, :, :])
        return self.output


class LambdaGateFusionV7(nn.Module):
    def __init__(self, featureNum, featureDim, headNum, fieldWeight, DNNLayerDims, hooker=None):
        super(LambdaGateFusionV7, self).__init__()
        self.Lambda = LambdaInteractionLayerV3(featureNum, featureDim, headNum)
        self.fieldWeight = fieldWeight
        self.DNN = ConcatMlpLayerV2(DNNLayerDims)
        self.AutoDNN = ConcatMlpLayerV2(DNNLayerDims)
        self.hooker = defaultHooker if hooker is None else hooker
        self.output = None

        # 所有feature转换成【B，F，D】

    def forward(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        feature = self.hooker(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        featureVec = [i.unsqueeze(1) for i in feature]
        cat = torch.cat(featureVec, dim=1)
        # print(cat.shape)
        dnn = self.DNN(cat)
        output = self.AutoDNN(self.Lambda(cat, self.fieldWeight[None, :, :]))
        self.output = torch.sigmoid(dnn + output)
        return self.output


if __name__ == '__main__':
    # pass
    model = FmFMFusionModule(150, 3)
    input = torch.arange(900).reshape((2, 150, 3)).type(dtype=torch.float) / ((1 + 900) * 900 / 2)
    loss = torch.nn.BCELoss()
    act = torch.nn.Sigmoid()
    label = torch.ones((2), requires_grad=False)
    # linear=LinearTransform(3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    for i in range(10000):
        model.train()
        optimizer.zero_grad()
        out = model(input)
        # print(out)
        a = torch.sum(out, dim=1).squeeze()
        # print(a)
        _out = act(a)
        # print(_out)
        ls = loss(_out, label)
        print(ls.detach().numpy(), _out.detach().numpy())
        ls.backward()
        optimizer.step()
