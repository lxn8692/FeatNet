import math

import torch
from Models.BaseTransforms.LTE import LTETransform
from Models.BaseTransforms.QKVTransform import QKVTransformV2, QKVTransform
from torch import nn
from torch import Tensor
from Models.BaseTransforms.Attention import DotAttention, DotWeight
from torch.autograd import Variable


class InteractionPruningLayer(nn.Module):
    def __init__(self, featureNumb, featureDim):
        super(InteractionPruningLayer, self).__init__()
        self.featureDim = featureDim
        self.output = None
        self.softmax = None
        self.featureNumb = featureNumb
        self.QKTrans = QKVTransformV2(2, featureDim, featureDim)
        self.QKVTrans = QKVTransformV2(3, featureDim, featureDim)
        self.DotWeight = DotWeight()
        self.LTE = LTETransform()
        self.output = None

    ##[B*1*F*1*1*D] * 【1*H*F*F*3*D】->[B*H*F*F*3*D]
    # score: [B,H,F,F,D]*[B,H,F,F,D]->[B,H,F,F]->[B,F,F]
    # sum: [B,F,F]*[B,F,F,D]->[B,H,F,D]
    def forward(self, feature, indicator):
        # [2,F,D]
        trans = self.QKTrans(indicator)
        gate = self.DotWeight(trans[0], trans[1])
        gate = self.LTE.apply(Variable(gate))
        # 【3,F,D】->[3,F,1,D,1] [3,1,F,1,D]
        QKV = self.QKVTrans(indicator)
        # [1,3,F,F,D,D]*[B,1,F,1,1,D]->[B,3,F,F,D]
        QKV = torch.matmul(QKV[:, :, None, :, None], QKV[:, None, :, None, :])
        input = torch.matmul(feature[:, None, :, None, None, :], QKV[None, :, :, :, :, :]).squeeze(-2)
        # [B,F,F,D]*[B,F,F,D] -> [B,F,F]

        score = torch.matmul((input[:, 0].unsqueeze(-2)), (input[:, 1].transpose(1, 2).unsqueeze(-1))).squeeze()

        scoreHat = score * gate[None, :, :]
        value = input[:, 2]
        self.output = torch.matmul(scoreHat[:, :, None, :], value).squeeze(-2)

        return self.output


# 不带feature
class InteractionPruningEluLayer(nn.Module):
    def __init__(self, featureNumb, featureDim):
        super(InteractionPruningEluLayer, self).__init__()
        self.featureDim = featureDim
        self.output = None
        self.softmax = None
        self.elu = nn.ELU()
        self.featureNumb = featureNumb
        self.QKTrans = QKVTransformV2(2, featureDim, featureDim)
        self.QKVTrans = QKVTransform(1, featureDim, featureDim)
        self.DotWeight = DotWeight()
        self.LTE = LTETransform()
        self.output = None

    ##[B*1*F*1*1*D] * 【1*H*F*F*3*D】->[B*H*F*F*3*D]
    # score: [B,H,F,F,D]*[B,H,F,F,D]->[B,H,F,F]->[B,F,F]
    # sum: [B,F,F]*[B,F,F,D]->[B,H,F,D]
    def forward(self, feature, indicator):
        # [2,F,D]
        trans = self.QKTrans(indicator)
        gate = self.DotWeight(trans[0], trans[1])
        gate = self.LTE.apply(torch.tanh(gate))
        # 【3,F,D】->[3,F,1,D,1] [3,1,F,1,D]
        QKV = self.QKVTrans(feature).squeeze(1)
        # [B,F,F,D]*[B,F,F,D] -> [B,F,F]
        query = self.elu(QKV[:, 0]) + 1
        key = self.elu(QKV[:, 1]) + 1
        score = torch.matmul(query[:, :, None, None, :], key[:, None, :, :, None]).squeeze(-1).squeeze(-1)
        softmax = torch.softmax(score, dim=-1)
        scoreHat = softmax * gate
        value = QKV[:, 2]
        self.output = torch.matmul(scoreHat, value)

        return self.output


## 带feature的
class InteractionPruningEluLayerV2(nn.Module):
    def __init__(self, featureNumb, featureDim):
        super(InteractionPruningEluLayerV2, self).__init__()
        self.featureDim = featureDim
        self.output = None
        self.softmax = None
        self.elu = nn.ELU()
        self.featureNumb = featureNumb
        self.QKTrans = QKVTransformV2(2, featureDim, featureDim)
        self.QKVTrans = QKVTransform(1, featureDim, featureDim)
        self.DotWeight = DotWeight()
        self.LTE = LTETransform()
        self.output = None

    ##[B*1*F*1*1*D] * 【1*H*F*F*3*D】->[B*H*F*F*3*D]
    # score: [B,H,F,F,D]*[B,H,F,F,D]->[B,H,F,F]->[B,F,F]
    # sum: [B,F,F]*[B,F,F,D]->[B,H,F,D]
    def forward(self, feature, indicator):
        # [2,F,D]
        fuseIndicator = feature * indicator[None, :, :]
        trans = self.QKTrans(fuseIndicator[:, None, :, :])
        gate1 = self.DotWeight(trans[:, 0], trans[:, 1])

        gate = self.LTE.apply(torch.tanh(gate1))
        # [B,3,F,D]
        QKV = self.QKVTrans(feature).squeeze(1)
        # [B,F,F,D]*[B,F,F,D] -> [B,F,F]
        query = self.elu(QKV[:, 0]) + 1
        key = self.elu(QKV[:, 1]) + 1
        score = torch.matmul(query[:, :, None, None, :], key[:, None, :, :, None]).squeeze(-1).squeeze(-1)
        softmax = torch.softmax(score, dim=-1)
        scoreHat = softmax * gate
        value = QKV[:, 2]
        self.output = torch.matmul(scoreHat, value)

        return self.output


## 不带feature,不做线性变换，外部传入
class InteractionPruningEluLayerV3(nn.Module):
    def __init__(self, featureNumb, featureDim):
        super(InteractionPruningEluLayerV3, self).__init__()
        self.featureDim = featureDim
        self.output = None
        self.softmax = None
        self.elu = nn.ELU()
        self.featureNumb = featureNumb
        # self.QKTrans = QKVTransformV2(2, featureDim, featureDim)
        # self.QKVTrans = QKVTransformV2(3, featureDim, featureDim)
        self.DotWeight = DotWeight()
        self.LTE = LTETransform()
        self.output = None

    ##[B*1*F*1*1*D] * 【1*H*F*F*3*D】->[B*H*F*F*3*D]
    # score: [B,H,F,F,D]*[B,H,F,F,D]->[B,H,F,F]->[B,F,F]
    # sum: [B,F,F]*[B,F,F,D]->[B,H,F,D]
    def forward(self, feature, indicator):
        # [2,F,D]
        trans = indicator
        gate = self.DotWeight(trans[0], trans[1])
        gate = self.LTE.apply(gate)
        # 【3,F,D】->[3,F,1,D,1] [3,1,F,1,D]
        QKV = indicator
        # [1,3,F,F,D,D]*[B,1,F,1,1,D]->[B,3,F,F,D]
        QKV = torch.matmul(QKV[:, :, None, :, None], QKV[:, None, :, None, :])
        input = torch.matmul(feature[:, None, :, None, None, :], QKV[None, :, :, :, :, :]).squeeze(-2)
        # [B,F,F,D]*[B,F,F,D] -> [B,F,F]
        query = self.elu(input[:, 0]) + 1
        key = self.elu(input[:, 1]) + 1
        score = torch.matmul((query.unsqueeze(-2)), (key.transpose(1, 2).unsqueeze(-1))).squeeze()
        softmax = torch.softmax(score, dim=-1)
        scoreHat = softmax * gate[None, :, :]
        value = input[:, 2]
        self.output = torch.matmul(scoreHat[:, :, None, :], value).squeeze(-2)

        return self.output


# 外部输入特征，只计算权重然后剪枝
# [B,F,F,D] [F,D] ->[F,F]->[B,F,F,D ]
class InteractionXPruneLayer(nn.Module):
    def __init__(self, featureNumb, featureDim):
        super(InteractionXPruneLayer, self).__init__()
        self.featureDim = featureDim
        self.output = None
        self.featureNumb = featureNumb
        self.QKTrans = QKVTransformV2(2, featureDim, featureDim)
        self.DotWeight = DotWeight()
        self.LTE = LTETransform()
        self.output = None

    ##[B*1*F*1*1*D] * 【1*H*F*F*3*D】->[B*H*F*F*3*D]
    # score: [B,H,F,F,D]*[B,H,F,F,D]->[B,H,F,F]->[B,F,F]
    # sum: [B,F,F]*[B,F,F,D]->[B,H,F,D]
    def forward(self, feature, indicator):
        # [2,F,D]
        trans = self.QKTrans(indicator)
        gate = self.DotWeight(trans[0], trans[1])
        gate = self.LTE.apply(gate)
        scoreHat = feature * gate[None, :, :, None]
        self.output = scoreHat
        return self.output
