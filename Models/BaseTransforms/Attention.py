from Models.BaseTransforms.L0Regularization import L0Regularization
from Models.BaseTransforms.GaussianTransform import GaussianTransformV2
from Models.BaseTransforms.STRTransform import STRTransform
from torch import nn, Tensor
import torch
import math


class DotAttention(nn.Module):
    def __init__(self, ):
        super(DotAttention, self).__init__()
        self.output = None
        self.softmax = None

    ##[B*F*1*1*D] * 【1*F*F*H*3*D】->[B*F*F*H*3*D]
    # score: [B,H,F,D] *[B,H,F,D] -> [B,H,F,F]
    # sum: 【B,H,F,F】 * [B,H,F,D] -> [B,H,F,D]
    def forward(self, query: Tensor, key: Tensor, value: Tensor):
        transb = key.transpose(-2, -1)
        shape = math.sqrt(query.shape[-1])
        shape = query.shape[-1]
        # print(transb.shape,query.shape)
        score = torch.matmul(query, transb) / shape
        # print(score.shape)

        self.softmax = torch.softmax(score, dim=-1)
        self.output = torch.matmul(self.softmax, value)
        # print(self.output.shape)
        return self.output


class DotReluAttention(nn.Module):
    def __init__(self, featureNumb, featureDim):
        super(DotReluAttention, self).__init__()
        self.output = None
        self.act = nn.ReLU()
        self.relu = None
        self.layerNorm = nn.LayerNorm([featureNumb, featureDim])

    ##[B*F*1*1*D] * 【1*F*F*H*3*D】->[B*F*F*H*3*D]
    # score: [B,H,F,D] *[B,H,F,D] -> [B,H,F,F]
    # sum: 【B,H,F,F】 * [B,H,F,D] -> [B,H,F,D]
    def forward(self, query: Tensor, key: Tensor, value: Tensor):
        transb = key.transpose(-2, -1)
        shape = math.sqrt(query.shape[-1])
        shape = query.shape[-1]
        # print(transb.shape,query.shape)
        score = torch.matmul(query, transb) / shape
        # print(score.shape)
        self.relu = self.act(score)
        self.output = torch.matmul(self.relu, value)
        # print(self.output.shape)
        # return self.layerNorm(self.output)
        return self.output


# 叉乘
class DotAttentionX(nn.Module):
    def __init__(self):
        super(DotAttentionX, self).__init__()
        self.output = None
        self.softmax = None

    ##[B*F*1*1*D] * 【1*F*F*H*3*D】->[B*F*F*H*3*D]
    # score: [B,H,F,D] *[B,H,F,D] -> [B,H,F,F]
    # sum: 【B,H,F,F】 * [B,H,F,D] -> [B,H,F,D]
    def forward(self, query: Tensor, key: Tensor, value: Tensor):
        transKey = key[:, :, None, :, None, :]
        # print('transb:shape:', transb.shape)
        query = query[:, :, :, None, :, None]
        shape = math.sqrt(query.shape[-1])
        score = (torch.matmul(query, transKey) / shape)
        score = score.sum(dim=(4, 5))
        self.softmax = torch.softmax(score, dim=-1)
        self.output = torch.matmul(self.softmax, value)
        # print(self.output.shape)
        return self.output


class DotAttentionV2(nn.Module):
    def __init__(self):
        super(DotAttentionV2, self).__init__()
        self.output = None
        self.softmax = None

    ##[B*1*F*1*1*D] * 【1*H*F*F*3*D】->[B*H*F*F*3*D]
    # score: [B,H,F,F,D]*[B,H,F,F,D]->[B,H,F,F]->[B,F,F]
    # sum: [B,F,F]*[B,F,F,D]->[B,H,F,D]
    def forward(self, query: Tensor, key: Tensor, value: Tensor):
        transb = key.permute(0, 1, 3, 2, 4)[:, :, :, :, :, None]
        # print('transb:shape:', transb.shape)
        query = query[:, :, :, :, None, :]
        shape = math.sqrt(query.shape[-1])
        score = (torch.matmul(query, transb) / shape).squeeze()
        if len(score.shape) == 2:
            score = score[None, :, :, ]
        # print("score Shape", score.shape)
        self.softmax = torch.softmax(score, dim=-1)[:, :, None, :]
        value = value.squeeze(1).permute(0, 2, 1, 3)
        self.output = torch.matmul(self.softmax, value).squeeze(dim=-2)
        # print(self.output.shape)
        return self.output


class DotAttentionV3(nn.Module):
    def __init__(self):
        super(DotAttentionV3, self).__init__()
        self.output = None
        self.softmax = None

    ##[B*1*F*1*1*D] * 【1*H*F*F*3*D】->[B*H*F*F*3*D]
    # score: [B,H,F,F,D]*[B,H,F,F,D]->[B,H,F,F]->[B,F,F]
    # sum: [B,F,F]*[B,F,F,D]->[B,H,F,D]
    def forward(self, query: Tensor, key: Tensor, value: Tensor):
        transb = key.permute(0, 1, 3, 2, 4)[:, :, :, :, None, :]
        # print('transb:shape:', transb.shape)
        query = query[:, :, :, :, :, None]
        shape = math.sqrt(query.shape[-1])
        score = (torch.matmul(query, transb) / shape).sum(dim=(4, 5)).squeeze()
        if len(score.shape) == 2:
            score = score[None, :, :, ]
        # print("score Shape", score.shape)
        self.softmax = torch.softmax(score, dim=-1)[:, :, None, :]
        value = value.squeeze(1).permute(0, 2, 1, 3)
        self.output = torch.matmul(self.softmax, value).squeeze(dim=-2)
        # print(self.output.shape)
        return self.output


# 叉乘的时候带pruning
class DotAttentionV4(nn.Module):
    def __init__(self, featureNumb, featureDim, beta, zeta, gamma):
        super(DotAttentionV4, self).__init__()
        self.featureDim = featureDim
        self.output = None
        self.softmax = None
        self.featureNumb = featureNumb
        self.L0Reg = L0Regularization(beta, zeta, gamma)
        self.matrix = nn.Parameter(torch.zeros(size=(featureNumb, featureNumb, featureDim, featureDim)))
        torch.nn.init.normal_(self.matrix.data, mean=0, std=0.001)
        self.L0Out = None
        self.L0 = None
        self.output = None

    ##[B*1*F*1*1*D] * 【1*H*F*F*3*D】->[B*H*F*F*3*D]
    # score: [B,H,F,F,D]*[B,H,F,F,D]->[B,H,F,F]->[B,F,F]
    # sum: [B,F,F]*[B,F,F,D]->[B,H,F,D]
    def forward(self, query: Tensor, key: Tensor, value: Tensor):
        transb = key.permute(0, 1, 3, 2, 4)[:, :, :, :, None, :]
        # print('transb:shape:', transb.shape)
        query = query[:, :, :, :, :, None]
        shape = math.sqrt(query.shape[-1])
        scoreHat = (torch.matmul(query, transb) / shape)
        self.L0Out, L0 = self.L0Reg(self.matrix)
        score = (scoreHat * self.L0Out[None, None, :, :, :, :]).sum(dim=(4, 5)).squeeze()
        if len(score.shape) == 2:
            score = score[None, :, :, ]
        # print("score Shape", score.shape)
        self.softmax = torch.softmax(score, dim=-1)[:, :, None, :]
        value = value.squeeze(1).permute(0, 2, 1, 3)
        self.output = torch.matmul(self.softmax, value).squeeze(dim=-2)
        # print(self.output.shape)
        return self.output, L0


class DotWeight(nn.Module):
    def __init__(self):
        super(DotWeight, self).__init__()
        self.output = None

    def forward(self, query: Tensor, key: Tensor):
        transb = key.transpose(-2, -1)
        # print(transb.shape,query.shape)
        dim = transb.shape[-1]
        # print(dim)
        score = torch.matmul(query, transb) / math.sqrt(dim)
        return score


class GaussianAttention(nn.Module):

    def __init__(self, featureDim, headNum=2):
        super(GaussianAttention, self).__init__()
        self.featureDim = featureDim
        self.gaussian = GaussianTransformV2(featureDim, featureDim, headNum)
        self.out = None
        self.softmax = None

    # score: [B,H,F,D] *[B,H,F,D] -> [B,H,F,F]
    # sum: 【B,H,F,F】 * [B,H,F,D] -> [B,H,F,D]
    def forward(self, query: Tensor, key: Tensor, value: Tensor):
        # print(transb.shape,query.shape)
        score = self.gaussian(query, key)
        # print(score.shape)
        self.softmax = torch.softmax(score, dim=-1)
        self.output = torch.matmul(self.softmax, value)
        # print(self.output.shape)
        return self.output


class DotAttentionEluX(nn.Module):
    def __init__(self):
        super(DotAttentionEluX, self).__init__()
        self.output = None
        self.STR = STRTransform()
        self.attention = None
        self.act = nn.ELU()

    ##[B*F*1*1*D] * 【1*F*F*H*3*D】->[B*F*F*H*3*D]
    # score: [B,H,F,D] *[B,H,F,D] -> [B,H,F,F]
    # sum: 【B,H,F,F】 * [B,H,F,D] -> [B,H,F,D]
    def forward(self, query: Tensor, key: Tensor, value: Tensor, indicator=None):
        key = self.act(key) + 1
        query = self.act(query) + 1
        transKey = key[:, :, None, :, None, :]
        # print('transb:shape:', transb.shape)
        query = query[:, :, :, None, :, None]
        shape = math.sqrt(query.shape[-1])
        score = (torch.matmul(query, transKey) / shape)
        score = score.sum(dim=(4, 5))
        if indicator is not None:
            score = self.STR(score, indicator)
        sum = torch.sum(score, dim=-1, keepdim=True)
        self.attention = score / sum
        self.output = torch.matmul(self.attention, value)
        # print(self.output.shape)
        return self.output
