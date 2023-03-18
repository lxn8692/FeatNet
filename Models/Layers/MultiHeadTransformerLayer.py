from typing import List
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from Models.BaseTransforms.ConcatResnetTransform import ConcatResnetTransform, ConcatResnetTransformV3
from Models.BaseTransforms.QKVTransform import QKVTransform, QKVFieldTransform
from Models.BaseTransforms.Attention import DotAttention, GaussianAttention, DotWeight, DotAttentionV3, DotAttentionV2, \
    DotAttentionV4, DotAttentionX, DotAttentionEluX, DotReluAttention
from Models.BaseTransforms.LinearTransform import LinearTransform
from enum import Enum


class MultiHeadTransformerSparseLayer(nn.Module):
    def __init__(self, featureNum, inDim, outDim, headNumb=2):
        super(MultiHeadTransformerSparseLayer, self).__init__()
        self.featureNum = featureNum
        self.inputDim = inDim
        self.outputDim = outDim
        self.headNumb = headNumb
        self.QKVKernel: QKVTransform = QKVTransform(self.headNumb, self.inputDim, self.outputDim)
        self.output = None
        self.QKV = None
        self.headOut = None
        self.dotAttention = DotReluAttention(featureNum, outDim)
        self.concatRes = ConcatResnetTransform(concatDim=[outDim for i in range(headNumb)], originDim=inDim)

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


class MultiHeadTransformerLayer(nn.Module):
    def __init__(self, featureNum, inDim, outDim, headNumb=2):
        super(MultiHeadTransformerLayer, self).__init__()
        self.featureNum = featureNum
        self.inputDim = inDim
        self.outputDim = outDim
        self.headNumb = headNumb
        self.QKVKernel: QKVTransform = QKVTransform(self.headNumb, self.inputDim, self.outputDim)
        self.output = None
        self.QKV = None
        self.headOut = None
        self.dotAttention = DotAttention()
        self.concatRes = ConcatResnetTransform(concatDim=[outDim for i in range(headNumb)], originDim=inDim)

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


class MultiHeadTransformerXLayer(nn.Module):
    def __init__(self, featureNum, inDim, outDim, headNumb=2):
        super(MultiHeadTransformerXLayer, self).__init__()
        self.featureNum = featureNum
        self.inputDim = inDim
        self.outputDim = outDim
        self.headNumb = headNumb
        self.QKVKernel: QKVTransform = QKVTransform(self.headNumb, self.inputDim, self.outputDim)
        self.output = None
        self.QKV = None
        self.headOut = None
        self.dotAttention = DotAttentionX()
        self.concatRes = ConcatResnetTransform(concatDim=[outDim for i in range(headNumb)], originDim=inDim)

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


class MultiHeadTransformerGauLayer(nn.Module):
    def __init__(self, featureNum, inDim, outDim, headNumb=2):
        super(MultiHeadTransformerGauLayer, self).__init__()
        self.featureNum = featureNum
        self.inputDim = inDim
        self.outputDim = outDim
        self.headNumb = headNumb
        self.QKVKernel: QKVTransform = QKVTransform(self.headNumb, self.inputDim, self.outputDim)
        self.output = None
        self.QKV = None
        self.headOut = None
        self.gaussianAttention = GaussianAttention(featureDim=outDim, headNum=headNumb)
        self.concatRes = ConcatResnetTransform(concatDim=[outDim for i in range(headNumb)], originDim=inDim)

    # QKV:[B,H,3,F,D]
    def forward(self, featureVec):
        # QKV :
        self.QKV = self.QKVKernel(featureVec)
        # attention:
        # score: [B,H,F,D] *[B,H,F,D] -> [B,H,F,F]
        # sum: 【B,H,F,F】 * [B,H,F,D] -> [B,H,F,D]
        self.headOut = self.gaussianAttention(self.QKV[:, :, 0, :, :], self.QKV[:, :, 1, :, :], self.QKV[:, :, 2, :, :])
        # concatHead:[B,H,F,D], [B,F,D] ->[B,F,D*N]
        self.output = self.concatRes(self.headOut, featureVec)
        # print(f'out:{self.output.shape}')
        return self.output


class SelfAttentionLayer(nn.Module):
    def __init__(self, featureNum, inDim, outDim, headNumb=2):
        super(SelfAttentionLayer, self).__init__()
        self.featureNum = featureNum
        self.inputDim = inDim
        self.outputDim = outDim
        self.headNumb = headNumb
        self.QKVKernel: QKVTransform = QKVTransform(self.headNumb, self.inputDim, self.outputDim)
        self.output = None
        self.QKV = None
        self.headOut = None
        self.dotAttention = DotWeight()
        self.act = nn.Tanh()

    # QKV:[B,H,3,F,D]
    def forward(self, featureVec):
        # QKV :
        self.QKV = self.QKVKernel(featureVec)
        # attention:
        # headout: [B,H,F,D] *[B,H,F,D] -> [B,H,F,F]
        self.headOut = self.dotAttention(self.QKV[:, :, 0, :, :], self.QKV[:, :, 1, :, :])
        temp = (self.headOut + torch.transpose(self.headOut, -1, -2)) / 2
        self.output = self.act(temp)
        return self.output


class MultiHeadTransformerWithFieldLayer(nn.Module):
    def __init__(self, featureNum, inDim, headNumb=2):
        super(MultiHeadTransformerWithFieldLayer, self).__init__()
        self.featureNum = featureNum
        self.inputDim = inDim
        self.headNumb = headNumb
        self.QKVKernel: QKVFieldTransform = QKVFieldTransform(headNumb, inDim, featureNum)
        self.output = None
        self.QKV = None
        self.headOut = None
        self.dotAttention = DotAttentionV2()
        self.concatRes = ConcatResnetTransform(concatDim=[inDim for i in range(headNumb)], originDim=inDim)

    # QKV:[B,H,3,F,D]
    def forward(self, featureVec):
        # QKV :
        ##[B*1*F*1*1*D] * 【1*H*F*F*3*D】->[B*H*F*F*3*D]
        self.QKV = self.QKVKernel(featureVec)
        # attention:
        # score: [B,H,F,D] *[B,H,F,D] -> [B,H,F,F]
        # sum: 【B,H,F,F】 * [B,H,F,D] -> [B,H,F,D]
        self.headOut = self.dotAttention(self.QKV[:, :, :, :, 0, :], self.QKV[:, :, :, :, 1, :],
                                         self.QKV[:, :, :, :, 2, :])[:, None, :, :]
        # concatHead:[B,H,F,D], [B,F,D] ->[B,F,D*N]
        self.output = self.concatRes(self.headOut, featureVec)
        print(f'out:{self.output.shape}')
        return self.output


# 叉乘求和
class MultiHeadTransformerWithFieldLayerV3(nn.Module):
    def __init__(self, featureNum, inDim, headNumb=2):
        super(MultiHeadTransformerWithFieldLayerV3, self).__init__()
        self.featureNum = featureNum
        self.inputDim = inDim
        self.headNumb = headNumb
        self.QKVKernel: QKVFieldTransform = QKVFieldTransform(headNumb, inDim, featureNum)
        self.output = None
        self.QKV = None
        self.headOut = None
        self.dotAttention = DotAttentionV3()
        self.concatRes = ConcatResnetTransform(concatDim=[inDim for i in range(headNumb)], originDim=inDim)

    # QKV:[B,H,3,F,D]
    def forward(self, featureVec):
        # QKV :
        ##[B*1*F*1*1*D] * 【1*H*F*F*3*D】->[B*H*F*F*3*D]
        self.QKV = self.QKVKernel(featureVec)
        # attention:
        # score: [B,H,F,D] *[B,H,F,D] -> [B,H,F,F]
        # sum: 【B,H,F,F】 * [B,H,F,D] -> [B,H,F,D]
        self.headOut = self.dotAttention(self.QKV[:, :, :, :, 0, :], self.QKV[:, :, :, :, 1, :],
                                         self.QKV[:, :, :, :, 2, :])[:, None, :, :]
        # concatHead:[B,H,F,D], [B,F,D] ->[B,F,D*N]
        self.output = self.concatRes(self.headOut, featureVec)
        print(f'out:{self.output.shape}')
        return self.output


# 叉乘求和+pruning
class MultiHeadTransformerWithFieldLayerV4(nn.Module):
    def __init__(self, featureNum, inDim, headNumb=2, featureDim=0, beta=0.66, zeta=1.1, gamma=-0.1):
        super(MultiHeadTransformerWithFieldLayerV4, self).__init__()
        self.gamma = gamma
        self.zeta = zeta
        self.beta = beta
        self.featureDim = featureDim
        self.featureNum = featureNum
        self.inputDim = inDim
        self.headNumb = headNumb
        self.QKVKernel: QKVFieldTransform = QKVFieldTransform(headNumb, inDim, featureNum)
        self.output = None
        self.QKV = None
        self.headOut = None
        self.dotAttention = DotAttentionV4(featureNum, featureDim, beta, zeta, gamma)
        self.L0 = None
        self.concatRes = ConcatResnetTransform(concatDim=[inDim for i in range(headNumb)], originDim=inDim)

    # QKV:[B,H,3,F,D]
    def forward(self, featureVec):
        # QKV :
        ##[B*1*F*1*1*D] * 【1*H*F*F*3*D】->[B*H*F*F*3*D]
        self.QKV = self.QKVKernel(featureVec)
        # attention:
        # score: [B,H,F,D] *[B,H,F,D] -> [B,H,F,F]
        # sum: 【B,H,F,F】 * [B,H,F,D] -> [B,H,F,D]
        self.headOut, self.L0 = self.dotAttention(self.QKV[:, :, :, :, 0, :], self.QKV[:, :, :, :, 1, :],
                                                  self.QKV[:, :, :, :, 2, :])
        self.headOut = self.headOut[:, None, :, :]
        # concatHead:[B,H,F,D], [B,F,D] ->[B,F,D*N]
        self.output = self.concatRes(self.headOut, featureVec)
        print(f'out:{self.output.shape}')
        return self.output, self.L0


class MultiHeadTransformerELUXLayer(nn.Module):
    def __init__(self, featureNum, inDim, outDim, headNumb=2, init=-15):
        super(MultiHeadTransformerELUXLayer, self).__init__()
        self.featureNum = featureNum
        self.inputDim = inDim
        self.outputDim = outDim
        self.interactionPrune = nn.Parameter(init * torch.ones(headNumb, featureNum, featureNum))
        self.headNumb = headNumb
        self.QKVKernel: QKVTransform = QKVTransform(self.headNumb, self.inputDim, self.outputDim)
        self.output = None
        self.QKV = None
        self.headOut = None
        self.dotAttention = DotAttentionEluX()
        self.concatRes = ConcatResnetTransform(concatDim=[outDim for i in range(headNumb)], originDim=inDim)

    # QKV:[B,H,3,F,D]
    def forward(self, featureVec):
        # QKV :
        self.QKV = self.QKVKernel(featureVec)
        # attention:
        # score: [B,H,F,D] *[B,H,F,D] -> [B,H,F,F]
        # sum: 【B,H,F,F】 * [B,H,F,D] -> [B,H,F,D]
        self.headOut = self.dotAttention(self.QKV[:, :, 0, :, :], self.QKV[:, :, 1, :, :], self.QKV[:, :, 2, :, :],
                                         self.interactionPrune[None, :, :, :].to(self.QKV.device))
        # concatHead:[B,H,F,D], [B,F,D] ->[B,F,D*N]
        self.output = self.concatRes(self.headOut, featureVec)
        # print(f'out:{self.output.shape}')
        return self.output


class MHTestTransformerLayer(nn.Module):
    def __init__(self, featureNum, featureDim, headNumb=3):
        super(MHTestTransformerLayer, self).__init__()
        self.featureNum = featureNum
        self.headNumb = headNumb
        self.QKVKernel: QKVTransform = QKVTransform(self.headNumb, featureDim, featureDim)
        self.output = None
        self.QKV = None
        self.headOut = None
        self.layerNorm1 = LayerNorm([featureNum, featureDim])
        self.layerNorm2 = LayerNorm([featureNum, featureDim])
        self.dotAttention = DotAttention()
        self.concatRes = ConcatResnetTransformV3(headNumb, originDim=featureDim)
        self.DNN = LinearTransform([featureDim, featureDim * 4, featureDim], True)

    # QKV:[B,H,3,F,D]
    def forward(self, featureVec):
        # QKV :
        self.QKV = self.QKVKernel(featureVec)
        # attention:
        # score: [B,H,F,D] *[B,H,F,D] -> [B,H,F,F]
        # sum: 【B,H,F,F】 * [B,H,F,D] -> [B,H,F,D]
        self.headOut = self.dotAttention(self.QKV[:, :, 0, :, :], self.QKV[:, :, 1, :, :], self.QKV[:, :, 2, :, :])
        # concatHead:[B,H,F,D], [B,F,D] ->[B,F,D*N]
        concat = self.concatRes(self.headOut)
        res = self.layerNorm1(concat + featureVec)
        DNNOut = self.DNN(res)
        res2 = self.layerNorm2(res + DNNOut)
        self.output = res2
        # print(f'out:{self.output.shape}')
        return self.output
