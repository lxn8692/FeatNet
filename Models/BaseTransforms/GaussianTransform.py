import torch
import torch.nn as nn
import numpy as np
from math import pi
from math import log


class GaussianTransform(nn.Module):
    def __init__(self, inputDim: int, outputDim: int):
        super(GaussianTransform, self).__init__()
        self.muTrans = nn.Linear(inputDim, outputDim, bias=False)
        self.sigmaTrans = nn.Linear(inputDim, outputDim, bias=False)
        self.mu = None,
        self.sigmaSquare = None
        self.output = None
        self.initParams()

    def initParams(self):
        nn.init.xavier_uniform(self.sigmaTrans.weight, gain=1.414)
        nn.init.xavier_uniform(self.muTrans.weight, gain=1.414)

    def forward(self, featureVec, matchVec):
        self.mu = self.muTrans(featureVec)
        logSigmaSquare = self.sigmaTrans(featureVec)
        self.sigmaSquare = torch.exp(logSigmaSquare)
        # self.output = (-0.5) * (((matchVec - self.mu) ** 2) / self.sigmaSquare + logSigmaSquare + log(2 * pi))
        self.output = (-0.5) * (
                ((matchVec - self.mu) ** 2) / (self.sigmaSquare + 1) + nn.functional.softplus(logSigmaSquare) + log(
            2 * pi))
        return self.output


class GaussianTransformV2(nn.Module):
    def __init__(self, inputDim: int, outputDim: int, headNum=2):
        super(GaussianTransformV2, self).__init__()
        self.headNum = headNum
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.trans = nn.Parameter(torch.zeros((1, headNum, inputDim, 2*outputDim)))
        self.mu = None,
        self.sigmaSquare = None
        self.output = None
        self.initParams()
        self.gaussianOut = None

    def initParams(self):
        nn.init.xavier_uniform(self.trans.data, gain=1.414)

    ##[b,h,f,1,d] [h,1,f,d] -> [B,F,F,D], Sum [B,F,F]
    def forward(self, featureVec: torch.Tensor, matchVec: torch.Tensor):
        shape = featureVec.shape
        matchVec = matchVec[:, :, None, :, :]
        trans = torch.matmul(featureVec, self.trans).reshape(shape[0], shape[1], shape[2], 2, -1)
        mu = trans[:, :, :, 0, :][:,:,:,None,:]
        logSigmaSquare = trans[:, :, :, 1, :][:,:,:,None,:]
        # [b,f,f]
        sigmaSquare = torch.exp(logSigmaSquare)
        # self.output = (-0.5) * (((matchVec - self.mu) ** 2) / self.sigmaSquare + logSigmaSquare + log(2 * pi))
        gaussianOut = (-0.5) * (
                ((matchVec - mu) ** 2) / (sigmaSquare + 1) + nn.functional.softplus(logSigmaSquare) + log(
            2 * pi))
        self.output = torch.sum(gaussianOut, dim=-1)
        return self.output
