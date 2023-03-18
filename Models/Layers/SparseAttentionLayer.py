from Models.BaseTransforms.LinearTransform import LinearTransform
from Models.Layers.InducedSetAttention import InducedSetAttention
from Models.Layers.SeNetLayer import SeNetLayer
from Models.Layers.DCNV2Layer import DCNV2Layer
from Models.BaseTransforms.Attention import DotAttention, DotWeight
from Models.Layers.LambdaLayer import LambdaLayerV2
from torch import nn
import torch


class SparseAttentionLayer(nn.Module):
    def __init__(self, featureNum, featureDim, topk=0):
        super(SparseAttentionLayer, self).__init__()
        self.featureDim = featureDim
        self.topk = topk if topk != 0 else 10
        self.featureNum = featureNum
        self.queryTrans = nn.Linear(featureDim, featureDim, bias=False)
        self.contextTrans = nn.Linear(featureDim, featureDim, bias=False)
        self.output = None
        self.QKV = None
        self.idx: torch.Tensor = torch.as_tensor([i for i in range(0, featureNum) for j in range(self.topk)],
                                                 dtype=torch.long)

        self.headOut = None
        self.dotAttention = DotWeight()

    # QKV:[B,H,3,F,D]
    def forward(self, featureVec: torch.Tensor, contextVec: torch.Tensor):
        # QKV :
        device = featureVec.device
        shape = [featureVec.shape[0], featureVec.shape[1], contextVec.shape[1]]
        query = self.queryTrans(featureVec)
        key = self.contextTrans(contextVec)
        # attention:
        # score: [B,F,F,D]
        value = featureVec.unsqueeze(2) * contextVec.unsqueeze(1)
        # [B,F,F]
        weight: torch.Tensor = self.dotAttention(query, key)
        # top-K
        topk, idx = weight.topk(self.topk, dim=-1)
        topk = torch.softmax(topk)
        idx = idx.unsqueeze(dim=3).expand(-1, -1, -1, self.featureDim)
        out = torch.gather(value, dim=2, index=idx)
        out = topk.unsqueeze(dim=3) * out
        return out.reshape(shape[0], -1, self.featureDim)


class SparseLambdaAttentionLayer(nn.Module):
    def __init__(self, featureNum, midDim, contextFeatureNum, featureDim, topk=0):
        super(SparseLambdaAttentionLayer, self).__init__()
        self.contextFeatureNum = contextFeatureNum
        self.midDim = midDim
        self.featureDim = featureDim
        self.topk = topk if topk != 0 else 10
        self.featureNum = featureNum
        self.LambdaNet = LambdaLayerV2(featureDim, featureDim, self.contextFeatureNum, self.contextFeatureNum)
        self.output = None
        self.QKV = None
        self.idx: torch.Tensor = torch.as_tensor([i for i in range(0, featureNum) for j in range(self.topk)],
                                                 dtype=torch.long)

        self.headOut = None

    # QKV:[B,H,3,F,D]
    def forward(self, featureVec: torch.Tensor, contextVec: torch.Tensor):
        # QKV :
        device = featureVec.device
        shape = [featureVec.shape[0], featureVec.shape[1], contextVec.shape[1]]
        # attention:
        # score: [B,F,F,D]
        value = featureVec.unsqueeze(2) * contextVec.unsqueeze(1)
        # [B,F,F]
        weight: torch.Tensor = self.LambdaNet(featureVec, contextVec.transpose(1, 2))
        # top-K
        topk, idx = weight.topk(self.topk, dim=-1)
        topk = torch.softmax(topk, dim=2)
        idx = idx.unsqueeze(dim=3).expand(-1, -1, -1, self.featureDim)
        out = torch.gather(value, dim=2, index=idx)
        out = topk.unsqueeze(dim=3) * out
        return out.reshape(shape[0], -1, self.featureDim)


class SparseDCNAttentionLayer(nn.Module):
    def __init__(self, featureNum, contextFeatureNum, featureDim, topk=0):
        super(SparseDCNAttentionLayer, self).__init__()
        self.contextFeatureNum = contextFeatureNum
        self.featureDim = featureDim
        self.topk = topk if topk != 0 else 10
        self.featureNum = featureNum
        self.queryTrans = nn.Linear(featureDim, featureDim, bias=False)
        self.contextTrans = nn.Linear(featureDim, featureDim, bias=False)
        self.DCN = DCNV2Layer(featureNum, self.contextFeatureNum, depth=3)
        self.output = None
        self.QKV = None
        self.idx: torch.Tensor = torch.as_tensor([i for i in range(0, featureNum) for j in range(self.topk)],
                                                 dtype=torch.long)
        self.layerNorm = nn.LayerNorm(self.featureDim)
        self.headOut = None

    # QKV:[B,H,3,F,D]
    def forward(self, featureVec: torch.Tensor, contextVec: torch.Tensor):
        # QKV :
        device = featureVec.device
        shape = [featureVec.shape[0], featureVec.shape[1], contextVec.shape[1]]
        featureVec = self.queryTrans(featureVec)
        contextVec = self.contextTrans(contextVec)
        # attention:
        # score: [B,F,F,D]
        # value = featureVec.unsqueeze(2) * contextVec.unsqueeze(1)
        weight = self.DCN(featureVec, contextVec)
        featureWeight = weight[:, :self.featureNum]
        contextWeight = weight[:, self.featureNum:]
        topKFeature = self.getTopKVec(featureVec, featureWeight, self.featureDim, self.topk)
        topKContext = self.getTopKVec(contextVec, contextWeight, self.featureDim, self.topk)
        out = topKFeature.unsqueeze(dim=2) * topKContext.unsqueeze(dim=1)
        norm = self.layerNorm(out)
        return norm.reshape(shape[0], -1, self.featureDim)

    def getTopKVec(self, feature, indicator: torch.Tensor, featureDim, topk):
        topk, idx = indicator.topk(topk, dim=1)
        idx = idx.unsqueeze(dim=2).expand(-1, -1, featureDim)
        out = torch.gather(feature, dim=1, index=idx)
        return out


class SENetMABLayer(nn.Module):
    def __init__(self, featureNum, featureDim):
        super(SENetMABLayer, self).__init__()
        self.featureDim = featureDim
        self.featureNum = featureNum
        self.keyTrans = nn.Linear(featureDim, featureDim, bias=False)
        self.valueTrans = nn.Linear(featureDim, featureDim, bias=False)
        self.seNet = SeNetLayer(featureNum)
        self.setTrans = InducedSetAttention(featureDim, featureNum)
        # self.idx: torch.Tensor = torch.as_tensor([i for i in range(0, featureNum) for j in range(self.topk)],
        #                                          dtype=torch.long)
        self.layerNorm = nn.LayerNorm(self.featureDim)
        self.FFN = LinearTransform([featureDim, featureDim * 4, featureDim])
        self.pooling = InducedSetAttention(featureDim, 1,)
        self.headOut = None
        self.poolingNorm = nn.LayerNorm(self.featureDim)

    # QKV:[B,H,3,F,D]
    def forward(self, featureVec: torch.Tensor):
        # QKV :
        shape = featureVec.shape
        device = featureVec.device
        featureVec, weight = self.seNet(featureVec)
        topkVec = self.getNoneZeroVec(featureVec, weight, self.featureDim)
        interaction = topkVec.unsqueeze(dim=2) * topkVec.unsqueeze(dim=1)
        norm = self.layerNorm(interaction)
        setVec = norm.reshape(shape[0], -1, self.featureDim)
        setTrans = self.setTrans(setVec)
        ffn = self.FFN(setTrans)
        pooling = self.pooling(ffn)
        poolingNorm = self.poolingNorm(pooling)
        return poolingNorm, setTrans

    def getNoneZeroVec(self, feature: torch.Tensor, indicator: torch.Tensor, featureDim):
        count = torch.count_nonzero(indicator.detach(), dim=1)
        maxCount = torch.max(count)
        topk, idx = indicator.topk(maxCount, dim=1)
        idx = idx.unsqueeze(dim=2).expand(-1, -1, featureDim)
        out = torch.gather(feature, dim=1, index=idx)
        return out
