from typing import List
import torch
from torch import nn
import torch.nn.functional as F
from Models.BaseModelV2 import BaseModelV2, BaseEmbedFormat
from Utils.HyperParamLoadModule import FeatureInfo, HyperParam, FEATURETYPE

class FM(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.w0 = nn.Parameter(torch.zeros([1, ]))
        self.w1 = nn.Parameter(torch.rand([num_embeddings, 1]))
        self.w2 = nn.Parameter(torch.rand([num_embeddings, embedding_dim]))

        # nn.init.xavier_normal_(self.w0)
        nn.init.xavier_normal_(self.w1)
        nn.init.xavier_normal_(self.w2)

    def forward(self, x):
        first_order = torch.mm(x, self.w1)
        second_order = 0.5 * torch.sum(
            torch.pow(torch.mm(x, self.w2), 2) - torch.mm(torch.pow(x, 2), torch.pow(self.w2, 2)),
            dim=1,
            keepdim=True
        )

        return self.w0 + first_order + second_order


class DNN(nn.Module):
    def __init__(self, hidden_units):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(in_features, out_features, bias=True) for in_features, out_features in
            zip(hidden_units[:-1], hidden_units[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x


class DeepFM(BaseModelV2):
    def __init__(self, featureInfo: List[FeatureInfo], embedFormat: BaseEmbedFormat):
        self.nameList = [i.featureName for i in featureInfo if (
                i.enable == True and i.featureType == FEATURETYPE.USER)]
        super().__init__(embedFormat)

        self.featureNum = HyperParam.AutoIntFeatureNum
        self.featureDim = HyperParam.AutoIntFeatureDim
        # self.layersDim = HyperParam.AutoIntMatchMlpDims
        self.layersDim = [self.featureNum * self.featureDim, 512, 128, 64]

        # # 解析特征信息
        # self.dense_features, self.sparse_features, self.sparse_features_nunique = features_info
        #
        # # 解析拿到所有 数值型 和 稀疏型特征信息
        # self.__dense_features_nums = len(self.dense_features)
        # self.__sparse_features_nums = len(self.sparse_features)
        #
        # # embedding
        # self.embeddings = nn.ModuleDict({
        #     "embed_" + key: nn.Embedding(num_embeds, embedding_dim)
        #     for key, num_embeds in self.sparse_features_nunique.items()
        # })

        # stack_dim = self.__dense_features_nums + self.__sparse_features_nums * embedding_dim
        # hidden_units.insert(0, stack_dim)

        self.fm = FM(self.featureNum * self.featureDim, self.featureDim)

        self.dnn = DNN(self.layersDim)

        self.dnn_last_linear = nn.Linear(self.layersDim[-1], 1, bias=False)

    def mainForward(self, feature: torch.Tensor):
        # # 从输入x中单独拿出 sparse_input 和 dense_input
        # dense_inputs, sparse_inputs = x[:, :self.__dense_features_nums], x[:, self.__dense_features_nums:]
        # sparse_inputs = sparse_inputs.long()
        #
        # embedding_feas = [self.embeddings["embed_" + key](sparse_inputs[:, idx]) for idx, key in
        #                   enumerate(self.sparse_features)]
        # embedding_feas = torch.cat(embedding_feas, dim=-1)
        #
        # input_feas = torch.cat([embedding_feas, dense_inputs], dim=-1)

        feature = torch.reshape(feature, [feature.shape[0], -1])
        fm = self.fm(feature)
        dnn = self.dnn_last_linear(self.dnn(feature))

        return torch.sigmoid(fm + dnn)
