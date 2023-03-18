from collections import OrderedDict

import torch
import torch.nn as  nn
import json
from itertools import chain
from Models.Modules.CatTransform import CatTrans
from Models.Modules.DIms2DimTransModule import LinearDimsTransV2Module
from Models.Modules.SeqTransform import GRULinearDimsTransModuleV2
from Data.WxBiz import WxBiz
from Data.WxBizV2 import WxBizV2
from Utils.HyperParamLoadModule import *
from abc import ABCMeta, abstractmethod
from typing import List
from enum import Enum
from datetime import datetime
from Utils.HyperParamLoadModule import HyperParam, FeatureInfo
from Data.DataFormatBase import *
from DataSource.WXBIZEmbed import *
from DataSource.BaseDataFormat import *


class BaseModelV2(nn.Module):
    def __init__(self, EmbedFormat: BaseEmbedFormat):
        super().__init__()
        self.EmbedFormat = EmbedFormat
        self.embedding: nn.ModuleDict[str:nn.Embedding] = self.EmbedFormat.buildEmbedding()
        self.buildModel()

    def buildModel(self):
        self.seq = GRULinearDimsTransModuleV2(HyperParam.UserFeatValVecFeatureDims,
                                              HyperParam.AutoIntFeatureDim,
                                              HyperParam.AutoIntFeatureDim * len(self.nameList), self.nameList,
                                              HyperParam.SequenceLenKey)
        transDim = {**HyperParam.ItemFeatValVecFeatureDims, **HyperParam.UserFeatValVecFeatureDims}
        transDim['dense'] = HyperParam.FeatValDenseFeatureDim
        self.DimsTran = LinearDimsTransV2Module(transDim,
                                                HyperParam.AutoIntFeatureDim, ) # above unused
        self.cat = CatTrans([HyperParam.FeatValDenseFeatureDim], HyperParam.AutoIntFeatureDim)
        # return model

    # 0 0 1
    def forward(self, feed_dict: dict):
        SeqFeature, Dims2DimFeature, CatFeature = self.EmbedFormat.preProcess(feed_dict, self.embedding)
        re = []
        if SeqFeature is not None:
            re.extend(self.seq(SeqFeature).data)
        if CatFeature is not None:
            re.extend(self.cat(CatFeature).data)
        if Dims2DimFeature is not None:
            re.extend(self.DimsTran(Dims2DimFeature).data)
        # [[b,d] ,[b,d] , [b,d] ..... f个]    v     v.data
        cat_array = [v[:, None, :] for v in re]
        # [b f d]
        cat = torch.cat(cat_array, dim=1)
        return self.mainForward(cat)

    @abstractmethod
    def mainForward(self, feature):
        pass

    def getAuxLoss(self):
        return 0

    def calRegLoss(self, p=2, alpha=0.0001, excludeList=[]):
        L = 0
        for name, param in self.named_parameters():
            if 'weight' in name and name not in excludeList:
                L = L + torch.norm(param, p=p)
        L = alpha * L
        return L

    def buildParameterGroup(self):
        return self.parameters()
