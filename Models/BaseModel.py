import torch
import torch.nn as  nn
import json

from Data.WxBiz import WxBiz
from Data.WxBizV2 import WxBizV2
from Utils.HyperParamLoadModule import *
from abc import ABCMeta, abstractmethod
from typing import List
from enum import Enum
from datetime import datetime
from Utils.HyperParamLoadModule import HyperParam, FeatureInfo
from Data.DataFormatBase import *


class BaseModel(nn.Module):
    def __init__(self, featureInfo, device, ):
        super().__init__()
        self.device = device
        self.featureInfo = featureInfo
        self.dataFormat: DataFormatBase = self.getDataFormat(Config.datasetType)
        self.L2 = HyperParam.L2
        self.tempEmbed = featureInfo[0].embeddingSize
        # self.embedding: nn.ModuleDict[str:nn.Embedding] = {}
        self.embedding: nn.ModuleDict[str:nn.Embedding] = nn.ModuleDict()
        # self.featureInfo: List[FeatureInfo] = featureInfo
        # self.hyperParams: HyperParam = HyperParam
        # self.userSequenceFeatureDim = [tempEmbed for i in range(self.userFeatureNumb)]
        self.dataFormat.buildEmbedding(self.embedding, featureInfo, device)
        self.contextTransModule = self.loadContextTransModule()
        self.userTransModule = self.loadUserTransModule()
        self.itemTransModule = self.loadItemTransModule()
        self.fusionModule = self.loadFusionModule()
        self.matchingModule = self.loadMatchingModule()
        self.matchingResult = None
        self.auxiliaryLoss = 0

    def getDataFormat(self, datasetType: DATASETTYPE, ):
        if datasetType == DATASETTYPE.WXBIZ:
            return WxBiz(HyperParam.popCatFeature, HyperParam.popDims2DimFeature, HyperParam.popSeqFeature)
        elif datasetType == DATASETTYPE.WXBIZV2:
            return WxBizV2(HyperParam.popCatFeature, HyperParam.popDims2DimFeature, HyperParam.popSeqFeature)

    def forwardForward(self, userFeature, itemFeature, contextFeature):
        time = []
        time.append(datetime.now())
        contextTrans = self.contextTransModule(contextFeature)
        time.append(datetime.now())
        userTrans = self.userTransModule(userFeature)
        time.append(datetime.now())
        itemTrans = self.itemTransModule(itemFeature)
        time.append(datetime.now())
        fusionFeature = self.fusionModule(userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans)
        time.append(datetime.now())
        self.matchingResult = self.matchingModule(userFeature, userTrans, itemFeature, itemTrans, contextFeature,
                                                  contextTrans, fusionFeature)
        time.append(datetime.now())
        self.endPoint()
        # print(
        #     f"forward Cost: context:{(time[1] - time[0]).total_seconds()} "
        #     f",userTrans:{(time[2] - time[1]).total_seconds()},"
        #     f"itemTrans:{(time[3] - time[2]).total_seconds()},"
        #     f"fusion:{(time[4] - time[3]).total_seconds()},"
        #     f"match:{(time[5] - time[4]).total_seconds()},"
        #     f"total:{(time[5] - time[0]).total_seconds()}")
        return self.matchingResult

    def forward(self, feed_dict: dict):
        SeqFeature, Dims2DimFeature, CatFeature = self.dataFormat.preProcess(feed_dict, self.embedding,
                                                                             self.featureInfo,
                                                                             self.device)

        # Seq = 'uin_vid_hour_id:2048,15,16|uin_vid_week_id:2048,15,16|uin_vid_cat1_id:2048,15,16|uin_vid_pos_id:2048,15,16|uin_vid_day_id:2048,15,16|uin_vid_id:2048,15,16|uin_vid_cat2_id:2048,15,16|uin_vid_len:2048'
        # Dims = 'dense:2048,286|kyk_vec:2048,128|uinvec:2048,64|oruinvec:2048,64|readuinvec:2048,64|uinvidvec:2048,64|uin_changdu_vec:2048,128|uin_kyk_vec:2048,128|uin_gnn_vec:2048,32|vvbizvec:2048,64|orbizvec:2048,64|readbizvec:2048,64|gnn_vec:2048,32|changdu_vec:2048,128'
        # Cat = 'vid_cat2_id:2048,16|vid_day_id:2048,16|vid_id:2048,16|vid_cat1_id:2048,16|vid_hour_id:2048,16|vid_week_id:2048,16|age_0:2048,16|gender_1:2048,16|language_2:2048,16|platform_3:2048,16|grade_4:2048,16|uin_prov_38:2048,16|uin_prov_39:2048,16|biz_halfyear_prov_81:2048,16|biz_halfyear_prov_82:2048,16|biz_quaterly_prov_89:2048,16|biz_quaterly_prov_90:2048,16|biz_monthly_prov_97:2048,16|biz_monthly_prov_98:2048,16|biz_cat1_str_70:2048,16|biz_cat1_str_71:2048,16|biz_cat1_str_72:2048,16|uin_short_vcat1_str_46:2048,16|uin_short_vcat1_str_47:2048,16|uin_short_vcat1_str_48:2048,16|uin_mid_vcat1_str_49:2048,16|uin_mid_vcat1_str_50:2048,16|uin_mid_vcat1_str_51:2048,16|uin_long_vcat1_str_52:2048,16|uin_long_vcat1_str_53:2048,16|uin_long_vcat1_str_54:2048,16|biz_vcat1_str_105:2048,16|biz_vcat1_str_106:2048,16|biz_vcat1_str_107:2048,16|short_cat1_v2_str_5:2048,16|short_cat1_v2_str_6:2048,16|short_cat1_v2_str_7:2048,16|mid_cat1_v2_str_8:2048,16|mid_cat1_v2_str_9:2048,16|mid_cat1_v2_str_10:2048,16|long_cat1_v2_str_11:2048,16|long_cat1_v2_str_12:2048,16|long_cat1_v2_str_13:2048,16|biz_cat1_v2_str_64:2048,16|biz_cat1_v2_str_65:2048,16|biz_cat1_v2_str_66:2048,16|biz_cat2_str_73:2048,16|biz_cat2_str_74:2048,16|biz_cat2_str_75:2048,16|uin_short_vcat2_str_55:2048,16|uin_short_vcat2_str_56:2048,16|uin_short_vcat2_str_57:2048,16|uin_mid_vcat2_str_58:2048,16|uin_mid_vcat2_str_59:2048,16|uin_mid_vcat2_str_60:2048,16|uin_long_vcat2_str_61:2048,16|uin_long_vcat2_str_62:2048,16|uin_long_vcat2_str_63:2048,16|biz_vcat2_str_108:2048,16|biz_vcat2_str_109:2048,16|biz_vcat2_str_110:2048,16|uin_city_40:2048,16|uin_city_41:2048,16|uin_city_42:2048,16|biz_halfyear_city_83:2048,16|biz_halfyear_city_84:2048,16|biz_halfyear_city_85:2048,16|biz_quaterly_city_91:2048,16|biz_quaterly_city_92:2048,16|biz_quaterly_city_93:2048,16|biz_monthly_city_99:2048,16|biz_monthly_city_100:2048,16|biz_monthly_city_101:2048,16|short_cat2_v2_str_14:2048,16|short_cat2_v2_str_15:2048,16|short_cat2_v2_str_16:2048,16|mid_cat2_v2_str_17:2048,16|mid_cat2_v2_str_18:2048,16|mid_cat2_v2_str_19:2048,16|long_cat2_v2_str_20:2048,16|long_cat2_v2_str_21:2048,16|long_cat2_v2_str_22:2048,16|biz_cat2_v2_str_67:2048,16|biz_cat2_v2_str_68:2048,16|biz_cat2_v2_str_69:2048,16|short_tag_str_23:2048,16|short_tag_str_24:2048,16|short_tag_str_25:2048,16|short_tag_str_26:2048,16|short_tag_str_27:2048,16|mid_tag_str_28:2048,16|mid_tag_str_29:2048,16|mid_tag_str_30:2048,16|mid_tag_str_31:2048,16|mid_tag_str_32:2048,16|long_tag_str_33:2048,16|long_tag_str_34:2048,16|long_tag_str_35:2048,16|long_tag_str_36:2048,16|long_tag_str_37:2048,16|biz_tag_str_76:2048,16|biz_tag_str_77:2048,16|biz_tag_str_78:2048,16|biz_tag_str_79:2048,16|biz_tag_str_80:2048,16|biz_vtag_str_111:2048,16|biz_vtag_str_112:2048,16|biz_vtag_str_113:2048,16|biz_vtag_str_114:2048,16|biz_vtag_str_115:2048,16|uin_short_vtag_str_116:2048,16|uin_short_vtag_str_117:2048,16|uin_short_vtag_str_118:2048,16|uin_short_vtag_str_119:2048,16|uin_short_vtag_str_120:2048,16|uin_long_vtag_str_121:2048,16|uin_long_vtag_str_122:2048,16|uin_long_vtag_str_123:2048,16|uin_long_vtag_str_124:2048,16|uin_long_vtag_str_125:2048,16|uin_county_43:2048,16|uin_county_44:2048,16|uin_county_45:2048,16|biz_halfyear_county_86:2048,16|biz_halfyear_county_87:2048,16|biz_halfyear_county_88:2048,16|biz_quaterly_county_94:2048,16|biz_quaterly_county_95:2048,16|biz_quaterly_county_96:2048,16|biz_monthly_county_102:2048,16|biz_monthly_county_103:2048,16|biz_monthly_county_104:2048,16'
        #
        #
        # SeqFeature = self.buildPack(Seq)
        # Dims2DimFeature = self.buildPack(Dims)
        # CatFeature = self.buildPack(Cat)

        return self.forwardForward(SeqFeature, Dims2DimFeature, CatFeature)

    def buildPack(self, str):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        shape = {val.split(':')[0]: [int(v) for v in val.split(":")[1].split(',')] for val in
                 str.split('|')}
        feature = {}
        for k, v in shape.items():
            feature[k] = DataPack(data=torch.zeros(v,device=device), fieldType=FIELDTYPE.SEQUENCE, name=k)
        return feature

    @abstractmethod
    def loadContextTransModule(self):
        raise NotImplementedError


    @abstractmethod
    def loadFusionModule(self):
        raise NotImplementedError


    @abstractmethod
    def loadItemTransModule(self):
        raise NotImplementedError


    @abstractmethod
    def loadUserTransModule(self):
        raise NotImplementedError


    @abstractmethod
    def loadMatchingModule(self):
        raise NotImplementedError


    def addAuxiliaryLoss(self, loss):
        self.auxiliaryLoss = self.auxiliaryLoss + loss


    def calRegLoss(self, p=2, alpha=0.0001, excludeList=[]):
        L = 0
        for name, param in self.named_parameters():
            if 'weight' in name and name not in excludeList:
                L = L + torch.norm(param, p=p)
        L = alpha * L
        return L


    def resetAuxLoss(self):
        self.auxiliaryLoss = 0


    def endPoint(self):
        return


    def calBeforeEachEpoch(self, epochNumb):
        return


    def buildParameterGroup(self):
        return self.parameters()
        # @abstractmethod
    # def loadFromJson(self,hyperJsonFilePath):
    #     raise NotImplementedError
