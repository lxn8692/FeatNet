import json
import os
from enum import Enum
from typing import Any, List
from Data.DataFormatBase import *

'''
文件目的是，对于每个模块，有属于自己的超参数
输入的方式有 从配置文件读取：
'''
from abc import abstractmethod, ABCMeta

'''
暴露唯一地址：配置文件1所在的地址；
配置文件1：解决是调用环境的问题：内容包括：模型持久化的路径/数据集的路径/模型超参数的路径/本次运行的模型的名称/数据特征的路径
配置文件*N：解决模型的问题：命名规则：持久化路径/模型模型名称/参数....：包括：模型超参数，本次选择加载的预训练模型。
持久化文件N*K：预训练模型文件：一个模型N个预训练结果。名字随意。。
'''

'''
  epoch: int = 100,
                 batch_size: int = 256, , patience: int = 15, evaluationTime=5
'''
'''
'''


class EnumEncoder(json.JSONEncoder):

    def default(self, o: Any) -> Any:
        j = type(o)
        if isinstance(o, Enum):
            return o.name
        return json.JSONEncoder.default(self, o)


class DATASETTYPE(Enum):
    WXBIZ = 1
    WXBIZV2 = 2
    AVAZU = 3
    CRITEO = 4
    MV = 5


class FEATURETYPE(Enum):
    USER = 1
    ITEM = 2
    CONTEXT = 3


class BUILDTYPE(Enum):
    RUN = 1
    TEST = 2
    TRAIN = 3
    SAVEONNX = 4


class Config(object):
    fatherPid: int = 0
    logPath: str = ''
    lightCC: bool = False
    absPath: str = ''
    epoch: int = 100
    needSave: bool = False
    earlyStop: int = 15
    validatePerTrain: int = 5
    savedModelPath: str = ''
    datasetPath: str = ''
    str2idxPath: str = ''
    lookupFeaturePath: str = ''
    shareEmbedFeatureJsonPath: str = ''
    datasetName: str = ''
    hyperParamPath: str = ''
    lossType: str = ''
    featureParamPath: str = ''
    modelName: str = ''
    loadPreTrainModel: bool = False
    preTrainModelName: str = ''
    buildState: BUILDTYPE = BUILDTYPE.TEST
    datasetType: DATASETTYPE = DATASETTYPE.WXBIZ
    embedType = "V2Embed"
    key = []

    @staticmethod
    def keys():
        dict = {i: getattr(Config, i) for i in Config.key}
        dict['buildState'] = str(dict['buildState'])
        dict['datasetType'] = str(dict['datasetType'])
        return dict

    def __getitem__(self, item):
        return getattr(self, item)

    @staticmethod
    def hooker(d):
        if "buildState" in d or "datasetType" in d:
            if "buildState" in d:
                name, member = d["buildState"].split(".")
                d['buildState'] = getattr(BUILDTYPE, member)
                # print(Config.__dict__)
            if "datasetType" in d:
                name, member = d["datasetType"].split(".")
                d['datasetType'] = getattr(DATASETTYPE, member)
                # print(Config.__dict__)
            for k in d.keys():
                setattr(Config, k, d[k])
                Config.key.append(k)
            return Config
        else:
            return d


class FeatureInfo(object):
    def __init__(self):
        self.featureName: str = ''
        self.featureType: FEATURETYPE = FEATURETYPE.USER
        self.enable: bool = True
        self.embeddingSize: int = -1
        self.inputDim: int = -1  # feature_index中的数据
        self.share: bool = True

    def keys(self):
        dict = self.__dict__
        dict['featureType'] = str(dict['featureType'])
        return dict

    def hooker(d):
        if "featureType" in d:
            name, member = d["featureType"].split(".")
            d['featureType'] = getattr(FEATURETYPE, member)
            p = FeatureInfo()
            p.__dict__.update(d)
            return p
        else:
            return d


class HyperParam(object):
    batchSize: int = 256
    LR: float = 0.1
    Dropout: float = 0.1
    L2: float = 0.1
    prefetch: int = 2000

    SequenceLenKey: str = ''
    FeatValDenseFeatureDim = 286
    FeatIdxLookupFeatureStart = 286
    FeatIdxLookupFeatureEnd = 0
    FeatValVecFeatureLen = 1024
    FeatValVecFeatureInfo: dict = {}
    FeatValVecFeatureKeys = []

    # Key类标签
    UserFeatValVecNames = []
    ItemFeatValVecNames = []

    # user 在featVal 里每一个向量特征的维度
    UserFeatValVecFeatureDims = {}

    ItemFeatValVecFeatureDims = {}

    # 长度类信息：
    # 用户序列特征的个数
    UserSeqFeatureNumb: int = 0
    # 用户lookup类的个数
    UserFeatIdxLookupFeatureNumb: int = 0
    # 用户向量类的个数
    UserFeatValVecFeatureNumb: int = 0

    ItemFeatIdxLookupFeatureNumb: int = 0
    ItemFeatValVecFeatureNumb: int = 0
    ItemSeqFeatureNumb: int = 0

    ItemTargetFeatureNumb: int = 0

    GMMBaseLinearItemTransLayerDims: list = []
    GMMV1GaussianFusionOutDims: list = []
    GMMV1ContextTransDims: list = []
    GMMV2AllMLPLayerDims: list = []
    GMMV3GaussianFusionFeatureNumb: int = 1
    GMMV3GaussianFusionLayersDims: list = []
    GMMV3GaussianFusionMlpLayerDims: list = []
    GMMV3GaussianFusionLayerNumb: int = 1

    AutoIntFeatureDim: int = 32
    AutoIntFeatureNum: int = 0
    AutoIntLayerDims: List = []
    AutoIntHeadNumList: List = []
    AutoIntMatchMlpDims: List = []

    AutoIntWithFieldLayerDims: List = []
    AutoIntWithFieldHeadNumList: List = []
    AutoIntWithFieldMatchMlpDims: List = []

    AutoPruningFeatureDim: int = 0
    AutoPruningFieldDim: int = 0
    AutoPruningBeta: list = [0.66, 0.66, 0.66]
    AutoPruningZeta: list = [1.1, 1.1, 1.1]
    AutoPruningGamma: list = [-0.1, -0.1, -0.1]
    AutoPruningFeatureL0 = 0
    AutoPruningInteractionL0 = 0
    AutoPruningStructureL0 = 0

    # STRV3
    STRStartEpoch = 0
    STREndEpoch = 3

    # popFeature
    popCatFeature = []
    popDims2DimFeature = []
    popSeqFeature = []

    @staticmethod
    def hooker(d):
        if d['flag'] == True:
            d.pop('flag')
            for k in d.keys():
                setattr(HyperParam, k, d[k])
            # print(HyperParam.__dict__)
            return HyperParam
        else:
            d.pop('flag')
            return d


def calConcatDim(featureInfo: List[FeatureInfo]):
    vecSum = HyperParam.FeatValVecFeatureLen
    userItemFeatureSum = (
                                 HyperParam.ItemFeatIdxLookupFeatureNumb + HyperParam.UserFeatIdxLookupFeatureNumb + 1 + HyperParam.ItemSeqFeatureNumb) * \
                         featureInfo[0].embeddingSize
    contextSum = HyperParam.FeatValDenseFeatureDim
    transSum = sum([i for i in HyperParam.GMMV1GaussianFusionOutDims]) + HyperParam.GMMBaseLinearItemTransLayerDims[-1]
    # transSum = HyperParam.GMMBaseLinearItemTransLayerDims[-1]
    return vecSum + contextSum + userItemFeatureSum + transSum


def calFeatVecName():
    with open(os.path.join(Config.absPath, Config.lookupFeaturePath), 'r', encoding='utf-8') as feature:
        info = json.load(feature)
        if Config.embedType == "Embed":
            for i in info.keys():
                i = info[i]
                if i['type'] == 'user':
                    HyperParam.UserFeatValVecNames.append(i['name'])
                else:
                    HyperParam.ItemFeatValVecNames.append(i['name'])
            HyperParam.ItemFeatIdxLookupFeatureNumb = len(HyperParam.ItemFeatValVecNames)
            HyperParam.UserFeatIdxLookupFeatureNumb = len(HyperParam.UserFeatValVecNames)

        elif Config.embedType == "V2Embed":
            for i in info.keys():
                i = info[i]
                base = int(i['fieldB']) - HyperParam.FeatValDenseFeatureDim
                for j in range(base, base + int(i['fieldLen'])):
                    if i['type'] == 'user':
                        HyperParam.UserFeatValVecNames.append(i['name'] + f"{j}")
                    else:
                        HyperParam.ItemFeatValVecNames.append(i['name'] + f"{j}")
            for i in info.keys():
                i = info[i]
                if i['type'] == 'user':
                    HyperParam.UserFeatIdxLookupFeatureNumb += int(i['fieldLen'])
                elif i['type'] == 'item':
                    HyperParam.ItemFeatIdxLookupFeatureNumb += int(i['fieldLen'])


def loadArgs(configPath):
    print(configPath)
    with open(configPath, 'r', encoding='utf-8') as config:
        json.load(config, object_hook=Config.hooker)
    with open(os.path.join(Config.absPath, Config.hyperParamPath), 'r', encoding='utf-8') as model:
        json.load(model, object_hook=HyperParam.hooker)
    with open(os.path.join(Config.absPath, Config.featureParamPath), 'r', encoding='utf-8') as feature:
        featureInfo: List[FeatureInfo] = json.load(feature, object_hook=FeatureInfo.hooker)
    with open(os.path.join(Config.absPath, Config.lookupFeaturePath), 'r', encoding='utf-8') as lookup:
        featLookup = json.load(lookup)

    calFeatVecName()
    # 统计featureNumb;
    itemFeatValVecDimSum = 0
    itemFeatValVecNumb = 0

    for i in featureInfo:
        if i.enable == False:
            continue
        if i.featureType == FEATURETYPE.USER:
            HyperParam.UserSeqFeatureNumb += 1
        if i.featureType == FEATURETYPE.ITEM:
            HyperParam.ItemSeqFeatureNumb += 1
    for i in HyperParam.FeatValVecFeatureInfo.keys():
        if 'uin' in i:
            HyperParam.UserFeatValVecFeatureDims[i] = HyperParam.FeatValVecFeatureInfo[i]
            HyperParam.UserFeatValVecFeatureNumb += 1
        else:
            HyperParam.ItemFeatValVecFeatureDims[i] = HyperParam.FeatValVecFeatureInfo[i]
            itemFeatValVecDimSum += HyperParam.FeatValVecFeatureInfo[i]
            HyperParam.ItemFeatValVecFeatureNumb += 1
    HyperParam.GMMBaseLinearItemTransLayerDims[0] = HyperParam.GMMBaseLinearItemTransLayerDims[
                                                        0] * (
                                                            HyperParam.ItemFeatIdxLookupFeatureNumb + HyperParam.ItemSeqFeatureNumb) + itemFeatValVecDimSum
    HyperParam.GMMV1GaussianFusionOutDims = [HyperParam.GMMV1GaussianFusionOutDims[0] for i in
                                             range(
                                                 1 + HyperParam.UserFeatValVecFeatureNumb + HyperParam.UserFeatIdxLookupFeatureNumb)]
    HyperParam.GMMV2AllMLPLayerDims[0] = calConcatDim(featureInfo)

    # seq feature+ Dense feature
    HyperParam.AutoIntFeatureNum = 2 + HyperParam.UserFeatIdxLookupFeatureNumb + HyperParam.UserFeatValVecFeatureNumb + \
                                   HyperParam.ItemSeqFeatureNumb + HyperParam.ItemFeatValVecFeatureNumb + \
                                   HyperParam.ItemFeatIdxLookupFeatureNumb - len(HyperParam.popSeqFeature) - len(
        HyperParam.popDims2DimFeature) - len(HyperParam.popCatFeature)
    HyperParam.AutoIntMatchMlpDims[0] = HyperParam.AutoIntFeatureNum * HyperParam.AutoIntLayerDims[-1] * \
                                        HyperParam.AutoIntHeadNumList[-1]
    HyperParam.AutoIntWithFieldMatchMlpDims[0] = HyperParam.AutoIntFeatureNum * HyperParam.AutoIntLayerDims[-1] * \
                                                 HyperParam.AutoIntHeadNumList[-1]

    # 兼容一下。。。
    if Config.datasetType == DATASETTYPE.AVAZU:
        HyperParam.AutoIntFeatureNum = 23
        HyperParam.AutoIntMatchMlpDims[0] = HyperParam.AutoIntFeatureNum * HyperParam.AutoIntFeatureDim

    if Config.datasetType == DATASETTYPE.CRITEO:
        HyperParam.AutoIntFeatureNum = 39
        HyperParam.AutoIntMatchMlpDims[0] = HyperParam.AutoIntFeatureNum * HyperParam.AutoIntFeatureDim

    if Config.datasetType == DATASETTYPE.MV:
        HyperParam.AutoIntFeatureNum = 7
        HyperParam.AutoIntMatchMlpDims[0] = HyperParam.AutoIntFeatureNum * HyperParam.AutoIntFeatureDim

    # if Config.datasetType == DATASETTYPE.MV:
    #     HyperParam.AutoIntFeatureNum = 264
    #     HyperParam.AutoIntFeatureDim = 16
    #     HyperParam.AutoIntMatchMlpDims[0] = HyperParam.AutoIntFeatureNum * HyperParam.AutoIntFeatureDim

    return featureInfo


if __name__ == '__main__':
    pass
