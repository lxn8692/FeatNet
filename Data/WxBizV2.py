import torch
import torch.nn as nn
from Data.DataFormatBase import *
from Utils.HyperParamLoadModule import *
from itertools import chain
from enum import Enum


class WxBizV2(DataFormatBase):
    def __init__(self, popCatFeature, popDims2DimFeature, popSeqFeature):
        super(WxBizV2, self, ).__init__()
        self.popCatFeature = popCatFeature
        self.popDims2DimFeature = popDims2DimFeature
        self.popSeqFeature = popSeqFeature

    def decompose(self, featValue, Dims2DimFeature: dict):
        start = HyperParam.FeatIdxLookupFeatureEnd
        for i in HyperParam.FeatValVecFeatureKeys:
            end = HyperParam.FeatValVecFeatureInfo[i] + start
            temp = featValue[:, start:end]
            Dims2DimFeature[i] = DataPack(name=i, fieldType=FIELDTYPE.DIMS2Dims, data=temp)
            start = HyperParam.FeatValVecFeatureInfo[i] + start

    # def dealWithFeatLookUp(self, featureName: str, content, CatFeature, embedding, device):
    #     with open(os.path.join(Config.absPath, Config.lookupFeaturePath), 'r', encoding='utf-8') as feature:
    #         info = json.load(feature)
    #     content = content.type(torch.LongTensor).to(device) - HyperParam.FeatValDenseFeatureDim
    #     temp = embedding[featureName](content.type(torch.LongTensor).to(device))
    #     for i in info.keys():
    #         i = info[i]
    #         base = int(i['fieldB']) - HyperParam.FeatValDenseFeatureDim
    #         record = None
    #         for j in range(base, base + int(i['fieldLen'])):
    #             CatFeature[i[f'name_{j}']] = DataPack(name=i['name'], fieldType=FIELDTYPE.CAT,
    #                                                  data=temp[:, j, :])
    #
    #     # return temp

    def dealWithFeatLookUp(self, featureName: str, content, CatFeature, embedding, device):
        with open(os.path.join(Config.absPath, Config.lookupFeaturePath), 'r', encoding='utf-8') as feature:
            info = json.load(feature)
        content = content.type(torch.LongTensor).to(device) - HyperParam.FeatValDenseFeatureDim
        temp = embedding[featureName](content.type(torch.LongTensor).to(device))
        for i in info.keys():
            i = info[i]
            base = int(i['fieldB']) - HyperParam.FeatValDenseFeatureDim
            for j in range(base, base + int(i['fieldLen'])):
                CatFeature[f'{i["name"]}_{j}'] = DataPack(name=i['name'], fieldType=FIELDTYPE.CAT, data=temp[:, j, :])
        return temp

    def lookup(self, featureName: str, content, embedding, device):
        temp = embedding[featureName](content.type(torch.LongTensor).to(device))
        return temp

    def preProcess(self, rawData: dict, embedding, featureInfo, device):
        if Config.buildState == BUILDTYPE.SAVEONNX:
            uidEmbed = rawData['uin_vid_emb']
            vidEmbed = rawData['vid_emb']

        SeqFeature = {i.featureName: DataPack(name=i.featureName, fieldType=FIELDTYPE.SEQUENCE,
                                              data=self.lookup(i.featureName, rawData[i.featureName], embedding,
                                                               device)) for i in
                      featureInfo
                      if
                      (i.enable == True and i.featureType == FEATURETYPE.USER)}

        SeqFeature[HyperParam.SequenceLenKey] = DataPack(name=HyperParam.SequenceLenKey, fieldType=FIELDTYPE.SIDEINFO,
                                                         data=rawData[HyperParam.SequenceLenKey])

        CatFeature = {i.featureName: DataPack(name=i.featureName, fieldType=FIELDTYPE.CAT,
                                              data=self.lookup(i.featureName, rawData[i.featureName],
                                                               embedding, device))
                      for i in
                      featureInfo
                      if
                      (i.enable == True and i.featureType == FEATURETYPE.ITEM)}

        Dims2DimFeature = {'dense': DataPack('dense', FIELDTYPE.DIMS2Dims,
                                             rawData['feat_value'][:, :HyperParam.FeatValDenseFeatureDim])}

        self.dealWithFeatLookUp('feat_index',
                                rawData['feat_index'][:,
                                HyperParam.FeatValDenseFeatureDim:HyperParam.FeatIdxLookupFeatureEnd],
                                CatFeature, embedding, device)

        self.decompose(rawData['feat_value'], Dims2DimFeature)
        if Config.buildState == BUILDTYPE.SAVEONNX:
            SeqFeature['uin_vid_id'].data = uidEmbed
            CatFeature['vid_id'].data = vidEmbed

        self.dropFeature(CatFeature, Dims2DimFeature, SeqFeature)

        # SeqFeature = {key: DataPack(data=torch.zeros_like(SeqFeature[key].data), fieldType=SeqFeature[key].fieldType,
        #                             name=SeqFeature[key].name) for key in SeqFeature.keys()}
        # Dims2DimFeature = {key: DataPack(data=torch.zeros_like(Dims2DimFeature[key].data), fieldType=Dims2DimFeature[key].fieldType,
        #                             name=Dims2DimFeature[key].name) for key in Dims2DimFeature.keys()}
        # CatFeature = {key: DataPack(data=torch.zeros_like(CatFeature[key].data), fieldType=CatFeature[key].fieldType,
        #                             name=CatFeature[key].name) for key in CatFeature.keys()}

        # SeqFeature_str = '|'.join(
        #     ["%s:%s" % (k, ','.join([str(vv) for vv in v.data.shape])) for k, v in SeqFeature.items()])
        # print(SeqFeature_str)
        #
        # Dims2DimFeature_str = '|'.join(
        #     ["%s:%s" % (k, ','.join([str(vv) for vv in v.data.shape])) for k, v in Dims2DimFeature.items()])
        # print(Dims2DimFeature_str)
        #
        # CatFeature_str = '|'.join(
        #     ["%s:%s" % (k, ','.join([str(vv) for vv in v.data.shape])) for k, v in CatFeature.items()])
        # print(CatFeature_str)


        # SeqFeature = self.buildPack(SeqFeature_str)
        # Dims2DimFeature = self.buildPack(Dims2DimFeature_str)
        # CatFeature = self.buildPack(CatFeature_str)
        return SeqFeature, Dims2DimFeature, CatFeature,



    def dropFeature(self, catFeature, Dims2DimFeature, seqFeature):
        for i in self.popCatFeature:
            if i in catFeature.keys():
                catFeature.pop(i)
        for i in self.popDims2DimFeature:
            if i in Dims2DimFeature.keys():
                Dims2DimFeature.pop(i)
        for i in self.popSeqFeature:
            if i in seqFeature.keys():
                seqFeature.pop(i)

    def buildEmbedding(self, embedding, featureInfo: FeatureInfo, device):
        for feature in featureInfo:
            if feature.enable:
                embedding[feature.featureName] = nn.Embedding(feature.inputDim, feature.embeddingSize).to(
                    device=device)
        # load pretrain Model
        self.buildShareEmbed(embedding)
        if Config.loadPreTrainModel:
            pass
        # due with share embedding
        else:
            for i in self.shareInfo.keys():
                nn.init.xavier_normal(embedding[i].weight, gain=1.414)
        return

    def buildShareEmbed(self, embedding):
        with open(os.path.join(Config.absPath, Config.shareEmbedFeatureJsonPath), 'r', encoding='utf-8') as file:
            self.shareInfo: dict[str:List] = json.load(file)
            for i in self.shareInfo.keys():
                embed = embedding[i]
                for j in self.shareInfo[i]:
                    embedding[j] = embed
