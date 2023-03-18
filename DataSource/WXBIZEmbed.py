import torch
import torch.nn as nn
from DataSource.BaseDataFormat import *
from Utils.HyperParamLoadModule import *
from DataSource.WXBIZData import *
from itertools import chain
from enum import Enum
import logging
import os
import time
import sys
import dis
import codecs

# for tensorflow env
try:
    import tensorflow as tf

    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%d-%m-%Y:%H:%M:%S')
    logging.getLogger().setLevel(logging.DEBUG)
    logger = logging.getLogger()
except:
    pass


class WXBIZEmbed(BaseEmbedFormat):
    def __init__(self, popCatFeature, popDims2DimFeature, popSeqFeature, featureInfo):

        self.popSeqFeature = popSeqFeature
        self.popDims2DimFeature = popDims2DimFeature
        self.popCatFeature = popCatFeature
        self.featureInfo = featureInfo

    def decompose(self, featValue, Dims2DimFeature: dict):
        start = HyperParam.FeatIdxLookupFeatureEnd
        for i in HyperParam.FeatValVecFeatureKeys:
            end = HyperParam.FeatValVecFeatureInfo[i] + start
            temp = featValue[:, start:end]
            Dims2DimFeature[i] = DataPack(name=i, fieldType=FIELDTYPE.DIMS2Dims, data=temp)
            start = HyperParam.FeatValVecFeatureInfo[i] + start

    def dealWithFeatLookUp(self, featureName: str, content, CatFeature, embedding, device):
        with open(os.path.join(Config.absPath, Config.lookupFeaturePath), 'r', encoding='utf-8') as feature:
            info = json.load(feature)
        content = content.type(torch.LongTensor).to(device) - HyperParam.FeatValDenseFeatureDim
        temp = embedding[featureName](content.type(torch.LongTensor).to(device))
        for i in info.keys():
            i = info[i]
            base = int(i['fieldB']) - HyperParam.FeatValDenseFeatureDim
            record = None
            for j in range(base, base + int(i['fieldLen'])):
                if record is None:
                    record = temp[:, j, :]
                else:
                    record = record + temp[:, j, :]
            CatFeature[i['name']] = DataPack(name=i['name'], fieldType=FIELDTYPE.CAT,
                                             data=record / int(i['fieldLen']))
        # return temp

        # def dealWithFeatLookUp(self, featureName: str, content, userFeature: dict, itemFeature: dict):
        #     with open(os.path.join(Config.absPath, Config.lookupFeaturePath), 'r', encoding='utf-8') as feature:
        #         info = json.load(feature)
        #     content = content.type(torch.LongTensor).to(self.device) - HyperParam.FeatValDenseFeatureDim
        #     temp = self.embedding[featureName](content.type(torch.LongTensor).to(self.device))
        #     for i in info.keys():
        #         i = info[i]
        #         base = int(i['fieldB']) - HyperParam.FeatValDenseFeatureDim
        #         for j in range(base, base + int(i['fieldLen'])):
        #             if i['type'] == 'user':
        #                 userFeature[i['name'] + f"{j}"] = temp[:, j, :]
        #             else:
        #                 itemFeature[i['name'] + f"{j}"] = temp[:, j, :]
        #     return temp

    def lookup(self, featureName: str, content, embedding, device):
        temp = embedding[featureName](content.type(torch.LongTensor).to(device))
        return temp

    def preProcess(self, rawData: dict, embedding, ):
        result={}
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if Config.buildState == BUILDTYPE.SAVEONNX:
            uidEmbed = rawData['uin_vid_emb']
            vidEmbed = rawData['vid_emb']

        SeqFeature = {i.featureName: DataPack(name=i.featureName, fieldType=FIELDTYPE.SEQUENCE,
                                              data=self.lookup(i.featureName, rawData[i.featureName], embedding,
                                                               device)) for i in
                      self.featureInfo
                      if
                      (i.enable == True and i.featureType == FEATURETYPE.USER)}

        SeqFeature[HyperParam.SequenceLenKey] = DataPack(name=HyperParam.SequenceLenKey, fieldType=FIELDTYPE.SEQUENCE,
                                                         data=rawData[HyperParam.SequenceLenKey])

        CatFeature = {i.featureName: DataPack(name=i.featureName, fieldType=FIELDTYPE.CAT,
                                              data=self.lookup(i.featureName, rawData[i.featureName],
                                                               embedding, device))
                      for i in
                      self.featureInfo
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
        return SeqFeature, Dims2DimFeature, CatFeature

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

    def buildEmbedding(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        embedding: nn.ModuleDict[str:nn.Embedding] = nn.ModuleDict()
        for feature in self.featureInfo:
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
        return embedding

    def buildShareEmbed(self, embedding):
        with open(os.path.join(Config.absPath, Config.shareEmbedFeatureJsonPath), 'r', encoding='utf-8') as file:
            self.shareInfo: dict[str:List] = json.load(file)
            for i in self.shareInfo.keys():
                embed = embedding[i]
                for j in self.shareInfo[i]:
                    embedding[j] = embed

    # def loadEmbed(self, savedPath, model):
    #     if os.path.exists(savedPath):
    #         save_info = torch.load(savedPath)
    #         assert 'model' in save_info.keys()
    #         model.load_state_dict(save_info['model'])
    #         self.buildShareEmbed()

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    abs_address = 'D:/Code/Pycharm/workspace/pytorch/FeatNet2/Config/FeatureConfig.json'
    featureInfo = loadArgs(abs_address)
    path = "D:/Code/data/WXBIZ/dataset/train_tfdata/data"
    test = WXBIZData(path)
    dataIter = test.getBatchData()
    with open(os.path.join(Config.absPath, Config.featureParamPath), 'r', encoding='utf-8') as feature:
        featureInfo: List[FeatureInfo] = json.load(feature, object_hook=FeatureInfo.hooker)
    embed = WXBIZEmbed([], [], [], featureInfo=featureInfo)
    embedding = embed.buildEmbedding()
    for i, count in dataIter:
        for j in i.keys():
            i[j] = torch.as_tensor(i[j]).to(device)
            result = embed.preProcess(i, embedding)
            print(result)
