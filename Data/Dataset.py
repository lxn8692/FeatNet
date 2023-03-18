import os

from Data.Input import TFRecordInput, TFOriginInput
from Utils.HyperParamLoadModule import *
import torch
from DataSource.MV import *
from DataSource.WXBIZData import *
from DataSource import *


# 444655
# 完成的任务主要是将file list不断地组装。
class Dataset():
    def __init__(self, device, trainWorker=10, testWoker=10, valWorker=1, collate=None):
        trainFileList = os.listdir(os.path.join(Config.datasetPath, Config.datasetName, 'train_tfdata/Data'))
        testFileList = os.listdir(os.path.join(Config.datasetPath, Config.datasetName, 'test_tfdata/Data'))
        # valFileList = os.listdir(os.path.join(Config.datasetPath, Config.datasetName, 'val_tfdata/Data'))
        try:
            trainFileList.remove(".DS_Store")
        except:
            pass
        try:
            trainFileList.remove('_SUCCESS')
        except:
            pass
        try:
            testFileList.remove(".DS_Store")
        except:
            pass
        try:
            testFileList.remove('_SUCCESS')
        except:
            pass
        # try:
        #     valFileList.remove(".DS_Store")
        # except:
        #     pass
        # try:
        #     valFileList.remove('_SUCCESS')
        # except:
        #     pass
        trainFileList.sort(key=lambda x: int(x[-5:]))
        testFileList.sort(key=lambda x: int(x[-5:]))
        print(trainFileList)
        print(testFileList)
        # valFileList = os.listdir(os.path.join(Config.datasetPath, Config.datasetName, 'val_tfdata/Data'))
        # valFileList.remove('_SUCCESS')

        # self.trainData = TFRecordInput(Config.datasetPath, os.path.join(Config.datasetName, 'train_tfdata/'),
        #                                trainFileList,
        #                                batchSize=HyperParam.batchSize, device=device, numWorker=trainWorker,
        #                                pinMemoryDataNum=1000, collate=collate, shuffle=True, lightCC=Config.lightCC)
        # self.testData = TFRecordInput(Config.datasetPath, os.path.join(Config.datasetName, 'test_tfdata/'),
        #                               testFileList,
        #                               batchSize=HyperParam.batchSize, device=device, numWorker=testWoker,
        #                               pinMemoryDataNum=1000, collate=collate, shuffle=False, lightCC=Config.lightCC)

        self.trainData = TFOriginInput(Config.datasetPath, os.path.join(Config.datasetName, 'train_tfdata/'),
                                       trainFileList,
                                       batchSize=HyperParam.batchSize, device=device, numWorker=trainWorker,
                                       pinMemoryDataNum=1000, collate=collate, shuffle=True, lightCC=Config.lightCC)
        self.testData = TFOriginInput(Config.datasetPath, os.path.join(Config.datasetName, 'test_tfdata/'),
                                      testFileList,
                                      batchSize=HyperParam.batchSize, device=device, numWorker=testWoker,
                                      pinMemoryDataNum=1000, collate=collate, shuffle=False, lightCC=Config.lightCC)
        # self.valData = TFRecordInput(Config.datasetPath, os.path.join(Config.datasetName, 'val_tfdata/'),
        #                               valFileList,
        #                               batchSize=HyperParam.batchSize, device=device, numWorker=testWoker,
        #                               pinMemoryDataNum=1000, collate=collate, shuffle=False, lightCC=Config.lightCC)
        # self.valData.generateIndex()


class DatasetV2:
    def __init__(self, datasetType, batch_size=2048, num_worker=10, prefetch=100, bufferSize=20000):
        self.prefetch = prefetch
        self.num_worker = num_worker
        self.batch_size = batch_size
        trainPath = os.path.join(Config.datasetPath, Config.datasetName, 'train_tfdata/Data')
        testPath = os.path.join(Config.datasetPath, Config.datasetName, 'test_tfdata/Data')
        self.train: BaseDataFormat = eval(f"{datasetType}Data.{datasetType}Data")(trainPath, batch_size=batch_size,
                                                                                  num_worker=num_worker,
                                                                                  prefetch=prefetch,
                                                                                  buffer_size=bufferSize)
        self.test: BaseDataFormat = eval(f"{datasetType}Data.{datasetType}Data")(testPath, batch_size=batch_size,
                                                                                 num_worker=num_worker,
                                                                                 prefetch=prefetch,
                                                                                 buffer_size=bufferSize)


def getVidNumb(dataPath, jsonPath):
    vidFileList = os.listdir(os.path.join(dataPath, 'vid_emb_tfdata/Data'))
    try:
        vidFileList.remove(".DS_Store")
    except:
        pass
    try:
        vidFileList.remove('_SUCCESS')
    except:
        pass
    data = TFRecordInput('', os.path.join(dataPath, 'vid_emb_tfdata/'), vidFileList, batchSize=1,
                         pinMemoryDataNum=0)
    counter = 0
    for i in data.getNextEpoch():
        counter += 1
    print(counter)
    with open(jsonPath, 'r+', encoding='utf-8') as file:
        featureInfo: List[FeatureInfo] = json.load(file, object_hook=FeatureInfo.hooker)
        file.seek(0)
        file.truncate()
        result = []
        for i in featureInfo:
            if i.embeddingSize == 32:
                i.embeddingSize = 16
            if i.featureName == 'vid_id':
                i.inputDim = counter
            result.append(i.keys())
        json.dump(result, file, ensure_ascii=False)


if __name__ == '__main__':
    file = "/Users/ivringwang/Desktop/tencent/GMM_torch/test/dataset/"
    getVidNumb(file, "/Users/ivringwang/Desktop/tencent/GMM_torch/Config/Parameter/test.json")
