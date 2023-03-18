from numpy import pi
import torch
import torch.nn as nn
import json
from abc import ABCMeta, abstractmethod
from typing import List
from enum import Enum
import pickle


def saved(obj,filepath):
    obj = pickle.dumps(obj)
    with open(filepath,'wb') as f:
        f.write(obj)
    
def load(filepath):
    with open(filepath,'rb') as f:
        obj = pickle.load(f)
    return obj

class FIELDTYPE(Enum):
    SEQUENCE = 1
    DIMS2Dims = 2
    CAT = 3
    SIDEINFO = 4


class DataPack(object):
    def __init__(self, name: str, fieldType: FIELDTYPE, data: torch.Tensor, memo: object = None):
        self.name = name
        self.fieldType = fieldType
        self.data = data
        self.memo = memo


class BaseEmbedPack(object):
    def __init__(self, popCatFeature, popDims2DimFeature, popSeqFeature, featureInfo):
        self.featureInfo = featureInfo
        self.popSeqFeature = popSeqFeature
        self.popDims2DimFeature = popDims2DimFeature
        self.popCatFeature = popCatFeature


class BaseDataFormat(object):
    def __init__(self):
        super(BaseDataFormat, self).__init__()

    @abstractmethod
    def loadDataParams(self):
        pass

    @abstractmethod
    def getBatchData(self):
        pass

    @abstractmethod
    def getBufferData(self):
        pass


        



class BaseEmbedFormat(object):

    @abstractmethod
    def buildEmbedding(self) -> dict:
        pass

    @abstractmethod
    def loadEmbedding(self,savedPath,model):
        pass

    @abstractmethod
    def preProcess(self, rawData, embedding) -> dict:
        pass
