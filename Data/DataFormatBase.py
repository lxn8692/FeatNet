import torch
import torch.nn as  nn
import json
from Utils.HyperParamLoadModule import *
from abc import ABCMeta, abstractmethod
from typing import List
from enum import Enum



class FIELDTYPE(Enum):
    SEQUENCE = 1
    DIMS2Dims = 2
    CAT = 3
    SIDEINFO = 4


class DataPack(object):
    def __init__(self, name: str, fieldType: FIELDTYPE, data: torch.Tensor):
        self.name = name
        self.fieldType = fieldType
        self.data = data


class DataFormatBase(object):
    def __init__(self):
        super(DataFormatBase, self).__init__()

    @abstractmethod
    def preProcess(self, rawData: dict, embedding, featureInfo, device):
        pass

    @abstractmethod
    def buildEmbedding(self, embedding, featureInfo, device):
        pass
