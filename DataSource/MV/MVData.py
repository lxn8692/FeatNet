# Copyright Verizon Media
# This project is licensed under the MIT. See license in project root for terms.
from operator import pos
import os
import sys

sys.path.append("/home/tianzhen/FeatNet/FeatNet")

from DataSource.BaseDataFormat import *
import tensorflow as tf
import logging
import json
import numpy as np
from enum import Enum
import csv
import torch
from tqdm import tqdm
from DataSource.BaseDataFormat import saved , load
logger = logging.getLogger(__name__)

class DATATYPE(Enum):
    test = 1
    train = 2


class MVData(BaseDataFormat):
    def __init__(self, data_path,
                 batch_size=2048,
                 num_worker=22,
                 buffer_size=10000,
                 prefetch=100):
        super(MVData, self).__init__()
        self.buffer_size = buffer_size
        self.prefetch = prefetch
        self.num_worker = num_worker
        self.batch_size = batch_size
        self.data_path = data_path
        self.getFileList()

        self.feature_names = ['user_id', 'item_id', 'label', 'weekday', 'hour', 'age', 'gender', 'occupation',
                              'zip_code', 'movie_title', 'release_year', 'genre']
        self.feature_defaults = [[0], [0], [0], [0], [0], [0], [0], [0], [0], ['x'], [0], ['x']]
        self.saved_path = os.path.join(self.data_path , f'{self.batch_size}_saved_dataset.pkl')
        sus_loaded = False
        if os.path.exists(self.saved_path):
            if self.fileType == DATATYPE.test:
                btsz , self.valData , self.testData = load(self.saved_path)
                sus_loaded =  btsz == self.batch_size
            elif self.fileType == DATATYPE.train:
                btsz , self.trainData = load(self.saved_path)
                sus_loaded = btsz == self.batch_size
        if not sus_loaded:
            self.getPrefetch()
            if self.fileType == DATATYPE.test:
                saved([self.batch_size , self.valData , self.testData],self.saved_path)
            elif self.fileType == DATATYPE.train:
                saved([self.batch_size , self.trainData] , self.saved_path)

    def getFileList(self):
        self.fileList = [os.path.join(self.data_path, name) for name in os.listdir(self.data_path)]
        if 'test_tfdata' in self.data_path:
            self.fileType = DATATYPE.test
            if 'test_data.csv' in self.fileList[0]:
                self.valFile = self.fileList[1]
                self.testFile = self.fileList[0]
            else:
                self.valFile = self.fileList[0]
                self.testFile = self.fileList[1]
        else:
            self.fileType = DATATYPE.train
            self.trainFile = self.fileList[0]

    def getPrefetch(self):
        if self.fileType == DATATYPE.test:
            self.valData = []
            for data, count in tqdm(self.readFromFile(self.valFile),position=10):
                self.valData.append((data, count))
                #break
                #print(f"val {count}")
            print(f"valFinish{len(self.valData)}")

            self.testData = []
            for data, count in tqdm(self.readFromFile(self.testFile),position=10):
                self.testData.append((data, count))
                #print(f"test {count}")
                #break
            print(f"testFinish{len(self.testData)}")

        if self.fileType == DATATYPE.train:
            self.trainData = []
            for data, count in tqdm(self.readFromFile(self.trainFile),position=10):
                #print(f"train {count}")
                self.trainData.append((data, count))
                #break

            print(f"trainFinish{len(self.trainData)}")

    def loadIntoMem(self, fileName):
        with open(fileName, 'r') as file:
            lines = csv.reader(file)
            out = list(lines)
        return out
    
    def preposs(self , x , lens):
        res = []
        for item in x:
            pred = bytes.decode(item)
            seq = []
            for token in pred.split(' '):
                seq.append(int(token))
            seq.extend([0] * (lens - len(seq)))
            res.append(np.array(seq,dtype=np.int32))
        return np.array(res)
    
    def readFromFile(self, fileName):
        dataset = tf.data.TextLineDataset(fileName)
        dataset = dataset.map(lambda x: self.parse_record(x, self.feature_names, self.feature_defaults),
                              num_parallel_calls=self.num_worker)
        dataset = dataset.batch(self.batch_size).prefetch(self.prefetch)
        count = 0
        for data in dataset.as_numpy_iterator():
            try:
                res_lt = data    
                count += res_lt[1].shape[0]
                res_lt[0]['label'] = res_lt[1]

                res_lt[0]['label'] = res_lt[0]['label'].astype(np.float32)
                res_lt[0]['movie_title'] = self.preposs(res_lt[0]['movie_title'] , 15)
                res_lt[0]['genre'] = self.preposs(res_lt[0]['genre'] , 6)
                yield res_lt[0], count
            except GeneratorExit:
                print("generator exit")
                break
        del dataset

    def getBatchData(self):
        if self.fileType == DATATYPE.train:
            dataIter = self.trainData
        else:
            dataIter = self.testData
        count = 0
        for data in dataIter:
            try:
                res_lt = data
                count += res_lt[1]
                yield res_lt[0], count
            except GeneratorExit:
                print("generator exit")
                break

    def getBufferData(self):
        if self.fileType == DATATYPE.test:
            dataIter = self.valData
        else:
            dataIter = self.trainData
        count = 0
        for data in dataIter:
            try:
                res_lt = data
                count += res_lt[1]
                yield res_lt[0], count
                if self.fileType == DATATYPE.train and count >= self.buffer_size:
                    print("buffer finish")
                    break
            except GeneratorExit:
                print("generator exit")
                break

    def load_cross_fields(self, cross_fields_file):
        if cross_fields_file is None:
            return None
        else:
            return set(json.load(open(cross_fields_file)))

    # Create a feature
    def parse_record(self, record, feature_names, feature_defaults):
        feature_array = tf.io.decode_csv(record, feature_defaults)
        features = dict(zip(feature_names, feature_array))
        label = features.pop('label')
        return features, label


if __name__ == "__main__":
    test = MVData("test/Movielens/dataset/train_tfdata")
    dataIter = test.getBufferData()
    for i, count in dataIter:
        print(count)
