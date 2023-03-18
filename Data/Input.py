import functools
from datetime import datetime
from Data.TensorflowIO import SampleDataTensorflow, DataDealArgs
import sys
import tensorflow as tf
import torch
import typing
import random
import numpy as np
from tfrecord import reader, tfrecord_loader, iterator_utils
from tfrecord.tools.tfrecord2idx import create_index
from Utils.HyperParamLoadModule import *
from tfrecord.torch.dataset import TFRecordDataset
import re
import collections.abc as container_abcs
int_classes = int
from torch._six import string_classes

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")

np_str_obj_array_pattern = re.compile(r'[SaUO]')
from tfrecord.torch.dataset import MultiTFRecordDataset
import tfrecord


class FILETYPE(Enum):
    TFRECORD = 1
    CSV = 2
    LIBSVM = 3


class Input():
    # 需要考虑io性能，这里设置的是一次读入文件的个数/最大上限内存
    def __init__(self, absPath: str, filePath, fileNameList: list, sep: str, maxFilePerIO: int,
                 maxMemPerIO: int):
        self.absPath = absPath
        self.filePath = filePath
        self.fileNameList = fileNameList
        self.sep = sep
        self.maxFilePerIO = maxFilePerIO
        self.maxMemPerIO = maxFilePerIO
        self.curIndex = 0  # 目前读到的文件
        pass

    # @abstractmethod
    # def readFile(self, ):
    #     raise NotImplemented
    # nameList = [self.absPath + self.filePath + self.fileNameList[self.curIndex]]
    # size = os.path.getsize(self.absPath + self.filePath + nameList[self.curIndex])
    # self.curIndex += 1
    # while self.curIndex < len(self.fileNameList):
    #     size = size + os.path.getsize(self.absPath + self.filePath + nameList[self.curIndex])
    #     if size > self.maxMemPerIO or len(nameList) > self.maxFilePerIO:
    #         break
    #     else:
    #         nameList.append(self.absPath + self.filePath + self.fileNameList[self.curIndex])
    #         self.curIndex += 1
    # else:
    #     self.curIndex = 0
    # return nameList
    #
    # if self.fileType == FILETYPE.TFRECORD:
    #     return self._readTFRecord()
    # elif self.fileType == FILETYPE.CSV:
    #     pass
    # elif self.fileType == FILETYPE.LIBSVM:
    #     pass
    # else:
    #     raise Exception('file type not support!')


class TFDataset(torch.utils.data.IterableDataset):

    def __init__(self,
                 data_pattern: str,
                 index_pattern: str,
                 fileNameList: List[str],
                 transform: typing.Callable[[dict], typing.Any] = None, ) -> None:
        super(TFDataset, self).__init__()
        self.data_pattern = data_pattern
        self.index_pattern = index_pattern
        self.fileNameList = fileNameList
        self.rmEmptyFile()
        self.transform = transform

    def shuffleFileList(self):
        random.shuffle(self.fileNameList)
        print(self.fileNameList)

    def rmEmptyFile(self):
        rmList = []
        for i in self.fileNameList:
            file = self.data_pattern.format(i)
            if os.path.getsize(file) == 0:
                rmList.append(i)
        for i in rmList:
            self.fileNameList.remove(i)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            shard = worker_info.id, worker_info.num_workers
            np.random.seed(worker_info.seed % np.iinfo(np.uint32).max)
        else:
            shard = None

        it = self.multi_tfrecord_loader(
            self.data_pattern, self.index_pattern, self.fileNameList, shard)
        return it

    def multi_tfrecord_loader(self, data_pattern: str,
                              index_pattern: typing.Union[str, None],
                              fileName: typing.List[str],
                              shard: typing.Optional[typing.Tuple[int, int]] = None,
                              description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
                              ) -> typing.Iterable:

        loaders = [functools.partial(tfrecord_loader, data_path=data_pattern.format(split),
                                     index_path=index_pattern.format(split) \
                                         if index_pattern is not None else None,
                                     description=description, shard=shard) for split in fileName]
        # loader = [iter() for iter in loaders]
        for i in loaders:
            for j in i():
                yield j

    # def bindAllDataFile(self):
    #     for i in range(len(self.input.fileNameList)):
    #         tfrecord_path = os.path.join(self.input.absPath + self.input.filePath + self.input.fileNameList[i])
    #         dataset = reader.tfrecord_loader(tfrecord_path, None, None)
    #     return itertools.chain.from_iterable(result)


def blank_collate(batch):
    return batch


def default_collate(batch):
    return batch
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        # if torch.utils.data.get_worker_info() is not None:
        #     # If we're in a background process, concatenate directly into a
        #     # shared memory tensor to avoid an extra copy
        #     numel = sum([x.numel() for x in batch])
        #     storage = elem.storage()._new_shared(numel)
        #     out = elem.new(storage)
        # return batch
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            # return default_collate(torch.as_tensor(np.asarray(batch)))
            return torch.as_tensor(np.asarray(batch))
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch, )
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


class TFRecordInput(object):
    def __init__(self, absPath: str, filePath, fileNameList: list, batchSize=32, sep: str = None, maxFilePerIO: int = 1,
                 maxMemPerIO: int = 1, collate=None, device=None, numWorker=2, pinMemoryDataNum: int = 0,
                 shuffle=False, lightCC=False):
        self.lightCC = lightCC
        self.shuffle = shuffle
        self.pinMemoryDataNum = pinMemoryDataNum
        self.col = collate if collate is not None else default_collate
        self.input = Input(absPath, filePath, fileNameList, sep, maxFilePerIO, maxMemPerIO)
        self.loader = None
        self.batchSize = batchSize
        self.device = device
        self.numWorker = numWorker
        self.dataPath = os.path.join(self.input.absPath, self.input.filePath, 'Data/{}')
        self.indexPath = os.path.join(self.input.absPath, self.input.filePath, 'Index/{}.index')
        self.dataIter = TFDataset(data_pattern=self.dataPath, index_pattern=self.indexPath,
                                  fileNameList=self.input.fileNameList, )
        # self.dataIter1 = TFDataset(data_pattern=self.dataPath, index_pattern=self.indexPath,
        #                            fileNameList=self.input.fileNameList)
        loader = torch.utils.data.DataLoader(self.dataIter, batch_size=self.batchSize, drop_last=False,
                                             collate_fn=default_collate,
                                             num_workers=numWorker)
        self.loader = loader
        # self.loader1 = torch.utils.data.DataLoader(self.dataIter1, batch_size=self.batchSize, drop_last=False,
        #                                            collate_fn=self.default_collate, num_workers=numWorker)
        # self.iter = None
        self.pinMemoryData = []
        if pinMemoryDataNum != 0:
            self.pinMemoryDataNum = self.getPinMemoryData()

    def transform(self, x):
        return x

    def generateIndex(self, ):
        path = os.path.join(self.input.absPath, self.input.filePath)
        for i in self.input.fileNameList:
            # create_index(os.path.join(path, i), os.path.join(path, 'index/', f'{i}.index'))
            print(os.path.join(path, 'Data/', i))
            os.system(
                f"python3 -m tfrecord.tools.tfrecord2idx {os.path.join(path, 'Data/', i)} {os.path.join(path, 'Index/', f'{i}.index')}")
        return

    def getNextEpoch(self):
        print(f"get next epoch: {os.getpid()}")
        if self.shuffle is True:
            self.dataIter.shuffleFileList()
        return self.loader

    def getPinData(self):
        loader = torch.utils.data.DataLoader(self.pinMemoryData, batch_size=self.batchSize, drop_last=False,
                                             collate_fn=default_collate, )
        return loader
        # return self.pinMemoryData

    # def checkStatistic(self, name):
    #     with open(f'./{name}_out.txt', 'w', encoding='utf-8') as file:
    #         counter = 0
    #         totalPos = 0
    #         totalNeg = 0
    #         for i in range(len(self.input.fileNameList)):
    #             pos = 0
    #             neg = 0
    #             subcounter = 0
    #             print(f'{self.input.fileNameList[i]}')
    #             # nameList = self.input.readFile()】
    #             # description = {"image": "byte", "label": "float"}
    #             for i, j in enumerate(iter(self.loader)):
    #                 print(f'NO:{i}   time:{datetime.now().timestamp().__str__()}')
    #                 if j['label'] == 0:
    #                     neg += 1
    #                 else:
    #                     pos += 1
    #                 subcounter += 1
    #             for i, j in enumerate(iter(self.loader)):
    #                 print(f'NO:{i}   time:{datetime.now().timestamp().__str__()}')

    def getPinMemoryData(self):
        loader = torch.utils.data.DataLoader(self.dataIter, batch_size=self.pinMemoryDataNum, drop_last=False,
                                             collate_fn=blank_collate)
        begin = datetime.now().timestamp()
        self.pinMemoryData = next(iter(loader))
        end = datetime.now().timestamp()
        print(f"get pin {self.dataPath} memory data number:{len(self.pinMemoryData)} time :{end - begin}")
        return len(self.pinMemoryData)


class TFOriginInput(object):
    def __init__(self, absPath: str, filePath, fileNameList: list, batchSize=32, sep: str = None, maxFilePerIO: int = 1,
                 maxMemPerIO: int = 1, collate=None, device=None, numWorker=2, pinMemoryDataNum: int = 0,
                 shuffle=False, lightCC=False):
        self.lightCC = lightCC
        self.shuffle = shuffle
        self.pinMemoryDataNum = pinMemoryDataNum
        self.col = collate if collate is not None else default_collate
        self.input = Input(absPath, filePath, fileNameList, sep, maxFilePerIO, maxMemPerIO)
        self.loader = None
        self.batchSize = batchSize
        self.device = device
        self.numWorker = numWorker
        self.dataPath = os.path.join(self.input.absPath, self.input.filePath, 'Data')
        self.dataset = SampleDataTensorflow(DataDealArgs(), self.dataPath, batchSize,prefetch=20000)

        # self.iter = None
        self.pinMemoryData = []
        if pinMemoryDataNum != 0:
            self.pinMemoryDataNum = self.getPinMemoryData()
        self.shuffleFileList()

    def shuffleFileList(self):
        random.shuffle(self.input.fileNameList)
        print(self.input.fileNameList)

    def getNextEpoch(self):
        # if self.shuffle is True:
        #     self.shuffleFileList()
        print(self.input.fileNameList)
        count = 0
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for batch_data, count in self.dataset.get_batch_data(sess, count, self.input.fileNameList):
                yield batch_data, count

        # print(f"get next epoch: {os.getpid()}")

        # return self.loader

    def getPinMemoryData(self):
        for batch_data, count in self.getNextEpoch():
            self.pinMemoryData.append(batch_data)
            if count >= self.pinMemoryDataNum:
                print(count, len(self.pinMemoryData))
                return count

    def getPinData(self):
        loader = torch.utils.data.DataLoader(self.pinMemoryData, batch_size=1, drop_last=False,
                                             collate_fn=default_collate, )
        return loader


def writeLookUpFeature():
    path = '/Users/ivringwang/Desktop/tencent/video_rec_meta_data/feat_map.txt'
    result = {}
    flag = False
    with open(path, 'r', encoding='utf-8') as feat:
        with open("/Users/ivringwang/Desktop/tencent/GMM_torch/Config/Parameter/lookupFeature.json", 'w',
                  encoding='utf-8') as lookup:
            for i in feat.readlines():
                i = i.split()
                if ('age' != i[0] and flag == False):
                    continue
                if (i[0] == 'kyk_vec'):
                    break
                else:
                    flag = True
                    temp = {
                        'name': i[0],
                        'offsetB': i[1],
                        'offsetLen': i[2],
                        'fieldB': i[3],
                        'fieldLen': i[4]
                    }
                    if 'uin' in i[0]:
                        temp['type'] = ('user')
                    else:
                        temp['type'] = ('item')
                    result[i[0]] = temp
            json.dump(result, lookup)

    # print("finish")


if __name__ == '__main__':
    pass
