import torch
import torch.nn as nn
from DataSource.BaseDataFormat import *
from Utils.HyperParamLoadModule import *
from itertools import chain
from enum import Enum
import logging
import os
import time
import sys
import dis
import codecs
import tracemalloc

# for tensorflow env
try:
    import tensorflow as tf

    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%d-%m-%Y:%H:%M:%S')
    logging.getLogger().setLevel(logging.DEBUG)
    logger = logging.getLogger()
except:
    pass


class DataDealArgs():
    uin_max_vid_len = 15
    pos_count = 10
    field_size = 1436
    vid_id_emb_sz = 128


class WXBIZData(BaseDataFormat):

    def __init__(self, data_path, args=DataDealArgs(),
                 batch_size=2048,
                 num_worker=10,
                 buffer_size=10000,
                 prefetch=100):
        super(WXBIZData, self, ).__init__()
        self.buffer_size = buffer_size
        self.prefetch = prefetch
        self.num_worker = num_worker
        self.batch_size = batch_size
        self.data_path = data_path
        self.tensor_names = "label,feat_index,feat_value,uin_vid_id,all_vid_id,uin_vid_len,uin_vid_pos_id,uin_vid_cat1_id,uin_vid_cat2_id,uin_vid_day_id,uin_vid_week_id,uin_vid_hour_id,uin_age_id,uin_gender_id,uin_language_id,uin_platform_id,uin_grade_id,vid_id,vid_cat1_id,vid_cat2_id,vid_day_id,vid_week_id,vid_hour_id".split(
            ',')
        self.uin_max_vid_len = args.uin_max_vid_len
        self.pos_count = args.pos_count
        self.field_size = args.field_size
        # 定义各自存入tfrecord的schema
        self.define_data_schema()
        self.getFileList()
        self.createBufferData()

    def getFileList(self):
        fileList = os.listdir(self.data_path)
        try:
            fileList.remove(".DS_Store")
        except:
            pass
        try:
            fileList.remove('_SUCCESS')
        except:
            pass

        """"""
        # fileList.sort(key=lambda x: int(x[-5:]))
        fileList.sort(key=lambda x: int(x[-5:-3]))
        """"""
        self.fileList = fileList
        print(self.fileList)

    def getBatchData(self):

        count = 0
        # 主函数
        features_tensor = self.create_dataset(self.fileList)
        # print(features_tensor)
        for data in features_tensor.as_numpy_iterator():
            try:
                # temp = [features_tensor[name] for name in self.tensor_names]
                # res_lt = sess.run(temp)
                res_lt = data
                # print(sys.getrefcount(res_lt))
                # print(res_lt)
                # input_dict = {name: res_lt[self.tensor_names2id[name]] for name in self.tensor_names}
                count += res_lt["label"].shape[0]
                # print(sys.getrefcount(input_dict))
                yield res_lt, count
            except GeneratorExit:
                print("generator exit")
                break
        del features_tensor

    def createBufferData(self):
        self.buffer = []
        data = self.getBatchData()
        for record, count in data:
            self.buffer.append((record, count))
            print(count)
            if count >= self.buffer_size:
                data.close()
                print(f"train buffer size:{count}")
                break

    def getBufferData(self):
        return self.buffer

    # 定义dataset结构
    def create_dataset(self, fileList):
        sub_path = self.data_path
        logger.info(sub_path)
        input_file_lt = ["%s/%s" % (sub_path, name) for name in fileList if
                         (name != ".DS_Store" and name != '_SUCCESS')]
        # print(input_file_lt)
        dataset_tmp = tf.data.TFRecordDataset(input_file_lt)
        dataset = dataset_tmp.map(self.read_one_sample_data, num_parallel_calls=self.num_worker) \
            .batch(self.batch_size) \
            .prefetch(self.prefetch)

        # iterator = dataset.make_one_shot_iterator()
        # features = iterator.get_next()
        return dataset

    # 依赖函数
    def read_one_sample_data(self, record):
        parsed = tf.io.parse_single_example(record, self.data_schema)
        # parsed = tf.parse_example(record, self.data_schema)
        # parsed["aaa"] = parsed["SSS"] + parsed["sdf"]
        return parsed

    def read_one_sample_data_v2(self, record):
        # parsed = tf.parse_single_example(record, self.data_schema)
        parsed = tf.io.parse_example(record, self.data_schema)
        return parsed

    def define_data_schema(self):
        self.tensor_names2id = {name: id for id, name in enumerate(self.tensor_names)}
        self.data_schema = {
            "uin_vid_id": tf.io.FixedLenFeature(shape=(self.uin_max_vid_len), dtype=tf.int64),
            "uin_vid_pos_id": tf.io.FixedLenFeature(shape=(self.uin_max_vid_len), dtype=tf.int64),
            "uin_vid_cat1_id": tf.io.FixedLenFeature(shape=(self.uin_max_vid_len), dtype=tf.int64),
            "uin_vid_cat2_id": tf.io.FixedLenFeature(shape=(self.uin_max_vid_len), dtype=tf.int64),
            "uin_vid_day_id": tf.io.FixedLenFeature(shape=(self.uin_max_vid_len), dtype=tf.int64),  # 周几--点击
            "uin_vid_hour_id": tf.io.FixedLenFeature(shape=(self.uin_max_vid_len), dtype=tf.int64),  # 几点--点击
            "uin_vid_week_id": tf.io.FixedLenFeature(shape=(self.uin_max_vid_len), dtype=tf.int64),
            "uin_vid_len": tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            "uin_age_id": tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            "uin_gender_id": tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            "uin_language_id": tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            "uin_platform_id": tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            "uin_grade_id": tf.io.FixedLenFeature(shape=(), dtype=tf.int64),

            "vid_id": tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            "vid_cat1_id": tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            "vid_cat2_id": tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            "vid_day_id": tf.io.FixedLenFeature(shape=(), dtype=tf.int64),  # 周几--曝光(推送近似)
            "vid_hour_id": tf.io.FixedLenFeature(shape=(), dtype=tf.int64),  # 几点--曝光(推送近似)
            "vid_week_id": tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            "label": tf.io.FixedLenFeature(shape=(), dtype=tf.float32),

            "all_vid_id": tf.io.FixedLenFeature(shape=(self.uin_max_vid_len + 1), dtype=tf.int64),
            "feat_index": tf.io.FixedLenFeature(shape=(self.field_size), dtype=tf.int64),
            "feat_value": tf.io.FixedLenFeature(shape=(self.field_size), dtype=tf.float32),
            # "context_emb": tf.FixedLenFeature(shape=(self.field_size), dtype=tf.float32),
        }


if __name__ == '__main__':
    path = "/Users/ivringwang/Desktop/tencent/GMM_torch/test/dataset/train_tfdata/Data"
    test = WXBIZData(path)
    dataIter = test.getBatchData()
    for i, count in dataIter:
        print(count)
        if count >= 20000:
            dataIter.close()
            print("close")
