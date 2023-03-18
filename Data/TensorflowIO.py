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


#####################################################
# 数据处理相关的参数集合
#####################################################
class DataDealArgs():
    uin_max_vid_len = 15
    pos_count = 10
    field_size = 1436
    vid_id_emb_sz = 128


#####################################################
# 训练数据样本的数据结构
#####################################################

class SampleData():
    def __init__(self):
        # self.tensor_names = ["label"]
        self.tensor_names = "label,feat_index,feat_value,uin_vid_id,all_vid_id,uin_vid_len,uin_vid_pos_id,uin_vid_cat1_id,uin_vid_cat2_id,uin_vid_day_id,uin_vid_week_id,uin_vid_hour_id,uin_age_id,uin_gender_id,uin_language_id,uin_platform_id,uin_grade_id,vid_id,vid_cat1_id,vid_cat2_id,vid_day_id,vid_week_id,vid_hour_id".split(
            ',')
        # self.tensor_names = "label,context_emb,feat_index,feat_value,uin_vid_id,all_vid_id,uin_vid_len,uin_vid_pos_id,uin_vid_cat1_id,uin_vid_cat2_id,uin_vid_day_id,uin_vid_week_id,uin_vid_hour_id,uin_age_id,uin_gender_id,uin_language_id,uin_platform_id,uin_grade_id,vid_id,vid_cat1_id,vid_cat2_id,vid_day_id,vid_week_id,vid_hour_id".split(
        #     ',')


# tfrecord for tensorflow
class SampleDataTensorflow(SampleData):
    def __init__(self, args, data_path, batch_size=2048, num_worker=10, prefetch=100):
        SampleData.__init__(self)
        # 提前存入数据处理依赖的参数
        self.uin_max_vid_len = args.uin_max_vid_len
        self.pos_count = args.pos_count
        self.field_size = args.field_size
        self.batch_size = batch_size
        self.data_path = data_path

        # data set 参数
        self.num_parallel_calls = num_worker
        self.shuffle_buff_sz = 3  # * self.batch_size
        self.prefetch_buff_sz = prefetch  # * self.batch_size
        # 定义各自存入tfrecord的schema
        self.define_data_schema()

    # 主函数
    def get_batch_data(self, sess, count, fileList):  # 获取一个batch数据
        features_tensor = self.create_dataset(fileList)
        # print(features_tensor)
        while True:
            try:
                temp=[features_tensor[name] for name in self.tensor_names]
                res_lt = sess.run(temp)
                # print(sys.getrefcount(res_lt))
                # print(res_lt)
                input_dict = {name: res_lt[self.tensor_names2id[name]] for name in self.tensor_names}
                count += input_dict["label"].shape[0]
                print(sys.getrefcount(input_dict))
                yield input_dict, count
                del input_dict
            except tf.errors.OutOfRangeError:
                logger.info("End of dataset, count %d" % count)
                break
        del features_tensor

    # 定义dataset结构
    def create_dataset(self, fileList):
        sub_path = self.data_path
        logger.info(sub_path)
        input_file_lt = ["%s/%s" % (sub_path, name) for name in fileList if
                         (name != ".DS_Store" and name != '_SUCCESS')]
        # print(input_file_lt)
        dataset_tmp = tf.data.TFRecordDataset(input_file_lt)
        dataset = dataset_tmp.map(self.read_one_sample_data, num_parallel_calls=self.num_parallel_calls) \
            .batch(self.batch_size) \
            .prefetch(self.prefetch_buff_sz)

        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()
        return features

    # 依赖函数
    def read_one_sample_data(self, record):
        parsed = tf.parse_single_example(record, self.data_schema)
        # parsed = tf.parse_example(record, self.data_schema)
        # parsed["aaa"] = parsed["SSS"] + parsed["sdf"]
        return parsed

    def read_one_sample_data_v2(self, record):
        # parsed = tf.parse_single_example(record, self.data_schema)
        parsed = tf.parse_example(record, self.data_schema)
        return parsed

    def define_data_schema(self):
        self.tensor_names2id = {name: id for id, name in enumerate(self.tensor_names)}
        self.data_schema = {
            "uin_vid_id": tf.FixedLenFeature(shape=(self.uin_max_vid_len), dtype=tf.int64),
            "uin_vid_pos_id": tf.FixedLenFeature(shape=(self.uin_max_vid_len), dtype=tf.int64),
            "uin_vid_cat1_id": tf.FixedLenFeature(shape=(self.uin_max_vid_len), dtype=tf.int64),
            "uin_vid_cat2_id": tf.FixedLenFeature(shape=(self.uin_max_vid_len), dtype=tf.int64),
            "uin_vid_day_id": tf.FixedLenFeature(shape=(self.uin_max_vid_len), dtype=tf.int64),  # 周几--点击
            "uin_vid_hour_id": tf.FixedLenFeature(shape=(self.uin_max_vid_len), dtype=tf.int64),  # 几点--点击
            "uin_vid_week_id": tf.FixedLenFeature(shape=(self.uin_max_vid_len), dtype=tf.int64),
            "uin_vid_len": tf.FixedLenFeature(shape=(), dtype=tf.int64),
            "uin_age_id": tf.FixedLenFeature(shape=(), dtype=tf.int64),
            "uin_gender_id": tf.FixedLenFeature(shape=(), dtype=tf.int64),
            "uin_language_id": tf.FixedLenFeature(shape=(), dtype=tf.int64),
            "uin_platform_id": tf.FixedLenFeature(shape=(), dtype=tf.int64),
            "uin_grade_id": tf.FixedLenFeature(shape=(), dtype=tf.int64),

            "vid_id": tf.FixedLenFeature(shape=(), dtype=tf.int64),
            "vid_cat1_id": tf.FixedLenFeature(shape=(), dtype=tf.int64),
            "vid_cat2_id": tf.FixedLenFeature(shape=(), dtype=tf.int64),
            "vid_day_id": tf.FixedLenFeature(shape=(), dtype=tf.int64),  # 周几--曝光(推送近似)
            "vid_hour_id": tf.FixedLenFeature(shape=(), dtype=tf.int64),  # 几点--曝光(推送近似)
            "vid_week_id": tf.FixedLenFeature(shape=(), dtype=tf.int64),
            "label": tf.FixedLenFeature(shape=(), dtype=tf.float32),

            "all_vid_id": tf.FixedLenFeature(shape=(self.uin_max_vid_len + 1), dtype=tf.int64),
            "feat_index": tf.FixedLenFeature(shape=(self.field_size), dtype=tf.int64),
            "feat_value": tf.FixedLenFeature(shape=(self.field_size), dtype=tf.float32),
            # "context_emb": tf.FixedLenFeature(shape=(self.field_size), dtype=tf.float32),
        }


if __name__ == '__main__':
    dataset = SampleDataTensorflow(DataDealArgs())
    count = 0
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for batch_data, count in dataset.get_batch_data(sess, count, dataset_type=1):
            print(batch_data)
            continue
