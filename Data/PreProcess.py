import os
import sys

os.system("pip install tfrecord")
sys.path.append("/cephfs/group/wxplat-wxbiz-offline-datamining/evanxcwang/ft_local/GMM_torch/")

import json

import tfrecord
import stat


from Utils.HyperParamLoadModule import Config, EnumEncoder


## 修改训练集合和测试集合
def generateIndex(path, fileList):
    for i in fileList:
        # create_index(os.path.join(path, i), os.path.join(path, 'index/', f'{i}.index'))
        print(os.path.join(path, 'Data/', i))
        tfrecord.tools.create_index(f"{os.path.join(path, 'Data/', i)}",
                                    f"{os.path.join(path, 'Index/', f'{i}.index')}")
        # os.system(
        #     f"python3 -m tfrecord.tools.tfrecord2idx {os.path.join(path, 'Data/', i)} {os.path.join(path, 'Index/', f'{i}.index')}")


def createDataIdxFolder(datasetPath, softLinkPath):
    ## 删除旧的文件夹

    # os.system(f"ln -svf {datasetPath} {os.path.join(softLinkPath, 'Data')}")
    fileList = os.listdir(os.path.join(softLinkPath, 'Data/'))
    if "_SUCCESS" in fileList:
        fileList.remove("_SUCCESS")
    if ".DS_Store" in fileList:
        fileList.remove(".DS_Store")
    print(fileList)
    ## 创建索引
    generateIndex(softLinkPath, fileList)


if __name__ == '__main__':
    # 划分DataIndex
    # path = sys.argv[0]
    # datasetPath = "/cephfs/group/wxplat-wxbiz-offline-datamining/zhiyuanxu/video_rec/data/hourly_train_data_for_din_xdeepfm_model_detail_info_after_deal"
    # softLinkPath = "/cephfs/group/wxplat-wxbiz-offline-datamining/evanxcwang/daily_train"
    path = "dataset/"
    datasetPath = "/Users/ivringwang/Desktop/tencent/GMM_torch/DataSet"
    softLinkPath = "/Users/ivringwang/Desktop/tencent/GMM_torch/test"
    # os.system(f"cp -r {softLinkPath}/tmp {softLinkPath}/{path}")
    # os.chmod(f"{os.path.join(softLinkPath, path)}", stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    softLinkPath = os.path.join(softLinkPath, path)
    datasetPath = os.path.join(datasetPath, path)
    createDataIdxFolder(os.path.join(datasetPath, "test_tfdata"), os.path.join(softLinkPath, "test_tfdata"))
    # createDataIdxFolder(os.path.join(datasetPath, "train_tfdata"), os.path.join(softLinkPath, "train_tfdata"))
    # createDataIdxFolder(os.path.join(datasetPath, "val_tfdata"), os.path.join(softLinkPath, "val_tfdata"))
    # createDataIdxFolder(os.path.join(datasetPath, "vid_emb_tfdata"), os.path.join(softLinkPath, "vid_emb_tfdata"))

    # 修改配置文件
    # with open(os.path.join(softLinkPath, "Config/FeatureConfigAutoGau.json"), "r", encoding="utf-8") as config:
    # with open("/Users/ivringwang/Desktop/tencent/GMM_torch/Config/FeatureConfig.json", "r+",
    #           encoding="utf-8") as config:
    #     json.load(config, object_hook=Config.hooker)
    #     config.seek(0)
    #     config.truncate()
    #     Config.datasetName = f"{path}"
    #     a = Config.keys()
    #     json.dump(a, config)
