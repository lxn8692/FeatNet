# _*_coding:utf-8_*_
import os
import sys
import subprocess
import time
from Utils.log import *
import hashlib
import codecs

tools_path = "/cephfs/group/wxplat-wxbiz-offline-datamining/evanxcwang/tools"


# hdfs交互的工具

def upload_hdfs(local_path, hdfs_path):
    sh = '/bin/bash %s/upload_data.sh %s %s' % (tools_path, local_path, hdfs_path)
    print(sh)
    p = subprocess.Popen(sh, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ret = p.wait()
    print(p.stdout.read())
    print(p.stderr.read())
    print(ret)


def chmod_local_path(path):
    sh = 'chmod -R 777 %s' % (path)
    print(sh)
    p = subprocess.Popen(sh, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ret = p.wait()
    print(p.stdout.read())
    print(p.stderr.read())
    print(ret)


# 上传pb模型数据
def upload_model(model_local_path, model_name, model_hdfs_path, vid_hdfs_path):
    model_base_path = model_local_path
    vid_hdfs_path = vid_hdfs_path
    model_name = model_name
    model_hdfs_path = model_hdfs_path

    # 模型文件
    local_path = os.path.join(model_local_path, "%s.onnx" % (modelName))
    hdfs_path_files = "%s/" % (model_hdfs_path)
    logger.info("local model path:%s" % local_path)
    logger.info("hdfs model path:%s" % hdfs_path_files)
    upload_hdfs(local_path, hdfs_path_files)

    # vid文件
    local_path = "%s /vid_info_summary.txt" % (model_base_path)
    hdfs_path_files = "%s/%s_vid_info_summary.txt" % (vid_hdfs_path, model_name)
    logger.info("local vid map path:%s" % local_path)
    logger.info("hdfs vid map path:%s" % hdfs_path_files)
    upload_hdfs(local_path, hdfs_path_files)


def add_format2tfsvrmodel(ModelExportPathWithVersion, RTXName, Overwrite=False):
    def get_file_md5(fname):
        m = hashlib.md5()
        with open(fname, 'rb') as fobj:
            while True:
                data = fobj.read(4096)
                if not data:
                    break
                m.update(data)

        return m.hexdigest()

    CheckFileOut = []
    for home, dirs, files in os.walk(ModelExportPathWithVersion):
        for filename in files:
            print("%s md5sum:(%s)" % (os.path.join(home, filename), get_file_md5(os.path.join(home, filename))))
            CheckFileOut.append((os.path.join(home, filename), get_file_md5(os.path.join(home, filename))))

    CheckFile = os.path.join(ModelExportPathWithVersion, "check.file")
    CheckFileWriter = codecs.open(CheckFile, 'w', encoding='utf-8')
    CheckFileWriter.write("%s\n" % (RTXName))
    for (file, md5sum_v) in CheckFileOut:
        CheckFileWriter.write("%s\t%s\n" % (file, md5sum_v))
    CheckFileWriter.close()


if __name__ == '__main__':
    timeStamp = int(round(time.time()))
    modelName = sys.argv[1]
    datasetName = sys.argv[2]
    datasetPath = sys.argv[3]
    vidInfoPath = os.path.join(sys.argv[4], datasetName)
    onnxhdfsPath = os.path.join(sys.argv[4], datasetName, modelName, str(timeStamp))
    savedPath = os.path.join(datasetPath, datasetName, "savedModel/", modelName, )
    print(savedPath, onnxhdfsPath)
    add_format2tfsvrmodel(savedPath, "evanxcwang")
    upload_model(savedPath, modelName, onnxhdfsPath, vidInfoPath)
