from typing import List

from torch import nn

from Data.Input import TFRecordInput
import os
import torch
import builtins
from tqdm import tqdm


class LoadOutEmbed():
    def __init__(self, absPath: str, filePath, outFileName=None):
        fileList = os.listdir(os.path.join(absPath, filePath, "Data"))
        if "_SUCCESS" in fileList:
            fileList.remove('_SUCCESS')
        if "_DS.STORE" in fileList:
            fileList.remove(".DS_Store")
        print(fileList)
        self.outFileName = outFileName
        self.fileNameList = fileList
        self.filePath = filePath
        self.absPath = absPath
        self.dataset = TFRecordInput(absPath, filePath, fileList, numWorker=0, batchSize=1)
        self.str2idxList = self.buildList()

    def list2str(self, data: torch.Tensor):
        data = data.detach().tolist()[0]
        # print(data)
        _str = ""
        for i in data:
            _str = _str + chr(i)
        return _str

    def buildList(self):
        nameList = [[self.list2str(i['vid']), i['tf_val'], i['vid_emb']] for i in iter(self.dataset.loader)]
        temp = sorted(nameList, key=lambda x: (x[1], x[0]))
        return temp

    def updateEmbed(self, embed: nn.Embedding):
        for i, embed in enumerate(embed.weight):
            self.str2idxList[i][2] = embed

    ## f"vid:{i[0]}|tf_val:{i[1]}|cat1_id:0|cat2_id:0|vid_emb:{i[2]}\n"
    def savedEmbed(self):
        with open(self.outFileName, 'w+', encoding='utf-8') as file:
            for i in tqdm(self.str2idxList):
                embed = "|".join(list(map(builtins.str, i[2].tolist())))
                frequency = float(float(i[1]))
                _str = f"{i[0]},0,0,{float(frequency)},{embed}\n"
                file.write(_str)


if __name__ == '__main__':
    fileList = os.listdir(os.path.join("/Users/ivringwang/Desktop/tencent/GMM_torch", "DataSet/vid_emb_tfdata", "Data"))
    absPath = "/Users/ivringwang/Desktop/tencent/GMM_torch"
    filePath = "DataSet/vid_emb_tfdata"
    data = LoadOutEmbed(absPath, filePath, fileList)
