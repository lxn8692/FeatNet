# -*- coding: utf-8 -*-
from Utils.HyperParamLoadModule import *
from DataSource.MV.MVData import *
import logging
import os

# for tensorflow env
try:
    import tensorflow as tf

    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%d-%m-%Y:%H:%M:%S')
    logging.getLogger().setLevel(logging.DEBUG)
    logger = logging.getLogger()
except:
    pass


class MVEmbed(BaseEmbedFormat):
    def __init__(self, args: BaseEmbedPack):
        self.dataPath = Config.datasetPath
        self.features = self.build_feature_meta(
            os.path.join(self.dataPath, 'feature.json'))
        self.seqfeatures = ['movie_title','genre']
        self.dep = ['weekday','hour']

    def lookup(self, featureName: str, content, embedding, device):
        temp = embedding[featureName](content.type(torch.LongTensor).to(device))
        return temp
    
    def seqlookup(self, featureName: str, content, embedding, device):
        #[b , s]
        masked = content != 0
        #[b , s , 1]
        masked = masked.float().unsqueeze(2)
        #[b , 1]
        avg = torch.sum(masked , dim= 1)
        #[b , s , d]
        tmp = self.lookup(featureName, content ,embedding, device)
        
        return torch.div( torch.sum(tmp , dim=1) , avg )
        

    def preProcess(self, rawData: dict, embedding, ):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        CatFeature = {key: DataPack(name=key, fieldType=FIELDTYPE.CAT,
                                    data=self.lookup(key, value,
                                                     embedding, device))
                      for key, value in rawData.items() if key != 'label' and key not in self.seqfeatures and key not in self.dep}
        #SeqFeature = {key:DataPack(name=key,fieldType=FIELDTYPE.CAT ,data=self.seqlookup(key, rawData[key],embedding, device)) for key in self.seqfeatures}
        #CatFeature.update(SeqFeature)
        return None, None, CatFeature

    def buildEmbedding(self):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        embedding: nn.ModuleDict[str:nn.Embedding] = nn.ModuleDict()
        for feature, numb in self.features.items():
            embedding[feature] = nn.Embedding(numb + 1, HyperParam.AutoIntFeatureDim).to(
                device=device)
        # load pretrain Model
        if Config.loadPreTrainModel:
            pass
        # due with share embedding
        else:
            with torch.no_grad():
                for key, value in embedding.items():
                    nn.init.xavier_normal(value.weight, gain=1.414)

        return embedding

    def build_feature_meta(self, features_meta_file='feature.json'):
        ans = None
        with open(features_meta_file) as f:
            ans = json.load(f)
        return ans

    # def loadEmbed(self, savedPath, model):
    #     if os.path.exists(savedPath):
    #         save_info = torch.load(savedPath)
    #         assert 'model' in save_info.keys()
    #         model.load_state_dict(save_info['model'])
    #         self.buildShareEmbed()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    abs_address = '/Users/ivringwang/Desktop/tencent/GMM_torch/Config/FeatureConfig.json'
    featureInfo = loadArgs(abs_address)
    path = "/DataSource/Avazu/all_data.csv"
    test = MVData(path)
    dataIter = test.getBatchData()
    embed = MVEmbed([], [], [], featureInfo=None)
    embedding = embed.buildEmbedding()
    for i, count in dataIter:
        for j in i.keys():
            i[j] = torch.as_tensor(i[j]).to(device)
        result = embed.preProcess(i, embedding)
        print(result)
