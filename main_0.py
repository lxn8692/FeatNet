import re
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")

np_str_obj_array_pattern = re.compile(r'[SaUO]')
from Utils.HyperParamLoadModule import *

print("main.py:pid:" + os.getpid().__str__())
from sklearn.metrics import roc_auc_score, log_loss
import numpy as np
import multiprocessing

from datetime import datetime
from DataSource.WXBIZEmbed import *
print("main.py:pid:" + os.getpid().__str__())
from torch.utils.tensorboard import SummaryWriter
from Data.Dataset import *
from Utils.HyperParamLoadModule import *
from Utils.loss import *
from Models.BaseModelV2 import BaseModelV2
import sys
sys.path.append("./Models/AuotInt/")

def load_model(featureInfo):
    preTrainModel = os.path.join(Config.absPath, Config.savedModelPath, Config.preTrainModelName)
    print(preTrainModel)
    if Config.buildState != BUILDTYPE.TRAIN and os.path.exists(preTrainModel) is False:
        raise Exception('running without model path!')
    # 加载模型（反射）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f'GPU:  {torch.cuda.get_device_name(0)}')
        torch.cuda.set_device(0)

    embedFormat = eval(f"{Config.datasetType.name}{Config.embedType}.{Config.datasetType.name}{Config.embedType}")(
        BaseEmbedPack(HyperParam.popCatFeature, HyperParam.popSeqFeature, HyperParam.popDims2DimFeature, featureInfo))
    model: BaseModelV2 = eval(f'{Config.modelName}.{Config.modelName}')(featureInfo, embedFormat).to(device)

    loss_fn = eval(Config.lossType)().to(device)
    optimizer = torch.optim.Adam(model.buildParameterGroup(), lr=HyperParam.LR, weight_decay=HyperParam.L2)
    epoch ,early_stop_cnt , best_auc = 0 , Config.earlyStop , 0.0
    # 初始化参数/加载参数
    if (Config.loadPreTrainModel is True and os.path.exists(preTrainModel)):
        save_info = torch.load(preTrainModel)
        optimizer.load_state_dict(save_info['optimizer'])
        model.load_state_dict(save_info['model'])
        epoch = save_info['epoch'] + 1
        best_auc = save_info['best_auc']
        early_stop_cnt = save_info['early_stop_cnt']
        print("model loaded !")
    return model, loss_fn, optimizer, device,epoch,best_auc,early_stop_cnt


def model_train(model, loss_fn, optimizer, metrics: list, dataset: DatasetV2, epoch: int, device,best_auc,early_stop_cnt):
    path = os.path.join(Config.datasetPath, Config.logPath, Config.modelName, f"lr={HyperParam.LR},l2={HyperParam.L2}",
                        f'{datetime.now().timestamp().__str__()}')
    print(path)
    print(multiprocessing.cpu_count())
    writer = SummaryWriter(path)
    print(f" chmod -R 777 {os.path.join(Config.datasetPath, Config.logPath)}")
    os.system(f" chmod -R 777 {os.path.join(Config.datasetPath, Config.logPath)}")
    res_cnt = early_stop_cnt

    ''' 统计每个子集中的特征个数 '''
    if Config.modelName == "FilterV4":
        setFeature = []

        model.eval()
        cross = model.crossLayer
        indicator = torch.sigmoid(cross.pruningWeight)
        weight = torch.relu(cross.param - indicator)
        weight = weight.squeeze(1).squeeze(-1)

        setFeature.append(weight.tolist())

    for step in range(epoch,Config.epoch):
        start_time = time.time()
        print(f'epoch:{step}\n')
        bat = 0
        for feed_dict, k in dataset.train.getBatchData():
            bat += 1
            if bat % 2000 == 0:
                print(f"--training--{bat}--")
            end = datetime.now()
            device_dict = {}
            for key, val in feed_dict.items():
                device_dict[key] = torch.as_tensor(val).to(device)

            model.train()
            optimizer.zero_grad()
            prediction = model(device_dict)
            loss: torch.Tensor = loss_fn(prediction.squeeze(-1), device_dict['label'].squeeze(-1))
            lossAux = loss + model.getAuxLoss()
            lossAux.backward()
            optimizer.step()

            del device_dict

        ''' 统计每个子集中的特征个数 '''
        if Config.modelName == "FilterV4":
            model.eval()
            cross = model.crossLayer
            indicator = torch.sigmoid(cross.pruningWeight)
            weight = torch.relu(cross.param - indicator)
            weight = weight.squeeze(1).squeeze(-1)

            setFeature.append(weight.tolist())

        aucVal, logloss = model_test(model, dataset.test, loss_fn, True)

        save(os.path.join(Config.absPath,
                            Config.savedModelPath) + f"main_0_{Config.modelName}lr={HyperParam.LR},l2={HyperParam.L2}.pt",
                optimizer, model,res_cnt,best_auc,step)
        print(f"val auc: {aucVal} , logloss: {logloss}")
        end_time = time.time()
        print(f"epoch {step} cost {end_time - start_time}s")
        if aucVal > best_auc:
            print("find a better model!")
            save(os.path.join(Config.absPath,
                            Config.savedModelPath) + f"main_0_{Config.modelName}lr={HyperParam.LR},l2={HyperParam.L2}_best.pt",
                optimizer, model,res_cnt,best_auc,step)
            best_auc = aucVal
            res_cnt = Config.earlyStop
        else:
            res_cnt -= 1
            if res_cnt == 0:
                print("trigging the early stop!")
                break
        writer.add_scalar('auc/globalTest', aucVal, global_step=step, walltime=None)
        writer.add_scalar('loss/globalTest', logloss, global_step=step, walltime=None)

    if Config.modelName == "FilterV4":
        np.save("setFeature_0.npy", setFeature)

    save_info = torch.load(
        os.path.join(Config.absPath,
                            Config.savedModelPath) + f"main_0_{Config.modelName}lr={HyperParam.LR},l2={HyperParam.L2}_best.pt"
    )
    model.load_state_dict(save_info['model'])
    aucVal, meanLoss = model_test(model, dataset.test, loss_fn, False)
    print(f"test auc: {aucVal} , meanLoss: {meanLoss}")


def model_test(model, dataset: BaseDataFormat, loss_fn, pinMemoryData=True): # True for val , False for test
    with torch.no_grad():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            print(f'GPU:  {torch.cuda.get_device_name(0)}')
            torch.cuda.set_device(0)
        model.eval()
        val = []
        truth = []
        if pinMemoryData:
            data = dataset.getBufferData()
        else:
            data = dataset.getBatchData()
        for feed_dict, _count in data:
            new_feed = {}
            for i, v in feed_dict.items():
                new_feed[i] = torch.as_tensor(v).to(device)
            prediction = model(new_feed)
            val.append(prediction.cpu().numpy())
            truth.append(new_feed['label'].cpu().numpy())
        try:
            auc = roc_auc_score(np.concatenate(truth, axis=0).squeeze(), np.concatenate(val, axis=0).squeeze())
            y_hat = np.concatenate(val, axis=0).squeeze()
            y = np.concatenate(truth, axis=0).squeeze()
            loss = - np.sum(y*np.log(y_hat + 1e-6) + (1-y) * np.log(1 - y_hat + 1e-6)) / len(y)
        except ValueError:
            auc = -1
    return auc, loss


def save(save_path, optimizer, model,early_stop_cnt,best_auc,epoch):
    assert (save_path is not None)
    save_info = {
        'optimizer': optimizer.state_dict(), 'model': model.state_dict() , 'early_stop_cnt': early_stop_cnt , 'best_auc':best_auc,'epoch':epoch
    }
    torch.save(save_info, save_path)
    print(f'model saved in {save_path}')

def main_process(featureInfo):

    model, loss_fn, optimizer, device , epoch , best_auc , early_stop_cnt = load_model(featureInfo)

    metrics = ['HR@5', 'NDCG@5', 'HR@10', 'NDCG@10', 'CNDCG@10', 'CNDCG@5', 'UNDCG@10', 'UNDCG@5']

    best_HR5, best_scores, pre_HR5, rank_distribution, counter = 0, {}, 0, {}, {}
    dataset = DatasetV2(Config.datasetType.name, HyperParam.batchSize, prefetch=HyperParam.prefetch)

    if Config.buildState == BUILDTYPE.TRAIN:
        model_train(model, loss_fn, optimizer, metrics, dataset, epoch, device=device,best_auc=best_auc,early_stop_cnt=early_stop_cnt)
    if Config.buildState == BUILDTYPE.TEST:
        model_test(model, dataset.test, loss_fn, False)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True

# def cosSim(vec1, vec2):
#     down = np.linalg.norm(vec1) * np.linalg.norm(vec2)
#     if down == 0:
#         return 0.0
#     else:
#         return np.dot(vec1, vec2) / down
#
# def setSim(set1, set2):
#     cnt = 0
#     for i in range(len(set1)):
#         if set1[i] == set2[i]:
#             cnt += 1
#     return cnt/len(set1)
#
# def cal():
#     setFeature = np.load("setFeature_0.npy")
#
#     train_epoch = setFeature.shape[0]
#     bucket = setFeature.shape[1]
#     feature = setFeature.shape[2]
#
#     x = [i for i in range(train_epoch)]
#
#     none_zero_feature = []
#
#     for i in range(bucket):
#         feature_list = []
#         for j in range(train_epoch):
#             cnt = 0
#             for k in range(feature):
#                 if setFeature[j][i][k] != 0:
#                     cnt += 1
#             feature_list.append(cnt)
#         none_zero_feature.append(feature_list)
#
#     final_feature_num = []
#     for i in none_zero_feature:
#         final_feature_num.append(i[-1])
#     print(f"所有bucket中保留特征数量最多的有{max(final_feature_num)}个，最少的有{min(final_feature_num)}个，平均每个bucket有{sum(final_feature_num)/len(final_feature_num)}个")
#
#     #-------------------------------------------------------------
#     # plt.figure()
#     #
#     # for y in none_zero_feature:
#     #     plt.plot(x, y)
#     #
#     # legends = [f"bucket {i}" for i in range(bucket)]
#     # plt.legend(legends)
#     # # plt.title("每个bucket中特征数量的变化")
#     #--------------------------------------------------------------
#     init_feature = setFeature[0]
#     end_feature = setFeature[train_epoch-1]
#     x0 = [i for i in range(feature)]
#     x1 = [i+0.3 for i in x0]
#
#     cof = 0
#     for i in range(bucket):
#         avg = sum(end_feature[i])/len(end_feature[i])
#         print(f"bucket {i} 中平均特征权重为{avg}")
#         cof += avg
#     print(f"所有bucket中的平均权重为{cof/bucket}")
#
#     # for i in range(bucket):
#     #     plt.figure()
#     #     # plt.bar(x0, init_feature[i], width=0.3)
#     #     # plt.bar(x1, end_feature[i], width=0.3)
#     #     plt.plot(x0, init_feature[i])
#     #     plt.plot(x1, end_feature[i])
#     #     plt.title(f"bucket {i}")
#     #     plt.legend(["init_weight", "trained_weight"])
#     #     # plt.title(f"训练开始和结束第{i}个bucket中特征权重的对比")
#
#     one_hot_feature = []
#
#     for i in range(bucket):
#         one_hot = []
#         for j in end_feature[i]:
#             if j > 0:
#                 one_hot.append(1)
#             else:
#                 one_hot.append(0)
#         one_hot_feature.append(one_hot)
#
#     one_hot_sim = []
#     print("-----------------------------one hot sim-----------------------------")
#     for i in range(bucket):
#         for j in range(i+1, bucket):
#             bucket_sim = setSim(one_hot_feature[i], one_hot_feature[j])
#             print(f"bucket {i} 和 bucket {j} 最终的相似度：{bucket_sim}")
#             one_hot_sim.append(bucket_sim)
#     print(f"所有的bucket之间相似性最大为{max(one_hot_sim)}，最小为{min(one_hot_sim)}，平均相似性为{sum(one_hot_sim)/len(one_hot_sim)}")
#
#     print("-----------------------------weight sim-----------------------------")
#
#     final_bucket_sim = []
#     for i in range(bucket):
#         for j in range(i+1, bucket):
#             bucket_sim = cosSim(end_feature[i], end_feature[j])
#             print(f"bucket {i} 和 bucket {j} 最终的相似度：{bucket_sim}")
#             final_bucket_sim.append(bucket_sim)
#     print(f"所有的bucket之间相似性最大为{max(final_bucket_sim)}，最小为{min(final_bucket_sim)}，平均相似性为{sum(final_bucket_sim)/len(final_bucket_sim)}")
#
#
#     init_weight_list = []
#     end_weight_list = []
#     for i in range(feature):
#         cnt1 = 0
#         cnt2 = 0
#         for j in range(bucket):
#             if init_feature[j][i] != 0:
#                 cnt1 += 1
#             if end_feature[j][i] != 0:
#                 cnt2 += 1
#         init_weight_list.append(cnt1)
#         end_weight_list.append(cnt2)
#
#     #--------------------------------------------------------------
#     # plt.figure()
#     # plt.bar(x0, init_weight_list, width=0.3)
#     # plt.bar(x1, end_weight_list, width=0.3)
#     # plt.title("init_weight & end_weight")
#     #
#     # plt.figure()
#     # plt.bar(x0, init_weight_list)
#     # plt.title("init_weight")
#     #
#     # plt.figure()
#     # plt.bar(x1, end_weight_list)
#     # plt.title("end_weight")
#
#     # plt.show()
#     #--------------------------------------------------------------
#
#     print(f"最终剩余的特征数量占比：{(len(end_weight_list)-end_weight_list.count(0)) * 100/len(end_weight_list)}%")


if __name__ == '__main__':
    '''
    python3 -u ./main.py TopDownIADNN  
    cephfs/group/wxplat-wxbiz-offline-datamining/evanxcwang/daily_train/Criteo/ Criteo  0.0001 0.00001 512 1000
    '''
    tf.random.set_seed(2022)
    setup_seed(2022)
    torch.multiprocessing.set_start_method("spawn")
    now = datetime.now()
    abs_address = "./Config/FeatureConfig.json"
    featureInfo = loadArgs(abs_address)
    print(Config)
    HyperParam.AutoPruningFeatureL0 = 0.001

    main_process(featureInfo)

    # if Config.modelName == "FilterV4":
    #     cal()
