import numpy as np
from enum import Enum
from sklearn.metrics import roc_auc_score


class MetricType(Enum):
    AP = 1
    NDCG = 2
    AUC = 3
    RECALL = 4
    PRECISION = 5


class Auc(object):
    def __init__(self, num_buckets):
        self._num_buckets = num_buckets
        self._table = np.zeros(shape=[2, self._num_buckets])

    def Reset(self):
        self._table = np.zeros(shape=[2, self._num_buckets])

    def Update(self, labels: np.ndarray, predicts: np.ndarray):
        """
        :param labels: 1-D ndarray
        :param predicts: 1-D ndarray
        :return: None
        """
        labels = labels.astype(np.int)
        predicts = self._num_buckets * predicts

        buckets = np.round(predicts).astype(np.int)
        buckets = np.where(buckets < self._num_buckets,
                           buckets, self._num_buckets-1)

        for i in range(len(labels)):
            self._table[labels[i], buckets[i]] += 1

    def Compute(self):
        tn = 0
        tp = 0
        area = 0
        for i in range(self._num_buckets):
            new_tn = tn + self._table[0, i]
            new_tp = tp + self._table[1, i]
            # self._table[1, i] * tn + self._table[1, i]*self._table[0, i] / 2
            area += (new_tp - tp) * (tn + new_tn) / 2
            tn = new_tn
            tp = new_tp
        if tp < 1e-3 or tn < 1e-3:
            return -0.5  # 样本全正例，或全负例
        return area / (tn * tp)



class Metrics(object):

    @staticmethod
    def AUC(self, label, prediction):
        return roc_auc_score(label, prediction)

    @staticmethod
    def simple_metric(r: np.ndarray, k: int, metric_type: MetricType, totalLen=None) -> np.ndarray:
        if metric_type == MetricType.AP:
            return np.where(r <= k, 1 / r, 0)
            # return 1 / r if r <= k else 0
        elif metric_type == MetricType.NDCG:
            return np.where(r <= k, 1 / np.log2(r + 1), 0)
        # return 1 / np.log2(r+1) if r <= k else 0
        elif metric_type == MetricType.RECALL:
            return np.where(r <= k, 1, 0)
            # return 1 if r <= k else 0
        elif metric_type == MetricType.AUC:
            if totalLen is None:
                raise Exception('totoLen=null when using AUC!')
            return (totalLen - r) / (totalLen - 1)
        elif metric_type == MetricType.PRECISION:
            return np.where(r <= k, 1 / r, 0)
            # return 1 / k if r <= k else 0
        else:
            raise Exception('wrong Metric Type', metric_type)
