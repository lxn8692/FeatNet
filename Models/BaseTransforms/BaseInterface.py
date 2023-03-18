from abc import ABCMeta, abstractmethod


class BaseTransform(ABCMeta):
    @abstractmethod
    def initParams(self):
        pass
