from Data.Input import FILETYPE
from Data.Dataset import getVidNumb

class TFRecordFeature():
    def __init__(self, dataIter):
        self.dataIter = dataIter

    def getNextBatch(self):
        return next(self.dataIter)

