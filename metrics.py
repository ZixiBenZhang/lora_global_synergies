from torchmetrics import Metric
from torchmetrics.text.rouge import ROUGEScore

import re


class MyRouge(Metric):
    def __init__(self, **kwargs):
        super().__init__()
        self.rouge = ROUGEScore(**kwargs)

    def update(self, preds, target):
        self.rouge.update(preds, target)

    def compute(self):
        res_dict = self.rouge.compute()
        res = {}
        for k, v in res_dict:
            if re.fullmatch(r"rouge\d_recall", k) is not None:
                res[k.split("_")[0]] = v
            elif re.fullmatch(r"rougeL.*_fmeasure", k) is not None:
                res[k.split("_")[0]] = v
        return res

    def plot(self, val, ax):
        return self.rouge.plot(val, ax)
