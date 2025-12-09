import torch
import torch.nn as nn
from rscd.losses.loss_func import CELoss, FocalLoss, dice_loss, BCEDICE_loss
from rscd.losses.mask2formerLoss import Mask2formerLoss

#针对DMINet的
class myLoss(nn.Module):
    def __init__(self, param, loss_name=['CELoss'], loss_weight=[1.0], **kwargs):
        super(myLoss, self).__init__()
        self.loss_weight = loss_weight
        self.loss = list()

        for _loss in loss_name:
            self.loss.append(eval(_loss)(**param[_loss], **kwargs))

    def forward(self, preds, target):
        loss = 0
        # 支持 preds 是 tuple 或 list，或单个 tensor
        if isinstance(preds, (tuple, list)):
            # preds和self.loss长度一致，逐个对应计算损失
            for i in range(len(self.loss)):
                loss += self.loss[i](preds[i], target) * self.loss_weight[i]
        else:
            # 兼容 preds是单个tensor的情况
            loss += self.loss[0](preds, target) * self.loss_weight[0]
        return loss

#原来的
'''
class myLoss(nn.Module):
    def __init__(self, param, loss_name=['CELoss'], loss_weight=[1.0], **kwargs):
        super(myLoss, self).__init__()
        self.loss_weight = loss_weight

        self.loss = list()#原来的

        #原来的
        for _loss in loss_name:
            self.loss.append(eval(_loss)(**param[_loss],**kwargs))


    def forward(self, preds, target):
        loss = 0
        for i in range(0, len(self.loss)):
            loss += self.loss[i](preds, target) * self.loss_weight[i]

        return loss
'''


def build_loss(cfg):
    loss_type = cfg.pop('type')
    obj_cls = eval(loss_type)
    obj = obj_cls(**cfg)
    return obj
