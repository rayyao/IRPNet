import torch.nn as nn
import numpy as np
import  torch
from model.cross_modal_fusion import *


def SoftIoULoss(pred, target):
        # Old One
        pred = torch.sigmoid(pred)
        smooth = 1

        intersection = pred * target
        loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() -intersection.sum() + smooth)

        loss = 1 - loss.mean()

        return loss


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

