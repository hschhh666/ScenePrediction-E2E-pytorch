# -*- coding: utf-8 -*- 

'''
程序说明：自定义loss
'''

import numpy as np
from cv2 import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import time
from datetime import datetime
from StateMapDataset import  FakeDataSet ,FakeAvgDataset, FakeSinglePairDataset
import os,sys
from logger import Logger
from AutoEncoder import BehaviorModelAutoEncoder
import itertools


def abs_loss(x,y):
    loss = torch.abs(x-y)
    return loss