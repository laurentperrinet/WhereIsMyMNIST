import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

import sys
sys.path.append("../figures")

from robust_what import WhatShift, WhatBackground, WhatNet, WhatTrainer, What, train, test, MNIST

from main import init
args = init(filename='../data/2019-06-12')

args.epochs = 60
args.save_model = True

args.lr = 1
args.do_adam = 'adadelta'

args.what_offset_std = 3.0
what = What(args=args, force=False)