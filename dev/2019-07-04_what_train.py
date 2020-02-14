import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

import sys
sys.path.append("../figures")

from what import WhatShift, WhatBackground, WhatNet, WhatTrainer, What, train, test

from main import init
args = init(filename='../data/2019-06-05')

args.epochs = 60
args.noise = 1
args.save_model = True
what = What(args=args, force= True)

