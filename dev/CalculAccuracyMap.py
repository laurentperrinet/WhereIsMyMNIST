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
args


import datetime


date = str(datetime.datetime.now())

f = open('AccuracyMap_{}_{}h{}.txt'.format(date[0:10], date[11:13], date[14:16]), "w+")

model = torch.load("../data/MNIST_cnn_0.1_0.1_0.75_0.7.pt")
accuracy_map = torch.zeros(55,55)
for i_offset in range(-1, 2):
    for j_offset in range(-1, 22):
        transform = transforms.Compose([
            WhatShift(i_offset=i_offset, j_offset=j_offset),
            WhatBackground(),
            transforms.ToTensor(),
            # transforms.Normalize((args.mean,), (args.std,))
        ])
        dataset_test = datasets.MNIST('../data',
                                      train=False,
                                      download=True,
                                      transform=transform,
                                      )
        test_loader = torch.utils.data.DataLoader(dataset_test,
                                                  batch_size=args.minibatch_size,
                                                  shuffle=True)
        whatTrainer = WhatTrainer(args, model=model, test_loader=test_loader)
        acc = whatTrainer.test()
        # print(acc)
        # accuracy_map[26-i_offset][26-j_offset] = acc
        f.write(str(acc))
    f.write('\n')
f.close()

