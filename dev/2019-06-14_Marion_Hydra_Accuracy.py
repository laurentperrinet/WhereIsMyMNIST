import torch
from torchvision import transforms

import sys
sys.path.append("../figures")

from robust_what import WhatShift, WhatBackground, WhatNet, WhatTrainer, What, train, test, MNIST

from main import init
args = init(filename='../data/2019-06-12')
args

model = torch.load("../data/MNIST_cnn_robust_what_0.1_0.1_1.0_0.7_5epoques_2019-06-14_15h00.pt")
transform = transforms.Compose([
    WhatShift(args, i_offset=None, j_offset=None),
    WhatBackground(),
    transforms.ToTensor(),
    # transforms.Normalize((args.mean,), (args.std,))
])
dataset_test = MNIST('../data',
                              train=False,
                              download=True,
                              transform=transform,
                              )
test_loader = torch.utils.data.DataLoader(dataset_test,
                                          batch_size=args.minibatch_size,
                                          shuffle=True)
whatTrainer = WhatTrainer(args, model=model, test_loader=test_loader)
acc = whatTrainer.test()