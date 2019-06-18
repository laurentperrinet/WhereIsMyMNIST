import sys
sys.path.append("../figures")

import torch
from torchvision import transforms

from robust_what import WhatShift, WhatBackground, WhatNet, WhatTrainer, What, train, test, MNIST

from main import init
args = init(filename='../data/2019-06-12')

args.epochs = 2
args.save_model = True

args.lr = 1
args.do_adam = 'adadelta'

args.what_offset_std = 3.0
what = What(args=args, force=False)




model = what.model
transform = transforms.Compose([
    WhatShift(args),
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
if True :
    print("test_loader")
    whatTrainer = WhatTrainer(args, model=model, test_loader=test_loader)
    acc = whatTrainer.test()
    print("what.test_loader")
    whatTrainer = WhatTrainer(args, model=model, test_loader=what.test_loader)
    acc = whatTrainer.test()