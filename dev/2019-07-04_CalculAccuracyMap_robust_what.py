import torch
from torchvision import datasets, transforms



import sys
sys.path.append("../figures")
from robust_what import WhatShift, WhatBackground, WhatNet, WhatTrainer, What, train, test, MNIST

from main import init
args = init(filename='../data/2019-06-12')
args


import datetime

debut = datetime.datetime.now()
date = str(debut)

reseau = "MNIST_cnn_robust_what_0.1_0.1_1.0_0.7_5epoques_2019-06-19_11h50.pt"
borne = 27


f = open('AccuracyMap_--{}--_{}_{}h{}.txt'.format(reseau[0:-3], date[0:10], date[11:13], date[14:16]), "w+")
compteur = 1

model = torch.load("../data/"+ reseau)
accuracy_map = torch.zeros(55, 55)
for i_offset in range(-borne, borne + 1):
    for j_offset in range(-borne, borne + 1):
        transform = transforms.Compose([
            WhatShift(args, i_offset=i_offset, j_offset=j_offset),
            WhatBackground(contrast=args.contrast, noise=args.noise, sf_0=args.sf_0, B_sf=args.B_sf, seed=args.seed),
            transforms.ToTensor(),
            transforms.Normalize((args.mean,), (args.std,))
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

        print("Avancement : ", float(int((compteur/(2*borne+1)**2*100)*100))/100, "%")
        compteur += 1
        # print(acc)
        # accuracy_map[26-i_offset][26-j_offset] = acc
        f.write(str(acc)+' ')
    f.write('\n')

fin = datetime.datetime.now()

f.write("Duree d'execution : " + str(fin-debut))

f.close()

