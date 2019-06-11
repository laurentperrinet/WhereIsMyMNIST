import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import datetime

import sys
sys.path.append("../figures")

from what import WhatShift, WhatBackground, WhatNet, WhatTrainer, What, train, test

from main import init
args = init(filename='../data/2019-06-05')

args.epochs = 60
args.noise = 1
args.save_model = False

# Parametre modifie du reseau par rapport au reseau de base :
do_adam = False

# Entrainement du reseau :
what = What(args=args, force=True)




# Calcul de l'accuracy map :

debut = datetime.datetime.now()
date = str(debut)

# reseau = "MNIST_cnn_0.1_0.1_1_0.7.pt"
borne = 13
nomPartielFichier = "doAdamFalse"


# f = open('AccuracyMap_{}_{}_{}h{}.txt'.format(reseau[0:-3], date[0:10], date[11:13], date[14:16]), "w+")
f = open('AccuracyMap_{}_{}_{}h{}.txt'.format(nomPartielFichier, date[0:10], date[11:13], date[14:16]), "w+")
compteur = 0

# model = torch.load("../data/"+ reseau)
model = what.model
accuracy_map = torch.zeros(55,55)
for i_offset in range(-borne, borne + 1):
    for j_offset in range(-borne, borne + 1):
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

        print("Avancement : ", float(int((compteur/(2*borne+1)**2*100)*100))/100, "%")
        compteur += 1
        # print(acc)
        # accuracy_map[26-i_offset][26-j_offset] = acc
        f.write(str(acc)+' ')
    f.write('\n')

fin = datetime.datetime.now()

f.write("Duree d'execution : " + str(fin-debut))

f.close()

