import torch
from torchvision import datasets, transforms
import sys

sys.path.append("../figures")
from robust_what import WhatShift, WhatBackground, WhatNet, WhatTrainer, What, train, test, MNIST
import datetime
from main import init

args = init(filename='../data/2019-06-12')
args

debut = datetime.datetime.now()
date = str(debut)

reseau = "MNIST_cnn_robust_what_0.1_0.1_1.0_0.7_5epoques_2019-06-19_11h50.pt"
borne = 27

f = open('AccuracyLign_--{}--_{}_{}h{}.txt'.format(reseau[0:-3], date[0:10], date[11:13], date[14:16]), "w+")
compteur = 1

ligne_test_posterior = ''
ligne_correct = ''

model = torch.load("../data/" + reseau)
for i_offset in range(-borne, borne + 1):
    transform = transforms.Compose([
        WhatShift(args, i_offset=i_offset, j_offset=0),
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
    test_posterior, correct = whatTrainer.posteriorTest()

    print("Avancement : ", float(int((compteur / (2 * borne + 1) * 100) * 100)) / 100, "%")
    compteur += 1
    # print(acc)
    # accuracy_map[26-i_offset][26-j_offset] = acc

    ligne_test_posterior += str(int(test_posterior * 10000) / 10000) + " "
    ligne_correct += str(correct) + " "

f.write(ligne_test_posterior + "\n" + ligne_correct)
f.write("\n1ere ligne : ligne_test_posterior; 2eme ligne : ligne_correct")

fin = datetime.datetime.now()

f.write("\nDuree d'execution : " + str(fin - debut))

f.close()
