import torch
from torchvision import transforms
import sys
import datetime

sys.path.append("../figures")
from robust_what import WhatShift, WhatBackground, WhatNet, WhatTrainer, What, train, test, MNIST
from main import init

args = init(filename='../data/2019-06-12')

args.epochs = 5  # 10 plus tard
args.save_model = True

debut = datetime.datetime.now()
date = str(debut)

liste_std = [i + 0.5 for i in range(0, 11)]  # pas de 1 de std en partant de 0.5 (essai)

args.do_adam = 'adam'
args.what_offset_std = liste_std[0]
print("En cours : std = 0\n")
what = What(args, force=True, seed=0)

seed = 1
for std in liste_std[1:]:
    print("En cours : std = " + str(std) + "\n")

    args.what_offset_std = std
    what_model = what.model
    what = What(args, model=what_model, force=False, seed=seed)
    seed += 1
    print("\n")

intermediaire = datetime.datetime.now()
print("\n\nDuree d'execution : " + str(intermediaire - debut))


model = what.model
transform = transforms.Compose([
    WhatShift(args, i_offset=0, j_offset=0),
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
    whatTrainer = WhatTrainer(args, model=model, test_loader=test_loader)
    acc = whatTrainer.test()

fin = datetime.datetime.now()
print("\n\nDuree d'execution totale : " + str(fin - debut))