import sys
import datetime
import random

sys.path.append("../figures")
from robust_what import WhatShift, WhatBackground, WhatNet, WhatTrainer, What, train, test, MNIST
from main import init

args = init(filename='../data/2019-06-12')

args.epochs = 5  # 10 plus tard
args.save_model = True

debut = datetime.datetime.now()
date = str(debut)

liste_std = [0, 1, 2, 3, 4, 5]
suite_std = [0, 1, 2, 3, 4, 5]
random.shuffle(suite_std)
liste_std += suite_std

args.do_adam = 'adam'
args.what_offset_std = liste_std[0]
print("En cours : std = 0\n")
what = What(args=args, force=True, seed=0)

seed = 1
for std in liste_std[1:]:
    print("En cours : std = " + str(std) + "\n")

    args.what_offset_std = std
    what = What(args=args, model=what.model, force=True, seed=seed)
    seed += 1
    print("\n")

fin = datetime.datetime.now()
print("\n\nDuree d'execution : " + str(fin - debut))

