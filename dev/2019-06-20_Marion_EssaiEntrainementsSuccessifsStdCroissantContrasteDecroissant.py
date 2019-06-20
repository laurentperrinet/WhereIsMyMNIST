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

liste_std = [i for i in range(0, 16)]  # pas de 1 de std

args.what_offset_max = 25
args.do_adam = 'adam'
args.what_offset_std = liste_std[0]
print("En cours : std = 0\n")
what = What(args, force=True, seed=0)

seed = 1
for std in liste_std[1:]:
    print("En cours : std = " + str(std) + "\n")

    args.what_offset_std = std
    what_model = what.model
    what = What(args, model=what_model, force=True, seed=seed)
    seed += 1
    print("\n")

liste_contrast = [i/10 for i in range(6, 0, -1)]

for contrast in liste_contrast:
    print("En cours : contrast = " + str(contrast) + "\n")

    args.contrast = contrast
    what_model = what.model
    what = What(args, model=what_model, force=True, seed=seed)
    seed += 1
    print("\n")


fin = datetime.datetime.now()
print("\n\nDuree d'execution : " + str(fin - debut))

