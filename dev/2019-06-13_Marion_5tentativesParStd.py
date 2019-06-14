import sys
import datetime
sys.path.append("../figures")
from robust_what import WhatShift, WhatBackground, WhatNet, WhatTrainer, What, train, test, MNIST
from main import init
args = init(filename='../data/2019-06-12')

args.epochs = 1
args.save_model = False

debut = datetime.datetime.now()
date = str(debut)

#f = open('5tentativesParStd_{}_{}h{}.txt'.format(date[0:10], date[11:13], date[14:16]), "w+")

liste_std = [i/2 for i in range(0,11)]

#f.write("fichier de parametres utilise : 2019-06-12_param\n\n")
print("fichier de parametres utilise : 2019-06-12_param\n\n")

for std in liste_std :
    # f.write("Pour std = " + str(std) + "\n")
    print("Pour std = " + str(std) + " :\n")
    for i in range(5):
        args.what_offset_std = std
        what = What(args=args, force=True, seed=i)
        print("\n")


fin = datetime.datetime.now()
print("\n\nDuree d'execution : "+str(fin-debut))