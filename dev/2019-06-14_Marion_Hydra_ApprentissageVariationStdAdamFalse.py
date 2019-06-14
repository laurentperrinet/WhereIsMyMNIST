import sys
sys.path.append("../figures")

from robust_what import WhatShift, WhatBackground, WhatNet, WhatTrainer, What, train, test, MNIST

from main import init
args = init(filename='../data/2019-06-12')

args.epochs = 2
args.save_model = False

args.do_adam = False

liste_std = [i/2 for i in range(4,11)]

for std in liste_std :
    print("En cours : std = ", std,"\n")
    args.what_offset_std = std
    what = What(args=args, force= True)