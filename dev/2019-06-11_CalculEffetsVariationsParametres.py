import sys
import torch
from torchvision import datasets, transforms
sys.path.append("../figures")
from what import WhatShift, WhatBackground, WhatNet, WhatTrainer, What, train, test
from main import init
args = init(filename='../data/2019-06-05')
import datetime

debut = datetime.datetime.now()
date = str(debut)

f = open('{}_{}h{}_Marion_EffetsVariationsParametres.txt'.format(date[0:10], date[11:13], date[14:16]), "w+")

f.write("On utilise what.py, pas robust_what.py\n\n")

f.write("Configuration de base :\n")


dicoParametres = {"w": 28, "minibatch_size": 100, "train_batch_size": 50000, "test_batch_size": 10000, "noise_batch_size": 1000, "mean": 0.1307, "std": 0.3081, "N_pic": 128, "offset_std": 30, "offset_max": 34, "noise": 1.0, "contrast": 0.7, "sf_0": 0.1, "B_sf": 0.1, "N_theta": 6, "N_azimuth": 24, "N_eccentricity": 10, "N_phase": 2, "rho": 1.41, "bias_deconv": True, "p_dropout": 0.0, "dim1": 1000, "dim2": 1000, "lr": 0.005, "do_adam": True, "bn1_bn_momentum": 0.5, "bn2_bn_momentum": 0.5, "momentum": 0.3, "epochs": 60, "num_processes": 1, "no_cuda": True, "log_interval": 100, "verbose": 1, "filename": "../data/2019-06-05", "seed": 2019, "N_cv": 10, "do_compute": True}

for parametre in dicoParametres:
    f.write(str(parametre)+" : "+str(dicoParametres[parametre])+"\n")


f.write("\nAccuracy de la configuration de base :\n")
print("Calcul accuracy de base")


model = torch.load("../data/MNIST_cnn_0.1_0.1_1_0.7.pt")
transform = transforms.Compose([
    WhatShift(i_offset=0, j_offset=0),
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
f.write(str(acc) + ' \n')


etape1 = datetime.datetime.now()

f.write("Duree d'execution : " + str(etape1-debut))

f.write("\n\nSi on met do_adam=False, alors :\n")
print("Calcul accuracy do_adam=False")

args.epochs = 60
args.noise = 1
args.save_model = False

# Parametre modifie du reseau par rapport au reseau de base :
args.do_adam = False

# Entrainement du reseau :
what = What(args=args, force=True)

# Test du reseau :
model = what.model
transform = transforms.Compose([
    WhatShift(i_offset=0, j_offset=0),
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
f.write(str(acc) + ' \n')


# Reinitialisation des parametres :
args.do_adam = True

etape2 = datetime.datetime.now()

f.write("Duree d'execution : " + str(etape2-etape1))

f.write("\n\nSi on met lr = 0.001, alors :\n")
print("Calcul accuracy lr = 0.001")


args.epochs = 60
args.noise = 1
args.save_model = False

# Parametre modifie du reseau par rapport au reseau de base :
args.lr = 0.001

# Entrainement du reseau :
what = What(args=args, force=True)

# Test du reseau :
model = what.model
transform = transforms.Compose([
    WhatShift(i_offset=0, j_offset=0),
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
f.write(str(acc) + ' \n')


# Reinitialisation des parametres :
args.lr = 0.005

etape3 = datetime.datetime.now()

f.write("Duree d'execution : " + str(etape3-etape2))

f.write("\n\nSi on met lr = 0.0001, alors :\n")
print("Calcul accuracy lr = 0.0001")


args.epochs = 60
args.noise = 1
args.save_model = False

# Parametre modifie du reseau par rapport au reseau de base :
args.lr = 0.0001

# Entrainement du reseau :
what = What(args=args, force=True)

# Test du reseau :
model = what.model
transform = transforms.Compose([
    WhatShift(i_offset=0, j_offset=0),
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
f.write(str(acc) + ' \n')


# Reinitialisation des parametres :
args.lr = 0.005

etape4 = datetime.datetime.now()

f.write("Duree d'execution : " + str(etape4-etape3))

f.write("\n\nSi on met contrast = 0.5, alors :\n")
print("Calcul accuracy contrast = 0.5")


args.epochs = 60
args.noise = 1
args.save_model = False

# Parametre modifie du reseau par rapport au reseau de base :
args.contrast = 0.5

# Entrainement du reseau :
what = What(args=args, force=True)

# Test du reseau :
model = what.model
transform = transforms.Compose([
    WhatShift(i_offset=0, j_offset=0),
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
f.write(str(acc) + ' \n')


# Reinitialisation des parametres :
args.contrast = 0.7

etape5 = datetime.datetime.now()

f.write("Duree d'execution : " + str(etape5-etape4))

f.write("\n\nSi on met contrast = 0.3, alors :\n")
print("Calcul accuracy contrast = 0.3")


args.epochs = 60
args.noise = 1
args.save_model = False

# Parametre modifie du reseau par rapport au reseau de base :
args.contrast = 0.3

# Entrainement du reseau :
what = What(args=args, force=True)

# Test du reseau :
model = what.model
transform = transforms.Compose([
    WhatShift(i_offset=0, j_offset=0),
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
f.write(str(acc) + ' \n')


# Reinitialisation des parametres :
args.contrast = 0.7

etape6 = datetime.datetime.now()

f.write("Duree d'execution : " + str(etape6-etape5))

f.write("\n\nSi on met p_dropout = 0.5, alors :\n")
print("Calcul accuracy p_dropout = 0.5")


args.epochs = 60
args.noise = 1
args.save_model = False

# Parametre modifie du reseau par rapport au reseau de base :
args.p_dropout = 0.5

# Entrainement du reseau :
what = What(args=args, force=True)

# Test du reseau :
model = what.model
transform = transforms.Compose([
    WhatShift(i_offset=0, j_offset=0),
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
f.write(str(acc) + ' \n')


# Reinitialisation des parametres :
args.p_dropout = 0.0

etape7 = datetime.datetime.now()

f.write("Duree d'execution : " + str(etape7-etape6))

f.write("\n\nSi on met p_dropout = 1.0, alors :\n")
print("Calcul accuracy p_dropout = 1.0")


args.epochs = 60
args.noise = 1
args.save_model = False

# Parametre modifie du reseau par rapport au reseau de base :
args.p_dropout = 1.0

# Entrainement du reseau :
what = What(args=args, force=True)

# Test du reseau :
model = what.model
transform = transforms.Compose([
    WhatShift(i_offset=0, j_offset=0),
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
f.write(str(acc) + ' \n')

etape8 = datetime.datetime.now()

f.write("Duree d'execution : " + str(etape8-etape7))



f.write("\n\nDuree d'execution totale: " + str(etape8-debut))

# Reinitialisation des parametres :
args.p_dropout = 0.0


f.close()
print("ok")
