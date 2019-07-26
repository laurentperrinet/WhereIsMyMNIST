import datetime

debut = datetime.datetime.now()
date = str(debut)

import sys
sys.path.append("../figures")

from where_chicago import WhereGrey, RetinaWhiten, TransformDico, ChicagoFacesDataset,  WhatGender, WhatGenderTrainer

import torch

from main import init
args = init(filename='../data/2019-07-23')
args.minibatch_size = 10
args.momentum = 0
args.bn1_bn_momentum = 0
args.bn2_bn_momentum = 0
args.p_dropout = 0.5

from retina_chicago import Retina
retina = Retina(args)

f = open('{}_{}h{}_Marion_EffetsVariationsParametresWhatFaces.txt'.format(date[0:10], date[11:13], date[14:16]), "w+")

f.write("On utilise where_chicago.py\n\n")

f.write("Configuration de base :\n")


dicoParametres = {"w": 28, "minibatch_size": 10, "train_batch_size": 1000, "test_batch_size": 126, "noise_batch_size": 1000, "mean": 0.1307, "std": 0.3081, "N_pic": 1718, "N_X": 1718, "N_Y":2444, "offset_std": 30, "offset_max": 34, "noise": 0.75, "contrast": 0.7, "sf_0": 0.1, "B_sf": 0.1, "N_theta": 6, "N_azimuth": 48, "N_eccentricity": 24, "N_phase": 2, "rho": 1.41, "bias_deconv": True, "p_dropout": 0.5, "dim1": 1000, "dim2": 1000, "lr": 0.005, "do_adam": "adam", "bn1_bn_momentum": 0, "bn2_bn_momentum": 0, "momentum": 0, "epochs": 60, "num_processes": 1, "no_cuda": True, "log_interval": 20, "verbose": 1, "filename": "../data/2019-07-23", "seed": 2019, "N_cv": 10, "do_compute": True, "save_model":  False}
for parametre in dicoParametres:
    f.write(str(parametre)+" : "+str(dicoParametres[parametre])+"\n")


f.write("\nPour chaque modele calcule, on recuperera la variation d'accuracy lors des 60 epoques d'apprentissga")
f.write("\n On n'enregistrera pas ces modeles.")
args.save_model = False
train_loader = torch.load("../tmp/train_loader_faces_2019-07-24_13h22.pt")
test_loader = torch.load("../tmp/test_loader_faces_2019-07-24_13h32.pt")

etape0 = datetime.datetime.now()

f.write("\n\nVariation d'accuracy pour la configuration de base :")

what_gender = WhatGender(args, retina, train_loader=train_loader, test_loader=test_loader)

f.write("\n[")
for acc in what_gender.list_acc[0:-1]:
    f.write(str(acc)+", ")
f.write("]")

etape1 = datetime.datetime.now()
f.write("\nDuree : " + str(etape1-etape0))


f.write("\n\nVariation d'accuracy pour la configuration lr = 0.001 :")

args.lr = 0.001
what_gender = WhatGender(args, retina, train_loader=train_loader, test_loader=test_loader)

f.write("\n[")
for acc in what_gender.list_acc[0:-1]:
    f.write(str(acc)+", ")
f.write("]")

etape2 = datetime.datetime.now()
f.write("\nDuree : " + str(etape2-etape1))

f.write("\n\nVariation d'accuracy pour la configuration lr = 0.0001 :")

args.lr = 0.0001
what_gender = WhatGender(args, retina, train_loader=train_loader, test_loader=test_loader)

f.write("\n[")
for acc in what_gender.list_acc[0:-1]:
    f.write(str(acc)+", ")
f.write("]")

etape3 = datetime.datetime.now()
f.write("\nDuree : " + str(etape3-etape2))

f.write("\n\nVariation d'accuracy pour la configuration lr = 0.00001 :")

args.lr = 0.00001
what_gender = WhatGender(args, retina, train_loader=train_loader, test_loader=test_loader)

f.write("\n[")
for acc in what_gender.list_acc[0:-1]:
    f.write(str(acc)+", ")
f.write(str(what_gender.list_acc[-1]) + "]")

etape4 = datetime.datetime.now()
f.write("\nDuree : " + str(etape4-etape3))


fin = datetime.datetime.now()
f.write("\n\nDuree totale : "+ str(fin-debut))

f.close()
print("ok")