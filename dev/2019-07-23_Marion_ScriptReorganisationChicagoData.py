###
### CONSIGNES
###

# # Creez prealablement les repertoires "../data/ChicagoFacesDataOrganized/train/women/", "../data/ChicagoFacesDataOrganized/train/men/"
# # "../data/ChicagoFacesDataOrganized/test/women/", "../data/ChicagoFacesDataOrganized/test/men/"

# # Remplacez False par True ligne 97

# # Rappels :
# # - Les fichiers ../data/ChicagoFacesData\BF-209\CFD-BF-209-172-N.jpg et ../data/ChicagoFacesData\WF-005\CFD-WF-005-016-HC.jpg
# #   ont une taille légèrement différentes de la normale dans le dataset téléchargé, il faut modifier leur taile pour qu'ils soient identiques au reste
# # - !!! Le fichier ../data/ChicagoFacesData\AF-215\CFD-AF-215-70-N.jpg doit être renommé en ../data/ChicagoFacesData\AF-215\CFD-AF-215-070-N.jpg pour pouvoir être traité comme les autres !!!


import sys
sys.path.append("../figures")
import os
from main import init
args = init(filename='../data/2019-07-08')
import numpy as np
import shutil


folder_directory = "../data/ChicagoFacesData"
list_files = []

for path, subdirs, files in os.walk(folder_directory):
    for name in files:
        if name[-4:]=='.jpg':
            list_files.append(os.path.join(path, name))
# print(list_files)

men, women = [], []

for idx in range(len(list_files)):
    #if idx % 10 == 0: print(idx)

    image_name = list_files[idx][-28:-4]
    #print(image_name)
    if image_name[-1] in ['O', 'C']:
        gender = image_name[-12]
        race = image_name[-13]
        expression = "H" + image_name[-1]
    else:
        gender = image_name[-11]
        race = image_name[-12]
        expression = image_name[-1]
    #print(race, gender, expression)
    races = ["A", "B", "L", "W"]
    genders = ["F", "M"]
    expressions = ["N", "A", "F", "HC", "HO"]

    if gender == "F":
        women.append(list_files[idx])
    if gender == "M":
        men.append(list_files[idx])

#print("femmes", len(women), women) # 644
#print("hommes", len(men), men) # 563

# 563*2 = 1126 images : 1000 images train, 126 images test
args.train_batch_size = 1000
args.test_batch_size = 126

train_men, train_women = [], []
test_men, test_women = [], []

index_utilises = []
while len(train_women) < args.train_batch_size//2:
    index = np.random.randint(len(women))
    if index not in index_utilises:
        train_women.append(women[index])
        index_utilises.append(index)
for index in range(len(women)):
    if len(test_women) < args.test_batch_size//2 and index not in index_utilises:
        test_women.append(women[index])


index_utilises = []
while len(train_men) < args.train_batch_size//2:
    index = np.random.randint(len(men))
    if index not in index_utilises:
        train_men.append(men[index])
        index_utilises.append(index)
for index in range(len(men)):
    if len(test_men) < args.test_batch_size//2 and index not in index_utilises:
        test_men.append(men[index])


# print("train_women", len(train_women), train_women)
# print("test_women", len(test_women), test_women)
# print("train_men", len(train_men), train_men)
# print("test_men", len(test_men), test_men)


# Creez prealablement "../data/ChicagoFacesDataOrganized/train/women/", "../data/ChicagoFacesDataOrganized/train/men/"
# "../data/ChicagoFacesDataOrganized/test/women/", "../data/ChicagoFacesDataOrganized/test/men/"
if False:
    print("Copie en cours...")
    for file in train_women:
        shutil.copy2(file, "../data/ChicagoFacesDataOrganized/train/women/")
    for file in test_women:
        shutil.copy2(file, "../data/ChicagoFacesDataOrganized/test/women/")
    for file in train_men:
        shutil.copy2(file, "../data/ChicagoFacesDataOrganized/train/men/")
    for file in test_men:
        shutil.copy2(file, "../data/ChicagoFacesDataOrganized/test/men/")
    print("Copie finie")