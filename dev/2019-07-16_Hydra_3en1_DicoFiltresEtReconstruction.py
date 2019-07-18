import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np

import sys
sys.path.append("../figures")

from where_chicago import WhereSquareCrop, OnlineRetinaTransform, RetinaFill, CollFill, WhereShift, RetinaBackground, RetinaMask, RetinaWhiten, TransformDico, InverseTransformDico
from where_chicago import ChicagoFacesDataset, RetinaTransform, WhereNet, CollTransform, Normalize, WhereTrainer, Where, WhereZoom, WhereGrey, WhereRotate

from main import init
args = init(filename='../data/2019-07-08')
args.train_batch_size = 1000
args.test_batch_size = 207
args.N_eccentricity = 24
args.N_azimuth = 48
args.contrast = 0.5
nom = "_{}_{}".format(args.N_azimuth, args.N_eccentricity)

from retina_chicago import Retina
retina = Retina(args)

transform=transforms.Compose([
                            WhereSquareCrop(args),
                            WhereGrey(args),
                            RetinaWhiten(N_pic=args.N_pic),
                            TransformDico(retina),
                            InverseTransformDico(retina)
                           ])

dataset = ChicagoFacesDataset(root_dir='../data/ChicagoFacesData/', transform=transform)

print(len(dataset))
index_liste = 2
donnees = dataset[index_liste]
filename, images, target = donnees[0], donnees[1], donnees[2]

pixel_fullfield = images[0]
rebuild_pixel_fullfield = images[1]


plt.figure(figsize=(20,20))
plt.imshow(pixel_fullfield[:,:].reshape((1718, 1718)))
plt.title('target : '+ target)
plt.savefig("pixel_fullfield" + nom + ".jpg")

plt.figure(figsize=(20,20))
plt.imshow(rebuild_pixel_fullfield[:,:].reshape((1718, 1718)))
plt.title("rebuild_pixel_fullfield")
plt.savefig("rebuild_pixel_fullfield" + nom + ".jpg")


i_theta = 0
i_phase = 0
i_azimuth = 0
N_X = args.N_pic
N_Y = args.N_pic

for i_eccentricity in range(args.N_eccentricity):
    filtre = retina.retina_dico[i_theta][i_phase][i_eccentricity][i_azimuth]
    #print(filtre.shape[0]**(1/2) - int(filtre.shape[0]**(1/2)))
    dimensions_filtre = int(filtre.shape[0]**(1/2))
    ecc_max = .8
    ecc = ecc_max * (1 / args.rho) ** ((args.N_eccentricity - i_eccentricity)/3)
    # ecc = ecc_max * (1 / self.args.rho) ** ((self.N_eccentricity - i_eccentricity)/5)
    # /5 ajoute sinon on obtient les memes coordonnees x et y pour environ la moitie des filtres crees 12/07
    r = np.sqrt(N_X ** 2 + N_Y ** 2) / 2 * ecc - 30 # radius
    psi = (i_azimuth + 1 * (i_eccentricity % 2) * .5) * np.pi * 2 / args.N_azimuth
    x = int(N_X / 2 + r * np.cos(psi))
    y = int(N_Y / 2 + r * np.sin(psi))
    plt.imshow(filtre.reshape((dimensions_filtre, dimensions_filtre)))
    plt.title("filtre d'eccentricite " + str(i_eccentricity) + " x, y=" + str(x) + ", "+ str(y))
    plt.savefig("filtre_eccentricity_" + str(i_eccentricity) + nom + ".jpg")

