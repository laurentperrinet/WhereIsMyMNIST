import sys
sys.path.append("../figures")
import matplotlib.pyplot as plt

from main import init
args = init(filename='../data/2019-07-08')
args.train_batch_size = 1000
args.test_batch_size = 207
args.contrast = 0.5
args.N_eccentricity = 12


from retina_chicago import Retina
retina = Retina(args)

i_theta = 0
i_phase = 0
i_azimuth = 0

for i_eccentricity in range(args.N_eccentricity):
    filtre = retina.retina_dico[i_theta][i_phase][i_eccentricity][i_azimuth]
    #print(filtre.shape[0]**(1/2) - int(filtre.shape[0]**(1/2)))
    dimensions_filtre = int(filtre.shape[0]**(1/2))
    plt.imshow(filtre.reshape((dimensions_filtre, dimensions_filtre)))
    plt.title("filtre d'eccentricite " + str(i_eccentricity))
    plt.savefig("filtre_eccentricity_" + str(i_eccentricity) + "_24_12.jpg")

