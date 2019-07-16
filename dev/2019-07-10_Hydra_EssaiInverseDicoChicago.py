import matplotlib.pyplot as plt
from torchvision import transforms

import sys
sys.path.append("../figures")

from where_chicago import WhereSquareCrop, OnlineRetinaTransform, RetinaFill, CollFill, WhereShift, RetinaBackground, RetinaMask, RetinaWhiten, TransformDico, InverseTransformDico
from where_chicago import ChicagoFacesDataset, RetinaTransform, WhereNet, CollTransform, Normalize, WhereTrainer, Where, WhereZoom, WhereGrey, WhereRotate

from main import init
args = init(filename='../data/2019-07-08')
args.train_batch_size = 1000
args.test_batch_size = 207
args.N_eccentricity = 12
args.contrast = 0.5

from retina_chicago import Retina
retina = Retina(args)

transform=transforms.Compose([
                            WhereSquareCrop(args),
                            WhereGrey(args),
                            RetinaWhiten(N_pic=args.N_pic),
                            TransformDico(retina),
                            InverseTransformDico(retina)
                           ])

dataset = ChicagoFacesDataset(csv_file='../data/ChicagoFacesData/CFD_2.0.3_Norming_Data_and_Codebook.csv',
                                    root_dir='../data/ChicagoFacesData/', transform=transform)
print(len(dataset))
index_liste = 2
donnees = dataset[index_liste]
filename, images, target = donnees[0], donnees[1], donnees[2]

pixel_fullfield = images[0]
rebuild_pixel_fullfield = images[1]


plt.figure(figsize=(20,20))
plt.imshow(pixel_fullfield[:,:].reshape((1718, 1718)))
plt.title('target : '+ target)
plt.savefig("pixel_fullfield_24_12.jpg")

plt.figure(figsize=(20,20))
plt.imshow(rebuild_pixel_fullfield[:,:].reshape((1718, 1718)))
plt.title("rebuild_pixel_fullfield")
plt.savefig("rebuild_pixel_fullfield_24_12.jpg")