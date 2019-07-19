import sys
sys.path.append("../figures")
from torchvision import transforms
import datetime

from main import init
args = init(filename='../data/2019-07-08')
args.N_azimuth = 48
args.N_eccentricity = 24

from where_chicago import WhereSquareCrop, WhereGrey, RetinaWhiten, TransformDico, ToFloatTensor, RetinaFaces

from retina_chicago import Retina
retina = Retina(args)

transform = transforms.Compose([
                                WhereSquareCrop(args),
                                WhereGrey(args),
                                RetinaWhiten(N_pic=args.N_pic),
                                TransformDico(retina)
                            ])

print(datetime.datetime.now())

dataset = RetinaFaces("../data/ChicagoFacesData/", transform, args)

print(datetime.datetime.now())