import sys
sys.path.append("../figures")

from main import init
args = init(filename='../data/2019-07-08')
args.train_batch_size = 1000
args.test_batch_size = 207
args.N_eccentricity = 18

print("N_pic", args.N_pic)
from retina_chicago import Retina
retina = Retina(args)

