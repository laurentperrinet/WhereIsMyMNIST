import sys
sys.path.append("../figures")

from where_chicago import WhereSquareCrop, WhereGrey, WhereNet, WhereTrainer, Where, train, test, ChicagoFacesDataset

from main import init
args = init(filename='../data/2019-07-08')

args.epochs = 2
args.save_model = True

where = Where(args=args)