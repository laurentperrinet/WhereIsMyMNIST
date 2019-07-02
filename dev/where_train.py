import sys
sys.path.append("../figures")

from where_copie import RetinaTransform, WhereNet, CollTransform, MNIST, Normalize, WhereTrainer, Where

import datetime

from main import init
args = init(filename='../data/2019-06-12')

print(datetime.datetime.now())

#args.epochs = 60
args.save_model = False

where = Where(args, what_model="MNIST_cnn_robust_what_0.1_0.1_1.0_0.7_5epoques_2019-06-19_11h50.pt")
# c'est le reseau robust what qui donne l'accuracy map etendue de tests2sur2.ipynb

acc = where.test()

print("acc = ", acc)

print(datetime.datetime.now())