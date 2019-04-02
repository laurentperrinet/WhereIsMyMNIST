import torch
import easydict

from main import init
#args = init(filename='debug')
#args = init(filename='../data/2019-03-27')
args = init()

from retina import Display, Retina
from where import Where, WhereNet
from what import WhatNet
where = Where(args)
filename_train = args.filename + '_train.pt'
where.train(filename_train)
