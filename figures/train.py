import torch
import easydict

from main import init
args = init()#'debug')

from retina import Display, Retina
from where import Where, WhereNet
from what import WhatNet
where = Where(args, Display(args), Retina(args))
filename_train = args.filename + '_train.pt'
where.train(filename_train)
