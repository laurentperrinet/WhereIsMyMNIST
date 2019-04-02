## SCRIPT

import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm_notebook as tqdm
from main_orig import init
import torch
import easydict
from retina import Display, Retina
from whatWhere import WhatWhere

args = init(filename='../data/2019-03-29')
FIC_NAME = '../data/2019-04-02-crowding-contrast'
result = {}

for period in range(1,21):
    args.sf_0 = 1/period
    result[period] = {} 
    for contrast in (0.5, 0.7):
        args.contrast = contrast
        whatWhere = WhatWhere(args, save = False)
        whatWhere.train()
        data_test, label_test = next(iter(whatWhere.loader_test)) 
        acc = whatWhere.pred_accuracy(data_test, label_test)
        print(f'sf_0 : %.2f, B_sf : %.2f, acc : %.2f'%(args.sf_0, args.B_sf, acc))
        result[period][prop] = acc
        np.save(FIC_NAME + '-result.npy', result)