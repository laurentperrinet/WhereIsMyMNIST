import torch
import easydict

args = easydict.EasyDict({'w': 28, 'minibatch_size': 100, 'train_batch_size': 600, 'test_batch_size': 1000, 
                          'noise_batch_size': 1000, 'mean': 0.1307, 'std': 0.3081, 'N_pic': 128, 
                          'offset_std': 30, 'offset_max': 35, 'noise': 1.0, 'contrast': 0.8, 
                          'sf_0': 0.2, 'B_sf': 0.3, 'N_theta': 6, 'N_azimuth': 16, 'N_eccentricity': 10, 
                          'N_phase': 2, 'N_X': 128, 'N_Y': 128, 'rho': 1.41, 'bias_deconv': True, 'p_dropout': 0.5, 
                          'dim1': 1000, 'dim2': 1000, 'loss_func': torch.nn.BCEWithLogitsLoss(), 'lr': 0.01, 
                          'do_adam': True, 'epochs': 10, 'bn1_bn_momentum': 0., 'bn2_bn_momentum': 0.,
                          'momentum': 0.1, 'n_epochs': 10, 'num_processes': 1, 'no_cuda': True, 'log_interval': 100, 
                          'verbose': 1, 'filename': '../data/2019-03-15', 'seed': 2019})

from retina import Display, Retina
from where import Where, WhereNet
from what import WhatNet
where = Where(args, Display(args), Retina(args))
filename_train = args.filename + '_train.pt'
where.train(filename_train)
