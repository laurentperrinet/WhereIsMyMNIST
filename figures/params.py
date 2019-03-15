import torch
from main import init, MetaML

opts = dict(filename='debug', verbose=0, log_interval=0)

print(50*'-')
print(' parameter scan')
print(50*'-')

if True:
    print(50*'-')
    print('Default parameters')
    print(50*'-')
    args = init(**opts)
    from where import Where as ML
    from what import WhatNet
    ml = ML(args)
    ml.main()

    args = init(**opts)
    mml = MetaML(args)
    if torch.cuda.is_available():
        mml.scan('no_cuda', [True, False])

    args = init(**opts)
    mml = MetaML(args)
    mml.scan('bias_deconv', [True, False])
   
# for base in [2]:#, 8]:
for base in [2, 8]:
    print(50*'-')
    print(' base=', base)
    print(50*'-')

    print(50*'-')
    print(' parameter scan : data')
    print(50*'-')
    args = init(**opts)
    mml = MetaML(args, base=base)
    for parameter in ['sf_0', 'B_sf', 'offset_std', 'noise', 'contrast']:
        mml.parameter_scan(parameter)
        
    # TODO:  'N_theta': 6, 'N_azimuth': 16, 'N_eccentricity': 10, 'rho': 1.41,

    print(50*'-')
    args = init(**opts)
    mml = MetaML(args)
    print(' parameter scan : network')
    print(50*'-')
    for parameter in ['dim1',
                      'bn1_bn_momentum',
                      'dim2',
                      'bn2_bn_momentum',
                      'p_dropout']:
        mml.parameter_scan(parameter)

    args = init(**opts)
    mml = MetaML(args, base=base)
    print(' parameter scan : learning ')
    print(50*'-')
    print('Using SGD')
    print(50*'-')
    for parameter in ['lr', 'momentum', 'minibatch_size', 'epochs']:
        mml.parameter_scan(parameter)
    print(50*'-')
    print('Using ADAM')
    print(50*'-')
    args = init(**opts)
    args.do_adam = True
    mml = MetaML(args, tag='adam')
    for parameter in ['lr', 'momentum', 'minibatch_size', 'epochs']:
        mml.parameter_scan(parameter)

