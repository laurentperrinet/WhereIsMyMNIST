import os
import time
import matplotlib.pyplot as plt
import numpy as np

import noise
import MotionClouds as mc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from LogGabor import LogGabor

#import SLIP for whitening and PIL for resizing
import SLIP
import PIL
whit = SLIP.Image(pe='https://raw.githubusercontent.com/bicv/LogGabor/master/default_param.py')

def MotionCloudNoise(sf_0=0.125, B_sf=3., alpha = .5):
    mc.N_X, mc.N_Y, mc.N_frame = 128, 128, 1
    fx, fy, ft = mc.get_grids(mc.N_X, mc.N_Y, mc.N_frame)
    name = 'static'
    env = mc.envelope_gabor(fx, fy, ft, sf_0=sf_0, B_sf=B_sf, B_theta=np.inf, V_X=0., V_Y=0., B_V=0, alpha= alpha)
    
    z = mc.rectif(mc.random_cloud(env))
    z = z.reshape((mc.N_X, mc.N_Y))
    return z, env

# TODO: passer les arguments par la ligne de commande
N_theta = 6 #1 
N_azimuth = 16
N_eccentricity = 8
N_phase = 2
N_X = 128
N_Y = 128
rho = 1.41
verbose = 1

def vectorization(N_theta=N_theta, N_azimuth=N_azimuth, N_eccentricity=N_eccentricity, N_phase=N_phase, \
                  N_X=N_X, N_Y=N_Y, rho=rho, ecc_max=.8, B_sf=.4, B_theta=np.pi/N_theta/2):
    retina = np.zeros((N_theta, N_azimuth, N_eccentricity, N_phase, N_X*N_Y))
    parameterfile = 'https://raw.githubusercontent.com/bicv/LogGabor/master/default_param.py'
    lg = LogGabor(parameterfile)
    lg.set_size((N_X, N_Y))
    # params = {'sf_0': .1, 'B_sf': lg.pe.B_sf,
    #           'theta': np.pi * 5 / 7., 'B_theta': lg.pe.B_theta}
    # phase = np.pi/4
    # edge = lg.normalize(lg.invert(lg.loggabor(
    #     N_X/3, 3*N_Y/4, **params)*np.exp(-1j*phase)))

    for i_theta in range(N_theta):
        for i_azimuth in range(N_azimuth):
            for i_eccentricity in range(N_eccentricity):
                ecc = ecc_max * (1/rho)**(N_eccentricity - i_eccentricity)
                r = np.sqrt(N_X**2+N_Y**2) / 2 * ecc  # radius
                psi = i_azimuth * np.pi * 2 / N_azimuth
                theta_ref = i_theta*np.pi/N_theta
                sf_0 = 0.5 * 0.03 / ecc
                x = N_X/2 + r * np.cos(psi)
                y = N_Y/2 + r * np.sin(psi)
                for i_phase in range(N_phase):
                    params = {'sf_0': sf_0, 'B_sf': B_sf,
                              'theta': theta_ref + psi, 'B_theta': B_theta}
                    phase = i_phase * np.pi/2
                    # print(r, x, y, phase, params)

                    retina[i_theta, i_azimuth, i_eccentricity, i_phase, :] = lg.normalize(
                        lg.invert(lg.loggabor(x, y, **params)*np.exp(-1j*phase))).ravel() * ecc

    return retina

retina = vectorization(N_theta, N_azimuth, N_eccentricity, N_phase, N_X, N_Y, rho)
print(retina.shape)

retina_vector = retina.reshape((N_theta*N_azimuth*N_eccentricity*N_phase, N_X*N_Y))
print(retina_vector.shape)

retina_inverse = np.linalg.pinv(retina_vector)
print(retina_inverse.shape)

colliculus = (retina**2).sum(axis=(0, 3))
colliculus = colliculus**.5
colliculus /= colliculus.sum(axis=-1)[:, :, None]
print(colliculus.shape)

colliculus_vector = colliculus.reshape((N_azimuth*N_eccentricity, N_X*N_Y))
print(colliculus_vector.shape)

colliculus_inverse = np.linalg.pinv(colliculus_vector)
print(colliculus_inverse.shape)

def get_data_loader(batch_size=100, train=True, num_workers = 4):
    data_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/tmp/data',
                       train=train,     # def the dataset as training data
                       download=True,  # download if dataset not present on disk
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=(0,), std=(1,))])),
                       batch_size=batch_size,
                       shuffle=True,
                       num_workers = num_workers)
    return data_loader

def mnist_fullfield(data, i_offset, j_offset, N_pic=128, CONTRAST=1., NOISE = 1., sf_0 = 0.1, B_sf = 0.1):
    N_stim = data.shape[0]
    center = (N_pic-N_stim)//2

    data_fullfield = (data.min().numpy()) * np.ones((N_pic, N_pic))
    data_fullfield[int(center+i_offset):int(center+N_stim+i_offset), int(center+j_offset):int(center+N_stim+j_offset)] = data

    # data normalization
    # data_fullfield -= data_fullfield.mean()
    # data_fullfield /= data_fullfield.std()
    # data_fullfield *= std
    # data_fullfield += mean
    data_fullfield = (data_fullfield - data_fullfield.min())/(data_fullfield.max() - data_fullfield.min())
    data_fullfield *= CONTRAST
    data_fullfield += 0.5

    if NOISE>0.:
        im_noise, _ = MotionCloudNoise(sf_0 = sf_0, B_sf = B_sf)
        im_noise = NOISE *  im_noise
        #data_fullfield += im_noise #randomized_perlin_noise() #
        indices_data = np.where(data_fullfield > data_fullfield.mean())
        im_noise[indices_data] = data_fullfield[indices_data]
        data_fullfield = im_noise
    
    

    if True:
        if True: #Whitening
            whit.set_size((N_pic,N_pic))
            data_fullfield = whit.whitening(data_fullfield)        
        data_retina = retina_vector @ np.ravel(data_fullfield)
        tensor_retina = data_retina.reshape(N_theta, N_azimuth, N_eccentricity, N_phase)
        slice1 = tensor_retina[N_theta - 1,:,:,:].reshape(1,N_azimuth,N_eccentricity,N_phase)
        slice2 = tensor_retina[0,:,:,:].reshape(1,N_azimuth,N_eccentricity,N_phase)
        tensor_retina = np.concatenate ((slice1, tensor_retina, slice2), axis = 0)
        tensor_retina = np.transpose(tensor_retina,(3,0,1,2))
    else:
        data_retina =colliculus_vector @ np.ravel(data_fullfield)
        tensor_retina = None
    
    

    return data_retina, tensor_retina, data_fullfield

def minmax(value, border):
    value = max(value, -border)
    value = min(value, border)
    return int(value)

minibatch_size = 100  # quantity of examples that'll be processed
lr = 1e-3 #0.05

OFFSET_STD = 15 #0 #
OFFSET_MAX = 30 #0 #
NOISE = 1 #0 #
CONTRAST = 0.3 #1 #
sf_0 = 0.2
B_sf = 0.3

do_cuda = torch.cuda.is_available()
if do_cuda:
    device = 'cuda:0'
else:
    device = 'cpu' #torch.cuda.device("0" if do_cuda else "cpu")
    
if device == 'cpu' :
    NUM_WORKERS = 10
else:
    NUM_WORKERS = 4
    
train_loader = get_data_loader(batch_size=minibatch_size, train = True, num_workers = NUM_WORKERS)
test_loader = get_data_loader(batch_size=1000, train = False, num_workers = NUM_WORKERS)

BIAS_CONV = True
BIAS_DECONV = True #True

class Net(torch.nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        self.bn1= torch.nn.Linear(N_theta*N_azimuth*N_eccentricity*N_phase, 200, bias = BIAS_DECONV)
        self.bn2 = torch.nn.Linear(200, 80, bias = BIAS_DECONV)
        self.bn3 = torch.nn.Linear(80, 10, bias = BIAS_DECONV)
                
    def forward(self, image):
       
        h_bn1 = F.relu(self.bn1(image))               
        h_bn2 = F.relu(self.bn2(h_bn1))
        h_bn2_drop = F.dropout(h_bn2, p = .5) 
        z = self.bn3(h_bn2_drop)
        
        return z

net = Net()
net.to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=lr)

loss_func = torch.nn.CrossEntropyLoss()


def train(net, minibatch_size, \
          optimizer=optimizer, \
          vsize = N_theta * N_azimuth * N_eccentricity * N_phase,\
          asize = 1, \
          offset_std=OFFSET_STD, \
          offset_max=OFFSET_MAX, \
          verbose=1, \
          CONTRAST=CONTRAST,
          NOISE = NOISE,
          sf_0 = sf_0, 
          B_sf = B_sf):
    
    t_start = time.time()
    
    if verbose: print('Starting training...')
    
    for batch_idx, (data, label) in enumerate(train_loader):
        optimizer.zero_grad()

        data_retina = np.zeros((minibatch_size, N_phase * N_theta * N_azimuth * N_eccentricity))
      
        # target = np.zeros((minibatch_size, asize))

        for i in range(minibatch_size):
            i_offset = minmax(np.random.randn() * offset_std, offset_max)
            j_offset = minmax(np.random.randn() * offset_std, offset_max)
            data_retina[i, :], _, _  = mnist_fullfield(data[i, 0, :, :], i_offset, j_offset, 
                                                        CONTRAST = CONTRAST, NOISE = NOISE,
                                                        sf_0 = sf_0, B_sf = B_sf)            

        data_retina = Variable(torch.FloatTensor(data_retina)).to(device)
        label = Variable(torch.LongTensor(label)).to(device)
        
        prediction = net(data_retina)
        loss = loss_func(prediction, label)
        loss.backward()
        optimizer.step()

        if verbose and batch_idx % 10 == 0:
            print('[{}/{}] Loss: {} Time: {:.2f} mn'.format(
                batch_idx*minibatch_size, len(train_loader.dataset),
                loss.data.cpu().numpy(), (time.time()-t_start)/60))
                                                        
    return net

def test(net, optimizer=optimizer,
         vsize=N_theta*N_azimuth*N_eccentricity*N_phase,
         asize=N_azimuth*N_eccentricity, offset_std=OFFSET_STD, offset_max=OFFSET_MAX, 
         CONTRAST=CONTRAST, NOISE = NOISE,
         sf_0 = sf_0, 
         B_sf = B_sf):
    #for batch_idx, (data, label) in enumerate(test_loader):
    data, label = next(iter(test_loader))
    batch_size = label.shape[0]

    data_retina = np.zeros((batch_size, N_phase * N_theta * N_azimuth * N_eccentricity))
    
    for i in range(batch_size):
        i_offset = minmax(np.random.randn() * offset_std, offset_max)
        j_offset = minmax(np.random.randn() * offset_std, offset_max)
        data_retina[i, :], _, _  = mnist_fullfield(data[i, 0, :, :], i_offset, j_offset, 
                                                    CONTRAST = CONTRAST, NOISE = NOISE,
                                                    sf_0 = sf_0, B_sf = B_sf)

    data_retina = Variable(torch.FloatTensor(data_retina)).to(device)
    label = Variable(torch.LongTensor(label)).to(device)
    
    with torch.no_grad():
        output = net(data_retina)
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        #pred = Variable(torch.LongTensor(pred))
        #print(pred.data.numpy())
        acc = pred.eq(label.view_as(pred)).sum().cpu().item()

    return acc / batch_size

FIC_NAME = '2019-02-13-crowding-test'
result = []

for r in range(1,11):
    sf_0 = 1/r
    B_sf = 1/r
    
    for epoch in range(10):
        print ('************ r = %d ************ turn = %d'%(r,epoch))
        train(net, minibatch_size, sf_0 = sf_0, B_sf = B_sf)
        Accuracy = test(net, sf_0 = sf_0, B_sf = B_sf)
        mes = 'r = %d, turn = %d, test set final Accuracy = %.3f'%(r,epoch,Accuracy*100) # print que le pourcentage de réussite final
        print(mes)
        with open(FIC_NAME + '.txt','a') as f:
            f.write(mes + '\n')
        result += [Accuracy]
        np.save(FIC_NAME + '-result.npy', result)


    