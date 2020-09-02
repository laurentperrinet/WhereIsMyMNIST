
import easydict


def init(filename=None, verbose=1, log_interval=100, do_compute=True):
    if filename is None:
        do_recompute = True
        import datetime
        filename = '../data/' + datetime.datetime.now().date().isoformat()
        print('Using filename=', filename)
    else:
        do_recompute = False

    import json
    filename_json = filename + '_param.json'
    if os.path.isfile(filename_json) and not do_recompute:
        with open(filename_json, 'r') as fp:
            args = json.load(fp)
            args = easydict.EasyDict(args)

    else:

        args = easydict.EasyDict(
                                # MNIST
                                w=28,
                                minibatch_size=100, # batch size
                                train_batch_size=50000, # size of training set
                                test_batch_size=10000,  # size of testing set
                                noise_batch_size=1000,
                                mean=0.1307,
                                std=0.3081,
                                what_offset_std=15,
                                what_offset_max=25,
                                # display
                                N_pic = 128,
                                offset_std = 30, #
                                offset_max = 34, # 128//2 - 28//2 *1.41 = 64 - 14*1.4 = 64-20
                                noise=.75, #0 #
                                contrast=.7, #
                                sf_0=0.1,
                                B_sf=0.1,
                                do_mask=True,
                                # foveation
                                N_theta=6,
                                N_azimuth=24,
                                N_eccentricity=10,
                                N_phase=2,
                                rho=1.41,
                                # network
                                bias_deconv=True,
                                p_dropout=.0,
                                dim1=1000,
                                dim2=1000,
                                # training
                                lr=5e-3,  # Learning rate
                                do_adam=True,
                                bn1_bn_momentum=0.5,
                                bn2_bn_momentum=0.5,
                                momentum=0.3,
                                epochs=60,
                                # simulation
                                num_processes=1,
                                no_cuda=False,
                                log_interval=log_interval, # period with which we report results for the loss
                                verbose=verbose,
                                filename=filename,
                                seed=2019,
                                N_cv=10,
                                do_compute=do_compute,
                                save_model=True,
                                )
        if filename == 'debug':
            args.filename = '../data/debug'
            args.train_batch_size = 100
            args.lr = 1e-2
            #args.noise = .5
            #args.contrast = .9
            #args.p_dropout = 0.
            args.epochs = 8
            args.test_batch_size = 20
            args.minibatch_size = 22
            #args.offset_std = 8
            args.N_cv = 2

        elif not do_recompute: # save if we want to keep the parameters
            with open(filename_json, 'w') as fp:
                json.dump(args, fp)

    return args

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from torchvision.datasets.mnist import MNIST as MNIST_dataset
class MNIST(MNIST_dataset):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform((img, index))

        if self.target_transform is not None:
            target = self.target_transform((target, index))

        return img, target

from torch.autograd import Variable
import MotionClouds as mc
import os

def minmax(value, border):
    value = max(value, -border)
    value = min(value, border)
    return value


from PIL import Image
import datetime
import sys


def MotionCloudNoise(sf_0=0.125, B_sf=3., alpha=.0, N_pic=28, seed=42):
    mc.N_X, mc.N_Y, mc.N_frame = N_pic, N_pic, 1
    fx, fy, ft = mc.get_grids(mc.N_X, mc.N_Y, mc.N_frame)
    env = mc.envelope_gabor(fx, fy, ft, sf_0=sf_0, B_sf=B_sf, B_theta=np.inf, V_X=0., V_Y=0., B_V=0, alpha=alpha)

    z = mc.rectif(mc.random_cloud(env, seed=seed), contrast=1., method='Michelson')
    z = z.reshape((mc.N_X, mc.N_Y))
    return z, env

class WhatShift(object):
    def __init__(self, args, i_offset=None, j_offset=None):
        if i_offset != None :
            self.i_offset = int(i_offset)
        else : self.i_offset = i_offset
        if j_offset != None :
            self.j_offset = int(j_offset)
        else : self.j_offset = j_offset
        self.args = args

    def __call__(self, sample_index):

        sample = np.array(sample_index[0])
        index = sample_index[1]

        # print(index)
        np.random.seed(index)

        if self.i_offset is not None:
            i_offset = self.i_offset
            if self.j_offset is None:
                j_offset_f = np.random.randn() * self.args.what_offset_std
                j_offset_f = minmax(j_offset_f, self.args.what_offset_max)
                j_offset = int(j_offset_f)
            else:
                j_offset = int(self.j_offset)
        else:
            if self.j_offset is not None:
                j_offset = int(self.j_offset)
                i_offset_f = np.random.randn() * self.args.what_offset_std
                i_offset_f = minmax(i_offset_f, self.args.what_offset_max)
                i_offset = int(i_offset_f)
            else:  # self.i_offset is None and self.j_offset is None
                i_offset_f = np.random.randn() * self.args.what_offset_std
                i_offset_f = minmax(i_offset_f, self.args.what_offset_max)
                i_offset = int(i_offset_f)
                j_offset_f = np.random.randn() * self.args.what_offset_std
                j_offset_f = minmax(j_offset_f, self.args.what_offset_max)
                j_offset = int(j_offset_f)


        N_pic = sample.shape[0]
        data = np.zeros((N_pic, N_pic))
        i_binf_patch = max(0, -i_offset)
        i_bsup_patch = min(N_pic, N_pic - i_offset)
        j_binf_patch = max(0, -j_offset)
        j_bsup_patch = min(N_pic, N_pic - j_offset)
        patch = sample[i_binf_patch:i_bsup_patch,
                       j_binf_patch:j_bsup_patch]

        i_binf_data = max(0, i_offset)
        i_bsup_data = min(N_pic, N_pic + i_offset)
        j_binf_data = max(0, j_offset)
        j_bsup_data = min(N_pic, N_pic + j_offset)
        data[i_binf_data:i_bsup_data,
             j_binf_data:j_bsup_data] = patch
        return data.astype('B')


class WhatBackground(object):
    def __init__(self, contrast=1., noise=1., sf_0=.1, B_sf=.1, seed = 0):
        self.contrast = contrast
        self.noise = noise
        self.sf_0 = sf_0
        self.B_sf = B_sf
        self.seed = seed

    def __call__(self, sample):

        data = np.array(sample)
        N_pic = data.shape[0]
        if data.min() != data.max():
            data = (data - data.min()) / (data.max() - data.min())
            data = 2 * data - 1 # go to [-1, 1] range
            if self.contrast is not None:
                data *= self.contrast
            else:
                contrast = np.random.uniform(low=0.3, high=0.7)
                data *= contrast
            data = data / 2 + 0.5 # back to [0, 1] range
        else:
            data = np.zeros((N_pic, N_pic))

        seed = self.seed + hash(tuple(data.flatten())) % (2**31 - 1)
        im_noise, env = MotionCloudNoise(sf_0=self.sf_0,
                                         B_sf=self.B_sf,
                                         seed=seed)
        im_noise = 2 * im_noise - 1  # go to [-1, 1] range
        im_noise = self.noise * im_noise
        im_noise /= 2  # back to [0, 1] range
        im_noise += .5  # back to a .5 baseline
        #plt.imshow(im_noise)
        #plt.show()

        #im = np.add(data, im_noise)
        data[data<=0.5] = -np.inf
        im = np.max((data, im_noise), axis=0)

        im = np.clip(im, 0., 1.)
        im = im.reshape((28,28,1))

        im *= 255
        return im.astype('B') #Variable(torch.DoubleTensor(im)) #.to(self.device)

class WhatNet(nn.Module):
    def __init__(self):
        super(WhatNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x #F.log_softmax(x, dim=1)

class WhatTrainer:
    def __init__(self, args, model = None, train_loader=None, test_loader=None, device='cpu', seed=0):
        self.args = args
        self.device = device
        self.seed = seed
        torch.manual_seed(seed)
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.device != 'cpu' else {}
        transform=transforms.Compose([
                               WhatShift(args),
                               WhatBackground(contrast=args.contrast,
                                              noise=args.noise,
                                              sf_0=args.sf_0,
                                              B_sf=args.B_sf),
                               transforms.ToTensor(),
                               #transforms.Normalize((args.mean,), (args.std,))
                           ])
        if train_loader is None:
            dataset_train = MNIST('../data',
                            train=True,
                            download=True,
                            transform=transform,
                            )
            self.train_loader = torch.utils.data.DataLoader(dataset_train,
                                             batch_size=args.minibatch_size,
                                             shuffle=True,
                                             **kwargs)
        else:
            self.train_loader = train_loader

        if test_loader is None:
            dataset_test = MNIST('../data',
                            train=False,
                            download=True,
                            transform=transform,
                            )
            self.test_loader = torch.utils.data.DataLoader(dataset_test,
                                             batch_size=args.minibatch_size,
                                             shuffle=True,
                                             **kwargs)
        else:
            self.test_loader = test_loader

        if not model:
            self.model = WhatNet().to(device)
        else:
            self.model = model

        #self.loss_func = F.nll_loss
        self.loss_func = nn.CrossEntropyLoss()

        if args.do_adam:
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum)

    def train(self, epoch):
        train(self.args, self.model, self.device, self.train_loader, self.loss_func, self.optimizer, epoch)

    def test(self):
        return test(self.args, self.model, self.device, self.test_loader, self.loss_func)

def train(args, model, device, train_loader, loss_function, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, args.epochs, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader, loss_function):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #test_loss += loss_function(output, target, reduction='sum').item() # sum up batch loss
            test_loss += loss_function(output, target).item()
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset)

class What:
    def __init__(self, args, train_loader=None, test_loader=None, force=False, seed=0, model=None):
        self.args = args
        self.seed = seed
        self.model = model # sinon hydra ne veut pas lors de l'entrainement d'un reseau where
        use_cuda = (not args.no_cuda) and torch.cuda.is_available()
        print('use_cuda', use_cuda)
        torch.manual_seed(args.seed)
        device = torch.device("cuda" if use_cuda else "cpu")
        # suffix = f"{self.args.sf_0}_{self.args.B_sf}_{self.args.noise}_{self.args.contrast}"
        suffix = "{}_{}_{}_{}_{}".format(self.args.sf_0,
                                         self.args.B_sf,
                                         self.args.noise,
                                         self.args.contrast,
                                         self.args.what_offset_std)

        # model_path = f"../data/MNIST_cnn_{suffix}.pt"
        model_path = "../data/MNIST_cnn_{}.pt".format(suffix)

        if model is not None and not force:
            self.model = model
            self.trainer = WhatTrainer(args,
                                       model=self.model,
                                       train_loader=train_loader,
                                       test_loader=test_loader,
                                       device=device,
                                       seed=self.seed)
        elif os.path.exists(model_path) and not force:
            self.model  = torch.load(model_path)
            self.trainer = WhatTrainer(args,
                                       model=self.model,
                                       train_loader=train_loader,
                                       test_loader=test_loader,
                                       device=device,
                                       seed=self.seed)
        else:
            self.trainer = WhatTrainer(args,
                                       model=self.model,
                                       train_loader=train_loader,
                                       test_loader=test_loader,
                                       device=device,
                                       seed=self.seed)
            if self.args.verbose:
                print("Training the What model")
            for epoch in range(1, args.epochs + 1):
                self.trainer.train(epoch)
                self.trainer.test()
            self.model = self.trainer.model
            print(model_path)
            if (args.save_model):
                #torch.save(model.state_dict(), "../data/MNIST_cnn.pt")
                torch.save(self.model, model_path)


def main(args=None, train_loader=None, test_loader=None, path="../data/MNIST_cnn.pt"):

    what = What(args, train_loader=train_loader, test_loader=test_loader)
    return what

if __name__ == '__main__':
    args = init(filename='../data/2019-06-13')
    main(args)
