import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import MotionClouds as mc

import matplotlib.pyplot as plt

def MotionCloudNoise(sf_0=0.125, B_sf=3., alpha=.0, N_pic=28, seed=42):

    mc.N_X, mc.N_Y, mc.N_frame = N_pic, N_pic, 1
    fx, fy, ft = mc.get_grids(mc.N_X, mc.N_Y, mc.N_frame)
    env = mc.envelope_gabor(fx, fy, ft, sf_0=sf_0, B_sf=B_sf, B_theta=np.inf, V_X=0., V_Y=0., B_V=0, alpha=alpha)

    z = mc.rectif(mc.random_cloud(env, seed=seed), contrast=1., method='Michelson')
    z = z.reshape((mc.N_X, mc.N_Y))
    return z, env

class WhatBackground(object):
    def __init__(self, contrast=1., noise=1., sf_0=.1, B_sf=.1):
        self.contrast = contrast
        self.noise = noise
        self.sf_0 = sf_0
        self.B_sf = B_sf

    def __call__(self, sample):

        # sample from the MNIST dataset
        data = np.array(sample)
        data = (data - data.min()) / (data.max() - data.min())
        data *= self.contrast

        seed = hash(tuple(data.flatten())) % (2**31 - 1)
        im_noise, env = MotionCloudNoise(sf_0=self.sf_0,
                                         B_sf=self.B_sf,
                                         seed=seed)
        im_noise = 2 * im_noise - 1  # go to [-1, 1] range
        im_noise = self.noise * im_noise
        
        #plt.imshow(im_noise)
        #plt.show()

        im = np.add(data, im_noise)
        im /= 2  # back to [0, 1] range
        im += .5  # back to a .5 baseline
        im = np.clip(im, 0, 1)
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
        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, args.epochs, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset)

def main(args=None, train_loader=None, test_loader=None, path="../data/MNIST_cnn.pt"):
    # Training settings
    if args is None:
        import argparse

        parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
        parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                            help='input batch size for training (default: 64)')
        parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                            help='input batch size for testing (default: 1000)')
        parser.add_argument('--epochs', type=int, default=10, metavar='N',
                            help='number of epochs to train (default: 10)')
        parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                            help='learning rate (default: 0.01)')
        parser.add_argument('--mean', type=float, default=0.1307, metavar='ME',
                            help='learning rate (default: 0.1307)')
        parser.add_argument('--std', type=float, default=0.3081, metavar='ST',
                            help='learning rate (default: 0.3081)')
        parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                            help='SGD momentum (default: 0.5)')
        parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                            help='how many batches to wait before logging training status')

        parser.add_argument('--save-model', action='store_true', default=True,
                            help='For Saving the current Model')

        args = parser.parse_args()
    else:
        args.batch_size = args.minibatch_size
        args.momentum = .5
        args.save_model = True

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if train_loader is None:
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data',
                           train=True,
                           download=True,
                           transform=transforms.Compose([
                               WhatBackground(),
                               transforms.ToTensor(),
                               transforms.Normalize((args.mean,), (args.std,))
                           ])),
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs)
    if test_loader is None:
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data',
                           train=False,
                           transform=transforms.Compose([
                               WhatBackground(),
                               transforms.ToTensor(),
                               transforms.Normalize((args.mean,), (args.std,))
                           ])),
            batch_size=args.test_batch_size,
            shuffle=True,
            **kwargs)

    model = WhatNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (args.save_model):
        #torch.save(model.state_dict(), "../data/MNIST_cnn.pt")
        torch.save(model, path)



if __name__ == '__main__':
    main()
