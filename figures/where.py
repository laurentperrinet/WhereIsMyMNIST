import os
import numpy as np
import time
import torch
torch.set_default_tensor_type('torch.FloatTensor')
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.mnist import MNIST as MNIST_dataset
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from display import Display, minmax
from retina import Retina
import MotionClouds as mc
from display import pe, minmax
from PIL import Image
import SLIP
from what import What
from tqdm import tqdm


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


class WhereFill:
    def __init__(self, accuracy_map=None, N_pic=128, keep_label = False, baseline=0):
        self.N_pic=N_pic
        self.accuracy_map = accuracy_map
        self.keep_label = keep_label
        self.baseline = baseline

    def __call__(self, sample_index):
        if self.accuracy_map is None:
            sample = np.array(sample_index[0])
        else:
            # !! target information is lost!
            sample = self.accuracy_map
        w = sample.shape[0]
        data = np.ones((self.N_pic, self.N_pic)) * self.baseline
        N_mid = self.N_pic//2
        w_mid = w // 2
        data[N_mid - w_mid: N_mid - w_mid + w,
             N_mid - w_mid: N_mid - w_mid + w] = sample
        if self.keep_label:
            # !! sample_index[0] contains target information!
            return (data, sample_index[1], sample_index[0])
        else:
            return (data, sample_index[1])    

class WhereShift:
    def __init__(self, args, i_offset=None, j_offset=None, radius=None, theta=None, baseline=0, keep_label = False):
        self.args = args
        self.i_offset = i_offset
        self.j_offset = j_offset
        self.radius = radius
        self.theta = theta
        self.baseline = baseline
        self.keep_label = keep_label

    def __call__(self, sample_index):
        #sample = np.array(sample)
        
        sample = sample_index[0]
        index = sample_index[1]
        
        #print(index)
        np.random.seed(index)
        
        if self.i_offset is not None:
            i_offset = self.i_offset
            if self.j_offset is None:
                j_offset_f = np.random.randn() * self.args.offset_std
                j_offset_f = minmax(j_offset_f, self.args.offset_max)
                j_offset = int(j_offset_f)
            else:
                j_offset = int(self.j_offset)
        else: 
            if self.j_offset is not None:
                j_offset = int(self.j_offset)
                i_offset_f = np.random.randn() * self.args.offset_std
                i_offset_f = minmax(i_offset_f, self.args.offset_max)
                i_offset = int(i_offset_f)
            else: #self.i_offset is None and self.j_offset is None
                if self.theta is None:
                    theta = np.random.rand() * 2 * np.pi
                    #print(theta)
                else:
                    theta = self.theta
                if self.radius is None:
                    radius_f = np.abs(np.random.randn()) * self.args.offset_std
                    radius = minmax(radius_f, self.args.offset_max)
                    #print(radius)
                else:
                    radius = self.radius
                i_offset = int(radius * np.cos(theta))
                j_offset = int(radius * np.sin(theta))
                
        N_pic = sample.shape[0]
        data = np.ones((N_pic, N_pic)) * self.baseline
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
        if self.keep_label:
            # !! sample_index[2] contains target information!
            return data, sample_index[2]
        else:
            return data #, index #.astype('B')
    
        
def MotionCloudNoise(sf_0=0.125, B_sf=3., alpha=.0, N_pic=28, seed=42):
    mc.N_X, mc.N_Y, mc.N_frame = N_pic, N_pic, 1
    fx, fy, ft = mc.get_grids(mc.N_X, mc.N_Y, mc.N_frame)
    env = mc.envelope_gabor(fx, fy, ft, sf_0=sf_0, B_sf=B_sf, B_theta=np.inf, V_X=0., V_Y=0., B_V=0, alpha=alpha)

    z = mc.rectif(mc.random_cloud(env, seed=seed), contrast=1., method='Michelson')
    z = z.reshape((mc.N_X, mc.N_Y))
    return z, env

class RetinaBackground:
    def __init__(self, contrast=1., noise=1., sf_0=.1, B_sf=.1):
        self.contrast = contrast
        self.noise = noise
        self.sf_0 = sf_0
        self.B_sf = B_sf

    def __call__(self, sample):

        # sample from the MNIST dataset
        #data = np.array(sample)
        data = sample
        N_pic = data.shape[0]
        if data.min() != data.max():
            data = (data - data.min()) / (data.max() - data.min())
        else:
            data = np.zeros((N_pic, N_pic))
        data *= self.contrast

        seed = hash(tuple(data.flatten())) % (2 ** 31 - 1)
        im_noise, env = MotionCloudNoise(sf_0=self.sf_0,
                                         B_sf=self.B_sf,
                                         N_pic=N_pic,
                                         seed=seed)
        im_noise = 2 * im_noise - 1  # go to [-1, 1] range
        im_noise = self.noise * im_noise

        # plt.imshow(im_noise)
        # plt.show()

        im = np.add(data, im_noise)
        im /= 2  # back to [0, 1] range
        im += .5  # back to a .5 baseline
        im = np.clip(im, 0, 1)
        im = im.reshape((N_pic, N_pic))
        im *= 255
        return im #.astype('B')  # Variable(torch.DoubleTensor(im)) #.to(self.device)

class RetinaMask:
    def __init__(self, N_pic=128):
        self.N_pic = N_pic
    def __call__(self, sample):
        data = np.array(sample)
        #d_min = data.min()
        #d_max = data.max()
        data -= 128 #/ 255 #(data - d_min) / (d_max - d_min)
        x, y = np.mgrid[-1:1:1j * self.N_pic, -1:1:1j * self.N_pic]
        R = np.sqrt(x ** 2 + y ** 2)
        mask = 1. * (R < 1)
        #print(data.shape, mask.shape)
        #print('mask', mask.min(), mask.max(), mask[0, 0])
        data *= mask.reshape((self.N_pic, self.N_pic))
        data += 128
        #data *= 255
        #data = np.clip(data, 0, 255)
        return data #.astype('B')
    
class RetinaWhiten:
    def __init__(self, N_pic=128):
        self.N_pic = N_pic
        self.whit = SLIP.Image(pe=pe)
        self.whit.set_size((self.N_pic, self.N_pic))
        # https://github.com/bicv/SLIP/blob/master/SLIP/SLIP.py#L611
        self.K_whitening = self.whit.whitening_filt()
    def __call__(self, sample):
        data = self.whit.FTfilter(sample, self.K_whitening) + 128
        return data.astype('B')

class RetinaTransform:
    def __init__(self, retina_transform_vector):
        self.retina_transform_vector = retina_transform_vector
    def __call__(self, sample):
        data = self.retina_transform_vector @ np.ravel(sample)
        return data
    
class FullfieldRetinaTransform:
    def __init__(self, retina_transform_vector):
        self.retina_transform_vector = retina_transform_vector
    def __call__(self, sample):
        transformed_data = self.retina_transform_vector @ np.ravel(sample)
        return (transformed_data, sample)
    
    
class CollTransform:
    def __init__(self, colliculus_transform_vector):
        self.colliculus_transform_vector = colliculus_transform_vector
    def __call__(self, target):
        data = self.colliculus_transform_vector @ np.ravel(target)
        return data

class FullfieldCollTransform:
    def __init__(self, colliculus_transform_vector, keep_label = False):
        self.colliculus_transform_vector = colliculus_transform_vector
        self.keep_label = keep_label
    def __call__(self, data):
        if self.keep_label:
            sample = data[0]
            label = data[1]
        else:
            sample = data
        transformed_data = self.colliculus_transform_vector @ np.ravel(sample)
        if self.keep_label:
            return (transformed_data, sample, label)
        else:
            return (transformed_data, sample)
    
class ToFloatTensor:
    def __init__(self):
        pass
    def __call__(self, data):
        return Variable(torch.FloatTensor(data))
    
class FullfieldToFloatTensor:
    def __init__(self):
        pass
    def __call__(self, data):
        return (Variable(torch.FloatTensor(data[0])), Variable(torch.FloatTensor(data[1])))
    
    
class Normalize:
    def __init__(self):
        pass
    def __call__(self, data):
        data -= data.mean()
        data /= data.std()
        return data

class WhereNet(torch.nn.Module):
    def __init__(self, args):
        super(WhereNet, self).__init__()
        self.args = args
        self.bn1 = torch.nn.Linear(args.N_theta*args.N_azimuth*args.N_eccentricity*args.N_phase, args.dim1, bias=args.bias_deconv)
        #https://raw.githubusercontent.com/MorvanZhou/PyTorch-Tutorial/master/tutorial-contents/504_batch_normalization.py
        self.bn1_bn = nn.BatchNorm1d(args.dim1, momentum=1-args.bn1_bn_momentum)
        self.bn2 = torch.nn.Linear(args.dim1, args.dim2, bias=args.bias_deconv)
        self.bn2_bn = nn.BatchNorm1d(args.dim2, momentum=1-args.bn2_bn_momentum)
        self.bn3 = torch.nn.Linear(args.dim2, args.N_azimuth*args.N_eccentricity, bias=args.bias_deconv)

    def forward(self, image):
        x = F.relu(self.bn1(image))
        if self.args.bn1_bn_momentum>0: x = self.bn1_bn(x)
        x = F.relu(self.bn2(x))
        if self.args.p_dropout>0: x = F.dropout(x, p=self.args.p_dropout)
        if self.args.bn2_bn_momentum>0: x = self.bn2_bn(x)
        x = self.bn3(x)
        return x

def where_suffix(args):
    suffix = '_{}_{}'.format(args.sf_0, args.B_sf)
    suffix += '_{}_{}'.format(args.noise, args.contrast)
    suffix += '_{}_{}'.format(args.offset_std, args.offset_max)
    suffix += '_{}_{}'.format(args.N_theta, args.N_azimuth)
    suffix += '_{}_{}'.format(args.N_eccentricity, args.N_phase)
    suffix += '_{}_{}'.format(args.rho, args.N_pic)
    return suffix

class WhereTrainer:
    def __init__(self, args, 
                 what_model=None, 
                 model=None, 
                 train_loader=None, 
                 test_loader=None, 
                 device='cpu', 
                 generate_data=True,
                 retina=None):
        self.args=args
        self.device=device
        if retina:
            self.retina=retina
        else:
            self.retina = Retina(args)
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.device != 'cpu' else {}
        
        # suffix = f"{args.sf_0}_{args.B_sf}_{args.noise}_{args.contrast}"
        suffix_what = "{}_{}_{}_{}".format(args.sf_0, args.B_sf, args.noise, args.contrast)
        
        ## DATASET TRANSFORMS     
        # accuracy_path = f"../data/MNIST_accuracy_{suffix}.pt"
        accuracy_path = "../data/MNIST_accuracy_{}.pt".format(suffix_what)
        if not os.path.isfile(accuracy_path):
            self.accuracy_map = np.load('../data/MNIST_accuracy.npy')
        else:
            self.accuracy_map = np.load(accuracy_path)
        
        ## DATASET TRANSFORMS     
        self.transform = transforms.Compose([
            WhereFill(N_pic=args.N_pic),
            WhereShift(args),
            RetinaBackground(contrast=args.contrast,
                             noise=args.noise,
                             sf_0=args.sf_0,
                             B_sf=args.B_sf),
            RetinaMask(N_pic=args.N_pic),
            RetinaWhiten(N_pic=args.N_pic),
            RetinaTransform(self.retina.retina_transform_vector),
            # ToFloatTensor()
            # transforms.Normalize((args.mean,), (args.std,))
        ])
        
        self.fullfield_transform = transforms.Compose([
            WhereFill(N_pic=args.N_pic),
            WhereShift(args),
            RetinaBackground(contrast=args.contrast,
                             noise=args.noise,
                             sf_0=args.sf_0,
                             B_sf=args.B_sf),
            RetinaMask(N_pic=args.N_pic),
            RetinaWhiten(N_pic=args.N_pic),
            FullfieldRetinaTransform(self.retina.retina_transform_vector),
            # FullfieldToFloatTensor()
            # transforms.Normalize((args.mean,), (args.std,))
        ])
        
        self.target_transform=transforms.Compose([
                               WhereFill(accuracy_map=self.accuracy_map, N_pic=args.N_pic, baseline=0.1),
                               WhereShift(args, baseline = 0.1),
                               CollTransform(self.retina.colliculus_transform_vector),
                               #ToFloatTensor()
                           ])
        
        self.fullfield_target_transform=transforms.Compose([
                               WhereFill(accuracy_map=self.accuracy_map, keep_label = True, N_pic=args.N_pic, baseline=0.1),
                               WhereShift(args, baseline = 0.1, keep_label = True),
                               FullfieldCollTransform(self.retina.colliculus_transform_vector, keep_label = True),
                               #FullfieldToFloatTensor()
                           ])
        
        suffix = where_suffix(args)
        
        if not train_loader:
            self.init_data_loader(args, suffix, 
                                  train=True, 
                                  generate_data=generate_data, 
                                  fullfield = False)
        else:
            self.train_loader = train_loader
        
        if not test_loader:
            self.init_data_loader(args, suffix, 
                                  train=False, 
                                  generate_data=generate_data, 
                                  fullfield = True)
        else:
            self.test_loader = test_loader
            
        if not model:
            self.model = WhereNet(args).to(device)
        else:
            self.model = model
            
        self.loss_func = torch.nn.BCEWithLogitsLoss()
        
        if args.do_adam:
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum)
            
        #if args.do_adam:
        #    # see https://heartbeat.fritz.ai/basics-of-image-classification-with-pytorch-2f8973c51864
        #    self.optimizer = optim.Adam(self.model.parameters(),
        #                                lr=args.lr, 
        #                                betas=(1.-args.momentum, 0.999), 
        #                                eps=1e-8)
        #else:
        #    self.optimizer = optim.SGD(self.model.parameters(),
        #                               lr=args.lr, 
        #                               momentum=args.momentum)
        
    def init_data_loader(self, args, suffix, train=True, generate_data=True, fullfield = False, force_generate = False):
        if train:
            use = 'train'
        else:
            use = 'test'
        data_loader_path = '/tmp/where_{}_dataset_{}_{}.pt'.format(use, suffix, args.minibatch_size)
        if os.path.isfile(data_loader_path) and not force_generate:
            if self.args.verbose: 
                print('Loading {}ing dataset'.format(use))
            data_loader = torch.load(data_loader_path)
        else:
            if fullfield:
                dataset = MNIST('../data',
                        train=train,
                        download=True,
                        transform=self.fullfield_transform,
                        target_transform=self.fullfield_target_transform,
                        )
            else:
                dataset = MNIST('../data',
                        train=train,
                        download=True,
                        transform=self.transform,
                        target_transform=self.target_transform,
                        )
            data_loader = DataLoader(dataset,
                                     batch_size=args.minibatch_size,
                                     shuffle=True)
            if generate_data:
                if self.args.verbose: 
                    print('Generating {}ing dataset'.format(use))
                for i, (data, acc) in enumerate(data_loader):
                    if self.args.verbose: 
                        print(i, (i+1) * args.minibatch_size)
                    if i == 0:
                        if fullfield:
                            full_data_features = data[0]
                            full_acc_features = acc[0]
                            full_data_fullfield = data[1]
                            full_acc_fullfield = acc[1]
                            full_label = acc[2]
                        else:
                            full_data = data
                            full_acc = acc
                    else:
                        if fullfield:
                            full_data_features = torch.cat((full_data_features, data[0]), 0)
                            full_acc_features = torch.cat((full_acc_features, acc[0]), 0)
                            full_data_fullfield = torch.cat((full_data_fullfield, data[1]), 0)
                            full_acc_fullfield = torch.cat((full_acc_fullfield, acc[1]), 0)
                            full_label = torch.cat((full_label, acc[2]), 0)
                        else:
                            full_data = torch.cat((full_data, data), 0)
                            full_acc = torch.cat((full_acc, acc), 0)
                if fullfield:
                    dataset = TensorDataset(full_data_features, 
                                            full_data_fullfield, 
                                            full_acc_features,
                                            full_acc_fullfield,
                                            full_label)
                else:
                    dataset = TensorDataset(full_data, full_acc)
                
                data_loader = DataLoader(dataset,
                                         batch_size=args.minibatch_size,
                                         shuffle=True)
                torch.save(data_loader, data_loader_path)
                if self.args.verbose: 
                    print('Done!')
        if train:
            self.train_loader = data_loader
        else:
            self.test_loader = data_loader
    
    def train(self, epoch):
        train(self.args, self.model, self.device, self.train_loader, self.loss_func, self.optimizer, epoch)
    
    def test(self):
        return test(self.args, self.model, self.device, self.test_loader, self.loss_func)

def train(args, model, device, train_loader, loss_function, optimizer, epoch):
    # setting up training
    '''if seed is None:
        seed = self.args.seed
    model.train() # set training mode
    for epoch in tqdm(range(1, self.args.epochs + 1), desc='Train Epoch' if self.args.verbose else None):
        loss = self.train_epoch(epoch, seed, rank=0)
        # report classification results
        if self.args.verbose and self.args.log_interval>0:
            if epoch % self.args.log_interval == 0:
                status_str = '\tTrain Epoch: {} \t Loss: {:.6f}'.format(epoch, loss)
                try:
                    #from tqdm import tqdm
                    tqdm.write(status_str)
                except Exception as e:
                    print(e)
                    print(status_str)'''
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = Variable(torch.FloatTensor(data.float())).to(device)
        target = Variable(torch.FloatTensor(target.float())).to(device)
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
    with torch.no_grad():
        for batch_idx, (data, data_fullfield, target, target_fullfield, label) in enumerate(test_loader):
            data = Variable(torch.FloatTensor(data.float())).to(device)
            target = Variable(torch.FloatTensor(target.float())).to(device)            
            output = model(data)
            test_loss += loss_function(output, target).item() # sum up batch loss
            #if batch_idx % args.log_interval == 0:
            #    print('i = {}, test done on {:.0f}% of the test dataset.'.format(batch_idx, 100. * batch_idx / len(test_loader.dataset)))

    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
    return test_loss
            
class Where():
    def __init__(self, args, 
                 save=True, 
                 batch_load=False, 
                 force_training=False, 
                 model=None,
                 train_loader=None, 
                 test_loader=None, 
                 generate_data=True,
                 what_model=None,
                 retina=None,
                 trainer=None):
        
        self.args = args

        # GPU boilerplate
        self.args.no_cuda = self.args.no_cuda or not torch.cuda.is_available()
        # if self.args.verbose: print('cuda?', not self.args.no_cuda)
        self.device = torch.device("cpu" if self.args.no_cuda else "cuda")
        torch.manual_seed(self.args.seed)

        #########################################################
        # loads a WHAT model (or learns it if not already done) #
        #########################################################
        if what_model:
            self.what_model = what_model
        else:
            what = What(args) # trains the what_model if needed
            self.what_model = what.model.to(self.device)
            
                
        '''from what import WhatNet
        # suffix = f"{self.args.sf_0}_{self.args.B_sf}_{self.args.noise}_{self.args.contrast}"
        suffix = "{}_{}_{}_{}".format(self.args.sf_0, self.args.B_sf, self.args.noise, elf.args.contrast)
        # model_path = f"../data/MNIST_cnn_{suffix}.pt"
        model_path = "../data/MNIST_cnn_{}.pt".format(suffix)
        if not os.path.isfile(model_path):
            train_loader = self.data_loader(suffix, 
                                            train=True,
                                            what = True,
                                            save=save, 
                                            batch_load=batch_load)
            test_loader = self.data_loader(suffix, 
                                            train=False, 
                                            what = True,
                                            save=save, 
                                            batch_load=batch_load)
            print('Training the "what" model ', model_path)
            from what import main
            main(args=self.args, 
                 train_loader=train_loader, 
                 test_loader=test_loader, 
                 path=model_path)
        self.What_model = torch.load(model_path)'''
        
        ######################
        # Accuracy map setup #
        ######################
        
        # TODO generate an accuracy map for different noise / contrast / sf_0 / B_sf
        '''path = "../data/MNIST_accuracy.npy"
        if os.path.isfile(path):
            self.accuracy_map =  np.load(path)
            if args.verbose:
                print('Loading accuracy... min, max=', self.accuracy_map.min(), self.accuracy_map.max())
        else:
            print('No accuracy data found.')'''
            
        
        ######################
        # WHERE model setup  #
        ######################
        
        # suffix = f'_{self.args.sf_0}_{self.args.B_sf}'
        # suffix += f'_{self.args.noise}_{self.args.contrast}'
        # suffix += f'_{self.args.offset_std}_{self.args.offset_max}'
        # suffix += f'_{self.args.N_theta}_{self.args.N_azimuth}'
        # suffix += f'_{self.args.N_eccentricity}_{self.args.N_phase}'
        # suffix += f'_{self.args.rho}_{self.args.N_pic}'

        '''suffix = '_{}_{}'.format(self.args.sf_0, self.args.B_sf)
        suffix += '_{}_{}'.format(self.args.noise, self.args.contrast)
        suffix += '_{}_{}'.format(self.args.offset_std, self.args.offset_max)
        suffix += '_{}_{}'.format(self.args.N_theta, self.args.N_azimuth)
        suffix += '_{}_{}'.format(self.args.N_eccentricity, self.args.N_phase)
        suffix += '_{}_{}'.format(self.args.rho, self.args.N_pic)'''
        
        suffix = where_suffix(args)
        model_path = '/tmp/where_model_{}.pt'.format(suffix)
        if model:
            self.model = model
            if trainer:
                self.trainer = trainer
            else:
                self.trainer = WhereTrainer(args, 
                                       model=self.model,
                                       train_loader=train_loader, 
                                       test_loader=test_loader, 
                                       device=self.device,
                                       retina=retina)
        elif trainer:
            self.model = trainer.model
            self.trainer = trainer
        elif os.path.exists(model_path) and not force_training:
            self.model  = torch.load(model_path)
            self.trainer = WhereTrainer(args, 
                                       model=self.model,
                                       train_loader=train_loader, 
                                       test_loader=test_loader, 
                                       device=self.device,
                                       retina=retina)
        else:                                                       
            self.trainer = WhereTrainer(args, 
                                       train_loader=train_loader, 
                                       test_loader=test_loader, 
                                       device=self.device,
                                       generate_data=generate_data,
                                       retina=retina)
            for epoch in range(1, args.epochs + 1):
                self.trainer.train(epoch)
                self.trainer.test()
            self.model = self.trainer.model
            print(model_path)
            if (args.save_model):
                #torch.save(model.state_dict(), "../data/MNIST_cnn.pt")
                torch.save(self.model, model_path) 
                print('Model saved at', model_path)
                
        self.accuracy_map = self.trainer.accuracy_map
            
        self.display = Display(args)
        if retina:
            self.retina = retina
        else:
            self.retina = self.trainer.retina
        # https://pytorch.org/docs/stable/nn.html#torch.nn.BCEWithLogitsLoss
        self.loss_func = self.trainer.loss_func #torch.nn.BCEWithLogitsLoss()        
        
        '''self.loader_train = self.data_loader(suffix, 
                                             train=True, 
                                             save=save, 
                                             batch_load=batch_load)
        self.loader_test = self.data_loader(suffix, 
                                            train=False, 
                                            save=save, 
                                            batch_load=batch_load)'''
        
        if train_loader:
            self.loader_train = train_loader
        else:
            self.loader_train = self.trainer.train_loader
        if test_loader:
            self.loader_test = test_loader
        else:
            self.loader_test = self.trainer.test_loader

        # MODEL
        '''self.model = WhereNet(self.args).to(self.device)'''
        if not self.args.no_cuda:
            # print('doing cuda')
            torch.cuda.manual_seed(self.args.seed)
            self.model.cuda()
        
            
    def data_loader(self, suffix, train=True, what=False, save=False, batch_load=False):
        """
        Arguments
        ---------
        suffix:
            temporary data file suffix (string)
        train:
            train/test dataset switch (boolean)
        what:
            what/where network switch (boolean)
        save:
            data save switch (boolean)
        batch_load:
            batch/unitary data read out switch (boolean)
        Returns
        -------
        a pytorch data loader
        """
        fullfield = True
        if train:
            batch_size = self.args.train_batch_size
            data_type = 'train'
        else:
            batch_size = self.args.test_batch_size
            data_type = 'test'
        if what:
            net = 'WHAT'
            do_extract = True
        else:
            net = ''
            do_extract = False
            if train:
                fullfield = False
        # filename_dataset = f'/tmp/{net}_dataset_{data_type}_{suffix}_{batch_size}.pt'
        filename_dataset = '/tmp/{}_dataset_{}_{}_{}.pt'.format(net, data_type, suffix, batch_size)
        if os.path.exists(filename_dataset):
            # if self.args.verbose: print(f'Loading {net} {data_type}ing dataset')
            if self.args.verbose: print('Loading {} {}ing dataset'.format(net, data_type))
            data_loader  = torch.load(filename_dataset)
        else:
            # SAVING DATASET
            # if self.args.verbose: print(f'Creating {net} {data_type}ing dataset')
            if self.args.verbose: print('Creating {} {}ing dataset'.format(net, data_type))
            retina_data, full_data, accuracy_maps, digit_labels = self.generate_data(batch_size, 
                                                                  train=train, 
                                                                  fullfield=True, 
                                                                  batch_load=batch_load, 
                                                                  do_extract=do_extract)
            # print('data_extract.shape=', data_extract.shape)
            # create your dataset, see dev/2019-03-18_precomputed dataset.ipynb
            if what:
                data_loader = DataLoader(TensorDataset(full_data, digit_labels), 
                                         batch_size=self.args.minibatch_size)
            else:
                data_loader = DataLoader(TensorDataset(retina_data, full_data, accuracy_maps, digit_labels),
                                         batch_size=self.args.minibatch_size)
            if save:
                torch.save(data_loader, filename_dataset)
            if self.args.verbose: print('Done!')
        return data_loader

    def generate_data(self, batch_size, train=True, fullfield=True, batch_load=False, do_extract=False):
        """
        Arguments
        ---------
        train:
            train/test dataset switch (boolean)
        fullfield:
            2D images return switch (boolean)
        batch_load:
            batch/unitary data read out switch (boolean)
        do_extract:
            central snippet extraction switch (boolean)
        Returns
        -------
        retina_data:
            a tensor of retina-encoded input vectors (torch FloatTensor)
        data_fullfield (optional):
            if not do_extract:
                a tensor of full-resolution 2D images (torch FloatTensor)
            else:
                a tensor of 2D snippets around the target position
        accuracy_colliculus:
            a tensor of retina-encoded output vector (torch FloatTensor)
        digit_labels:
            a tensor of integer labels (torch LongTensor)
        """
        
        # loading data
        from display import get_data_loader
        # loader_full = get_data_loader(batch_size=1, train=train, mean=self.args.mean, std=self.args.std, seed=self.args.seed+train)

        # init variables
        if fullfield: # warning = this matrix may fill your memory :-)
            if do_extract:
                data_fullfield = np.zeros((batch_size, 1, self.args.w, self.args.w))
            else:
                data_fullfield = np.zeros((batch_size, self.args.N_pic, self.args.N_pic))
        else:
            data_fullfield = None
        retina_data = np.zeros((batch_size, self.retina.feature_vector_size))
        accuracy_colliculus = np.zeros((batch_size, self.args.N_azimuth * self.args.N_eccentricity))
        digit_labels = np.zeros(batch_size)
        # cycling over digits
        label = None
        if batch_load:
            if train:
                size = self.args.train_batch_size
                if self.args.verbose: print('train dataset, size = ', size)
            else:
                size = self.args.test_batch_size
                if self.args.verbose: print('test dataset, size = ', size)
            loader_full = get_data_loader(batch_size=size, train=train, mean=self.args.mean, std=self.args.std, seed=self.args.seed+1)
            data, label = next(iter(loader_full))
            for i in range(size):
                if i%1000 == 0: print(i)
                data_fullfield_, i_offset, j_offset = self.display.draw(data[i, 0, :, :].numpy())
                if fullfield:
                    if do_extract:
                        data_fullfield[i, 0, :, :] = self.extract(data_fullfield_, i_offset, j_offset)
                    else:
                        data_fullfield[i, :, :] =  data_fullfield_
                if not do_extract:
                    retina_data[i, :]  =  self.retina.retina(data_fullfield_)
                    accuracy_colliculus[i,:], _ = self.retina.accuracy_fullfield(self.accuracy_map, i_offset, j_offset)
            digit_labels = label
        else:
            loader_full = get_data_loader(batch_size=1, train=train, mean=self.args.mean, std=self.args.std, seed=self.args.seed+train)
            for i, (data, label) in enumerate(loader_full):
                if i >= batch_size : break
                data_fullfield_, i_offset, j_offset = self.display.draw(data[0, 0, :, :].numpy())
                if fullfield:
                    if do_extract:
                        data_fullfield[i, 0, :, :] = self.extract(data_fullfield_, i_offset, j_offset)
                    else:
                        data_fullfield[i, :, :] =  data_fullfield_
                if not do_extract:
                    retina_data[i, :]  =  self.retina.retina(data_fullfield_)
                    accuracy_colliculus[i,:], _ = self.retina.accuracy_fullfield(self.accuracy_map, i_offset, j_offset)
                digit_labels[i] = label#.detach.numpy()
            digit_labels = Variable(torch.LongTensor(digit_labels))

       # converting to torch format
        retina_data = Variable(torch.FloatTensor(retina_data)).to(self.device)
        if fullfield:
            data_fullfield = Variable(torch.FloatTensor(data_fullfield)).to(self.device)
        accuracy_colliculus = Variable(torch.FloatTensor(accuracy_colliculus)).to(self.device)
        #digit_labels = Variable(torch.FloatTensor(digit_labels)).to(self.device)
        digit_labels = digit_labels.to(self.device)
        # returning
        return retina_data, data_fullfield, accuracy_colliculus, digit_labels
    
    
    def minibatch(self, data):
        # TODO: utiliser https://laurentperrinet.github.io/sciblog/posts/2018-09-07-extending-datasets-in-pytorch.html
        batch_size = data.shape[0]
        retina_data = np.zeros((batch_size, self.retina.feature_vector_size))
        accuracy_colliculus = np.zeros((batch_size, self.args.N_azimuth * self.args.N_eccentricity))
        data_fullfield = np.zeros((batch_size, self.args.N_pic, self.args.N_pic))
        positions =[]

        for i in range(batch_size):
            #print(i, data[i, 0, :, :].numpy().shape)
            data_fullfield[i, :, :], i_offset, j_offset = self.display.draw(data[i, 0, :, :].numpy())
            positions.append(dict(i_offset=i_offset, j_offset=j_offset))
            # TODO use one shot matrix multiplication
            retina_data[i, :]  =  self.retina.retina(data_fullfield[i, :, :])
            accuracy_colliculus[i,:], _ = self.retina.accuracy_fullfield(self.accuracy_map, i_offset, j_offset)

        retina_data = Variable(torch.FloatTensor(retina_data))
        accuracy_colliculus = Variable(torch.FloatTensor(accuracy_colliculus))
        retina_data, accuracy_colliculus = retina_data.to(self.device), accuracy_colliculus.to(self.device)

        return positions, data_fullfield, retina_data, accuracy_colliculus

    def extract(self, data_fullfield, i_offset, j_offset):
        mid = self.args.N_pic//2
        rad = self.args.w//2

        im = data_fullfield[(mid+i_offset-rad):(mid+i_offset+rad),
                            (mid+j_offset-rad):(mid+j_offset+rad)]

        im = np.clip(im, 0.5, 1)
        im = (im-.5)*2
        return im

    def classify_what(self, im):
        im = (im-self.args.mean)/self.args.std
        if im.ndim ==2:
            im = Variable(torch.FloatTensor(im[None, None, ::]))
        else:
            im = Variable(torch.FloatTensor(im[:, None, ::]))
        with torch.no_grad():
            output = self.what_model(im)

        return np.exp(output)


    def pred_accuracy(self, retina_data):
        # Predict classes using images from the train set
        #retina_data = Variable(torch.FloatTensor(retina_data))
        prediction = self.model(retina_data)
        # transform in a probability in collicular coordinates
        pred_accuracy_colliculus = F.sigmoid(prediction).detach().numpy()
        return pred_accuracy_colliculus

    def index_prediction(self, pred_accuracy_colliculus, do_shortcut=False):
        if do_shortcut:
            test = pred_accuracy_colliculus.reshape((self.args.N_azimuth, self.args.N_eccentricity))
            indices_ij = np.where(test == max(test.flatten()))
            azimuth = indices_ij[0][0]
            eccentricity = indices_ij[1][0]
            if true: #eccentricity < 5:
                im_colliculus = self.retina.colliculus_transform[azimuth, eccentricity, :].reshape((self.args.N_pic, self.args.N_pic))
            else:
                im_colliculus = self.retina.accuracy_invert(pred_accuracy_colliculus)
        else:
            im_colliculus = self.retina.accuracy_invert(pred_accuracy_colliculus)

        # see https://laurentperrinet.github.io/sciblog/posts/2016-11-17-finding-extremal-values-in-a-nd-array.html
        i, j = np.unravel_index(np.argmax(im_colliculus.ravel()), im_colliculus.shape)
        i_pred = i - self.args.N_pic//2
        j_pred = j - self.args.N_pic//2
        return i_pred, j_pred

    def test_what(self, data_fullfield, pred_accuracy_colliculus, digit_labels, do_control=False):
        batch_size = pred_accuracy_colliculus.shape[0]
        # extract foveal images
        im = np.zeros((batch_size, self.args.w, self.args.w))
        for idx in range(batch_size):
            if do_control:
                i_pred, j_pred = 0, 0
            else:
                i_pred, j_pred = self.index_prediction(pred_accuracy_colliculus[idx, :])
            # avoid going beyond the border (for extraction)
            border = self.args.N_pic//2 - self.args.w//2
            i_pred, j_pred = minmax(i_pred, border), minmax(j_pred, border)
            im[idx, :, :] = self.extract(data_fullfield[idx, :, :], i_pred, j_pred)
        # classify those images
        proba = self.classify_what(im).numpy()
        pred = proba.argmax(axis=1) # get the index of the max log-probability
        #print(im.shape, batch_size, proba.shape, pred.shape, label.shape)
        return (pred==digit_labels.numpy())*1.


    def train(self, path=None, seed=None):
        if not path is None:
            # using a data_cache
            if os.path.isfile(path):
                #self.model.load_state_dict(torch.load(path))
                self.model = torch.load(path)
                print('Loading file', path)
            else:
                #print('Training model...')
                self.train(path=None, seed=seed)
                torch.save(self.model, path)
                #torch.save(self.model.state_dict(), path) #save the neural network state
                print('Model saved at', path)
        else:
            from tqdm import tqdm
            # setting up training
            if seed is None:
                seed = self.args.seed

            self.model.train() # set training mode
            for epoch in tqdm(range(1, self.args.epochs + 1), desc='Train Epoch' if self.args.verbose else None):
                loss = self.train_epoch(epoch, seed, rank=0)
                # report classification results
                if self.args.verbose and self.args.log_interval>0:
                    if epoch % self.args.log_interval == 0:
                        status_str = '\tTrain Epoch: {} \t Loss: {:.6f}'.format(epoch, loss)
                        try:
                            #from tqdm import tqdm
                            tqdm.write(status_str)
                        except Exception as e:
                            print(e)
                            print(status_str)
            self.model.eval()

    def train_epoch(self, epoch, seed, rank=0):
        torch.manual_seed(seed + epoch + rank*self.args.epochs)
        for retina_data, accuracy_colliculus in self.loader_train:
            # Clear all accumulated gradients
            self.optimizer.zero_grad()

            # Predict classes using images from the train set
            prediction = self.model(retina_data)
            # Compute the loss based on the predictions and actual labels
            loss = self.loss_func(prediction, accuracy_colliculus)
            # TODO try with the reverse divergence
            # loss = self.loss_func(accuracy_colliculus, prediction)
            # Backpropagate the loss
            loss.backward()
            # Adjust parameters according to the computed gradients
            self.optimizer.step()

        return loss.item()

    def test(self, dataloader=None):
        if dataloader is None:
            dataloader = self.loader_test
        self.model.eval()
        accuracy = []
        for retina_data, data_fullfield, accuracy_colliculus, accuracy_fullfield, digit_labels in dataloader:
            retina_data = Variable(torch.FloatTensor(retina_data.float())).to(self.device)
            pred_accuracy_colliculus = self.pred_accuracy(retina_data)
            # use that predicted map to extract the foveal patch and classify the image
            correct = self.test_what(data_fullfield.numpy(), pred_accuracy_colliculus, digit_labels.squeeze())
            accuracy.append(correct.mean())

        return np.mean(accuracy)

    def show(self, gamma=.5, noise_level=.4, transpose=True, only_wrong=False):
        for idx, (data, target) in enumerate(self.display.loader_test):
            #data, target = next(iter(self.dataset.test_loader))
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            if only_wrong and not pred == target:
                #print(target, self.dataset.dataset.imgs[self.dataset.test_loader.dataset.indices[idx]])
                print('target:' + ' '.join('%5s' % self.dataset.dataset.classes[j] for j in target))
                print('pred  :' + ' '.join('%5s' % self.dataset.dataset.classes[j] for j in pred))
                #print(target, pred)

                from torchvision.utils import make_grid
                npimg = make_grid(data, normalize=True).numpy()
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=((13, 5)))
                import numpy as np
                if transpose:
                    ax.imshow(np.transpose(npimg, (1, 2, 0)))
                else:
                    ax.imshow(npimg)
                plt.setp(ax, xticks=[], yticks=[])

                return fig, ax
            else:
                return None, None


    def main(self, path=None, seed=None):
        self.train(path=path, seed=seed)
        Accuracy = self.test()
        return Accuracy

if __name__ == '__main__':

    from main import init, MetaML
    import os
    filename = 'figures/accuracy.pdf'
    if not os.path.exists(filename) :
        args = init(verbose=0, log_interval=0, epochs=20)
        from gaze import MetaML
        mml = MetaML(args)
        Accuracy = mml.protocol(args, 42)
        print('Accuracy', Accuracy[:-1].mean(), '+/-', Accuracy[:-1].std())
        import numpy as np
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=((8, 5)))
        n, bins, patches = ax.hist(Accuracy[:-1]*100, bins=np.linspace(0, 100, 100), alpha=.4)
        ax.vlines(np.median(Accuracy[:-1])*100, 0, n.max(), 'g', linestyles='dashed', label='median')
        ax.vlines(25, 0, n.max(), 'r', linestyles='dashed', label='chance level')
        ax.vlines(100, 0, n.max(), 'k', label='max')
        ax.set_xlabel('Accuracy (%)')
        ax.set_ylabel('Smarts')
        ax.legend(loc='best')
        plt.show()
        plt.savefig(filename)
        plt.savefig(filename.replace('.pdf', '.png'))
