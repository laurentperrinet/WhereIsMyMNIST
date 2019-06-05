import os
import numpy as np
import time
import torch
torch.set_default_tensor_type('torch.FloatTensor')
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from display import Display, minmax
from retina import Retina
import MotionClouds as mc
from display import pe
import SLIP


class WhereFill(object):
    def __init__(self, N_pic=128):
        self.N_pic=N_pic

    def __call__(self, sample):
        sample = np.array(sample)
        w = sample.shape[0]
        data = np.zeros((self.N_pic, self.N_pic))
        N_mid = self.N_pic//2
        w_mid = w // 2
        data[N_mid - w_mid: N_mid - w_mid + w,
             N_mid - w_mid: N_mid - w_mid + w] = sample
        return data

class WhereShift(object):
    def __init__(self, i_offset=0, j_offset=0):
        self.i_offset = int(i_offset)
        self.j_offset = int(j_offset)

    def __call__(self, sample):
        #sample = np.array(sample)
        N_pic = sample.shape[0]
        data = np.zeros((N_pic, N_pic))
        i_binf_patch = max(0, -self.i_offset)
        i_bsup_patch = min(N_pic, N_pic - self.i_offset)
        j_binf_patch = max(0, -self.j_offset)
        j_bsup_patch = min(N_pic, N_pic - self.j_offset)
        patch = sample[i_binf_patch:i_bsup_patch,
                j_binf_patch:j_bsup_patch]

        i_binf_data = max(0, self.i_offset)
        i_bsup_data = min(N_pic, N_pic + self.i_offset)
        j_binf_data = max(0, self.j_offset)
        j_bsup_data = min(N_pic, N_pic + self.j_offset)
        data[i_binf_data:i_bsup_data,
             j_binf_data:j_bsup_data] = patch
        return data #.astype('B')

def MotionCloudNoise(sf_0=0.125, B_sf=3., alpha=.0, N_pic=28, seed=42):
    mc.N_X, mc.N_Y, mc.N_frame = N_pic, N_pic, 1
    fx, fy, ft = mc.get_grids(mc.N_X, mc.N_Y, mc.N_frame)
    env = mc.envelope_gabor(fx, fy, ft, sf_0=sf_0, B_sf=B_sf, B_theta=np.inf, V_X=0., V_Y=0., B_V=0, alpha=alpha)

    z = mc.rectif(mc.random_cloud(env, seed=seed), contrast=1., method='Michelson')
    z = z.reshape((mc.N_X, mc.N_Y))
    return z, env

class WhereBackground(object):
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

class WhereMask(object):
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
    
class WhereWhiten:
    def __init__(self, N_pic=128):
        self.N_pic = N_pic
        self.whit = SLIP.Image(pe=pe)
        self.whit.set_size((self.N_pic, self.N_pic))
        # https://github.com/bicv/SLIP/blob/master/SLIP/SLIP.py#L611
        self.K_whitening = self.whit.whitening_filt()
    def __call__(self, sample):
        data = self.whit.FTfilter(sample, self.K_whitening) + 128
        return data.astype('B')

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

class WhereTrainer():
    def __init__(self, args, device='cpu'):
        self.args=args
        self.device=device
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.device != 'cpu' else {}
        transform = transforms.Compose([
            WhereShift(),
            WhereBackground(contrast=args.contrast,
                           noise=args.noise,
                           sf_0=args.sf_0,
                           B_sf=args.B_sf),
            transforms.ToTensor(),
            # transforms.Normalize((args.mean,), (args.std,))
        ])




class Where():
    def __init__(self, args, save=True, batch_load=False):
        self.args = args
        self.display = Display(args)
        self.retina = Retina(args)
        # https://pytorch.org/docs/stable/nn.html#torch.nn.BCEWithLogitsLoss
        self.loss_func = torch.nn.BCEWithLogitsLoss()

        # GPU boilerplate
        self.args.no_cuda = self.args.no_cuda or not torch.cuda.is_available()
        # if self.args.verbose: print('cuda?', not self.args.no_cuda)
        self.device = torch.device("cpu" if self.args.no_cuda else "cuda")
        torch.manual_seed(self.args.seed)

        #########################################################
        # loads a WHAT model (or learns it if not already done) #
        #########################################################
        
        from what import WhatNet
        suffix = f"{self.args.sf_0}_{self.args.B_sf}_{self.args.noise}_{self.args.contrast}"
        model_path = f"../data/MNIST_cnn_{suffix}.pt"
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
        self.What_model = torch.load(model_path)

        ######################
        # Accuracy map setup #
        ######################
        
        # TODO generate an accuracy map for different noise / contrast / sf_0 / B_sf
        path = "../data/MNIST_accuracy.npy"
        if os.path.isfile(path):
            self.accuracy_map =  np.load(path)
            if args.verbose:
                print('Loading accuracy... min, max=', self.accuracy_map.min(), self.accuracy_map.max())
        else:
            print('No accuracy data found.')

        ######################
        # WHERE model setup  #
        ######################
        
        suffix = f'_{self.args.sf_0}_{self.args.B_sf}'
        suffix += f'_{self.args.noise}_{self.args.contrast}'
        suffix += f'_{self.args.offset_std}_{self.args.offset_max}'
        suffix += f'_{self.args.N_theta}_{self.args.N_azimuth}'
        suffix += f'_{self.args.N_eccentricity}_{self.args.N_phase}'
        suffix += f'_{self.args.rho}_{self.args.N_pic}'
            
        self.loader_train = self.data_loader(suffix, 
                                             train=True, 
                                             save=save, 
                                             batch_load=batch_load)
        self.loader_test = self.data_loader(suffix, 
                                            train=False, 
                                            save=save, 
                                            batch_load=batch_load)
        

        # MODEL
        self.model = WhereNet(self.args).to(self.device)
        if not self.args.no_cuda:
            # print('doing cuda')
            torch.cuda.manual_seed(self.args.seed)
            self.model.cuda()

        if self.args.do_adam:
            # see https://heartbeat.fritz.ai/basics-of-image-classification-with-pytorch-2f8973c51864
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=self.args.lr, 
                                        betas=(1.-self.args.momentum, 0.999), 
                                        eps=1e-8)
        else:
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=self.args.lr, 
                                       momentum=self.args.momentum)
            
    def data_loader(self, suffix, train=True, what = False, save=False, batch_load=False):
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
        filename_dataset = f'/tmp/{net}_dataset_{data_type}_{suffix}_{batch_size}.pt'
        if os.path.exists(filename_dataset):
            if self.args.verbose: print(f'Loading {net} {data_type}ing dataset')
            data_loader  = torch.load(filename_dataset)
        else:
            # SAVING DATASET
            if self.args.verbose: print(f'Creating {net} {data_type}ing dataset')
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
            output = self.What_model(im)

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
            if False: #eccentricity < 5:
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
        for retina_data, data_fullfield, accuracy_colliculus, digit_labels in dataloader:
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
