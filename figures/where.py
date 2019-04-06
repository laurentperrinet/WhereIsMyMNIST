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
from retina import Display, Retina, minmax


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


class Where():
    def __init__(self, args, save=True, batch_load=False):
        self.args = args
        self.display = Display(args)
        self.retina = Retina(args)
        # https://pytorch.org/docs/stable/nn.html#torch.nn.BCEWithLogitsLoss
        self.loss_func = torch.nn.BCEWithLogitsLoss()
        from what import WhatNet
        model_path = "../data/MNIST_cnn.pt"
        self.What_model = torch.load(model_path)
        #model = torch.load(model_path)
        #self.What_model = WhatNet()
        # torch.save(model.state_dict(), "../data/MNIST_cnn.pt")
        #self.What_model.load_state_dict(torch.load(model_path))
        #self.What_model.eval()
        path = "../data/MNIST_accuracy.npy"
        if os.path.isfile(path):
            self.accuracy_map =  np.load(path)
            if args.verbose:
                print('Loading accuracy... min, max=', self.accuracy_map.min(), self.accuracy_map.max())
        else:
            print('No accuracy data found.')

        # GPU boilerplate
        self.args.no_cuda = self.args.no_cuda or not torch.cuda.is_available()
        # if self.args.verbose: print('cuda?', not self.args.no_cuda)
        self.device = torch.device("cpu" if self.args.no_cuda else "cuda")
        torch.manual_seed(self.args.seed)

        # DATA
        suffix = '_%.3f_%.3f' % (self.args.sf_0, self.args.B_sf) #f'_{self.args.sf_0}_{self.args.B_sf}'
        suffix += '_%.3f_%.3f' % (self.args.noise, self.args.contrast) #f'_{self.args.noise}_{self.args.contrast}'
        suffix += '_%.3f_%.3f' % (self.args.offset_std, self.args.offset_max) #f'_{self.args.offset_std}_{self.args.offset_max}'
        suffix += '_%d_%d' % (self.args.N_theta, self.args.N_azimuth) #f'_{self.args.N_theta}_{self.args.N_azimuth}'
        suffix += '_%d_%d' % (self.args.N_eccentricity, self.args.N_phase) #f'_{self.args.N_eccentricity}_{self.args.N_phase}'
        suffix += '_%.3f_%d' % (self.args.rho, self.args.N_pic) #f'_{self.args.rho}_{self.args.N_pic}'
        # TRAINING DATASET
        filename_dataset = '/tmp/dataset_train' + suffix + '_%d.pt'% self.args.train_batch_size #f'/tmp/dataset_train_{suffix}_{self.args.train_batch_size}.pt'
        if os.path.exists(filename_dataset):
            if args.verbose: print('Loading training dataset')
            self.loader_train  = torch.load(filename_dataset)
        else:
            # MNIST DATASET
            from retina import get_data_loader        
            # SAVING DATASET
            if args.verbose: print('Creating training dataset')
            retina_data, _, accuracy_colliculus, _ = self.generate_data(self.args.train_batch_size, train=True, fullfield=False, batch_load=batch_load)
            # create your dataset, see dev/2019-03-18_precomputed dataset.ipynb
            self.loader_train = DataLoader(TensorDataset(retina_data, accuracy_colliculus), batch_size=args.minibatch_size)
            if save:
                torch.save(self.loader_train, filename_dataset)
            if args.verbose: print('Done!')

        # TESTING DATASET
        filename_dataset = filename_dataset = '/tmp/dataset_test' + suffix + '_%d.pt'% self.args.test_batch_size #f'/tmp/dataset_test_{suffix}_{self.args.test_batch_size}.pt'
        if os.path.exists(filename_dataset):
            if args.verbose: print('Loading testing dataset')
            self.loader_test  = torch.load(filename_dataset)
        else:
            if args.verbose: print('Creating testing dataset')
            retina_data, data_fullfield, accuracy_colliculus, digit_labels = self.generate_data(self.args.test_batch_size, train=False, fullfield=True, batch_load=batch_load)
            # create your dataset, see dev/2019-03-18_precomputed dataset.ipynb
            self.loader_test = DataLoader(TensorDataset(retina_data, data_fullfield, accuracy_colliculus, digit_labels), batch_size=args.test_batch_size)
            if save:
                torch.save(self.loader_test, filename_dataset)
            if args.verbose: print('Done!')


        # MODEL
        self.model = WhereNet(self.args).to(self.device)
        if not self.args.no_cuda:
            # print('doing cuda')
            torch.cuda.manual_seed(self.args.seed)
            self.model.cuda()

        if self.args.do_adam:
            # see https://heartbeat.fritz.ai/basics-of-image-classification-with-pytorch-2f8973c51864
            self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.args.lr, betas=(1.-self.args.momentum, 0.999), eps=1e-8)
        else:
            self.optimizer = optim.SGD(self.model.parameters(),
                                    lr=self.args.lr, momentum=self.args.momentum)

    def generate_data(self, batch_size, train=True, fullfield=True, batch_load=False):
        # loading data
        from retina import get_data_loader
        # loader_full = get_data_loader(batch_size=1, train=train, mean=self.args.mean, std=self.args.std, seed=self.args.seed+train)
        
        # init variables
        if fullfield: # warning = this matrix may fill your memory :-)
            data_fullfield = np.zeros((batch_size, self.args.N_pic, self.args.N_pic))
        else:
            data_fullfield = None
        retina_data = np.zeros((batch_size, self.retina.vsize))
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
                    data_fullfield[i, :, :] =  data_fullfield_
                retina_data[i, :]  =  self.retina.retina(data_fullfield_)
                accuracy_colliculus[i,:], _ = self.retina.accuracy_fullfield(self.accuracy_map, i_offset, j_offset)
            digit_labels = label
        else:
            loader_full = get_data_loader(batch_size=1, train=train, mean=self.args.mean, std=self.args.std, seed=self.args.seed+train)
            for i, (data, label) in enumerate(loader_full):
                if i >= self.args.train_batch_size : print(i); break
                data_fullfield_, i_offset, j_offset = self.display.draw(data[0, 0, :, :].numpy())
                if fullfield:
                    data_fullfield[i, :, :] =  data_fullfield_
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
        retina_data = np.zeros((batch_size, self.retina.vsize))
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
                im_colliculus = self.retina.colliculus[azimuth,eccentricity,:].reshape((self.args.N_pic, self.args.N_pic))
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
        for data_fullfield, retina_data, accuracy_colliculus, digit_labels in dataloader:
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
