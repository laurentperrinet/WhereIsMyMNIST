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
from retina import Retina
from display import Display

class WhatWhereNet(torch.nn.Module):
    def __init__(self, args):
        super(WhatWhereNet, self).__init__()
        self.args = args
        self.bn1 = torch.nn.Linear(args.N_theta*args.N_azimuth*args.N_eccentricity*args.N_phase, args.dim1, bias=args.bias_deconv)
        #https://raw.githubusercontent.com/MorvanZhou/PyTorch-Tutorial/master/tutorial-contents/504_batch_normalization.py
        self.bn1_bn = nn.BatchNorm1d(args.dim1, momentum=1-args.bn1_bn_momentum)
        self.bn2 = torch.nn.Linear(args.dim1, args.dim2, bias=args.bias_deconv)
        self.bn2_bn = nn.BatchNorm1d(args.dim2, momentum=1-args.bn2_bn_momentum)
        self.bn3 = torch.nn.Linear(args.dim2, 10, bias=args.bias_deconv)

    def forward(self, image):
        x = F.relu(self.bn1(image))
        if self.args.bn1_bn_momentum>0: x = self.bn1_bn(x)
        x = F.relu(self.bn2(x))
        if self.args.p_dropout>0: x = F.dropout(x, p=self.args.p_dropout)
        if self.args.bn2_bn_momentum>0: x = self.bn2_bn(x)
        x = self.bn3(x)
        return x


class WhatWhere():
    def __init__(self, args, save = True):
        self.args = args
        print(1)
        self.display = Display(args, save = save)
        print(2)
        self.retina = Retina(args)
        print(3)
        # https://pytorch.org/docs/stable/nn.html#torch.nn.BCEWithLogitsLoss
        self.loss_func = torch.nn.CrossEntropyLoss()
        
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
            self.loader_train  = torch.load(filename_dataset)
        else:
            # MNIST DATASET
            from retina import get_data_loader        
            # SAVING DATASET
            print('Creating training dataset')
            retina_data, _, label = self.generate_data(train = True, fullfield = False)
            # create your dataset, see dev/2019-03-18_precomputed dataset.ipynb
            self.loader_train = DataLoader(TensorDataset(retina_data, label), batch_size=args.minibatch_size)
            if save:
                torch.save(self.loader_train, filename_dataset)
            print('Done!')

        # TESTING DATASET
        filename_dataset = filename_dataset = '/tmp/dataset_test' + suffix + '_%d.pt'% self.args.test_batch_size #f'/tmp/dataset_test_{suffix}_{self.args.test_batch_size}.pt'
        if os.path.exists(filename_dataset):
            self.loader_test  = torch.load(filename_dataset)
        else:
            print('Creating testing dataset')
            retina_data, _, label = self.generate_data(train = False, fullfield = False)
            # create your dataset, see dev/2019-03-18_precomputed dataset.ipynb
            self.loader_test = DataLoader(TensorDataset(retina_data, label), batch_size=args.test_batch_size)
            if save:
                torch.save(self.loader_test, filename_dataset)
            print('Done!')

        # MODEL
        self.model = WhatWhereNet(self.args).to(self.device)
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

    def generate_data(self, train = True, fullfield = True):
        from retina import get_data_loader
        if train:
            size = self.args.train_batch_size
            print('train dataset, size = ', size)
        else:
            size = self.args.test_batch_size
            print('test dataset, size = ', size)
        loader_full = get_data_loader(batch_size=size, train=train, mean=self.args.mean, std=self.args.std, seed=self.args.seed+1)
        if fullfield:
            data_fullfield = np.zeros((size, self.args.N_pic, self.args.N_pic))
        retina_data = np.zeros((size, self.retina.feature_vector_size))
        data, label = next(iter(loader_full)) 
        for i in range(size):
            if i%1000 == 0: print(i)
            if fullfield:
                data_fullfield[i, :, :], i_offset, j_offset = self.display.draw(data[i, 0, :, :].numpy())
                retina_data[i, :]  =  self.retina.retina(data_fullfield[i, :, :])
            else:
                data_fullfield, i_offset, j_offset = self.display.draw(data[i, 0, :, :].numpy())
                retina_data[i, :]  =  self.retina.retina(data_fullfield)
        retina_data = Variable(torch.FloatTensor(retina_data)).to(self.device)
        if fullfield:
            data_fullfield = Variable(torch.FloatTensor(data_fullfield)).to(self.device)
        else:
            data_fullfield = None
        label = label.to(self.device)
        return retina_data, data_fullfield, label
    
    def minibatch(self, data):
        # TODO: utiliser https://laurentperrinet.github.io/sciblog/posts/2018-09-07-extending-datasets-in-pytorch.html
        batch_size = data.shape[0]
        retina_data = np.zeros((batch_size, self.retina.feature_vector_size))
        data_fullfield = np.zeros((batch_size, self.args.N_pic, self.args.N_pic))
        positions =[]

        for i in range(batch_size):
            #print(i, data[i, 0, :, :].numpy().shape)
            data_fullfield[i, :, :], i_offset, j_offset = self.display.draw(data[i, 0, :, :].numpy())
            positions.append(dict(i_offset=i_offset, j_offset=j_offset))
            # TODO use one shot matrix multiplication
            retina_data[i, :]  =  self.retina.retina(data_fullfield[i, :, :])

        retina_data = Variable(torch.FloatTensor(retina_data)).to(self.device)
        label = Variable(torch.LongTensor(loader_full[1])).to(self.device)

        return positions, data_fullfield, retina_data, label

    def pred_accuracy(self, retina_data, label):
        # Predict classes using images from the train set
        #retina_data = Variable(torch.FloatTensor(retina_data))
        output = self.model(retina_data)
        # transform in a probability in collicular coordinates
        pred = output.argmax(dim=1, keepdim=True)
        pred_acc = pred.eq(label.view_as(pred)).detach().numpy()
        return pred_acc.mean()


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
        for retina_data, label in self.loader_train:
            # Clear all accumulated gradients
            self.optimizer.zero_grad()

            # Predict classes using images from the train set
            prediction = self.model(retina_data)
            # Compute the loss based on the predictions and actual labels
            loss = self.loss_func(prediction, label)
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
        for data_fullfield, retina_data, digit_labels in dataloader:
            pred_acc = self.pred_accuracy(retina_data)
            accuracy.append(pred_acc.mean())

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
