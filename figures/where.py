import os
import numpy as np
import time
import torch
torch.set_default_tensor_type('torch.FloatTensor')
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn


class WhereNet(torch.nn.Module):
    def __init__(self, args):
        super(WhereNet, self).__init__()
        self.args = args       
        #self.bn1= torch.nn.Linear(N_theta*N_azimuth*N_eccentricity*N_phase, 200, bias = BIAS_DECONV)
        self.bn1 = torch.nn.Linear(args.N_theta*args.N_azimuth*args.N_eccentricity*args.N_phase, args.dim1, bias=args.bias_deconv)
        self.bn1_bn = nn.BatchNorm1d(args.dim1, momentum=1-args.bn1_bn_momentum)
        #self.bn2 = torch.nn.Linear(200, 80, bias = BIAS_DECONV)
        #https://raw.githubusercontent.com/MorvanZhou/PyTorch-Tutorial/master/tutorial-contents/504_batch_normalization.py
        #self.conv2_bn = nn.BatchNorm2d(args.conv2_dim, momentum=1-args.conv2_bn_momentum)
        self.bn2 = torch.nn.Linear(args.dim1, args.dim2, bias=args.bias_deconv)
        self.bn2_bn = nn.BatchNorm1d(args.dim2, momentum=1-args.bn2_bn_momentum)
        #self.bn3 = torch.nn.Linear(80, N_azimuth*N_eccentricity, bias = BIAS_DECONV)
        self.bn3 = torch.nn.Linear(args.dim2, args.N_azimuth*args.N_eccentricity, bias=args.bias_deconv)
                
    def forward(self, image):  
        x = F.relu(self.bn1(image))  
        if self.args.bn1_bn_momentum>0: x = self.bn1_bn(x)
        x = F.relu(self.bn2(x))
        x = F.dropout(x, p=self.args.p_dropout) 
        if self.args.bn2_bn_momentum>0: x = self.bn2_bn(x)
        x = self.bn3(x)
        return x


class Where():
    def __init__(self, args, display, retina):
        self.args = args
        from what import WhatNet
        model_path = "../data/MNIST_cnn.pt"
        self.What_model = torch.load(model_path)
        #model = torch.load(model_path)
        #self.What_model = WhatNet()
        # torch.save(model.state_dict(), "../data/MNIST_cnn.pt")
        #self.What_model.load_state_dict(torch.load(model_path))        
        #self.What_model.eval()

        # GPU boilerplate
        self.args.no_cuda = self.args.no_cuda or not torch.cuda.is_available()
        # if self.args.verbose: print('cuda?', not self.args.no_cuda)
        self.device = torch.device("cpu" if self.args.no_cuda else "cuda")
        torch.manual_seed(self.args.seed)
        # DATA
        self.display = display
        self.retina = retina
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

        path = "../data/MNIST_accuracy.npy"
        if os.path.isfile(path):
            self.accuracy_map =  np.load(path)
            if args.verbose:
                print('Loading accuracy... min, max=', self.accuracy_map.min(), self.accuracy_map.max())
        else:
            print('No accuracy data found.')


    def minibatch(self, data):
        # TODO: utiliser https://laurentperrinet.github.io/sciblog/posts/2018-09-07-extending-datasets-in-pytorch.html
        batch_size = data.shape[0]
        retina_data = np.zeros((batch_size, self.retina.vsize))
        accuracy_colliculus = np.zeros((batch_size, self.args.N_azimuth * self.args.N_eccentricity))
        full =[]

        for i in range(batch_size):
            #print(i, data[i, 0, :, :].numpy().shape)
            data_fullfield, i_offset, j_offset = self.display.draw(data[i, 0, :, :].numpy())
            full.append(dict(data_fullfield=data_fullfield, i_offset=i_offset, j_offset=j_offset))
            # TODO use one shot matrix multiplication
            retina_data[i, :]  =  self.retina.retina(data_fullfield)
            accuracy_colliculus[i,:], _ = self.retina.accuracy_fullfield(self.accuracy_map, i_offset, j_offset)

        retina_data = Variable(torch.FloatTensor(retina_data))
        accuracy_colliculus = Variable(torch.FloatTensor(accuracy_colliculus))
        retina_data, accuracy_colliculus = retina_data.to(self.device), accuracy_colliculus.to(self.device)
                              
        return full, retina_data, accuracy_colliculus

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
        
    def extract(self, data_fullfield, i_offset, j_offset):
        mid = self.args.N_pic//2
        rad = self.args.w//2

        im = data_fullfield[(mid+i_offset-rad):(mid+i_offset+rad),
                            (mid+j_offset-rad):(mid+j_offset+rad)]

        im = np.clip(im, 0.5, 1)
        im = (im-.5)*2
        return im
        
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
        for batch_idx, (data, label) in enumerate(self.display.loader_train):
            # Clear all accumulated gradients
            self.optimizer.zero_grad()
            
            # get a minibatch of the same digit at different positions and noises
            full, retina_data, accuracy_colliculus = self.minibatch(data)
            
            # Predict classes using images from the train set
            prediction = self.model(retina_data)
            # Compute the loss based on the predictions and actual labels
            loss = self.args.loss_func(prediction, accuracy_colliculus)
            # Backpropagate the loss
            loss.backward()
            # Adjust parameters according to the computed gradients
            self.optimizer.step()
            
            # stops prematurely (for testing purposes)
            if batch_idx > self.args.train_batch_size : break
            
        return loss.item()

    def test(self, dataloader=None):
        if dataloader is None:
            dataloader = self.display.loader_test
        self.model.eval()
        test_loss = 0
        correct = 0
        for data, target in dataloader:
            # get a minibatch of the same digit at different positions and noises
            full, retina_data, accuracy_colliculus = self.minibatch(data)
            
            # Predict classes using images from the train set
            prediction = self.model(retina_data)

            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

        test_loss /= len(self.dataloader.dataset)

        if self.args.log_interval>0:
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.dataloader.dataset),
            100. * correct / len(self.dataloader.dataset)))
        return correct.numpy() / len(self.dataloader.dataset)

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


import time
class MetaML:
    def __init__(self, args, base = 2, N_scan = 9, tag='', verbose=0, log_interval=0):
        self.args = args
        self.seed = args.seed

        self.base = base
        self.N_scan = N_scan
        self.tag = tag
        self.default = dict(verbose=verbose, log_interval=log_interval)

    def test(self, args, seed):
        # makes a loop for the cross-validation of results
        Accuracy = []
        for i_cv in range(self.args.N_cv):
            ml = ML(args)
            ml.train(seed=seed + i_cv)
            Accuracy.append(ml.test())
        return np.array(Accuracy)

    def protocol(self, args, seed):
        t0 = time.time()
        Accuracy = self.test(args, seed)
        t1 = time.time() - t0
        Accuracy = np.hstack((Accuracy, [t1]))
        return Accuracy

    def scan(self, parameter, values):
        import os
        try:
            os.mkdir('_tmp_scanning')
        except:
            pass
        print('scanning over', parameter, '=', values)
        seed = self.seed
        Accuracy = {}
        for value in values:
            if isinstance(value, int):
                value_str = str(value)
            else:
                value_str = '%.3f' % value
            path = '_tmp_scanning/' + parameter + '_' + self.tag + '_' + value_str.replace('.', '_') + '.npy'
            print ('For parameter', parameter, '=', value_str, ', ', end=" ")
            if not(os.path.isfile(path + '_lock')):
                if not(os.path.isfile(path)):
                    open(path + '_lock', 'w').close()
                    try:
                        args = easydict.EasyDict(self.args.copy())
                        args[parameter] = value
                        Accuracy[value] = self.protocol(args, seed)
                        np.save(path, Accuracy[value])
                        os.remove(path + '_lock')
                    except Exception as e:
                        print('Failed with error', e)
                else:
                    Accuracy[value] = np.load(path)

                try:
                    print('Accuracy={:.1f}% +/- {:.1f}%'.format(Accuracy[value][:-1].mean()*100, Accuracy[value][:-1].std()*100),
                  ' in {:.1f} seconds'.format(Accuracy[value][-1]))
                except Exception as e:
                    print('Failed with error', e)

            else:
                print(' currently locked with ', path + '_lock')
            seed += 1
        return Accuracy

    def parameter_scan(self, parameter, display=False):
        if parameter in ['momentum', 'conv1_bn_momentum', 'conv2_bn_momentum', 'dense_bn_momentum']:
            values = np.linspace(0, 1, self.N_scan, endpoint=True)
        else:
            values = self.args[parameter] * np.logspace(-1, 1, self.N_scan, base=self.base)
        if isinstance(self.args[parameter], int):
            # print('integer detected') # DEBUG
            values =  [int(k) for k in values]
        Accuracy = self.scan(parameter, values)
        if display:
            fig, ax = plt.subplots(figsize=(8, 5))



        return Accuracy


if __name__ == '__main__':
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

    print(50*'-')
    print(' parameter scan')
    print(50*'-')

    if False :
        print(50*'-')
        print('Default parameters')
        print(50*'-')
        args = init(verbose=0, log_interval=0)
        ml = ML(args)
        ml.main()
    if False :
        args = init(verbose=0, log_interval=0)
        mml = MetaML(args)
        if torch.cuda.is_available():
            mml.scan('no_cuda', [True, False])
        else:
            mml.scan('no_cuda', [True])

    # for base in [2]:#, 8]:
    for base in [2, 8]:
        print(50*'-')
        print(' base=', base)
        print(50*'-')

        print(50*'-')
        print(' parameter scan : data')
        print(50*'-')
        args = init(verbose=0, log_interval=0)
        mml = MetaML(args, base=base)
        for parameter in ['size', 'fullsize', 'crop', 'mean', 'std']:
            mml.parameter_scan(parameter)

        print(50*'-')
        args = init(verbose=0, log_interval=0)
        mml = MetaML(args)
        print(' parameter scan : network')
        print(50*'-')
        for parameter in ['conv1_kernel_size',
                          'conv1_dim',
                          'conv1_bn_momentum',
                          'conv2_kernel_size',
                          'conv2_dim',
                          'conv2_bn_momentum',
                          'stride1', 'stride2',
                          'dense_bn_momentum',
                          'dimension']:
            mml.parameter_scan(parameter)

        args = init(verbose=0, log_interval=0)
        mml = MetaML(args, base=base)
        print(' parameter scan : learning ')
        print(50*'-')
        print('Using SGD')
        print(50*'-')
        for parameter in ['lr', 'momentum', 'batch_size', 'epochs']:
            mml.parameter_scan(parameter)
        print(50*'-')
        print('Using ADAM')
        print(50*'-')
        args = init(verbose=0, log_interval=0, do_adam=True)
        mml = MetaML(args, tag='adam')
        for parameter in ['lr', 'momentum', 'batch_size', 'epochs']:
            mml.parameter_scan(parameter)
