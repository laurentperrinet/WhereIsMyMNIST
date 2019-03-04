dataset_folder = 'dataset_faces'
batch_size = 16
no_cuda = False
test_batch_size = 1
valid_size = .2
do_adam = False
epochs = 20
lr = 0.05
momentum = 0.18
num_processes = 1
seed = 42
log_interval = 0
fullsize = 175
crop = 175 # int(.9*fullsize)
size = 150
mean = .36
std = .3
conv1_dim = 5
conv1_kernel_size = 7
conv2_dim = 13
conv2_kernel_size = 7
dimension = 25
verbose = False
stride1 = 2
stride2 = 4
N_cv = 10
# DEBUG
# epochs = 2
# N_cv = 2

import easydict
def init(dataset_folder=dataset_folder, batch_size=batch_size, test_batch_size=test_batch_size, valid_size=valid_size, epochs=epochs,
            do_adam=do_adam, lr=lr, momentum=momentum, no_cuda=no_cuda, num_processes=num_processes, seed=seed,
            log_interval=log_interval, fullsize=fullsize, crop=crop, size=size, mean=mean, std=std,
            conv1_dim=conv1_dim, conv1_kernel_size=conv1_kernel_size,
            conv2_dim=conv2_dim, conv2_kernel_size=conv2_kernel_size,
            stride1=stride1, stride2=stride2, N_cv=N_cv,
            dimension=dimension, verbose=verbose):
    # Training settings
    kwargs = {}
    kwargs.update(dataset_folder=dataset_folder, batch_size=batch_size, test_batch_size=test_batch_size, valid_size=valid_size, epochs=epochs,
                do_adam=do_adam, lr=lr, momentum=momentum, no_cuda=no_cuda, num_processes=num_processes, seed=seed,
                log_interval=log_interval, fullsize=fullsize, crop=crop, size=size, mean=mean, std=std,
                conv1_dim=conv1_dim, conv1_kernel_size=conv1_kernel_size,
                conv2_dim=conv2_dim, conv2_kernel_size=conv2_kernel_size,
                stride1=stride1, stride2=stride2, N_cv=N_cv,
                dimension=dimension, verbose=verbose
                )
    # print(kwargs, locals())
    return easydict.EasyDict(kwargs)

import numpy as np
import torch
torch.set_default_tensor_type('torch.FloatTensor')
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
# import torch.multiprocessing as mp
# import torchvision.models as models
import torchvision
# torchvision.set_image_backend('accimage')
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim


class Data:
    def __init__(self, args):
        self.args = args
        # GPU boilerplate
        if self.args.verbose:
            if not self.args.no_cuda and not torch.cuda.is_available():
                print('Trying to load cuda, but it is not available')
        self.args.no_cuda = self.args.no_cuda or not torch.cuda.is_available()
        if self.args.verbose:
            print('no cuda?', self.args.no_cuda)
        kwargs = {'num_workers': 1, 'pin_memory': True} if not args.no_cuda else {'num_workers': 1, 'shuffle': True}

        t = transforms.Compose([
            # https://pytorch.org/docs/master/torchvision/transforms.html#torchvision.transforms.Resize
            # Resize the input PIL Image to the given size. size (sequence or int) – Desired output size. If size is a sequence like (h, w), output size will be matched to this. If size is an int, smaller edge of the image will be matched to this number. i.e, if height > width, then image will be rescaled to (size * height / width, size)
            transforms.Resize(args.fullsize),
            # https://pytorch.org/docs/master/torchvision/transforms.html#torchvision.transforms.RandomAffine
            #
            #transforms.RandomAffine(degrees=10, scale=(.8, 1.2), shear=10, resample=False, fillcolor=0),
            #transforms.RandomVerticalFlip(),
            # transforms.CenterCrop((args.crop, int(args.crop*4/3))),
            transforms.CenterCrop(args.crop),
            #torchvision.transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2)[
            transforms.Resize(args.size),
            # transforms.RandomAffine(args.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[args.mean]*3, std=[args.std]*3),
            ])
        self.dataset = ImageFolder(self.args.dataset_folder, t)
        #self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
        #self.test_loader = torch.utils.data.DataLoader(self.dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=1)

        # https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
        num_train = len(self.dataset)
        # indices = list(range(num_train))
        split = int(np.floor(self.args.valid_size * num_train))
        if self.args.verbose:
            print('Found', num_train, 'sample images; ', num_train-split, ' to train', split, 'to test')
        #
        # np.random.seed(self.args.seed)
        # np.random.shuffle(indices)
        #
        # train_idx, valid_idx = indices[split:], indices[:split]
        # train_sampler = SubsetRandomSampler(train_idx)
        # valid_sampler = SubsetRandomSampler(valid_idx)

        # N-batch_size, C-num_channels , H-height, W-width

        from torch.utils.data import random_split
        train_dataset, test_dataset = random_split(self.dataset, [num_train-split, split])

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.test_batch_size, **kwargs)
        self.classes = self.dataset.classes #'blink', 'left ', 'right', ' fix '

    def show(self, gamma=.5, noise_level=.4, transpose=True):

        images, foo = next(iter(self.train_loader))

        from torchvision.utils import make_grid
        npimg = make_grid(images, normalize=True).numpy()
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=((13, 5)))
        import numpy as np
        if transpose:
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
        else:
            ax.imshow(npimg)
        plt.setp(ax, xticks=[], yticks=[])

        return fig, ax


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        # data is in the format (N, C, H, W)
        self.conv1 = nn.Conv2d(3, args.conv1_dim, kernel_size=args.conv1_kernel_size)
        padding1 = args.conv1_kernel_size - 1 # total padding in layer 1 (before max pooling)
        # https://pytorch.org/docs/stable/nn.html#torch.nn.MaxPool2d
        out_width_1 = (args.size - padding1 - args.stride1) // args.stride1 + 1
        # TODO : self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(args.conv1_dim, args.conv2_dim, kernel_size=args.conv2_kernel_size)
        #self.conv2_drop = nn.Dropout2d()
        padding2 = args.conv2_kernel_size - 1 # total padding in layer 2
        out_width_2 = (out_width_1 - padding2 - args.stride2) // args.stride2 + 1
        fc1_dim = (out_width_2**2) * args.conv2_dim
        self.fc1 = nn.Linear(fc1_dim, args.dimension)
        self.fc2 = nn.Linear(args.dimension, len(self.args.classes))

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=[self.args.stride1, self.args.stride1]))#, stride=[self.args.stride1, self.args.stride1]))
            # s = self.bn1(self.conv1(s))        # batch_size x 32 x 64 x 64
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), kernel_size=[2, 2], stride=[2, 2]))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=[self.args.stride2, self.args.stride2]))#, stride=[self.args.stride2, self.args.stride2]))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        #         # apply 2 fully connected layers with dropout
        # s = F.dropout(F.relu(self.fcbn1(self.fc1(s))),
        #     p=self.dropout_rate, training=self.training)    # batch_size x 128
        #x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# class Net(nn.Module):
#     def __init__(self, args):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
#         #self.pool = nn.MaxPool2d(2, 2)#
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, args.dimension)
#         self.fc2 = nn.Linear(args.dimension, 10)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 4))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 4))
#         x = x.view(-1, 20480)
#         #x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         #x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         #x = F.dropout(x, training=self.training)
#         x = self.fc3(x)
#         return F.log_softmax(x, dim=1)


class ML():
    def __init__(self, args):
        self.args = args
        # GPU boilerplate
        self.args.no_cuda = self.args.no_cuda or not torch.cuda.is_available()
        if self.args.verbose:
            print('cuda?', not self.args.no_cuda)
        self.device = torch.device("cpu" if self.args.no_cuda else "cuda")
        torch.manual_seed(self.args.seed)
        # DATA
        self.dataset = Data(self.args)
        self.args.classes = self.dataset.classes
        # MODEL
        self.model = Net(self.args).to(self.device)
        if not self.args.no_cuda:
            # print('doing cuda')
            torch.cuda.manual_seed(self.args.seed)
            self.model.cuda()

        if self.args.do_adam:
            # see https://heartbeat.fritz.ai/basics-of-image-classification-with-pytorch-2f8973c51864
            scale = 10
            self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.args.lr/scale, weight_decay=1-self.args.momentum/scale)
        else:
            self.optimizer = optim.SGD(self.model.parameters(),
                                    lr=self.args.lr, momentum=self.args.momentum)
    # def forward(self, img):
    #     # normalize img
    #     return (img - self.mean) / self.std

    def train(self, path=None, seed=None):
        if seed is None:
            seed = self.args.seed
        # cosmetics
        try:
            from tqdm import tqdm
            #from tqdm import tqdm_notebook as tqdm
            verbose = 1
        except ImportError:
            verbose = 0
        if self.args.verbose == 0 or verbose == 0:
            def tqdm(x, desc=None):
                if desc is not None: print(desc)
                return x

        # setting up training
        self.model.train()
        if path is not None:
            # using a data_cache
            import os
            import torch
            if os.path.isfile(path):
                ml.model.load_state_dict(torch.load(path))
                print('Loading file', path)
            else:
                print('Training model...')
                for epoch in tqdm(range(1, self.args.epochs + 1), desc='Train Epoch' if self.args.verbose else None):
                    self.train_epoch(epoch, rank=0)
                    torch.save(ml.model.state_dict(), path) #save the neural network state
                print('Model saved at', path)
        else:
            for epoch in tqdm(range(1, self.args.epochs + 1), desc='Train Epoch' if self.args.verbose else None):
                self.train_epoch(epoch, seed, rank=0)

    def train_epoch(self, epoch, seed, rank=0):
        torch.manual_seed(seed + epoch + rank*self.args.epochs)
        for batch_idx, (data, target) in enumerate(self.dataset.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            # Clear all accumulated gradients
            self.optimizer.zero_grad()
            # Predict classes using images from the train set
            output = self.model(data)
            # Compute the loss based on the predictions and actual labels
            loss = F.nll_loss(output, target)
            # Backpropagate the loss
            loss.backward()
            # Adjust parameters according to the computed gradients
            self.optimizer.step()
            if self.args.verbose and self.args.log_interval>0:
                if batch_idx % self.args.log_interval == 0:
                    status_str = '\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(self.dataset.train_loader.dataset),
                        100. * batch_idx / len(self.dataset.train_loader), loss.item())
                    try:
                        from tqdm import tqdm
                        tqdm.write(status_str)
                    except Exception as e:
                        print(e)
                        print(status_str)

    def test(self, dataloader=None):
        if dataloader is None:
            dataloader = self.dataset.test_loader
        self.model.eval()
        test_loss = 0
        correct = 0
        for data, target in dataloader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            test_loss += F.nll_loss(output, target, reduction='elementwise_mean').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

        test_loss /= len(self.dataset.test_loader.dataset)

        if self.args.log_interval>0:
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.dataset.test_loader.dataset),
            100. * correct / len(self.dataset.test_loader.dataset)))
        return correct.numpy() / len(self.dataset.test_loader.dataset)

    def show(self, gamma=.5, noise_level=.4, transpose=True, only_wrong=False):
        data, target = next(iter(self.dataset.test_loader))
        data, target = data.to(self.device), target.to(self.device)
        output = self.model(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        if only_wrong and not pred == target:
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
        print('scanning over', parameter, '=', values)
        seed = self.seed
        for value in values:
            if isinstance(value, int):
                value_str = str(value)
            else:
                value_str = '%.3f' % value
            path = '_tmp_scanning_' + parameter + '_' + self.tag + '_' + value_str.replace('.', '_') + '.npy'
            print ('For parameter', parameter, '=', value_str, ', ', end=" ")
            if not(os.path.isfile(path + '_lock')):
                if not(os.path.isfile(path)):
                    open(path + '_lock', 'w').close()
                    try:
                        args = easydict.EasyDict(self.args.copy())
                        args[parameter] = value
                        Accuracy = self.protocol(args, seed)

                    except Exception as e:
                        print('Failed with error', e)
                    np.save(path, Accuracy)
                    os.remove(path + '_lock')
                else:
                    Accuracy = np.load(path)

                print('Accuracy={:.1f}% +/- {:.1f}%'.format(Accuracy[:-1].mean()*100, Accuracy[:-1].std()*100),
                  ' in {:.1f} seconds'.format(Accuracy[-1]))
            else:
                print(' currently locked with ', path + '_lock')
            seed += 1

    def parameter_scan(self, parameter):
        values = self.args[parameter] * np.logspace(-1, 1, self.N_scan, base=self.base)
        if isinstance(self.args[parameter], int):
            # print('integer detected') # DEBUG
            values =  [int(k) for k in values]
        self.scan(parameter, values)


if __name__ == '__main__':
    if False :
        print(50*'-')
        print('Default parameters')
        print(50*'-')
        args = init(verbose=0, log_interval=0)
        ml = ML(args)
        ml.main()
    if True :
        args = init(verbose=0, log_interval=0)
        mml = MetaML(args)
        if torch.cuda.is_available():
            mml.scan('no_cuda', [True, False])
        else:
            mml.scan('no_cuda', [True])
    print(50*'-')
    print(' parameter scan')
    print(50*'-')
    for base in [10, 2]:
        print(50*'-')
        print(' base=', base)
        print(50*'-')
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

        print(50*'-')
        args = init(verbose=0, log_interval=0)
        mml = MetaML(args)
        print(' parameter scan : network')
        print(50*'-')
        for parameter in ['conv1_kernel_size',
                          'conv1_dim',
                          'conv2_kernel_size',
                          'conv2_dim',
                          'stride1', 'stride2',
                          'dimension']:
            mml.parameter_scan(parameter)

        print(50*'-')
        print(' parameter scan : data')
        print(50*'-')
        for parameter in ['size', 'fullsize', 'crop', 'mean', 'std']:
            mml.parameter_scan(parameter)
