import os
import numpy as np
import matplotlib.pyplot as plt

#import SLIP for whitening and PIL for resizing
import SLIP

verbose = 1
##########################################################################################################@
##########################################################################################################@
##########################################################################################################@
class Retina:
    def __init__(self, args):
        self.args = args
        delta = 1./args.N_azimuth
        self.log_r, self.theta = np.meshgrid(np.linspace(0, 1, args.N_eccentricity + 1), np.linspace(-np.pi*(.5 + delta), np.pi*(1.5 - delta), args.N_azimuth + 1))
        
        try:
            self.retina_transform = np.load('../data/retina_transform.npy')
        except:
            self.retina_transform = vectorization(self.args.N_theta, self.args.N_azimuth,
                                                  self.args.N_eccentricity, 
                                                  self.args.N_phase, 
                                                  self.args.N_pic, self.args.N_pic, 
                                                  self.args.rho)
            np.save('../data/retina_transform.npy', self.retina_transform)
            
        self.vsize =  self.args.N_theta*self.args.N_azimuth*self.args.N_eccentricity*self.args.N_phase 
        self.retina_transform_vector = self.retina_transform.reshape((self.vsize, self.args.N_pic**2))
        
        try:
            self.retina_inverse_transform = np.load('../data/retina_inverse_transform.npy')
        except:
            #self.retina_inverse_transform = retina_inverse(self.retina_transform)
            self.retina_inverse_transform = np.linalg.pinv(self.retina_transform_vector)
    
            np.save('../data/retina_inverse_transform.npy', self.retina_inverse_transform)
            
        self.whit = SLIP.Image(pe='https://raw.githubusercontent.com/bicv/LogGabor/master/default_param.py')
        self.whit.set_size((args.N_pic, args.N_pic))
        # https://github.com/bicv/SLIP/blob/master/SLIP/SLIP.py#L611
        self.K_whitening = self.whit.whitening_filt()
        
        self.colliculus = (self.retina_transform**2).sum(axis=(0, 3))
        #colliculus = colliculus**.5
        self.colliculus /= self.colliculus.sum(axis=-1)[:, :, None]
        
        self.colliculus_vector = self.colliculus.reshape((self.args.N_azimuth*self.args.N_eccentricity, self.args.N_pic**2))
        self.colliculus_inverse = np.linalg.pinv(self.colliculus_vector)


    def retina(self, data_fullfield):
        # https://github.com/bicv/SLIP/blob/master/SLIP/SLIP.py#L674
        # data_fullfield = self.whit.whitening(data_fullfield)
        # https://github.com/bicv/SLIP/blob/master/SLIP/SLIP.py#L518
        data_fullfield = self.whit.FTfilter(data_fullfield, self.K_whitening)
        data_retina = self.retina_transform_vector @ np.ravel(data_fullfield)
        return data_retina #retina(data_fullfield, self.retina_transform)     
    
    def retina_invert(self, data_retina, do_dewhitening=False):
        im = self.retina_inverse_transform @ data_retina
        im = im.reshape((self.args.N_pic, self.args.N_pic))
        if do_dewhitening: im = self.whit.dewhitening(im)
        return im
    
    def accuracy_fullfield(self, accuracy_map, i_offset, j_offset):
        accuracy_colliculus, accuracy_fullfield_map = accuracy_fullfield(accuracy_map, i_offset, j_offset, self.args.N_pic, self.colliculus_vector)
        return accuracy_colliculus, accuracy_fullfield_map
    
    def accuracy_invert(self, accuracy_colliculus):
        im = self.colliculus_inverse @ accuracy_colliculus

        return im.reshape(self.args.N_pic, self.args.N_pic)
    
    
    def show(self, ax, im, rmin=None, rmax=None, ms=26, markeredgewidth=1, alpha=.6, lw=.75, do_cross=True):
        if rmin is None: rmin = im.min()
        if rmax is None: rmax = im.max()
        ax.imshow(im, cmap=plt.viridis(), vmin=rmin, vmax=rmax)
        if do_cross:
            mid = self.args.N_pic//2
            w = self.args.w

            ax.plot([mid], [mid], '+g', ms=ms, markeredgewidth=markeredgewidth, alpha=alpha)

            ax.plot([mid-w/2, mid+w/2, mid+w/2, mid-w/2, mid-w/2], 
                    [mid-w/2, mid-w/2, mid+w/2, mid+w/2, mid-w/2], '--', color='r', lw=lw, markeredgewidth=1)
        
        ax.set_xticks([])
        ax.set_yticks([])
        return ax
######################

def vectorization(N_theta=6, N_azimuth=16, N_eccentricity=10, N_phase=2,
                  N_X=128, N_Y=128, rho=1.41, ecc_max=.8, B_sf=.4, B_theta=np.pi/12):
    
    retina = np.zeros((N_theta, N_azimuth, N_eccentricity, N_phase, N_X*N_Y))
    
    from LogGabor import LogGabor
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
                #psi = i_azimuth * np.pi * 2 / N_azimuth
                psi = (i_azimuth + 1 * (i_eccentricity % 2)*.5) * np.pi * 2 / N_azimuth
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
                        lg.invert(lg.loggabor(x, y, **params)*np.exp(-1j*phase))).ravel()

    return retina

def retina_data(data_fullfield, retina_transform):
    N_pic = data_fullfield.shape[0]
    whit.set_size((N_pic, N_pic))
    data_fullfield = whit.whitening(data_fullfield)
    
    N_theta, N_azimuth, N_eccentricity, N_phase, N_pixel = retina_transform.shape
    retina_vector = retina_transform.reshape((N_theta*N_azimuth*N_eccentricity*N_phase, N_pixel))

    data_retina = retina_vector @ np.ravel(data_fullfield)

    return data_retina


def retina_tensor(data_retina, N_theta, N_azimuth, N_eccentricity, N_phase, N_pixel):
    
    tensor_retina = data_retina.reshape(N_theta, N_azimuth, N_eccentricity, N_phase)
    slice1 = tensor_retina[N_theta-1, ...].reshape(1, N_azimuth, N_eccentricity, N_phase)
    slice2 = tensor_retina[0, ...].reshape(1, N_azimuth, N_eccentricity, N_phase)
    tensor_retina = np.concatenate ((slice1, tensor_retina, slice2), axis = 0)
    tensor_retina = np.transpose(tensor_retina, (3, 0, 1, 2))

    return tensor_retina

def retina(data_fullfield, retina_transform):
    data_retina = retina_data(data_fullfield, retina_transform)
    tensor_retina = retina_tensor(data_retina, N_theta, N_azimuth, N_eccentricity, N_phase, N_pixel)
    return data_retina, tensor_retina

def retina_inverse(retina_transform):
    N_theta, N_azimuth, N_eccentricity, N_phase, N_pixel = retina_transform.shape
    retina_vector = retina_transform.reshape((N_theta*N_azimuth*N_eccentricity*N_phase, N_pixel))
    retina_inverse_transform = np.linalg.pinv(retina_vector)
    return retina_inverse_transform


def accuracy_fullfield(accuracy_map, i_offset, j_offset, N_pic, colliculus_vector):
    
    accuracy_fullfield_map = do_offset(data=accuracy_map, i_offset=i_offset, j_offset=j_offset, N_pic=N_pic, data_min=0.1)
    
    accuracy_colliculus = colliculus_vector @ accuracy_fullfield_map.ravel()

    return accuracy_colliculus, accuracy_fullfield_map


##########################################################################################################@
##########################################################################################################@
class Display:
    def __init__(self, args):
        self.args = args
        self.loader_train = get_data_loader(batch_size=args.minibatch_size, train=True, mean=args.mean, std=args.std, seed=args.seed)
        self.loader_test = get_data_loader(batch_size=args.test_batch_size, train=False, mean=args.mean, std=args.std, seed=args.seed)
        self.N_classes = len(self.loader_test.dataset.classes)
        
        np.random.seed(seed=args.seed+1)
        # cache noise
        path = f"../data/MotionClouds_{self.args.sf_0}_{self.args.B_sf}.npy"
        # print(path)
        if os.path.isfile(path):
            self.noise =  np.load(path)
        else:
            self.noise = np.zeros((args.noise_batch_size, args.N_pic, args.N_pic))
            for i_noise in range(args.noise_batch_size):
                self.noise[i_noise, :, :], _ = MotionCloudNoise(sf_0=args.sf_0, B_sf=args.B_sf, seed=self.args.seed+i_noise)
            np.save(path, self.noise)
            
    def place_object(self, data, i_offset, j_offset):
        if True:
            im_noise = self.noise[np.random.randint(self.args.noise_batch_size), :, :]
            im_noise = np.roll(im_noise, np.random.randint(self.args.N_pic), 0)
            im_noise = np.roll(im_noise, np.random.randint(self.args.N_pic), 1)            
        else:
            im_noise = None
        return place_object(data, i_offset, j_offset, im_noise=im_noise, N_pic=self.args.N_pic,
                                    contrast=self.args.contrast, noise=self.args.noise,
                                    sf_0=self.args.sf_0, B_sf=self.args.B_sf)
    def draw(self, data):
        # radial draw
        radius = minmax(np.random.randn() * self.args.offset_std, self.args.offset_max)
        theta = np.random.rand() * np.pi
        i_offset = int(radius * np.cos(theta))
        j_offset = int(radius * np.sin(theta))
        return self.place_object(data, i_offset, j_offset), i_offset, j_offset


    def show(self, ax, data_fullfield, ms=26, markeredgewidth=6, do_cross=True):
        ax.imshow(data_fullfield, cmap=plt.gray(), vmin=0, vmax=1)
        if do_cross: ax.plot([self.args.N_pic//2], [self.args.N_pic//2], '+', ms=ms, markeredgewidth=markeredgewidth)
        ax.set_xticks([])
        ax.set_yticks([])
        return ax
##########################################################################################################@
##########################################################################################################@


whit = SLIP.Image(pe='https://raw.githubusercontent.com/bicv/LogGabor/master/default_param.py')

def get_data_loader(batch_size=100, train=True, mean=0.1307, std=0.3081, seed=2019):
    import torch
    torch.manual_seed(seed=seed)
    from torchvision import datasets, transforms
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    data_loader = torch.utils.data.DataLoader(
        # https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.MNIST
        datasets.MNIST('../data',
                       train=train,     # def the dataset as training data
                       download=True,  # download if dataset not present on disk
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((mean,), (std,))
                       ])),
                       batch_size=batch_size,
                       shuffle=True)
    return data_loader

def minmax(value, border):
    value = max(value, -border)
    value = min(value, border)
    return int(value)


def do_offset(data, i_offset, j_offset, N_pic, data_min=None):
    # place data in a big image with some known offset
    N_stim = data.shape[0]
    center = (N_pic-N_stim)//2
    if data_min is None:
        data_min = data.min()
        
    data_fullfield = data_min * np.ones((N_pic, N_pic))
    data_fullfield[int(center+i_offset):int(center+N_stim+i_offset), int(center+j_offset):int(center+N_stim+j_offset)] = data
    return data_fullfield
    
def place_object(data, i_offset, j_offset, im_noise=None, N_pic=128, contrast=1., noise=.5, sf_0=0.1, B_sf=0.1, do_mask=True, do_max=False):
    #print(data.min(), data.max())
    data = (data - data.min())/(data.max() - data.min())
    
    # place data in a big image with some known offset
    data_fullfield = do_offset(data=data, i_offset=i_offset, j_offset=j_offset, N_pic=N_pic, data_min=0.)

    # normalize data in [0, 1]
    data_fullfield = (data_fullfield - data_fullfield.min())/(data_fullfield.max() - data_fullfield.min())
    #data_fullfield = 2 * data_fullfield - 1 # [-1, 1] range
    #data_fullfield /= 2 # back to [0, 1] range
    data_fullfield *= contrast
    #data_fullfield += .5 # back to [0, 1] range
    
    # add noise
    if noise>0.:
        if im_noise is None:
            im_noise, _ = MotionCloudNoise(sf_0=sf_0, B_sf=B_sf)
        # print(im_noise.min(), im_noise.max())
        #im_noise = 2 * im_noise - 1
        im_noise = noise *  im_noise
        #im_noise = .5 * im_noise + .5 # back to [0, 1] range
        #print(im_noise.min(), im_noise.max())
        if do_max:
            data_fullfield = np.max((im_noise, data_fullfield), axis=0)
        else:
            data_fullfield = np.mean((im_noise, data_fullfield), axis=0)
    #print(data_fullfield.min(), data_fullfield.max())
        
        
    # add a circular mask
    if do_mask:
        x, y = np.mgrid[-1:1:1j*N_pic, -1:1:1j*N_pic]
        R = np.sqrt(x**2 + y**2)
        mask = 1. * (R<1)
        #print('mask', mask.min(), mask.max(), mask[0, 0])
        data_fullfield = data_fullfield*mask
    
    # normalize data in [0, 1]
    data_fullfield /= 2 # back to [0, 1] range
    data_fullfield += .5 # back to a .5 baseline
    data_fullfield = np.clip(data_fullfield, 0, 1)
    return data_fullfield

def MotionCloudNoise(sf_0=0.125, B_sf=3., alpha=.0, N_pic=128, seed=42):
    import MotionClouds as mc
    mc.N_X, mc.N_Y, mc.N_frame = N_pic, N_pic, 1
    fx, fy, ft = mc.get_grids(mc.N_X, mc.N_Y, mc.N_frame)
    env = mc.envelope_gabor(fx, fy, ft, sf_0=sf_0, B_sf=B_sf, B_theta=np.inf, V_X=0., V_Y=0., B_V=0, alpha=alpha)
    
    z = mc.rectif(mc.random_cloud(env, seed=seed), contrast=1., method='Michelson')
    z = z.reshape((mc.N_X, mc.N_Y))
    return z, env

