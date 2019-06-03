import numpy as np
import matplotlib.pyplot as plt

#import SLIP for whitening and PIL for resizing
import SLIP
# copied from https://raw.githubusercontent.com/bicv/LogGabor/master/default_param.py
pe = {
    # Image
    # 'N_image' : None, #use all images in the folder
    'N_image' : 100, #use 100 images in the folder
    # 'N_image' : 10, #use 4 images in the folder
    'seed' : None, # a seed for the Random Number Generator (RNG) for picking images in databases, set to None xor set to a given number to freeze the RNG
    'N_X' : 256, # size of images
    'N_Y' : 256, # size of images
    # 'N_X' : 64, # size of images
    # 'N_Y' : 64, # size of images
    'noise' : 0.1, # level of noise when we use some
    'do_mask'  : True, # self.pe.do_mask
    'mask_exponent': 3., #sharpness of the mask
    # whitening parameters:
    'do_whitening'  : True, # = self.pe.do_whitening
    'white_name_database' : 'kodakdb',
    'white_n_learning' : 0,
    'white_N' : .07,
    'white_N_0' : .0, # olshausen = 0.
    'white_f_0' : .4, # olshausen = 0.2
    'white_alpha' : 1.4,
    'white_steepness' : 4.,
    'white_recompute' : False,
    # Log-Gabor
    #'base_levels' : 2.,
    'base_levels' : 1.618,
    'n_theta' : 24, # number of (unoriented) angles between 0. radians (included) and np.pi radians (excluded)
    'B_sf' : .4, # 1.5 in Geisler
    'B_theta' : 3.14159/18.,
    # PATHS
    'use_cache' : True,
    'figpath': 'results',
    'edgefigpath': 'results/edges',
    'matpath': 'cache_dir',
    'edgematpath': 'cache_dir/edges',
    'datapath': 'database',
    'ext' : '.pdf',
    'figsize': 14.,
    'formats': ['pdf', 'png', 'jpg'],
    'dpi': 450,
    'verbose': 0,
    }

##########################################################################################################@
##########################################################################################################@
##########################################################################################################@

class Retina:
    """ Class implementing the retina transform
    """
    def __init__(self, args):

        self.args = args
        self.N_theta = args.N_theta
        self.N_azimuth = args.N_azimuth
        self.N_eccentricity = args.N_eccentricity
        self.N_phase = args.N_phase
        self.N_pic = args.N_pic

        self.feature_vector_size = self.N_theta * self.N_azimuth * self.N_eccentricity * self.N_phase

        self.whit = SLIP.Image(pe=pe)
        self.whit.set_size((self.N_pic, self.N_pic))
        # https://github.com/bicv/SLIP/blob/master/SLIP/SLIP.py#L611
        self.K_whitening = self.whit.whitening_filt()

        self.init_grid()
        self.init_retina_transform()
        self.init_inverse_retina()
        self.init_colliculus_transform()
        self.init_colliculus_inverse()

    def init_grid(self):
        delta = 1. / self.N_azimuth
        self.log_r_grid, self.theta_grid = \
        np.meshgrid(np.linspace(0, 1, self.N_eccentricity + 1),
                    np.linspace(-np.pi * (.5 + delta), np.pi * (1.5 - delta), self.N_azimuth + 1))

    def get_suffix(self):
        suffix = f'_{args.N_theta}_{args.N_azimuth}'
        suffix += f'_{args.N_eccentricity}_{args.N_phase}'
        suffix += f'_{args.rho}_{args.N_pic}'
        return suffix

    def init_retina_transform(self):
        filename = '/tmp/retina' + self.get_suffix() + '_transform.npy'
        try:
            self.retina_transform = np.load(filename)
        except:
            if self.args.verbose: print('Retina vectorizing...')
            self.retina_transform = self.vectorization()
            np.save(filename, self.retina_transform)
            if self.args.verbose: print('Done vectorizing...')
        self.retina_transform_vector = self.retina_transform.reshape((self.feature_vector_size, self.N_pic ** 2))

    def init_inverse_retina(self):
        filename = '/tmp/retina' + self.get_suffix() + '_inverse_transform.npy'
        try:
            self.retina_inverse_transform = np.load(filename)
        except:
            if self.args.verbose: print('Inversing retina transform...')
            self.retina_inverse_transform = np.linalg.pinv(self.retina_transform_vector)
            np.save(filename, self.retina_inverse_transform)
            if self.args.verbose: print('Done Inversing retina transform...')

    def init_colliculus_transform(self):
        # TODO : make a different transformation for the clliculus (more eccentricties?)
        self.colliculus_transform = (self.retina_transform ** 2).sum(axis=(0, 3))
        # colliculus = colliculus**.5
        self.colliculus_transform /= self.colliculus_transform.sum(axis=-1)[:, :, None]  # normalization as a probability
        self.colliculus_transform_vector = self.colliculus_transform.reshape((self.args.N_azimuth * self.args.N_eccentricity, self.args.N_pic ** 2))

    def init_colliculus_inverse(self):
        self.colliculus_inverse = np.linalg.pinv(self.colliculus_transform_vector)

    def vectorization(self):
        #N_theta=6, N_azimuth=16, N_eccentricity=10, N_phase=2,
        #              N_X=128, N_Y=128, rho=1.41, ecc_max=.8, sf_0_max=0.45, sf_0_r=0.03, B_sf=.4, B_theta=np.pi / 12):

        retina = np.zeros((self.N_theta, self.N_azimuth, self.N_eccentricity, self.N_phase, self.N_pic**2))

        from LogGabor import LogGabor
        lg = LogGabor(pe=pe)
        lg.set_size((self.N_pic, self.N_pic))

        for i_theta in range(self.N_theta):
            for i_azimuth in range(self.N_azimuth):
                for i_eccentricity in range(self.N_eccentricity):
                    for i_phase in range(self.N_phase):
                        retina[i_theta, i_azimuth, i_eccentricity, i_phase, :] = self.local_filter(i_theta,
                                                                                              i_azimuth,
                                                                                              i_eccentricity,
                                                                                              i_phase,
                                                                                              lg,
                                                                                              N_X=self.N_pic,
                                                                                              N_Y=self.N_pic)
                                                                                              #rho=self.args.rho,
                                                                                              #ecc_max=self.args.ecc_max,
                                                                                              #sf_0_max=self.args.sf_0_max,
                                                                                              #sf_0_r=self.args.sf_0_r,
                                                                                              #B_sf=self.args.B_sf,
                                                                                              #B_theta=self.args.B_theta)

        return retina

    def local_filter(self, i_theta, i_azimuth, i_eccentricity, i_phase, lg,
                     N_X=128, N_Y=128):
                     #rho=1.41, ecc_max=.8,
                     #sf_0_max=0.45, sf_0_r=0.03,
                     #B_sf=.4, B_theta=np.pi / 12):

        ecc = self.args.ecc_max * (1 / self.args.rho) ** (self.N_eccentricity - i_eccentricity)
        r = np.sqrt(N_X ** 2 + N_Y ** 2) / 2 * ecc  # radius
        # psi = i_azimuth * np.pi * 2 / N_azimuth
        psi = (i_azimuth + 1 * (i_eccentricity % 2) * .5) * np.pi * 2 / self.N_azimuth
        theta_ref = i_theta * np.pi / self.N_theta
        sf_0 = 0.5 * self.args.sf_0_r / ecc
        sf_0 = np.min((sf_0, .45))
        # TODO : find the good ref for this                print(sf_0)
        x = N_X / 2 + r * np.cos(psi)
        y = N_Y / 2 + r * np.sin(psi)
        params = {'sf_0': sf_0,
                  'B_sf': self.args.B_sf,
                  'theta': theta_ref + psi,
                  'B_theta': self.args.B_theta}
        phase = i_phase * np.pi / 2
        return lg.normalize(lg.invert(lg.loggabor(x, y, **params) * np.exp(-1j * phase))).ravel()

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
        #accuracy_colliculus, accuracy_fullfield_map = accuracy_fullfield(accuracy_map, i_offset, j_offset, self.args.N_pic, self.colliculus_transform_vector)
        from display import do_offset
        accuracy_fullfield_map = do_offset(data=accuracy_map,
                                           i_offset=i_offset,
                                           j_offset=j_offset,
                                           N_pic=self.N_pic,
                                           data_min=0.1)
        accuracy_colliculus = self.colliculus_transform_vector @ accuracy_fullfield_map.ravel()
        return accuracy_colliculus, accuracy_fullfield_map
    
    def accuracy_invert(self, accuracy_colliculus):
        im = self.colliculus_inverse @ accuracy_colliculus

        return im.reshape(self.args.N_pic, self.args.N_pic)
    
    
    def show(self, ax, im, rmin=None, rmax=None, ms=26, markeredgewidth=1, alpha=.6, lw=.75, do_cross=True):
        # ???? Rien à voir avec la choucroute ?????
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




