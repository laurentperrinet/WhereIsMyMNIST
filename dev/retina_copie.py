import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../figures")

#import SLIP for whitening and PIL for resizing
import SLIP
from display import pe
from LogGabor import LogGabor

##########################################################################################################@
##########################################################################################################@
##########################################################################################################@

def affiche(donnees, titre, c_a, commentaire=None):
    if c_a:
        if commentaire:
            print(commentaire)
        plt.imshow(donnees)
        plt.title(titre)
        plt.show()

class Retina:
    """ Class implementing the retina transform
    """
    def __init__(self, args):

        self.args = args

        self.N_pic = args.N_pic
        self.whit = SLIP.Image(pe=pe)
        self.whit.set_size((self.N_pic, self.N_pic))
        # https://github.com/bicv/SLIP/blob/master/SLIP/SLIP.py#L611
        self.K_whitening = self.whit.whitening_filt()

        self.N_theta = args.N_theta
        self.N_azimuth = args.N_azimuth
        self.N_eccentricity = args.N_eccentricity
        self.N_phase = args.N_phase
        self.feature_vector_size = self.N_theta * self.N_azimuth * self.N_eccentricity * self.N_phase

        self.init_retina_dico()
        #self.init_colliculus_transform_dico()
        self.init_grid()
        self.init_retina_transform()
        #self.init_inverse_retina() # il faudra le decommenter plus tard
        self.init_colliculus_transform()
        #self.init_colliculus_inverse()

    def init_grid(self):
        delta = 1. / self.N_azimuth
        self.log_r_grid, self.theta_grid = \
        np.meshgrid(np.linspace(0, 1, self.N_eccentricity + 1),
                    np.linspace(-np.pi * (.5 + delta), np.pi * (1.5 - delta), self.N_azimuth + 1))

    def get_suffix(self):
        # suffix = f'_{self.N_theta}_{self.N_azimuth}'
        # suffix += f'_{self.N_eccentricity}_{self.N_phase}'
        # suffix += f'_{self.args.rho}_{self.N_pic}'

        suffix = '_{}_{}'.format(self.N_theta, self.N_azimuth)
        suffix += '_{}_{}'.format(self.N_eccentricity, self.N_phase)
        suffix += '_{}_{}'.format(self.args.rho, self.N_pic)
        return suffix

    def init_retina_transform(self):
        filename = '../tmp/retina' + self.get_suffix() + '_transform.npy'
        print(filename)
        try:
            self.retina_transform = np.load(filename)
            print("Fichier retina_transform charge avec succes")
        except:
            if self.args.verbose: print('Retina vectorizing...')
            self.retina_transform = self.vectorization()
            print("ok")
            np.save(filename, self.retina_transform)
            print("Fichier retina_transform ecrit et sauvegarde avec succes")
            if self.args.verbose: print('Done vectorizing...')
        self.retina_transform_vector = self.retina_transform.reshape((self.feature_vector_size, self.N_pic ** 2))

    def init_inverse_retina(self):
        filename = '../tmp/retina' + self.get_suffix() + '_inverse_transform.npy'
        print(filename)
        try:
            self.retina_inverse_transform = np.load(filename)
            print("Fichier retina_inverse_transform charge avec succes")
        except:
            if self.args.verbose: print('Inversing retina transform...')
            self.retina_inverse_transform = np.linalg.pinv(self.retina_transform_vector)
            print("ok2")
            np.save(filename, self.retina_inverse_transform)
            print("Fichier retina_inverse_transform ecrit et sauvegarde avec succes")
            if self.args.verbose: print('Done Inversing retina transform...')

    def init_colliculus_transform(self):
        # TODO : make a different transformation for the clliculus (more eccentricties?)
        #print(self.retina_transform)
        self.colliculus_transform = (self.retina_transform ** 2).sum(axis=(0, 3))
        # colliculus = colliculus**.5
        self.colliculus_transform /= self.colliculus_transform.sum(axis=-1)[:, :, None]  # normalization as a probability
        self.colliculus_transform_vector = self.colliculus_transform.reshape((self.N_azimuth * self.N_eccentricity, self.N_pic ** 2))

    def init_colliculus_transform_dico(self):
        filename = '../tmp/colliculus_transform' + self.get_suffix() + '_dico.npy'
        if self.args.verbose: print(filename)
        try:
            self.colliculus_transform_dico = np.load(filename).item()
            if self.args.verbose: print("Fichier colliculus_transform_dico charge avec succes")
        except:
            if self.args.verbose: print("Creation de colliculus_transform_dico en cours")
            self.colliculus_transform_dico = {}
            for i_azimuth in range(self.N_azimuth):
                self.colliculus_transform_dico[i_azimuth] = {}
                for i_eccentricity in range(self.N_eccentricity):
                    self.colliculus_transform_dico[i_azimuth][i_eccentricity] = {}
                    dimension_filtres = int(self.retina_dico[0][0][i_eccentricity][i_azimuth].shape[0] ** (1 / 2))
                    for pixel in range(dimension_filtres ** 2):
                        self.colliculus_transform_dico[i_azimuth][i_eccentricity][pixel] = 0
                        for i_theta in range(self.N_theta):
                            for i_phase in range(self.N_phase):
                                #print(pixel)
                                self.colliculus_transform_dico[i_azimuth][i_eccentricity][pixel] += (self.retina_dico[i_theta][i_phase][i_eccentricity][i_azimuth][pixel])**2
            somme_valeurs_pixels = {}
            for i_azimuth in range(self.N_azimuth):
                somme_valeurs_pixels[i_azimuth] = {}
                for i_eccentricity in range(self.N_eccentricity):
                    somme_valeurs_pixels[i_azimuth][i_eccentricity] = 0
                    dimension_filtres = int(self.retina_dico[0][0][i_eccentricity][i_azimuth].shape[0] ** (1 / 2))
                    for pixel in range(dimension_filtres ** 2):
                        somme_valeurs_pixels[i_azimuth][i_eccentricity] += self.colliculus_transform_dico[i_azimuth][i_eccentricity][pixel]
            for i_azimuth in range(self.N_azimuth):
                for i_eccentricity in range(self.N_eccentricity):
                    dimension_filtres = int(self.retina_dico[0][0][i_eccentricity][i_azimuth].shape[0] ** (1 / 2))
                    for pixel in range(dimension_filtres ** 2):
                        self.colliculus_transform_dico[i_azimuth][i_eccentricity][pixel] /= somme_valeurs_pixels[i_azimuth][i_eccentricity] # normalization as a probability
            final_colliculus_transform_dico = {}
            for i_azimuth in range(self.N_azimuth):
                final_colliculus_transform_dico[i_azimuth] = {}
                for i_eccentricity in range(self.N_eccentricity):
                    final_colliculus_transform_dico[i_azimuth][i_eccentricity] = []
                    dimension_filtres = int(self.retina_dico[0][0][i_eccentricity][i_azimuth].shape[0] ** (1 / 2))
                    for pixel in range(dimension_filtres ** 2):
                        final_colliculus_transform_dico[i_azimuth][i_eccentricity].append(self.colliculus_transform_dico[i_azimuth][i_eccentricity][pixel])
            self.colliculus_transform_dico = final_colliculus_transform_dico
            if self.args.verbose : print("Colliculus_transform_dico cree")
            np.save(filename, self.retina_dico)
            if self.args.verbose: print("Colliculus_transform_dico sauvegarde")

    def init_colliculus_inverse(self):
        self.colliculus_inverse = np.linalg.pinv(self.colliculus_transform_vector)

    def local_filter_dico(self, i_theta, i_azimuth, i_eccentricity, i_phase, lg=LogGabor(pe=pe),
                               N_X=128, N_Y=128):
        # rho=1.41, ecc_max=.8,
        # sf_0_max=0.45, sf_0_r=0.03,
        # B_sf=.4, B_theta=np.pi / 12): # on enleve self pour l'instant

        # !!?? Magic numbers !!??
        ecc_max = .8  # self.args.ecc_max
        sf_0_r = 0.03  # self.args.sf_0_r
        B_theta = np.pi / self.N_theta / 2  # self.args.B_theta
        B_sf = .4
        sf_0_max = 0.45

        ecc = ecc_max * (1 / self.args.rho) ** (self.N_eccentricity - i_eccentricity)
        r = np.sqrt(N_X ** 2 + N_Y ** 2) / 2 * ecc  # radius
        # print(r)
        dimension_filtre = min(2 * int(2 * r),
                               self.args.N_pic)  # 2*int(2*r) pour avoir des filtres vraiment de la meme taille qu'avant
        lg.set_size((dimension_filtre, dimension_filtre))
        # psi = i_azimuth * np.pi * 2 / N_azimuth
        psi = (i_azimuth + 1 * (i_eccentricity % 2) * .5) * np.pi * 2 / self.N_azimuth
        theta_ref = i_theta * np.pi / self.N_theta
        sf_0 = 0.5 * sf_0_r / ecc
        sf_0 = np.min((sf_0, sf_0_max))
        # TODO : find the good ref for this                print(sf_0)
        x = N_X / 2 + r * np.cos(psi)  # c'est bien le centre du filtre ?
        y = N_Y / 2 + r * np.sin(psi)  # c'est bien le centre du filtre ?
        params = {'sf_0': sf_0,
                  'B_sf': B_sf,
                  'theta': theta_ref + psi,
                  'B_theta': B_theta}
        phase = i_phase * np.pi / 2
        # lg.show_loggabor(x, y, **params)
        # print('taille sortie', lg.loggabor(x, y, **params).ravel().shape)
        return lg.normalize(
            lg.invert(lg.loggabor(dimension_filtre // 2, dimension_filtre // 2, **params) * np.exp(-1j * phase)))

    def init_retina_dico(self):
        filename = '../tmp/retina' + self.get_suffix() + '_dico.npy'
        if self.args.verbose: print(filename)
        try:
            self.retina_dico = np.load(filename, allow_pickle=True).item()
            if self.args.verbose: print("Fichier retina_dico charge avec succes")
        except:
            if self.args.verbose: print('Creation du dictionnaire de filtres en cours...')
            self.retina_dico = {}
            lg = LogGabor(pe=pe)
            for i_theta in range(self.N_theta):
                self.retina_dico[i_theta] = {}
                for i_phase in range(self.N_phase):
                    self.retina_dico[i_theta][i_phase] = {}
                    for i_eccentricity in range(self.N_eccentricity):
                        self.retina_dico[i_theta][i_phase][i_eccentricity] = {}
                        for i_azimuth in range(self.N_azimuth):
                            self.retina_dico[i_theta][i_phase][i_eccentricity][i_azimuth] = np.ravel(self.local_filter_dico(i_theta, i_azimuth, i_eccentricity, i_phase, lg, N_X=self.args.N_pic, N_Y=self.args.N_pic))
            if self.args.verbose: print("Dico cree")
            np.save(filename, self.retina_dico)
            if self.args.verbose: print("len finale", len(self.retina_dico),len(self.retina_dico[0]),len(self.retina_dico[0][0]),len(self.retina_dico[0][0][0]),len(self.retina_dico[0][0][0][0]))
            if self.args.verbose: print("Fichier retina_dico ecrit et sauvegarde avec succes")

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

    def online_vectorization(self, pixel_fullfield):   # pixel_fullfield = image
        fullfield_dot_filters = np.zeros(self.N_theta * self.N_azimuth * self.N_eccentricity * self.N_phase)

        from LogGabor import LogGabor
        lg = LogGabor(pe=pe)
        lg.set_size((self.N_pic, self.N_pic))

        for i_theta in range(self.N_theta):
            for i_azimuth in range(self.N_azimuth):
                for i_eccentricity in range(self.N_eccentricity):
                    for i_phase in range(self.N_phase):
                        filter = self.local_filter(i_theta, i_azimuth, i_eccentricity, i_phase, lg, N_X=self.N_pic,
                                                  N_Y=self.N_pic)
                        indice = i_theta + i_azimuth + i_eccentricity + i_phase
                        fullfield_dot_filters[indice] = np.dot(np.ravel(filter), np.ravel(pixel_fullfield))

        return fullfield_dot_filters

    def transform_dico(self, pixel_fullfield):
        fullfield_dot_retina_dico = np.zeros(self.N_theta * self.N_azimuth * self.N_eccentricity * self.N_phase)

        N_X, N_Y = self.N_pic, self.N_pic

        indice = 0
        for i_theta in range(self.N_theta):
            for i_azimuth in range(self.N_azimuth):
                for i_eccentricity in range(self.N_eccentricity):
                    for i_phase in range(self.N_phase):

                        c_a = i_azimuth == 0 and i_theta == 0 and i_phase == 0  # conditions d'affichage
                        c_a = False

                        fenetre_filtre = self.retina_dico[i_theta][i_phase][i_eccentricity][i_azimuth]
                        dimension_filtre = int(fenetre_filtre.shape[0] ** (1 / 2))
                        fenetre_filtre = fenetre_filtre.reshape((dimension_filtre, dimension_filtre))

                        ecc_max = .8
                        ecc = ecc_max * (1 / self.args.rho) ** (self.args.N_eccentricity - i_eccentricity)
                        r = np.sqrt(N_X ** 2 + N_Y ** 2) / 2 * ecc  # radius
                        psi = (i_azimuth + 1 * (i_eccentricity % 2) * .5) * np.pi * 2 / self.args.N_azimuth
                        x = int(N_X / 2 + r * np.cos(psi))
                        y = int(N_Y / 2 + r * np.sin(psi))
                        r = int(r)

                        r = dimension_filtre // 2

                        affiche(fenetre_filtre, "fenetre_filtre", c_a)

                        fenetre_image = pixel_fullfield[int(x - r):int(x + r), int(y - r):int(y + r)]

                        if np.ravel(fenetre_image).shape != np.ravel(fenetre_filtre).shape:
                            fenetre_image = np.zeros((dimension_filtre, dimension_filtre))

                            if y + r > self.N_pic:  # ca depasse à droite
                                morceau_interne_fullfield = pixel_fullfield[x - r:x + r, y - r:self.N_pic]
                                morceau_externe_fullfield = pixel_fullfield[x - r:x + r,
                                                            0:y + r - self.N_pic]  # qu'on est donc alle chercher ailleurs dans l'image
                                nb_lignes = morceau_externe_fullfield.shape[0]
                                if nb_lignes == 2 * r:  # ce n'est pas un coin
                                    fenetre_image[0:2 * r, 0:r + self.N_pic - y] = morceau_interne_fullfield
                                    affiche(fenetre_image, "Construction fenetre_image : ajout partie gauche", c_a)
                                    fenetre_image[0:2 * r, r + self.N_pic - y:2 * r] = morceau_externe_fullfield
                                    affiche(fenetre_image, "Construction fenetre_image : ajout partie droite", c_a)

                                elif x - r < 0:  # contient le coin superieur droit
                                    morceau_externe_fullfield = pixel_fullfield[0:x + r, 0:r - self.N_pic + y]
                                    fenetre_image[r - x:2 * r, r + self.N_pic - y:2 * r] = morceau_externe_fullfield
                                    affiche(fenetre_image, "Construction image : ajout bas droit", c_a)
                                    fenetre_image[0:r - x, r - y + self.N_pic:2 * r] = pixel_fullfield[
                                                                                       self.N_pic - r + x:self.N_pic,
                                                                                       0:y + r - self.N_pic]
                                    affiche(fenetre_image, "Construction fenetre_image : ajout haut droit", c_a)

                                elif x + r > self.N_pic:  # contient le coin inferieur droit
                                    fenetre_image[0:nb_lignes, r + self.N_pic - y:2 * r] = morceau_externe_fullfield
                                    affiche(fenetre_image, "Construction fenetre_image : ajout haut droit", c_a)
                                    fenetre_image[r - x + self.N_pic:2 * r, r - y + self.N_pic:2 * r] = pixel_fullfield[
                                                                                                        0:x + r - self.N_pic,
                                                                                                        0:y + r - self.N_pic]
                                    affiche(fenetre_image, "Construction fenetre_image : ajout bas droit", c_a)


                            elif y - r < 0:  # ca depasse a gauche
                                morceau_externe_fullfield = pixel_fullfield[x - r:x + r, self.N_pic - r + y:self.N_pic]
                                morceau_interne_fullfield = pixel_fullfield[x - r:x + r, 0:y + r]
                                nb_lignes = morceau_externe_fullfield.shape[0]
                                if nb_lignes == 2 * r:  # ce n'est pas un coin
                                    fenetre_image[0:2 * r, 0:r - y] = morceau_externe_fullfield
                                    affiche(fenetre_image, "Construction fenetre_image : ajout partie gauche", c_a)
                                    fenetre_image[0:2 * r, r - y:2 * r] = morceau_interne_fullfield
                                    affiche(fenetre_image, "Construction fenetre_image : ajout partie droite", c_a)

                                elif x - r < 0:  # contient le coin superieur gauche
                                    morceau_externe_fullfield = pixel_fullfield[0:x + r, self.N_pic - r + y:self.N_pic]
                                    fenetre_image[r - x:2 * r, 0:r - y] = morceau_externe_fullfield
                                    affiche(fenetre_image, "Construction fenetre_image : ajout bas gauche", c_a)
                                    fenetre_image[0:r - x, 0:r - y] = pixel_fullfield[self.N_pic - r + x:self.N_pic,
                                                                      self.N_pic - r + y:self.N_pic]
                                    affiche(fenetre_image, "Construction fenetre_image : ajout haut gauche", c_a)

                                elif x + r > self.N_pic:  # contient le coin inferieur gauche
                                    fenetre_image[r + self.N_pic - x:2 * r, 0:r - y] = pixel_fullfield[
                                                                                       0:x + r - self.N_pic,
                                                                                       self.N_pic - r + y:self.N_pic]
                                    affiche(fenetre_image, "Construction fenetre_image : ajout bas gauche", c_a)
                                    fenetre_image[0:r + self.N_pic - x, 0:r - y] = morceau_externe_fullfield
                                    affiche(fenetre_image, "Construction fenetre_image : ajout haut gauche", c_a)

                            if x - r < 0:  # ca depasse en haut
                                morceau_externe_fullfield = pixel_fullfield[self.N_pic - r + x:self.N_pic, y - r:y + r]
                                morceau_interne_fullfield = pixel_fullfield[0:r + x, y - r:y + r]
                                nb_colonnes = morceau_externe_fullfield.shape[1]
                                if nb_colonnes == 2 * r:  # ce n'est pas un coin
                                    fenetre_image[r - x:2 * r, 0:2 * r] = morceau_interne_fullfield
                                    affiche(fenetre_image, "Construction fenetre_image : ajout partie basse", c_a)
                                    fenetre_image[0:r - x, 0:2 * r] = morceau_externe_fullfield
                                    affiche(fenetre_image, "Construction fenetre_image : ajout partie haute", c_a)

                                elif y - r < 0:  # contient le coin superieur gauche
                                    morceau_externe_fullfield = pixel_fullfield[self.N_pic - r + x:self.N_pic, 0:y + r]
                                    morceau_interne_fullfield = pixel_fullfield[0:r + x, 0:y + r]
                                    fenetre_image[0:r - x, r - y:2 * r] = morceau_externe_fullfield
                                    affiche(fenetre_image, "Construction fenetre_image : ajout haut droit", c_a)
                                    fenetre_image[r - x:2 * r, r - y:2 * r] = morceau_interne_fullfield
                                    affiche(fenetre_image, "Construction fenetre_image : ajout bas droit", c_a)

                                elif y + r > self.N_pic:  # contient le coin superieur droit
                                    nb_colonnes = r + self.N_pic - y
                                    fenetre_image[0:r - x, 0:nb_colonnes] = morceau_externe_fullfield
                                    affiche(fenetre_image, "Construction fenetre_image : ajout haut gauche", c_a)
                                    fenetre_image[r - x:2 * r, 0:nb_colonnes] = morceau_interne_fullfield
                                    affiche(fenetre_image, "Construction fenetre_image : ajout bas gauche", c_a)


                            elif x + r > self.N_pic:  # ca depasse en bas
                                morceau_interne_fullfield = pixel_fullfield[x - r:self.N_pic, y - r:y + r]
                                morceau_externe_fullfield = pixel_fullfield[0:r - self.N_pic + x, y - r:y + r]
                                nb_colonnes = morceau_externe_fullfield.shape[1]
                                if nb_colonnes == 2 * r:  # ce n'est pas un coin
                                    fenetre_image[0:self.N_pic - x + r, 0:2 * r] = morceau_interne_fullfield
                                    affiche(fenetre_image, "Construction fenetre_image : ajout partie haute", c_a)
                                    fenetre_image[self.N_pic - x + r:2 * r, 0:2 * r] = morceau_externe_fullfield
                                    affiche(fenetre_image, "Construction fenetre_image : ajout partie basse", c_a)

                                elif y - r < 0:  # contient le coin inferieur gauche
                                    morceau_interne_fullfield = pixel_fullfield[x - r:self.N_pic, 0:r + y]
                                    morceau_externe_fullfield = pixel_fullfield[0:r - self.N_pic + x, 0:r + y]
                                    fenetre_image[0:self.N_pic - x + r, r - y:2 * r] = morceau_interne_fullfield
                                    affiche(fenetre_image, "Construction fenetre_image : ajout haut droit", c_a)
                                    fenetre_image[self.N_pic - x + r:2 * r, r - y:2 * r] = morceau_externe_fullfield
                                    affiche(fenetre_image, "Construction fenetre_image : ajout bas droit", c_a)

                                elif y + r > self.N_pic:  # contient le coin inferieur droit
                                    fenetre_image[0:self.N_pic - x + r, 0:nb_colonnes] = morceau_interne_fullfield
                                    affiche(fenetre_image, "Construction fenetre_image : ajout haut gauche", c_a)
                                    fenetre_image[self.N_pic - x + r:2 * r, 0:nb_colonnes] = morceau_externe_fullfield
                                    affiche(fenetre_image, "Construction fenetre_image : ajout bas gauche", c_a)

                        affiche(fenetre_image, "image correspondant au filtre ci-dessus", c_a)

                        a = np.dot(np.ravel(fenetre_filtre), np.ravel(fenetre_image))
                        fullfield_dot_retina_dico[indice] = a

                        indice += 1

        return pixel_fullfield, fullfield_dot_retina_dico

    def inverse_transform_dico(self, retina_features):
        N_X, N_Y = self.N_pic, self.N_pic
        rebuild_pixel_fullfield = np.zeros((N_X, N_Y))
        indice_coefficient = 0
        for i_theta in range(self.N_theta):
            for i_azimuth in range(self.N_azimuth):
                inter_rebuild_pixel_fullfield = np.zeros((N_X, N_Y))
                for i_eccentricity in range(self.N_eccentricity):
                    for i_phase in range(self.N_phase):
                        fenetre_filtre = self.retina_dico[i_theta][i_phase][i_eccentricity][i_azimuth]
                        dimension_filtre = int(fenetre_filtre.shape[0] ** (1 / 2))
                        fenetre_filtre = fenetre_filtre.reshape((dimension_filtre, dimension_filtre))
                        coefficient = float(retina_features[indice_coefficient])

                        type_affichage = 3 # 1 affichage habituel, 2 affichage graduel, 3 affichage par azimuth croissant

                        if type_affichage == 1:
                            c_a = i_azimuth == 16 and i_theta == 0 and i_phase == 0  # conditions d'affichage
                        else:
                            c_a = False

                        # print("coeff", type(coefficient))
                        # print("filtre", type(fenetre_filtre))
                        morceau_image_reconstituee = coefficient * fenetre_filtre

                        # il faut a present placer le morceau au bon endroit de l'image

                        ecc_max = .8
                        ecc = ecc_max * (1 / self.args.rho) ** (self.N_eccentricity - i_eccentricity)
                        r = np.sqrt(N_X ** 2 + N_Y ** 2) / 2 * ecc  # radius
                        psi = (i_azimuth + 1 * (i_eccentricity % 2) * .5) * np.pi * 2 / self.N_azimuth
                        x = int(N_X / 2 + r * np.cos(psi))
                        y = int(N_Y / 2 + r * np.sin(psi))

                        r = dimension_filtre // 2

                        # if c_a :
                        #    print("x, y, r", x,y,r)

                        # print(c_a)
                        # if c_a :
                        #    print(coefficient)

                        affiche(fenetre_filtre, "fenetre_filtre", c_a)
                        affiche(morceau_image_reconstituee, "morceau_image_reconstituee", c_a)

                        if type_affichage == 3 and i_theta == 0 and i_phase == 0 and i_azimuth == 12:
                            c_a = True
                            affiche(fenetre_filtre, "fenetre_filtre", c_a)
                            c_a = False

                        fenetre_image = inter_rebuild_pixel_fullfield[int(x - r):int(x + r), int(y - r):int(y + r)]

                        # if c_a :
                        #    print(np.ravel(fenetre_image).shape, np.ravel(fenetre_filtre).shape)

                        if np.ravel(fenetre_image).shape == np.ravel(fenetre_filtre).shape:
                            inter_rebuild_pixel_fullfield[int(x - r):int(x + r),
                            int(y - r):int(y + r)] += morceau_image_reconstituee

                        if np.ravel(fenetre_image).shape != np.ravel(fenetre_filtre).shape:
                            fenetre_image = np.zeros((dimension_filtre, dimension_filtre))

                            if y + r > self.N_pic:  # ca depasse à droite
                                morceau_interne_fullfield = inter_rebuild_pixel_fullfield[x - r:x + r, y - r:self.N_pic]
                                morceau_externe_fullfield = inter_rebuild_pixel_fullfield[x - r:x + r,
                                                            0:y + r - self.N_pic]  # qu'on est donc alle chercher ailleurs dans l'image
                                nb_lignes = morceau_externe_fullfield.shape[0]
                                if nb_lignes == 2 * r:  # ce n'est pas un coin
                                    # print("rebuild1", rebuild_pixel_fullfield[0:x + r, 0:r - self.N_pic + y].shape)
                                    # print("reconstit1", morceau_image_reconstituee[r - x:2 * r, r + self.N_pic - y:2 * r].shape)
                                    inter_rebuild_pixel_fullfield[x - r:x + r,
                                    y - r:self.N_pic] += morceau_image_reconstituee[0:2 * r, 0:r + self.N_pic - y]
                                    # affiche(inter_rebuild_pixel_fullfield, "Construction rebuild_pixel_fullfield : ajout partie gauche", c_a)
                                    inter_rebuild_pixel_fullfield[x - r:x + r,
                                    0:y + r - self.N_pic] += morceau_image_reconstituee[0:2 * r,
                                                             r + self.N_pic - y:2 * r]
                                    # affiche(inter_rebuild_pixel_fullfield, "Construction rebuild_pixel_fullfield : ajout partie droite", c_a)


                                elif x - r < 0:  # contient le coin superieur droit
                                    # print("rebuild2", rebuild_pixel_fullfield[0:x + r, 0:r - self.N_pic + y].shape)
                                    # print("reconstit2", morceau_image_reconstituee[r - x:2 * r, r + self.N_pic - y:2 * r].shape)
                                    inter_rebuild_pixel_fullfield[0:x + r,
                                    0:r - self.N_pic + y] += morceau_image_reconstituee[r - x:2 * r,
                                                             r + self.N_pic - y:2 * r]
                                    # affiche(inter_rebuild_pixel_fullfield, "Construction rebuild_pixel_fullfield : ajout bas droit", c_a)
                                    inter_rebuild_pixel_fullfield[self.N_pic - r + x:self.N_pic,
                                    0:y + r - self.N_pic] += morceau_image_reconstituee[0:r - x,
                                                             r - y + self.N_pic:2 * r]
                                    # affiche(inter_rebuild_pixel_fullfield, "Construction rebuild_pixel_fullfield : ajout haut droit", c_a)


                                elif x + r > self.N_pic:  # contient le coin inferieur droit

                                    inter_rebuild_pixel_fullfield[x - r:x + r,
                                    0:y + r - self.N_pic] += morceau_image_reconstituee[0:nb_lignes,
                                                             r + self.N_pic - y:2 * r]
                                    # affiche(inter_rebuild_pixel_fullfield, "Construction rebuild_pixel_fullfield : ajout haut droit", c_a)
                                    # print("rebuild3", rebuild_pixel_fullfield[0:x + r - self.N_pic, 0:y + r - self.N_pic].shape)
                                    # print("reconstit3", morceau_image_reconstituee[0:nb_lignes, r + self.N_pic - y:2 * r].shape)
                                    inter_rebuild_pixel_fullfield[0:x + r - self.N_pic,
                                    0:y + r - self.N_pic] += morceau_image_reconstituee[r - x + self.N_pic:2 * r,
                                                             r - y + self.N_pic:2 * r]
                                    # affiche(inter_rebuild_pixel_fullfield, "Construction rebuild_pixel_fullfield : ajout bas droit", c_a)


                            elif y - r < 0:  # ca depasse a gauche
                                morceau_externe_fullfield = inter_rebuild_pixel_fullfield[x - r:x + r,
                                                            self.N_pic - r + y:self.N_pic]
                                morceau_interne_fullfield = inter_rebuild_pixel_fullfield[x - r:x + r, 0:y + r]
                                nb_lignes = morceau_externe_fullfield.shape[0]
                                if nb_lignes == 2 * r:  # ce n'est pas un coin
                                    inter_rebuild_pixel_fullfield[x - r:x + r,
                                    self.N_pic - r + y:self.N_pic] += morceau_image_reconstituee[0:2 * r, 0:r - y]
                                    # affiche(inter_rebuild_pixel_fullfield, "Construction rebuild_pixel_fullfield : ajout partie gauche", c_a)
                                    inter_rebuild_pixel_fullfield[x - r:x + r, 0:y + r] += morceau_image_reconstituee[
                                                                                           0:2 * r, r - y:2 * r]
                                    # affiche(inter_rebuild_pixel_fullfield, "Construction rebuild_pixel_fullfield : ajout partie droite", c_a)

                                elif x - r < 0:  # contient le coin superieur gauche
                                    inter_rebuild_pixel_fullfield[0:x + r,
                                    self.N_pic - r + y:self.N_pic] += morceau_image_reconstituee[r - x:2 * r, 0:r - y]
                                    # affiche(inter_rebuild_pixel_fullfield, "Construction rebuild_pixel_fullfield : ajout bas gauche", c_a)
                                    inter_rebuild_pixel_fullfield[self.N_pic - r + x:self.N_pic,
                                    self.N_pic - r + y:self.N_pic] += morceau_image_reconstituee[0:r - x, 0:r - y]
                                    # affiche(inter_rebuild_pixel_fullfield, "Construction rebuild_pixel_fullfield : ajout haut gauche", c_a)

                                elif x + r > self.N_pic:  # contient le coin inferieur gauche
                                    inter_rebuild_pixel_fullfield[0:x + r - self.N_pic,
                                    self.N_pic - r + y:self.N_pic] += morceau_image_reconstituee[
                                                                      r + self.N_pic - x:2 * r, 0:r - y]
                                    # affiche(inter_rebuild_pixel_fullfield, "Construction rebuild_pixel_fullfield : ajout bas gauche", c_a)
                                    inter_rebuild_pixel_fullfield[x - r:x + r,
                                    self.N_pic - r + y:self.N_pic] += morceau_image_reconstituee[0:r + self.N_pic - x,
                                                                      0:r - y]
                                    # affiche(inter_rebuild_pixel_fullfield, "Construction rebuild_pixel_fullfield : ajout haut gauche", c_a)

                            if x - r < 0:  # ca depasse en haut
                                morceau_externe_fullfield = inter_rebuild_pixel_fullfield[self.N_pic - r + x:self.N_pic,
                                                            y - r:y + r]
                                morceau_interne_fullfield = inter_rebuild_pixel_fullfield[0:r + x, y - r:y + r]
                                nb_colonnes = morceau_externe_fullfield.shape[1]
                                if nb_colonnes == 2 * r:  # ce n'est pas un coin
                                    inter_rebuild_pixel_fullfield[0:r + x, y - r:y + r] += morceau_image_reconstituee[
                                                                                           r - x:2 * r, 0:2 * r]
                                    # affiche(inter_rebuild_pixel_fullfield, "Construction rebuild_pixel_fullfield : ajout partie basse", c_a)
                                    inter_rebuild_pixel_fullfield[self.N_pic - r + x:self.N_pic,
                                    y - r:y + r] += morceau_image_reconstituee[0:r - x, 0:2 * r]
                                    # affiche(inter_rebuild_pixel_fullfield, "Construction rebuild_pixel_fullfield : ajout partie haute", c_a)

                                elif y - r < 0:  # contient le coin superieur gauche
                                    inter_rebuild_pixel_fullfield[self.N_pic - r + x:self.N_pic,
                                    0:y + r] += morceau_image_reconstituee[0:r - x, r - y:2 * r]
                                    # affiche(inter_rebuild_pixel_fullfield, "Construction rebuild_pixel_fullfield : ajout haut droit", c_a)
                                    inter_rebuild_pixel_fullfield[0:r + x, 0:y + r] += morceau_image_reconstituee[
                                                                                       r - x:2 * r, r - y:2 * r]
                                    # affiche(inter_rebuild_pixel_fullfield, "Construction rebuild_pixel_fullfield : ajout bas droit", c_a)

                                elif y + r > self.N_pic:  # contient le coin superieur droit
                                    inter_rebuild_pixel_fullfield[self.N_pic - r + x:self.N_pic,
                                    y - r:y + r] += morceau_image_reconstituee[0:r - x, 0:r + self.N_pic - y]
                                    # affiche(inter_rebuild_pixel_fullfield, "Construction rebuild_pixel_fullfield : ajout haut gauche", c_a)
                                    inter_rebuild_pixel_fullfield[0:r + x, y - r:y + r] += morceau_image_reconstituee[
                                                                                           r - x:2 * r, 0:nb_colonnes]
                                    # affiche(inter_rebuild_pixel_fullfield, "Construction rebuild_pixel_fullfield : ajout bas gauche", c_a)


                            elif x + r > self.N_pic:  # ca depasse en bas
                                morceau_interne_fullfield = inter_rebuild_pixel_fullfield[x - r:self.N_pic, y - r:y + r]
                                morceau_externe_fullfield = inter_rebuild_pixel_fullfield[0:r - self.N_pic + x,
                                                            y - r:y + r]
                                nb_colonnes = morceau_externe_fullfield.shape[1]
                                if nb_colonnes == 2 * r:  # ce n'est pas un coin
                                    inter_rebuild_pixel_fullfield[x - r:self.N_pic,
                                    y - r:y + r] += morceau_image_reconstituee[0:self.N_pic - x + r, 0:2 * r]
                                    affiche(inter_rebuild_pixel_fullfield,
                                            "Construction rebuild_pixel_fullfield : ajout partie gauche", c_a)
                                    inter_rebuild_pixel_fullfield[0:r - self.N_pic + x,
                                    y - r:y + r] += morceau_image_reconstituee[self.N_pic - x + r:2 * r, 0:2 * r]
                                    affiche(inter_rebuild_pixel_fullfield,
                                            "Construction rebuild_pixel_fullfield : ajout partie droite", c_a)

                                elif y - r < 0:  # contient le coin inferieur gauche
                                    inter_rebuild_pixel_fullfield[x - r:self.N_pic,
                                    0:r + y] += morceau_image_reconstituee[0:self.N_pic - x + r, r - y:2 * r]
                                    affiche(inter_rebuild_pixel_fullfield,
                                            "Construction rebuild_pixel_fullfield : ajout haut droit", c_a)
                                    inter_rebuild_pixel_fullfield[0:r - self.N_pic + x,
                                    0:r + y] += morceau_image_reconstituee[self.N_pic - x + r:2 * r, r - y:2 * r]
                                    affiche(inter_rebuild_pixel_fullfield,
                                            "Construction rebuild_pixel_fullfield : ajout bas droit", c_a)

                                elif y + r > self.N_pic:  # contient le coin inferieur droit
                                    inter_rebuild_pixel_fullfield[x - r:self.N_pic,
                                    y - r:y + r] += morceau_image_reconstituee[0:self.N_pic - x + r, 0:nb_colonnes]
                                    affiche(inter_rebuild_pixel_fullfield,
                                            "Construction rebuild_pixel_fullfield : ajout haut gauche", c_a)
                                    inter_rebuild_pixel_fullfield[0:r - self.N_pic + x,
                                    y - r:y + r] += morceau_image_reconstituee[self.N_pic - x + r:2 * r, 0:nb_colonnes]
                                    affiche(inter_rebuild_pixel_fullfield,
                                            "Construction rebuild_pixel_fullfield : ajout bas gauche", c_a)

                        affiche(inter_rebuild_pixel_fullfield,
                                "Image reconstituee apres l'application de ce filtre (et de tous les precedents de même azimuth)",
                                c_a)
                        c_a = False

                        if type_affichage == 2:
                            if indice_coefficient % 100 == 0:
                                print("coefficient numero", indice_coefficient)
                                c_a = True

                        affiche(rebuild_pixel_fullfield + inter_rebuild_pixel_fullfield,
                                "rebuild_pixel_fullfield en cours de construction", c_a)
                        c_a = False

                        indice_coefficient += 1

                if type_affichage == 3 and i_theta == 0:
                    print("coefficient numero", indice_coefficient)
                    c_a = True
                affiche(inter_rebuild_pixel_fullfield,
                        "rebuild_pixel_fullfield après application des filtre d'azimuth =" + str(i_azimuth), c_a)
                c_a = False
                rebuild_pixel_fullfield += inter_rebuild_pixel_fullfield

        if type_affichage != 0 : c_a = True
        affiche(rebuild_pixel_fullfield, "image finale", c_a)

        return rebuild_pixel_fullfield




    def local_filter(self, i_theta, i_azimuth, i_eccentricity, i_phase, lg,
                     N_X=128, N_Y=128):
                     #rho=1.41, ecc_max=.8,
                     #sf_0_max=0.45, sf_0_r=0.03,
                     #B_sf=.4, B_theta=np.pi / 12):

        # !!?? Magic numbers !!??
        ecc_max = .8 # self.args.ecc_max
        sf_0_r = 0.03 # self.args.sf_0_r
        B_theta = np.pi/self.N_theta/2 #self.args.B_theta
        B_sf = .4
        sf_0_max = 0.45
        
        ecc = ecc_max * (1 / self.args.rho) ** (self.N_eccentricity - i_eccentricity)
        r = np.sqrt(N_X ** 2 + N_Y ** 2) / 2 * ecc  # radius
        # psi = i_azimuth * np.pi * 2 / N_azimuth
        psi = (i_azimuth + 1 * (i_eccentricity % 2) * .5) * np.pi * 2 / self.N_azimuth
        theta_ref = i_theta * np.pi / self.N_theta
        sf_0 = 0.5 * sf_0_r / ecc
        sf_0 = np.min((sf_0, sf_0_max))
        # TODO : find the good ref for this                print(sf_0)
        x = N_X / 2 + r * np.cos(psi)
        y = N_Y / 2 + r * np.sin(psi)
        params = {'sf_0': sf_0,
                  'B_sf': B_sf,
                  'theta': theta_ref + psi,
                  'B_theta': B_theta}
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
        if do_dewhitening:
            im = self.whit.dewhitening(im)
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




