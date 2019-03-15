import os
import numpy as np
import time

def init(filename=None, verbose=1, log_interval=100):
    if filename is None:
        import datetime
        filename = '../data/' + datetime.datetime.now().date().isoformat()
        print('Using filename=', filename)

    import easydict
    args = easydict.EasyDict(
                            # MNIST
                            w=28,
                            minibatch_size = 100,  # quantity of examples that'll be processed
                            train_batch_size=60000, # train
                            test_batch_size=1000, 
                            noise_batch_size=1000, 
                            mean=0.1307, 
                            std=0.3081, 
                            # display
                            N_pic = 128,
                            offset_std = 30, #
                            offset_max = 35, #
                            noise=1., #0 #
                            contrast=0.8, #1 #
                            sf_0=0.2,
                            B_sf=0.3,
                            # foveation
                            N_theta = 6,
                            N_azimuth = 16,
                            N_eccentricity = 10,
                            N_phase = 2,
                            rho = 1.41,
                            # network
                            bias_deconv=True,
                            p_dropout=.5,
                            dim1=1000,
                            dim2=1000,
                            # training
                            lr = 1e-4, #1e-3  #0.05
                            do_adam=True,
                            epochs=40,
                            bn1_bn_momentum=0.,
                            bn2_bn_momentum=0.,
                            momentum=0.1,    
                            n_epochs=10,
                            # simulation
                            num_processes=1,
                            no_cuda=True,
                            log_interval=log_interval, # period with which we report results for the loss
                            verbose=verbose,
                            filename=filename,
                            seed=2019,
                            N_cv=20,
                                )
    if filename == 'debug':
        args.filename = '../data/debug'
        args.train_batch_size = 100
        args.lr = 1e-2
        #args.noise = .5
        #args.contrast = .9
        #args.p_dropout = 0.
        args.epochs = 2
        args.test_batch_size = 20
        args.minibatch_size = 22
        #args.offset_std = 8

    return args

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
            path = '../data/_tmp_scanning/' + parameter + '_' + self.tag + '_' + value_str.replace('.', '_') + '.npy'
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