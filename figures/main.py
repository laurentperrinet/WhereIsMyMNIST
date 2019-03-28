import os
import numpy as np
import time
import easydict

from what import WhatNet
from where import Where as ML

def init(filename=None, verbose=1, log_interval=100, do_compute=True):
    if filename is None:
        import datetime
        filename = '../data/' + datetime.datetime.now().date().isoformat()
        print('Using filename=', filename)

    import json
    filename_json = filename + '_param.json'
    

    return args

class MetaML:
    def __init__(self, args, base = 2, N_scan = 9, tag='', verbose=0, log_interval=0, do_compute=True):
        self.args = args
        self.seed = args.seed
        self.do_compute = do_compute

        self.base = base
        self.N_scan = N_scan
        self.tag = tag
        self.default = dict(verbose=verbose, log_interval=log_interval)
        self.scan_folder = '../data/_tmp_scanning'
        os.makedirs(self.scan_folder, exist_ok=True)

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

    def scan(self, parameter, values, verbose=True):
        import os
        print('scanning over', parameter, '=', values)
        seed = self.seed
        Accuracy = {}
        for value in values:
            if isinstance(value, int):
                value_str = str(value)
            else:
                value_str = '%.3f' % value
            filename = parameter + '_' + self.tag + '_' + value_str.replace('.', '_') + '.npy'
            path = os.path.join(self.scan_folder, filename)
            print ('For parameter', parameter, '=', value_str, ', ', end=" ")
            if not(os.path.isfile(path + '_lock')):
                if not(os.path.isfile(path)) and self.do_compute:
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

                if verbose:
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
        if parameter in ['bn1_bn_momentum', 'bn2_bn_momentum', 'p_dropout']:
            values = np.linspace(0, 1, self.N_scan, endpoint=True)
        else:
            values = self.args[parameter] * np.logspace(-1, 1, self.N_scan, base=self.base, endpoint=True)
        if isinstance(self.args[parameter], int):
            # print('integer detected') # DEBUG
            values =  [int(k) for k in values]
            
        accuracies = self.scan(parameter, values)
        # print('accuracies=', accuracies)
        if display:
            fig, ax = plt.subplots(figsize=(8, 5))
            # TODO
        return accuracies


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
