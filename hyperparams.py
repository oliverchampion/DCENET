'''
Mar 2018 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/deep-learning/ivim
'''
import torch


class network_training_hyper_parameters:
    def __init__(self):
        self.lr = 1e-4
        self.lr_mult = 0.1
        self.epochs = 2
        self.optim = 'adam'  # adam 0.0001; sgd 0.1
        self.patience = self.epochs
        self.optim_patience = 3  # 0 is disabled
        self.batch_size = 128  # alias = N
        self.val_batch_size = 1280
        self.split = 0.9
        self.totalit = 1000
        self.save_train_fig = True
        self.weight_decay = 0


class network_building_hyper_parameters:
    def __init__(self):
        self.dropout = 0
        self.nn = 'lstm'  # ['linear', 'convlin', 'lstm']
        self.layers = [32, 4]
        self.attention = False
        self.weighted_loss = False
        self.aif = True
        self.constrained = True
        self.dual_path = False
        self.bidirectional = False


class simulation_hyper_parameters:
    def __init__(self):
        self.num_samples = 500000
        self.num_samples_leval = 5000
        self.data_length = 160
        self.vp_min = 0.001
        self.vp_max = 0.05  # was 0.02
        self.ve_min = 0.01
        self.ve_max = 0.7  # was 1
        self.kep_min = 0.1
        self.kep_max = 2.
        self.R1_min = 1/2
        self.R1_max = 1/0.3
        self.time = 1.75 # 1.632 - 2.894
        self.Tonset_min = self.time * self.data_length//6
        self.Tonset_max = self.time * self.data_length//5
        self.dt_min = self.Tonset_min/60
        self.dt_max = self.Tonset_max/60
        self.what_to_sim = "nn"  # T1fit, lsq or nn
        self.plot = True

        self.bounds = torch.FloatTensor(((1e-8, 1e-6, 1e-6, 1e-2),
                                         (3, 1, 0.1, 1)
                                         ))  # ke, ve, vp, dt, ((min), (max))


class acquisition_parameters:
    def __init__(self):
        self.S0 = 1000.
        self.r1 = 5.0
        self.TR = 3.2e-3  # 7.2e-3 in ms for the new toolbox
        self.FA1 = 4./180*3.14159  # 4./180*3.14159
        self.FA2 = 20./180*3.14159  # 24./180*3.14159
        self.rep0 = 10
        self.rep1 = 1
        self.rep2 = 161


class AIF_parameters:
    def __init__(self):
        self.Hct = 0.40
        self.aif = {'ab': 7.9785,
                    'ae': 0.5216,
                    'ar': 0.0482,
                    'mb': 32.8855,
                    'me': 0.1811,
                    'mm': 9.1868,
                    'mr': 15.8167,
                    't0': 0,  # 0.4307
                    'tr': 0.2533}


class Hyperparams:
    def __init__(self):
        '''Hyperparameters'''
        self.create_name = 'simulations_data.p'
        self.supervised = False
        self.pretrained = False
        self.max_rep = 160
        # main
        self.training = network_training_hyper_parameters()
        self.network = network_building_hyper_parameters()
        self.simulations = simulation_hyper_parameters()
        self.acquisition = acquisition_parameters()
        self.aif = AIF_parameters()
        self.use_cuda = True
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.jobs = 4

        self.out_fold = 'results'
