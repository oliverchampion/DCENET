import hyperparams
import argparse
import numpy as np
import simulations as sim
import os
import torch

# np.random.seed(42)
# torch.manual_seed(42)

parser = argparse.ArgumentParser()

#parameters for training
parser.add_argument('--nn', type=str, default='linear')
parser.add_argument('--layers', type=int, nargs='+', default=[160, 160, 160])
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--attention', action='store_true', default=False)
parser.add_argument('--bidirectional', action='store_true', default=False)
parser.add_argument('--supervised', action='store_true', default=False)
parser.add_argument('--cpu', action='store_true', default=False)
parser.add_argument('--exp', type=int, default=0)
#parameters for evaluation
parser.add_argument('--var_seq', action='store_true', default=False)
parser.add_argument('--var_hct', action='store_true', default=False)
parser.add_argument('--results', action='store_true', default=False)

args = parser.parse_args()

hp = hyperparams.Hyperparams()

hp.training.lr = args.lr
hp.training.batch_size = args.batch_size
hp.network.nn = args.nn
hp.network.layers = args.layers
hp.network.attention = args.attention
hp.network.bidirectional = args.bidirectional
hp.supervised = args.supervised
if args.cpu:
    hp.device = torch.device('cpu')

# create save name for framework
hp.exp_name = ''
arg_dict = vars(args)
for i, arg in enumerate(arg_dict):
    if i == len(arg_dict)-6:
        hp.exp_name += str(arg_dict[arg])
        break
    else:
        hp.exp_name += '{}_'.format(arg_dict[arg])

if args.exp!=0:
    hp.exp_name += '_'+str(args.exp)

print(hp.exp_name)

if os.path.exists(hp.create_name):
    hp.create_data = False
else:
    hp.create_data = True

# execute the non-linear least squares method
if hp.network.nn == 'lsq':
    hp.simulations.num_samples_leval = 10000
    SNRs = [5, 7, 10, 15, 20, 25, 40, 100]
    datapoints = [80, 90, 100, 110, 120, 130, 140, 150, 160]
    Hcts = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

    if args.var_seq:
        params = np.zeros((len(datapoints), 4, 2))
        for i, datapoint in enumerate(datapoints):
            hp.acquisition.rep2 = datapoint+1

            file = hp.create_name.replace('.p', '')+'_seq_'+str(hp.acquisition.rep2-1)+'.p'
            if os.path.exists(file):
                hp.create_data = False
            else:
                hp.create_data = True

            params[i] = sim.run_simulations(hp, args=args, SNR=20, eval=True)
        
        np.save('results'+hp.exp_name+'_seq.npy', params)
    
    elif args.var_hct:
        params = np.zeros((len(Hcts), 4, 2))
        for i, Hct in enumerate(Hcts):
            file = hp.create_name.replace('.p', '')+'_hct_'+str(Hct)+'.p'
            if os.path.exists(file):
                hp.create_data = False
            else:
                hp.create_data = True

            hp.aif.Hct = Hct

            params[i] = sim.run_simulations(hp, args=args, SNR=20, eval=True)

        np.save('results'+hp.exp_name+'_hct.npy', params)
    
    else:
        params = np.zeros((len(SNRs), 4, 2))
        for i, SNR in enumerate(SNRs):
            file = hp.create_name.replace('.p', '')+'_'+str(SNR)+'.p'
            if os.path.exists(file):
                hp.create_data = False
            else:
                hp.create_data = True

            params[i] = sim.run_simulations(hp, args=args, SNR=20, eval=True)

        np.save('results'+hp.exp_name+'.npy', params)

# train a neural network based approach
elif not args.results:
    sim.run_simulations(hp, args, SNR='all')

# passing --results will perform evaluation on the given framework
else:
    hp.simulations.num_samples = 10000
    SNRs = [5, 7, 10, 15, 20, 25, 40, 100]
    datapoints = [80, 90, 100, 110, 120, 130, 140, 150, 160]
    Hcts = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

    if args.var_seq:
        params = np.zeros((len(datapoints), 4, 2))
        for i, datapoint in enumerate(datapoints):
            hp.acquisition.rep2 = datapoint+1

            file = hp.create_name.replace('.p', '')+'_seq_'+str(hp.acquisition.rep2-1)+'.p'
            if os.path.exists(file):
                hp.create_data = False
            else:
                hp.create_data = True

            params[i] = sim.run_simulations(hp, args=args, SNR=20, eval=True)

        np.save('results'+hp.exp_name+'_var_seq.npy', params)
    
    elif args.var_hct:
        params = np.zeros((len(Hcts), 4, 2))
        for i, Hct in enumerate(Hcts):
            file = hp.create_name.replace('.p', '')+'_hct_'+str(Hct)+'.p'
            if os.path.exists(file):
                hp.create_data = False
            else:
                hp.create_data = True

            hp.aif.Hct = Hct

            params[i] = sim.run_simulations(hp, args=args, SNR=20, eval=True)

        np.save('results'+hp.exp_name+'_var_hct.npy', params)

    else:
        params = np.zeros((len(SNRs), 4, 2))
        for i, SNR in enumerate(SNRs):
            file = hp.create_name.replace('.p', '')+'_'+str(SNR)+'.p'
            if os.path.exists(file):
                hp.create_data = False
            else:
                hp.create_data = True

            params[i] = sim.run_simulations(hp, args=args, SNR=SNR, eval=True)

        np.save('results'+hp.exp_name+'.npy', params)