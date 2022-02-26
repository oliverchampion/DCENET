import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
import simulations as sim
import pickle
matplotlib.use('TkAgg')

rootdir = 'params/'
SNRs = [5, 7, 10, 15, 20, 25, 40, 100]
datapoints = [80, 90, 100, 110, 120, 130, 140, 150, 160]
Hcts = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
ylim = np.array([1.9, 0.69, 0.049])
ylim_std = ylim*0.25
ylim_abs = ylim*0.05
sns.set()


def make_plot(method_scores, save=False):
    params = ['$k_{ep}$', '$v_e$', '$v_p$', 'dt']

    for i, param in enumerate(params):
        all_values_mean = []
        all_values_std = []
        for k, (scores, label) in enumerate(method_scores):
            values_mean = []
            values_std = []
            for j in range(len(SNRs)):
                values_std.append(scores[j][i][0])
                values_mean.append(scores[j][i][1])

            all_values_mean.append((values_mean, label))
            all_values_std.append((values_std, label))

        for (values, label) in all_values_mean:
            if label=='NLLS':
                plt.plot(values, label=label, color='r')
            else:
                plt.plot(values, label=label)

        size_ticks = 14
        size_labels = 15

        plt.xticks(np.arange(len(SNRs)), SNRs, fontsize=size_ticks)
        plt.yticks(fontsize=size_ticks)
        plt.xlabel('Noise (SD)', fontsize=size_labels)
        plt.ylabel('Systematic Error', fontsize=size_labels)
        plt.title(param, fontsize=20)
        plt.legend(fontsize=13)
        plt.tight_layout()

        if save:
            plt.savefig('Figure_{}_SNRs.png'.format(i*2+1))
        else:
            plt.show()

        plt.close()

        for (values, label) in all_values_std:
            if label=='NLLS':
                plt.plot(values, label=label, color='r')
            else:
                plt.plot(values, label=label)

        plt.xticks(np.arange(len(SNRs)), SNRs, fontsize=size_ticks)
        plt.yticks(fontsize=size_ticks)
        plt.xlabel('Noise (SD)', fontsize=size_labels)
        plt.ylabel('Random Error', fontsize=size_labels)
        plt.title(param, fontsize=20)
        plt.legend(fontsize=13)
        plt.tight_layout()

        if save:
            plt.savefig('Figure_{}_SNRs.png'.format(i*2+2))
        else:
            plt.show()

        plt.close()

method_scores = []
for i, method in enumerate(sorted(os.listdir(rootdir))):
    scores = np.load(rootdir+method)
    if 'gru' in method:
        method = 'GRU'
    elif 'lstm' in method:
        method = 'LSTM'
    elif 'linear' in method:
        method = 'FCN'
    else:
        method = 'NLLS'

    method_scores.append((scores, method))
    

make_plot(method_scores, save=True)
