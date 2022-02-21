from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import random
import os
import sys
import copy
import pandas as pd

def plot_tnse_ihdp(all_feat, treatment, perplexity=50, iters=1000, model_name=''):
    colors = ['#77AADD', '#EE8866', '#EEDD88', 'grey']
    marker = ['x', 'D', '*', 's', '+', 'v', 'p']
    list_perplexity = [10]#[5, 10, 20]
    list_iter = [2000]#[250, 500] 
    list_type_name = [0, 1]
    output_dir = 'tsne/ihdp'
    all_feat = all_feat.cpu().detach().numpy()
    treatment = treatment.squeeze(0).cpu().detach().numpy()
    for perplexity in list_perplexity:
        for iter in list_iter:
            plt.rc('font',family='Times New Roman')
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(1, 1, 1)
            for i in range(2):
                new_idx = (treatment == i).nonzero()[0]
                new_feat = all_feat[new_idx]
                tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=iter)
                tsne_ref = tsne.fit_transform(new_feat)
                ax.scatter(tsne_ref[:, 0],
                    tsne_ref[:, 1],
                    marker=marker[i], color=colors[i],
                    label='t = '+str(i), s=30)
            ax.legend(loc='best', fontsize=20)
            save_name = model_name + 'p' + str(perplexity) + '_i' + str(iter) + '.pdf'
            plt.yticks(size = 24)
            plt.xticks(size = 24)
            pdf = PdfPages(output_dir+save_name)
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            pdf.close()