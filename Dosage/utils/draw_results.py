import os 
import re
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


style_dict = {
    '0':dict(linestyle='-', marker='o',markersize=0.1,color='#dd7e6b'),
    '1':dict(linestyle='-',marker='*',markersize=0.1,color='#b6d7a8'),
    '2':dict(linestyle='-',marker='s',markersize=0.1,color='#b4a7d6'),
    '3':dict(linestyle='-',marker='v',markersize=0.1,color='#a4c2f4'), 
    '4':dict(linestyle='-',marker='+',markersize=0.1,color='#f9cb9c')
}

style_dict_interval = {
    '0':dict(color='#dd7e6b'),
    '1':dict(color='#b6d7a8'),
    '2':dict(color='#b4a7d6'),
    '3':dict(color='#a4c2f4'), 
    '4':dict(color='#f9cb9c')
}
path = 'results/All_dcga_bias/'
font_y = {'family': 'serif',
        'color':  'Black',
        'size': 14,
}
font_x = {'family': 'serif',
        'size': 14,
}

x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Algs = ['Truth', 'Tarnet (D)', 'Drnet (D)', 'SINGAN', 'VCNet (D)', 'TransTEE']

Algs_means = np.array([[2.015, 5.491230010986328,6.204881095886231, 5.755753374099731, 5.6752], [2.1549, 3.0255686402320863, 3.8774894952774046, 3.365570831298828, 3.37], [0.115, 0.31368621699512006, 0.16449128091335297, 0.15431835912168027, 0.1621]])
Algs_stds = np.array([[1.07, 1.7616978608825207,0.5366308459534302,0.5147732802226582, 0.449025615873285], [1.04,  0.9776664583432995,0.485082947270075, 0.41199697667493157, 0.37 ], [0.102,.3352929610442719, 0.08206265835276705,0.1539, 0.1443]])

def draw( name):
    for a in range(len(Algs)):
        plt.plot(x, Algs_means[a] , **style_dict[str(a)])
        low_CI_bound, high_CI_bound = Algs_means[a] - Algs_stds[a], Algs_means[a] + Algs_stds[a]
        plt.fill_between(x, low_CI_bound, high_CI_bound, alpha=0.2, **style_dict_interval[str(a)])

    plt.ylabel('Response', fontdict=font_x)
    plt.grid(linestyle="--", alpha=0.2) # 网格线
    plt.legend(Algs, frameon=False, loc='best',prop=font_x)
    pdf = PdfPages(name)
    #plt.tight_layout()
    pdf.savefig()
    plt.close()
    pdf.close()

draw('idhp_treatments.pdf')