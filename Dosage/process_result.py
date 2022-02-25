import os 
import re
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json

def get_result_dir(path):
    file = path + '/result.json'
    with open(file, encoding='utf-8') as f:
        line = f.readline()
        d = json.loads(line)
        in_mean, in_std = np.mean(d['in']), np.std(d['in'])
        out_mean, out_std =  np.mean(d['out']), np.std(d['out'])
    return in_mean, in_std, out_mean, out_std


path = 'saved_models'
x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
Names = ['SCIGAN', 'Tarnet', 'Drnet', 'Vcnet', 'TransTEE']
Algs = ['scigan', 'tarnet', 'drnet', 'vcnet', 'TransTEE']
style_dict = {
    '0':dict(linestyle='-', marker='o',markersize=0.1,color='#dd7e6b'),
    '1':dict(linestyle='-',marker='*',markersize=0.1,color='#b6d7a8'),
    '2':dict(linestyle='-',marker='s',markersize=0.1,color='#f9cb9c'),
    '3':dict(linestyle='-',marker='v',markersize=0.1,color='#a4c2f4'), 
    '4':dict(linestyle='-',marker='+',markersize=0.1,color='#b4a7d6')
}

style_dict_interval = {
    '0':dict(color='#dd7e6b'),
    '1':dict(color='#b6d7a8'),
    '2':dict(color='#f9cb9c'),
    '3':dict(color='#a4c2f4'), 
    '4':dict(color='#b4a7d6')
}
Bias_mean_in = np.array([[0. for j in range(len(Algs)) ] for i in range(len(x))])
Bias_mean_out = np.array([[0. for j in range(len(Algs)) ] for i in range(len(x))])
Bias_std_in = np.array([[0. for j in range(len(Algs)) ] for i in range(len(x))])
Bias_std_out = np.array([[0. for j in range(len(Algs)) ] for i in range(len(x))])

plt.style.use(['light','grid'])
font_y = {'family': 'serif',
        #'color':  'darkred',
        #'size': 20,
}
font_x = {'family': 'serif',
        #'size': 20,
}

for a in range(len(Algs)):
    for b in range(len(x)):
        #path_ab = "saved_models/treatments/{}/tr3/trb{}/dob2.0".format(Algs[a], x[b])
        path_ab = "saved_models/dosages/{}/tr3/trb2.0/dob{}".format(Algs[a], x[b])
        in_mean, in_std, out_mean, out_std = get_result_dir(path_ab)
        Bias_mean_in[b][a], Bias_std_in[b][a], Bias_mean_out[b][a], Bias_std_out[b][a] = in_mean, in_std, out_mean, out_std

def draw(means, stds, name):
    for a in range(len(Algs)):
        y_mean, y_std = means[:,a], stds[:,a]
        low_CI_bound, high_CI_bound = y_mean - y_std, y_mean + y_std

        plt.plot(x, y_mean, **style_dict[str(a)])
        plt.fill_between(x, low_CI_bound, high_CI_bound, alpha=0.3, **style_dict_interval[str(a)])
        
    plt.ylabel('AMSE', fontdict=font_y)
    plt.xlabel('Dosage Selection Bias', fontdict=font_y)
    #plt.xlabel('Treatment Selection Bias', fontdict=font_y)
    plt.legend((r'SCIGAN', 'Tarnet (D)', 'Drnet (D)', 'Vcnet (D)', 'TransTEE'), frameon=False, loc='best')
    pdf = PdfPages(name)
    #plt.ylim(0, 10)
    #plt.tight_layout()
    pdf.savefig()
    plt.close()
    pdf.close()

draw(Bias_mean_in, Bias_std_in, 'all_bias_treatment_in.pdf')
draw(Bias_mean_out, Bias_std_out, 'all_bias_treatment_out.pdf')