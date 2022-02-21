import os 
import re
import numpy as np
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def get_result_dir(path):
    g = os.walk(path) 
    files = 10
    results_u = [[0 for i in range(18)] for j in range(files)]
    results_w = [[0 for i in range(18)] for j in range(files)]
    num = 0
    for path,dir_list,file_list in g:  
        for file_name in file_list:  
            patt1 = "'unweighted': (\d+\.\d+)"
            patt2 = ", 'weighted': (\d+\.\d+)"
            f = open(os.path.join(path, file_name)).read()
            res1 = re.findall(patt1, f)
            res2 = re.findall(patt2, f)
            for i in range(18):
                results_u[num][i] = float(res1[i])
                results_w[num][i] = float(res2[i])
            num += 1
    in_mean, in_std = [0 for i in range(9)], [0 for i in range(9)]
    out_mean, out_std = [0 for i in range(9)], [0 for i in range(9)]
    for i in range(9):
        x_in = [results_u[j][i*2] for j in range(num)]
        x_out = [results_u[j][i*2+1] for j in range(num)]
        in_mean[i], in_std[i] = np.mean(x_in), np.std(x_in)
        out_mean[i], out_std[i] = np.mean(x_out), np.std(x_out)
        # print("{:.2f} ± {:.2f}".format(np.mean(x_in), np.std(x_in)))
        # print("{:.2f} ± {:.2f}".format(np.mean(x_out), np.std(x_out)))
    return in_mean, in_std, out_mean, out_std


path = 'results/All_sw_bias/'
Bias = [0, 5, 10, 15, 20, 25, 30, 35, 40] #SW
#Bias = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] # TCGA
Algs = ['gin', 'gnn', 'graphite', 'zero', 'transtee']
Names = ['SIN', 'GNN', 'GraphITE', 'Zero', 'TransTEE']
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
Bias_mean_in = np.array([[[0. for k in range(9)] for j in range(5) ] for i in range(len(Bias))])
Bias_mean_out = np.array([[[0. for k in range(9)] for j in range(5) ] for i in range(len(Bias))])
Bias_std_in = np.array([[[0. for k in range(9)] for j in range(5) ] for i in range(len(Bias))])
Bias_std_out = np.array([[[0. for k in range(9)] for j in range(5) ] for i in range(len(Bias))])

plt.style.use(['light','grid'])
font_y = {'family': 'serif',
        'color':  'darkred',
        #'size': 20,
}
font_x = {'family': 'serif',
        #'size': 20,
}

for b in range(len(Bias)):
    path_b = path + '/bias' + str(Bias[b])
    for a in range(len(Algs)):
        path_ba = path_b + '/' + Algs[a]
        in_mean, in_std, out_mean, out_std = get_result_dir(path_ba)
        Bias_mean_in[b][a], Bias_std_in[b][a], Bias_mean_out[b][a], Bias_std_out[b][a] = in_mean, in_std, out_mean, out_std

def draw(means, stds, name):
    fig, axs = plt.subplots(3, 3, sharex=True, sharey=True)
    for i in range(9):
        for a in range(len(Algs)):
            y_mean, y_std = means[:,a,i], stds[:,a,i]
            low_CI_bound, high_CI_bound = y_mean - y_std, y_mean + y_std

            axs[i//3, i%3].plot(Bias, y_mean , **style_dict[str(a)])
            axs[i//3, i%3].fill_between(Bias, low_CI_bound, high_CI_bound, alpha=0.4, **style_dict_interval[str(a)])
            axs[i//3, i%3].set_ylabel('UPEHE@'+str(i+2), fontdict=font_y)

    fig.legend(('SIN', 'GNN', 'GraphITE', 'Zero', 'TransTEE'), frameon=False, loc='upper center',ncol=5,handlelength=3)
    pdf = PdfPages(name)
    #plt.tight_layout()
    pdf.savefig()
    plt.close()
    pdf.close()

# draw(Bias_mean_in, Bias_std_in, 'all_bias_dcga_in_unweight.pdf')
# draw(Bias_mean_out, Bias_std_out, 'all_bias_dcga_out_unweight.pdf')
draw(Bias_mean_in, Bias_std_in, 'all_bias_sw_in_unweight.pdf')
draw(Bias_mean_out, Bias_std_out, 'all_bias_sw_out_unweight.pdf')