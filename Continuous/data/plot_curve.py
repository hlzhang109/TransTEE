import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_curve(train_matrix, test_matrix, t_grid, train_grid):
    plt.figure(figsize=(5, 5))
    truth_grid = t_grid[:,t_grid[0,:].argsort()]
    t_truth_grid = train_grid[:,train_grid[0,:].argsort()]
    x = truth_grid[0, :]
    y = truth_grid[1, :]
    y2 = train_matrix[:, 0].squeeze()
    print(torch.max(x), torch.max(t_truth_grid[0, :]))
    plt.plot(x, y, marker='', ls='-', label='Test', linewidth=4, color='gold')
    plt.plot(t_truth_grid[0, :], t_truth_grid[1, :], marker='', ls='-', label='Train',alpha=0.5, linewidth=1, color='grey')
    #plt.yticks(np.arange(-2.0, 2.1, 0.5), fontsize=4, family='Times New Roman')
    #plt.xticks(np.arange(0, 2.1, 0.2), fontsize=4, family='Times New Roman')
    plt.legend()
    plt.xlabel('Treatment')
    plt.ylabel('Response')

    plt.savefig( "news.pdf", bbox_inches='tight')