import numpy as np
import json
import pandas as pd
import torch
import os

from tqdm import tqdm
from plot_curve import plot_curve
import matplotlib.pyplot as plt

train_h = 1.9
test_h = 2.0
h_max = max(train_h, test_h)

path = 'dataset/ihdp/ihdp.csv'
save_path = 'dataset/ihdp/ihdp_v2/tr_h_{:02}_te_l_{:02}_h{:02}'.format(train_h, test_h-train_h, test_h)
ihdp = pd.read_csv(path)
ihdp = ihdp.to_numpy()
ihdp = ihdp[:, 2:27]  # delete the first column (data idx)/ delete the second coloum (treatment)
ihdp = torch.from_numpy(ihdp)
ihdp = ihdp.float()

n_feature = ihdp.shape[1]
n_data = ihdp.shape[0]

if not os.path.exists(save_path):
    os.makedirs(save_path)
# 0 1 2 4 5 -> continuous

# normalize the data

for _ in range(n_feature):
    minval = min(ihdp[:, _]) * 1.
    maxval = max(ihdp[:, _]) * 1.
    ihdp[:, _] = (1. * (ihdp[:, _] - minval))/maxval

# cate_idx = torch.tensor([3,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
cate_idx1 = torch.tensor([3,6,7,8,9,10,11,12,13,14])
cate_idx2 = torch.tensor([15,16,17,18,19,20,21,22,23,24])

alpha = 5.
cate_mean1 = torch.mean(ihdp[:, cate_idx1], dim=1).mean()
cate_mean2 = torch.mean(ihdp[:, cate_idx1], dim=1).mean()
tem = torch.tanh((torch.sum(ihdp[:, cate_idx2], dim=1)/10. - cate_mean2) * alpha)

def x_t(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[4]
    x5 = x[5]
    t = x1/(1. + x2) + max(x3, x4, x5)/(0.2 + min(x3, x4, x5)) + torch.tanh((torch.sum(x[cate_idx2])/10. - cate_mean2) * alpha) - 2.

    return t

def x_t_link(t):
    return 1. / (1. + torch.exp(-2. * t))


def t_x_y(t, x, h):
    # only x1, x3, x4 are useful
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[4]
    x5 = x[5]

    # v1
    factor1 = 0.5
    factor2 = 1.5

    # v2
    factor1 = 1.5
    factor2 = 0.5

    # original
    factor1 = 1.
    factor2 = 1.

    #y = 1./(1.2 - t) * torch.sin(t * 2. * 3.14159) * (factor1 * torch.tanh((torch.sum(x[cate_idx1])/10. - cate_mean1) * alpha) +
    #                                                  factor2 * 0.2 * torch.exp(0.2 * (x1 - x5))/(0.1 + min(x2, x3, x4)))
    y = 1. / (1.2 - t/h_max) * torch.sin((t/h_max) * 3. * 3.14159) * (
                factor1 * torch.tanh((torch.sum(x[cate_idx1]) / 10. - cate_mean1) * alpha) +
                factor2 * torch.exp(0.2 * (x1 - x5)) / (0.1 + min(x2, x3, x4)))

    return y


def ihdp_matrix(h=1.):
    train_matrix = torch.zeros(n_data, n_feature+2)
    for _ in range(n_data):
        x = ihdp[_, :]
        t = x_t(x)
        t += torch.randn(1)[0] * 0.5
        t = x_t_link(t) * train_h + (test_h - train_h)
        y = t_x_y(t, x, h)
        y += torch.randn(1)[0] * 0.5

        train_matrix[_, 0] = t
        train_matrix[_, n_feature+1] = y
        train_matrix[_, 1: n_feature+1] = x

    train_grid = torch.zeros(2, n_data)
    train_grid[0, :] = train_matrix[:, 0].squeeze()
    train_grid[1, :] = train_matrix[:, -1].squeeze()

    data_matrix = torch.zeros(n_data, n_feature+2)
    # get data matrix
    for _ in range(n_data):
        x = ihdp[_, :]
        t = x_t(x)
        t += torch.randn(1)[0] * 0.5
        t = x_t_link(t) * test_h
        y = t_x_y(t, x, h)
        y += torch.randn(1)[0] * 0.5

        data_matrix[_, 0] = t
        data_matrix[_, n_feature+1] = y
        data_matrix[_, 1: n_feature+1] = x

    # get t_grid
    t_grid = torch.zeros(2, n_data)
    t_grid[0, :] = data_matrix[:, 0].squeeze()

    for i in tqdm(range(n_data)):
        psi = 0
        t = t_grid[0, i]
        for j in range(n_data):
            x = data_matrix[j, 1: n_feature+1]
            psi += t_x_y(t, x, h)
        psi /= n_data
        t_grid[1, i] = psi
    
    plt.figure(figsize=(5, 5))
    truth_grid = t_grid[:,t_grid[0,:].argsort()]
    t_truth_grid = train_grid[:,train_grid[0,:].argsort()]
    x = truth_grid[0, :]
    y = truth_grid[1, :]
    y2 = train_matrix[:, 0].squeeze()
    print(torch.max(x), torch.max(t_truth_grid[0, :]))
    print(x, t_truth_grid[0, :])
    plt.plot(x, y, marker='', ls='-', label='Test', linewidth=4, color='gold')
    plt.plot(t_truth_grid[0, :], t_truth_grid[1, :], marker='', ls='-', label='Train',alpha=0.5, linewidth=1, color='grey')
    plt.legend()
    plt.xlabel('Treatment')
    plt.ylabel('Response')

    plt.savefig( "ihdp.pdf", bbox_inches='tight')

    return train_matrix, train_grid, data_matrix, t_grid


dt, trg, dm, tg = ihdp_matrix()

torch.save(dt, save_path+'/train_matrix.pt')
torch.save(dm, save_path+'/data_matrix.pt')
torch.save(tg, save_path+'/t_grid.pt')

# generate splitting
for _ in range(100):
    print('generating eval set: ', _)
    data_path = os.path.join(save_path, 'eval', str(_))
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    idx_list = torch.randperm(n_data)
    idx_train = idx_list[0:471]
    idx_test = idx_list[471:]

    torch.save(idx_train, data_path + '/idx_train.pt')
    torch.save(idx_test, data_path + '/idx_test.pt')

    np.savetxt(data_path + '/idx_train.txt', idx_train.numpy())
    np.savetxt(data_path + '/idx_test.txt', idx_test.numpy())