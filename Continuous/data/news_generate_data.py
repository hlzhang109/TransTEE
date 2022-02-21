import numpy as np
import json
import pandas as pd
import torch
import os

from tqdm import tqdm
from plot_curve import plot_curve

train_h = 4.5
test_h = 5.0
h_max = max(train_h, test_h)

path = 'dataset/news/news_pp.npy'
save_path = 'dataset/news/tr_h_{:02}_te_h_{:02}'.format(train_h, test_h)
# load data
news = np.load(path)
if not os.path.exists(save_path):
    os.makedirs(save_path)
# # normalize data
for _ in range(news.shape[1]):
    max_freq = max(news[:,_])
    news[:,_] = news[:,_] / max_freq

num_data = news.shape[0]
num_feature = news.shape[1]

np.random.seed(5)
v1 = np.random.randn(num_feature)
v1 = v1/np.sqrt(np.sum(v1**2))
v2 = np.random.randn(num_feature)
v2 = v2/np.sqrt(np.sum(v2**2))
v3 = np.random.randn(num_feature)
v3 = v3/np.sqrt(np.sum(v3**2))


def x_t(x):
    alpha = 2
    tt = np.sum(v3 * x) / (2. * np.sum(v2 * x))
    beta = (alpha - 1)/tt + 2 - alpha
    beta = np.abs(beta) + 0.0001
    t = np.random.beta(alpha, beta, 1)
    return t

def t_x_y(t, x):
    # return 10. * (np.sum(v1 * x) + 5. * np.sin(3.14159 * np.sum(v2 * x) / np.sum(v3 * x) * t))
    res1 = max(-2, min(2, np.exp(0.3 * (np.sum(3.14159 * np.sum(v2 * x) / np.sum(v3 * x)) - 1))))
    res2 = 20. * (np.sum(v1 * x))
    # print("-"*89)
    # print(res1)
    # print(res2)
    # res = 2 * (2 * np.sin(2 * 3.14159 * t + 4. * res1)) * (res2)
    res = 2 * (4 * (t/h_max - 0.5)**2 * np.sin(0.5 * 3.14159 * t/h_max)) * (res1 + res2)
    return res
    # return 10. * (np.sum(v1 * x) + 5. * np.sin(3.14159 * np.sum(v2 * x) / np.sum(v3 * x) * t))


def news_matrix():
    train_matrix = torch.zeros(num_data, num_feature+2)

    # get data matrix
    for _ in range(num_data):
        x = news[_, :]
        t = x_t(x) * train_h
        y = torch.from_numpy(t_x_y(t, x))
        x = torch.from_numpy(x)
        t = torch.from_numpy(t)
        y += torch.randn(1)[0] * np.sqrt(0.5)

        train_matrix[_, 0] = t
        train_matrix[_, num_feature+1] = y
        train_matrix[_, 1: num_feature+1] = x

    train_grid = torch.zeros(2, num_data)
    train_grid[0, :] = train_matrix[:, 0].squeeze()
    train_grid[1, :] = train_matrix[:, -1].squeeze()

    data_matrix = torch.zeros(num_data, num_feature+2)

    # get data matrix
    for _ in range(num_data):
        x = news[_, :]
        t = x_t(x) * test_h
        y = torch.from_numpy(t_x_y(t, x))
        x = torch.from_numpy(x)
        t = torch.from_numpy(t)
        y += torch.randn(1)[0] * np.sqrt(0.5)

        data_matrix[_, 0] = t
        data_matrix[_, num_feature+1] = y
        data_matrix[_, 1: num_feature+1] = x

    # get t_grid
    t_grid = torch.zeros(2, num_data)
    t_grid[0, :] = data_matrix[:, 0].squeeze()

    for i in tqdm(range(num_data)):
        psi = 0
        t = t_grid[0, i].numpy()
        for j in range(num_data):
            x = data_matrix[j, 1: num_feature+1].numpy()
            psi += t_x_y(t, x)
        psi /= num_data
        t_grid[1, i] = psi

    return train_matrix, train_grid, data_matrix, t_grid

dt, trg, dm, tg = news_matrix()

plot_curve(dt, dm, tg, trg)

torch.save(dt, save_path+'/train_matrix.pt')
torch.save(dm, save_path+'/data_matrix.pt')
torch.save(tg, save_path+'/t_grid.pt')

# generate splitting
for _ in range(20):
    print('generating eval set: ', _)
    data_path = os.path.join(save_path, 'eval', str(_))
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    idx_list = torch.randperm(num_data)
    idx_train = idx_list[0:2000]
    idx_test = idx_list[2000:]

    torch.save(idx_train, data_path + '/idx_train.pt')
    torch.save(idx_test, data_path + '/idx_test.pt')

    np.savetxt(data_path + '/idx_train.txt', idx_train.numpy())
    np.savetxt(data_path + '/idx_test.txt', idx_test.numpy())