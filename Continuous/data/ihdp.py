import os
import numpy as np
import torch

import argparse
import numpy as np
from sklearn.model_selection import train_test_split


class IHDP(object):
    def __init__(self, path_data="./dataset/ihdp/continuous", replications=10):
        self.path_data = path_data
        self.replications = replications
        # which features are binary
        self.binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        # which features are continuous
        self.contfeats = [i for i in range(25) if i not in self.binfeats]

    def __iter__(self):
        for i in range(self.replications):
            data = np.loadtxt(self.path_data + '/ihdp_npci_' + str(4 + 1) + '.csv', delimiter=',')
            t, y, y_cf = data[:, 0], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
            mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
            yield (x, t, y), (y_cf, mu_0, mu_1)

    def get_train_valid_test(self):
        for i in range(self.replications): 
            data = np.loadtxt(self.path_data + '/ihdp_npci_' + str(4 + 1) + '.csv', delimiter=',')
            data = torch.from_numpy(data).float()
            t, y, y_cf = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
            mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]

            # this binary feature is in {1, 2}
            #x[:, 13] -= 1
            itr, ite = train_test_split(np.arange(x.shape[0]), test_size=0.33, random_state=1)
            #itr, iva = train_test_split(idxtrain, test_size=0.33, random_state=1)
            train = (x[itr], t[itr], y[itr]), (y_cf[itr], mu_0[itr], mu_1[itr])
            #valid = (x[iva], t[iva], y[iva]), (y_cf[iva], mu_0[iva], mu_1[iva])
            test = (x[ite], t[ite], y[ite]), (y_cf[ite], mu_0[ite], mu_1[ite])
            yield train, test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate simulate data')
    parser.add_argument('--save_dir', type=str, default='./dataset/ihdp/revised', help='dir to save generated data')
    parser.add_argument('--train_h', type=float, default=4.0, help='training h')
    parser.add_argument('--test_h', type=float, default=5.0, help='test h')

    args = parser.parse_args()
    save_path = args.save_dir

    for _ in range(args.num_eval):
        print('generating eval set: ', _)
        data_path = os.path.join(save_path, 'eval', str(_))
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        train_matrix, test_matrix, t_grid = simu_data1(500, 200, args.train_h, args.test_h)

        data_file = os.path.join(data_path, 'train.txt')
        np.savetxt(data_file, train_matrix.numpy())
        data_file = os.path.join(data_path, 'test.txt')
        np.savetxt(data_file, test_matrix.numpy())
        data_file = os.path.join(data_path, 't_grid.txt')
        np.savetxt(data_file, t_grid.numpy())