import os
import numpy as np

from data.simu1 import simu_data1
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate simulate data')
    parser.add_argument('--save_dir', type=str, default='dataset/simu/ood4to5', help='dir to save generated data')
    parser.add_argument('--num_eval', type=int, default=100, help='num of dataset for evaluating the methods')
    parser.add_argument('--num_tune', type=int, default=0, help='num of dataset for tuning the parameters')
    parser.add_argument('--train_h', type=float, default=4.0, help='training h')
    parser.add_argument('--test_h', type=float, default=5.0, help='test h')

    args = parser.parse_args()
    save_path = args.save_dir

    for _ in range(args.num_eval):
        print('generating eval set: ', _)
        data_path = os.path.join(save_path, 'eval', str(_))
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        train_matrix, test_matrix, t_grid = simu_data1(5000, 200, args.train_h, args.test_h)

        data_file = os.path.join(data_path, 'train.txt')
        np.savetxt(data_file, train_matrix.numpy())
        data_file = os.path.join(data_path, 'test.txt')
        np.savetxt(data_file, test_matrix.numpy())
        data_file = os.path.join(data_path, 't_grid.txt')
        np.savetxt(data_file, t_grid.numpy())
