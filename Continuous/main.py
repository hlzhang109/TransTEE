import torch
import math
import numpy as np
import os
import pandas as pd
from torch.nn.functional import dropout
from torch.autograd import Variable

from models.dynamic_net import Vcnet, Drnet, TR, TransPlus
from models.itermidate_models import VcnetAtt
from data.data import get_iter
from utils.eval import curve

import argparse

def adjust_learning_rate(optimizer, init_lr, epoch):
    if lr_type == 'cos':  # cos without warm-up
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * epoch / num_epoch))
    elif lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = init_lr * (decay ** (epoch // step))
    elif lr_type == 'fixed':
        lr = init_lr
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def save_checkpoint(state, checkpoint_dir='.'):
    filename = os.path.join(checkpoint_dir, model_name + '_ckpt.pth.tar')
    print('=> Saving checkpoint to {}'.format(filename))
    torch.save(state, filename)

# criterion
def criterion(out, y, alpha=0.5, epsilon=1e-6):
    return ((out[1].squeeze() - y.squeeze())**2).mean() #- alpha * torch.log(out[0] + epsilon).mean()

def criterion_TR(out, trg, y, beta=1., epsilon=1e-6):
    # out[1] is Q
    # out[0] is g
    return beta * ((y.squeeze() - trg.squeeze()/(out[0].squeeze() + epsilon) - out[1].squeeze())**2).mean()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train with simulate data')

    # i/o
    parser.add_argument('--data_dir', type=str, default='./dataset/ihdp/continuous/0', help='dir of eval dataset')
    parser.add_argument('--save_dir', type=str, default='dataset/ihdp/continuous/', help='dir to save result')

    # training
    parser.add_argument('--n_epochs', type=int, default=100, help='num of epochs to train')
    parser.add_argument('--h', type=int, default=1.0, help='the maximal value of treatmemts')

    # print train info
    parser.add_argument('--verbose', type=int, default=50, help='print train info freq')

    parser.add_argument('--att_layers', type=int, default=1, help='whther use attention')
    parser.add_argument('--class_head', type=str, default='mean', help='choose from max, mean, token, sum, last, first')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--lr', type=float, default=0.0001, help='dropout rate')
    parser.add_argument('--hidden_dim', type=int, default=10, help='dimension for hidden state')
    parser.add_argument('--num_heads', type=int, default=2, help='dimension for hidden state')

    # plot adrf
    parser.add_argument('--plt_adrf', type=bool, default=True, help='whether to plot adrf curves. (only run two methods if set true; '
                                                                    'the label of fig is only for drnet and vcnet in a certain order)')

    args = parser.parse_args()

    print(args)

    # optimizer
    lr_type = 'fixed'
    wd = 5e-3
    momentum = 0.9
    # targeted regularization optimizer
    tr_wd = 5e-3

    num_epoch = args.n_epochs

    # check val loss
    verbose = args.verbose

    load_path = args.data_dir
    save_path = args.save_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data = pd.read_csv(load_path + '/train.txt', header=None, sep=' ')
    train_matrix = torch.from_numpy(data.to_numpy()).float()
    data = pd.read_csv(load_path + '/test.txt', header=None, sep=' ')
    test_matrix = torch.from_numpy(data.to_numpy()).float()
    data = pd.read_csv(load_path + '/t_grid.txt', header=None, sep=' ')
    t_grid = torch.from_numpy(data.to_numpy()).float()

    train_loader = get_iter(train_matrix, batch_size=500, shuffle=True)
    test_loader = get_iter(test_matrix, batch_size=test_matrix.shape[0], shuffle=False)

    grid = []
    MSE = []

    # choose from {'Tarnet', 'Tarnet_tr','Drnet', 'Drnet_tr', 'Vcnet', 'Vcnet_tr', 'TransPlus', 'TransPlus_tr'}
    method_list = ['VcnetAtt', 'Tarnet', 'Drnet', 'Vcnet']

    for model_name in method_list:
        # import model
        if model_name == 'Vcnet' or model_name == 'Vcnet_tr':
            cfg_density = [(25, 50, 1, 'relu'), (50, 50, 1, 'relu')]
            num_grid = 10
            cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
            degree = 2
            knots = [0.33, 0.66]
            model = Vcnet(cfg_density, num_grid, cfg, degree, knots)
            model._initialize_weights()

        elif model_name == 'VcnetAtt' or model_name == 'VcnetAtt_tr':
            cfg_density = [(25, 50, 1, 'relu'), (50, 50, 1, 'relu')]
            num_grid = 10
            cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
            degree = 2
            knots = [0.33, 0.66]
            model = VcnetAtt(cfg_density, num_grid, cfg, degree, knots, att_layers=args.att_layers)
            model._initialize_weights()

        elif model_name == 'Drnet' or model_name == 'Drnet_tr':
            cfg_density = [(25, 50, 1, 'relu'), (50, 50, 1, 'relu')]
            num_grid = 10
            cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
            isenhance = 1
            model = Drnet(cfg_density, num_grid, cfg, isenhance=isenhance, att_layers=args.att_layers, h=args.h)
            model._initialize_weights()

        elif model_name == 'Tarnet' or model_name == 'Tarnet_tr':
            cfg_density = [(25, 50, 1, 'relu'), (50, 50, 1, 'relu')]
            num_grid = 10
            cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
            isenhance = 0
            model = Drnet(cfg_density, num_grid, cfg, isenhance=isenhance, att_layers=args.att_layers, h=args.h)
            model._initialize_weights()

        elif model_name == 'TransPlus' or model_name == 'TransPlus_tr':
            cfg_density = [(25, 50, 1, 'relu'), (50, 50, 1, 'relu')]
            num_grid = 10
            cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
            isenhance = 0
            dropout = args.dropout
            nhead=args.num_heads
            model = TransPlus(cfg_density, num_grid, cfg, isenhance=isenhance, att_layers=args.att_layers, dropout=dropout, nhead=nhead, class_head=args.class_head)
            model._initialize_weights()

        model.cuda()
        # use Target Regularization?
        if 'tr' in model_name:
            isTargetReg = 1
        else:
            isTargetReg = 0

        if isTargetReg:
            tr_knots = list(np.arange(0.1, 1, 0.1))
            tr_degree = 2
            TargetReg = TR(tr_degree, tr_knots)
            TargetReg._initialize_weights()

        # best cfg for each model
        if model_name == 'Tarnet':
            init_lr = 0.05
            alpha = 1.0
        elif model_name == 'Tarnet_tr':
            init_lr = 0.05
            alpha = 0.5
            tr_init_lr = 0.001
            beta = 1.
        elif model_name == 'Drnet':
            init_lr = 0.05
            alpha = 1.
        elif model_name == 'Drnet_tr':
            init_lr = 0.05
            # init_lr = 0.05 tuned
            alpha = 0.5
            tr_init_lr = 0.001
            beta = 1.
        elif model_name == 'Vcnet':
            init_lr = 0.0005
            #init_lr = 0.01
            alpha = 0.5
        elif model_name == 'Vcnet_tr':
            init_lr = 0.0001
            alpha = 0.5
            tr_init_lr = 0.001
            beta = 1.
        elif model_name == 'VcnetAtt':
            init_lr = 0.0005
            #init_lr = 0.01
            alpha = 0.5
        elif model_name == 'VcnetAtt_tr':
            init_lr = 0.0001
            alpha = 0.5
            tr_init_lr = 0.001
            beta = 1.
        elif model_name == 'TransPlus':
            init_lr = args.lr
            alpha = 0.5
        elif model_name == 'TransPlus_tr':
            init_lr = args.lr
            alpha = 0.5
            tr_init_lr = 0.001
            beta = 1.
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=momentum, weight_decay=wd, nesterov=True)

        if isTargetReg:
            tr_optimizer = torch.optim.SGD(TargetReg.parameters(), lr=tr_init_lr, weight_decay=tr_wd)

        print('model = ', model_name)
        for epoch in range(num_epoch):

            for idx, (inputs, y) in enumerate(train_loader):
                inputs = Variable(inputs.cuda().detach())
                y = Variable(y.cuda().detach())
                t = inputs[:, 0]
                x = inputs[:, 1:]

                if isTargetReg:
                    optimizer.zero_grad()
                    out = model.forward(t, x)
                    trg = TargetReg(t)
                    loss = criterion(out, y, alpha=alpha) + criterion_TR(out, trg, y, beta=beta)
                    loss.backward()
                    optimizer.step()

                    tr_optimizer.zero_grad()
                    out = model.forward(t, x)
                    trg = TargetReg(t)
                    tr_loss = criterion_TR(out, trg, y, beta=beta)
                    tr_loss.backward()
                    tr_optimizer.step()
                else:
                    optimizer.zero_grad()
                    out = model.forward(t, x)
                    loss = criterion(out, y, alpha=alpha)
                    loss.backward()
                    optimizer.step()

            if epoch % verbose == 0:
                print('current epoch: ', epoch)
                print('loss: ', loss.data)
                t_grid_hat, mse = curve(model, test_matrix, t_grid)
                print('current test loss: ', mse)

        if isTargetReg:
            t_grid_hat, mse = curve(model, test_matrix, t_grid, targetreg=TargetReg)
        else:
            t_grid_hat, mse = curve(model, test_matrix, t_grid)

        mse = float(mse)
        print('current loss: ', float(loss.data))
        print('current test loss: ', mse)
        print('-----------------------------------------------------------------')

        save_checkpoint({
            'model': model_name,
            'best_test_loss': mse,
            'model_state_dict': model.state_dict(),
            'TR_state_dict': TargetReg.state_dict() if isTargetReg else None,
        }, checkpoint_dir=save_path)

        print('-----------------------------------------------------------------')

        grid.append(t_grid_hat)
        MSE.append(mse)


    if args.plt_adrf:
        import matplotlib.pyplot as plt

        font1 = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 14,
        }

        font_legend = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 14,
        }
        plt.figure(figsize=(5, 5))

        c1 = 'gold'
        c2 = 'grey'
        c3 = '#d7191c'
        c4 = 'red'
        c0 = '#2b83ba'

        truth_grid = t_grid[:,t_grid[0,:].argsort()]
        x = truth_grid[0, :]
        y = truth_grid[1, :]
        plt.plot(x, y, marker='', ls='-', label='Truth', linewidth=4, color=c1)


        x = grid[3][0, :]
        y = grid[3][1, :]
        plt.scatter(x, y, marker='h', label='Vcnet', alpha=0.5, zorder=2, color='#abdda4', s=15)

        x = grid[2][0, :]
        y = grid[2][1, :]
        plt.scatter(x, y, marker='x', label='Drnet', alpha=0.5, zorder=2, color=c2, s=15)

        x = grid[1][0, :]
        y = grid[1][1, :]
        plt.scatter(x, y, marker='H', label='Tarnet', alpha=0.5, zorder=2, color='#2b83ba', s=15)

        x = grid[0][0, :]
        y = grid[0][1, :]
        plt.scatter(x, y, marker='*', label='TransITE', alpha=0.9, zorder=3, color='#d7191c', s=15)

        plt.yticks(np.arange(-2.0, 2 + 0.1, 0.5), fontsize=4, family='Times New Roman')
        plt.xticks(np.arange(0, args.h + 0.1, 0.2), fontsize=4, family='Times New Roman')
        plt.grid()
        plt.legend(prop=font_legend, loc='lower left')
        plt.xlabel('Treatment', font1)
        plt.ylabel('Response', font1)

        plt.savefig(save_path +"/Vc_Dr.pdf", bbox_inches='tight')
