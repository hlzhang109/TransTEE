import torch
import math
import numpy as np
import pandas as pd
import torch.nn.functional as F

import os
import json

import matplotlib.pyplot as plt
from models.dynamic_net import Vcnet, Drnet, TR
from models.itermidate_models import VcnetAtt, VcnetAttv2
from models.DisCri import DisCri
from datas.data import get_iter
from utils.eval import curve
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, MultiStepLR

import argparse

def adjust_learning_rate(optimizer, init_lr, epoch, lr_type='cos', num_epoch=50):
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

def save_checkpoint(state, model_name='', checkpoint_dir='.'):
    filename = os.path.join(checkpoint_dir, model_name + '_ckpt.pth.tar')
    print('=> Saving checkpoint to {}'.format(filename))
    torch.save(state, filename)

# criterion
def criterion(out, y, alpha=0.5, epsilon=1e-6):
    return ((out[1].squeeze() - y.squeeze())**2).mean()# - alpha * torch.log(out[0] + epsilon).mean()

def neg_guassian_likelihood(d, u):
    """return: -N(u; mu, var)"""
    B, dim = u.shape[0], 1
    assert (d.shape[1] == dim * 2)
    mu, logvar = d[:, :dim], d[:, dim:]
    return 0.5 * (((u - mu) ** 2) / torch.exp(logvar) + logvar).mean()

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train with simulate data')

    # i/o
    parser.add_argument('--data_dir', type=str, default='dataset/news/tr_h_1.0_te_h_1.0', help='dir of eval dataset')
    parser.add_argument('--save_dir', type=str, default='logs/news/tr_h_1.0_te_h_1.0/', help='dir to save result')

    # common
    parser.add_argument('--num_dataset', type=int, default=20, help='num of datasets to train')
    parser.add_argument('--h', type=int, default=1., help='the maximal value of treatmemts')
    parser.add_argument('--clip', type=int, default=1., help='gradient clip')
    # training
    parser.add_argument('--n_epochs', type=int, default=1500, help='num of epochs to train')
    parser.add_argument('--batch_size', type=int, default=500, help='num of epochs to train')
    parser.add_argument('--alpha', type=float, default=0.5, help='tradeoff parameter for advesarial loss')
    parser.add_argument('--p', type=int, default=2, help='dim for outputs of treatments discriminator, 1 for value, 2 for mean and var')

    # print train info
    parser.add_argument('--verbose', type=int, default=500, help='print train info freq')
    parser.add_argument('--plt_adrf', type=bool, default=False, help='whether to plot adrf curves.')

    args = parser.parse_args()

    # fixed parameter for optimizer
    lr_type = 'fixed'
    wd = 5e-3
    momentum = 0.9

    # targeted regularization optimizer
    tr_wd = 5e-3

    num_epoch = args.n_epochs

    # check val loss
    verbose = args.verbose

    # data
    load_path = args.data_dir
    num_dataset = args.num_dataset

    # save
    save_path = args.save_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.plt_adrf:
        grid = [] ;MSE = []; num_dataset=1
    Result = {}
    for model_name in ['Tarnet', 'Tarnet_tr', 'Drnet', 'Drnet_tr', 'Vcnet', 'Vcnet_tr', 'VcnetAttv2_tr', 'VcnetAttv2']:
        Result[model_name]=[]
    method_list = ['VcnetAttv2_tr']
    for model_name in method_list:
        # import model
        if model_name == 'Vcnet' or model_name == 'Vcnet_tr':
            cfg_density = [(498, 50, 1, 'relu'), (50, 50, 1, 'relu')]
            num_grid = 10
            cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
            degree = 2
            knots = [0.33, 0.66]
            model = Vcnet(cfg_density, num_grid, cfg, degree, knots)
            model._initialize_weights()

        elif model_name == 'VcnetAtt' or model_name == 'VcnetAtt_tr':
            cfg_density = [(498, 50, 1, 'relu'), (50, 50, 1, 'relu')]
            num_grid = 10
            cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
            degree = 2
            knots = [0.33, 0.66]
            model = VcnetAtt(cfg_density, num_grid, cfg, degree, knots)
            model._initialize_weights()
        
        elif model_name == 'VcnetAttv2' or model_name == 'VcnetAttv2_tr':
            num_cov=498
            num_t=1
            num_heads = 2
            att_layers = 1
            dropout = 0.0
            init_range_f = 0.1
            init_range_t = 0.2
            embed_size = 10
            model = VcnetAttv2(embed_size=embed_size, num_t=num_t, num_cov=num_cov, num_heads=num_heads, att_layers=att_layers, dropout=dropout, init_range_f=init_range_f, init_range_t=init_range_t)
            model._initialize_weights()

        elif model_name == 'Drnet' or model_name == 'Drnet_tr':
            cfg_density = [(498, 50, 1, 'relu'), (50, 50, 1, 'relu')]
            num_grid = 10
            cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
            isenhance = 1
            model = Drnet(cfg_density, num_grid, cfg, isenhance=isenhance, h=args.h)
            model._initialize_weights()

        elif model_name == 'Tarnet' or model_name == 'Tarnet_tr':
            cfg_density = [(498, 50, 1, 'relu'), (50, 50, 1, 'relu')]
            num_grid = 10
            cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
            isenhance = 0
            model = Drnet(cfg_density, num_grid, cfg, isenhance=isenhance, h=args.h)
            model._initialize_weights()

        model.cuda()
        # use Target Regularization
        if 'tr' in model_name:
            isTargetReg = 1
        else:
            isTargetReg = 0
        # best cfg for each model
        if model_name == 'Tarnet':
            init_lr = 0.05
            alpha = 1.0

            Result['Tarnet'] = []

        elif model_name == 'Drnet':
            init_lr = 0.05
            alpha = 1.

            Result['Drnet'] = []

        elif model_name == 'Vcnet':
            init_lr = 0.0005
            alpha = 0.5

            Result['Vcnet'] = []
        elif model_name == 'VcnetAtt':
            init_lr = 0.0005
            alpha = 0.5

            Result['VcnetAtt'] = []  
        
        elif model_name == 'VcnetAttv2':
            init_lr = 0.01
            #init_lr = 0.01
            alpha = 0.5
        
            Result['VcnetAttv2'] = [] 

        elif model_name == 'VcnetAttv2_tr':
            init_lr = 0.01
            alpha = 0.5
            tr_init_lr = 0.001
        
            Result['VcnetAttv2_tr'] = [] 

        if args.h != 1:
            data_matrix = torch.load(args.data_dir + '/train_matrix.pt')
            data_test_matrix = torch.load(args.data_dir + '/data_matrix.pt')
        else:
            data_matrix = torch.load(args.data_dir + '/data_matrix.pt')
        data_grid = torch.load(args.data_dir + '/t_grid.pt')
        for _ in range(num_dataset):

            train_idx = torch.load(args.data_dir + '/eval/' + str(_) + '/idx_train.pt')
            test_idx = torch.load(args.data_dir + '/eval/' + str(_) + '/idx_test.pt')
            train_matrix = data_matrix[train_idx]
            test_matrix = data_matrix[test_idx] if args.h == 1 else data_test_matrix[test_idx]
            t_grid = data_grid[:,test_idx]
            
            # train_matrix, test_matrix, t_grid = simu_data1(500, 200)
            train_loader = get_iter(train_matrix, batch_size=args.batch_size, shuffle=True)
            test_loader = get_iter(test_matrix, batch_size=test_matrix.shape[0], shuffle=False)

            # reinitialize model
            model._initialize_weights()

            # define optimizer
            optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=momentum, weight_decay=wd, nesterov=True)

            if isTargetReg:
                if 'Att' in model_name:
                    in_dim = embed_size
                else:
                    in_dim = 50
                TargetReg = DisCri(in_dim, 50, args.p)
                TargetReg.cuda()
                tr_optimizer = torch.optim.SGD(TargetReg.parameters(), lr=tr_init_lr, weight_decay=tr_wd)

            print('model : ', model_name)
            for epoch in range(num_epoch):

                for idx, (inputs, y) in enumerate(train_loader):
                    inputs = Variable(inputs.cuda().detach())
                    y = Variable(y.cuda().detach())
                    t = inputs[:, 0]
                    x = inputs[:, 1:]

                    if isTargetReg:
                        out = model.forward(t, x)

                        set_requires_grad(TargetReg, True)
                        tr_optimizer.zero_grad()
                        trg = TargetReg(out[0].detach())
                        if args.p == 1:
                            loss_D = F.mse_loss(trg.squeeze(), t)
                        elif args.p == 2:
                            loss_D = neg_guassian_likelihood(trg.squeeze(), t)
                        loss_D.backward()
                        tr_optimizer.step()
                    
                        set_requires_grad(TargetReg, False)
                        optimizer.zero_grad()
                        trg = TargetReg(out[0])
                        if args.p == 1:
                            loss_D = F.mse_loss(trg.squeeze(), t)
                        elif args.p == 2:
                            loss_D = neg_guassian_likelihood(trg.squeeze(), t)
                        loss = criterion(out, y, alpha=alpha) - args.alpha * loss_D
                        loss.backward()
                        optimizer.step()
                        adjust_learning_rate(optimizer, init_lr, epoch, lr_type='cos', num_epoch=num_epoch)
                    else:
                        optimizer.zero_grad()
                        out = model.forward(t, x)
                        loss = criterion(out, y, alpha=alpha)
                        loss.backward()
                        optimizer.step()
                        adjust_learning_rate(optimizer, init_lr, epoch, lr_type='cos', num_epoch=num_epoch)

                if epoch % verbose == 0:
                    print('current epoch: ', epoch)
                    print('loss: ', loss.data)
                    if isTargetReg:
                        print('adv_loss: ', loss_D.data)
                    model.eval()
                    t_grid_hat, mse = curve(model, test_matrix, t_grid)
                    model.train()
                    print('current test loss: ', mse)

            model.eval()
            t_grid_hat, mse = curve(model, test_matrix, t_grid)

            mse = float(mse)
            print('current loss: ', float(loss.data))
            print('current test loss: ', mse)
            if args.plt_adrf:
                grid.append(t_grid_hat);MSE.append(mse)
            Result[model_name].append(mse)
            if not args.plt_adrf:
                with open(save_path + '/p' + str(args.p) + str(args.alpha) + 'result.json', 'w') as fp:
                    json.dump(Result, fp)

    if args.plt_adrf:
        plt.figure(figsize=(5, 5))

        c1 = 'gold'
        c2 = 'grey'
        c3 = '#d7191c'
        c4 = 'red'
        c0 = '#2b83ba'

        truth_grid = data_grid[:,data_grid[0,:].argsort()]
        x = truth_grid[0, :]
        y = truth_grid[1, :]
        plt.plot(x, y, marker='', ls='-', label='Truth', linewidth=4, color=c1)


        x = grid[3][0, :]
        y = grid[3][1, :]
        plt.scatter(x, y, marker='h', label='Tarnet', alpha=0.5, zorder=2, color='#abdda4', s=15)

        x = grid[2][0, :]
        y = grid[2][1, :]
        plt.scatter(x, y, marker='x', label='Drnet', alpha=0.5, zorder=2, color=c2, s=15)

        x = []
        y = []
        plt.scatter(x, y, marker='H', label='Vcnet', alpha=0.5, zorder=2, color='#2b83ba', s=15)

        x = grid[0][0, :]
        y = grid[0][1, :]
        plt.scatter(x, y, marker='*', label='TransITE', alpha=0.9, zorder=3, color='#d7191c', s=15)

        plt.grid()
        plt.legend()
        plt.xlabel('Treatment')
        plt.ylabel('Response')

        plt.savefig(save_path +"/ihdp_curve.pdf", bbox_inches='tight')