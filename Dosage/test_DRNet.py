# Copyright (c) 2020, Ioana Bica

import argparse
import os
import shutil
import logging
import tqdm
import numpy as np
import torch.nn.functional as F
import json

from data_simulation import get_dataset_splits, TCGA_Data, get_iter
from SCIGAN import SCIGAN_Model
from utils.evaluation_torch import compute_eval_metrics
import torch
from torch.autograd import Variable
from DRNet import Drnet, Vcnet
from utils.utils import get_optimizer_scheduler

def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_treatments", default=3, type=int)
    parser.add_argument("--treatment_selection_bias", default=2.0, type=float)
    parser.add_argument("--dosage_selection_bias", default=2.0, type=float)
    parser.add_argument("--save_dataset", default=True)
    parser.add_argument("--validation_fraction", default=0.1, type=float)
    parser.add_argument("--test_fraction", default=0.2, type=float)
    parser.add_argument("--rep", default=10, type=int)
    parser.add_argument("--model_name", default="tarnet")
    parser.add_argument("--verbose", default=False)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--h_dim", default=48, type=int)
    parser.add_argument("--h", default=1., type=float)
    parser.add_argument("--h_inv_eqv_dim", default=64, type=int)
    parser.add_argument("--num_dosage_samples", default=5, type=int)
    parser.add_argument("--alpha", default=1.0, type=float)

    #optimizer and scheduler
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["adam", "amsgrad"]
    )
    parser.add_argument(
            "--log_interval",
            type=int,
            default=10,
            help="How many batches to wait before logging training status",
    )
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--cov_dim", type=int, default=100)
    parser.add_argument(
        "--initialiser",
        type=str,
        default="xavier",
        choices=["xavier", "orthogonal", "kaiming", "none"],
    )
    return parser.parse_args()

if __name__ == "__main__":

    args = init_arg()

    dataset_params = dict()
    dataset_params['num_treatments'] = args.num_treatments
    dataset_params['treatment_selection_bias'] = args.treatment_selection_bias
    dataset_params['dosage_selection_bias'] = args.dosage_selection_bias
    dataset_params['save_dataset'] = args.save_dataset
    dataset_params['validation_fraction'] = args.validation_fraction
    dataset_params['test_fraction'] = args.test_fraction

    export_dir = 'saved_models/' + args.model_name + '/tr' +str(args.num_treatments) + '/trb' + str(args.treatment_selection_bias) +'/dob' +str(args.dosage_selection_bias)
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    model_name = args.model_name
    
    if model_name == 'vcnet':
        init_lr = 0.005
        cfg_density = [(4000, 100, 1, 'relu'), (100, 64, 1, 'relu')]
        num_grid = 10
        cfg = [(64, 64, 1, 'relu'), (64, 1, 1, 'id')]
        degree = 2
        knots = [0.33, 0.66]
        model = Vcnet(cfg_density, num_grid, cfg, degree, knots, num_t=args.num_treatments)

    elif model_name == 'drnet':
        init_lr = 0.005
        cfg_density = [(4000, 100, 1, 'relu'), (100, 64, 1, 'relu')]
        num_grid = 10
        cfg = [(64, 64, 1, 'relu'), (64, 1, 1, 'id')]
        isenhance = 1
        model = Drnet(cfg_density, num_grid, cfg, isenhance=isenhance, h=args.h, num_t=args.num_treatments)

    elif model_name == 'tarnet':
        init_lr = 0.005
        cfg_density = [(4000, 100, 1, 'relu'), (100, 64, 1, 'relu')]
        num_grid = 10
        cfg = [(64, 64, 1, 'relu'), (64, 1, 1, 'id')]
        isenhance = 0
        model = Drnet(cfg_density, num_grid, cfg, isenhance=isenhance, h=args.h, num_t=args.num_treatments)
    print("model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    result = {}
    result['in'] = []
    result['out'] = []
    model.cuda()

    for r in range(args.rep):
        data_class = TCGA_Data(dataset_params)
        dataset = data_class.dataset
        dataset_train, dataset_val, dataset_test = get_dataset_splits(dataset)
        dataset_train['x'] = torch.from_numpy(dataset_train['x']).float()
        dataset_train['t'] = torch.from_numpy(dataset_train['t']).float()
        dataset_train['d'] = torch.from_numpy(dataset_train['d']).float()
        dataset_train['y'] = torch.from_numpy(dataset_train['y']).float()
        train_loader = get_iter(dataset_train, batch_size=args.batch_size, shuffle=True)

        model._initialize_weights(args.initialiser)
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9, weight_decay=5e-3, nesterov=True)

        for it in range(args.max_epochs):
            for idx, (x, t, d, y) in enumerate(train_loader):
                optimizer.zero_grad()
                X_mb = Variable(x).cuda().detach().float()
                T_mb = Variable(t).cuda().detach().float()
                D_mb = Variable(d).cuda().detach().float()
                Y_mb = Variable(y).cuda().detach().float()
                pred_outcome = model(X_mb, T_mb, D_mb)
                loss = F.mse_loss(input=pred_outcome.squeeze(), target=Y_mb)
                loss.backward()
                optimizer.step()
            if it % args.log_interval == 0:
                tr_mise, tr_dpe, tr_pe, tr_ite = compute_eval_metrics(dataset, X_mb.cpu().detach().numpy(), num_treatments=args.num_treatments,num_dosage_samples=args.num_dosage_samples, model_folder=export_dir, model=model, train=True)
                mise, dpe, pe, ite = compute_eval_metrics(dataset, dataset_test['x'], num_treatments=args.num_treatments,num_dosage_samples=args.num_dosage_samples, model_folder=export_dir, model=model, train=True)
                print( "Train Epoch: [{}/{}]\tLoss: {:.6f}, tr_mise: {:.6f}, tr_dpe: {:.6f}, tr_pe: {:.6f}, tr_ite: {:.6}, mise: {:.6f}, dpe: {:.6f}, pe: {:.6f}, ite: {:.6}".format(it, args.max_epochs, loss.item(),  tr_mise, tr_dpe, tr_pe, tr_ite, mise, dpe, pe, ite )) 

        print('-----------------Test------------------')
        mise, dpe, pe, ate = compute_eval_metrics(dataset, dataset_test['x'], num_treatments=args.num_treatments,
                                            num_dosage_samples=args.num_dosage_samples, model_folder=export_dir, model=model)
        print("Mise: %s" % str(mise))
        print("DPE: %s" % str(dpe))
        print("PE: %s" % str(pe))
        print("ATE: %s" % str(ate))
        result['out'].append(ate)

        mise, dpe, pe, ate = compute_eval_metrics(dataset, dataset_val['x'], num_treatments=args.num_treatments,
                                            num_dosage_samples=args.num_dosage_samples, model_folder=export_dir, model=model)
        print('-----------------Val------------------')
        print("Mise: %s" % str(mise))
        print("DPE: %s" % str(dpe))
        print("PE: %s" % str(pe))
        print("ATE: %s" % str(ate))
        result['in'].append(ate)

        with open(export_dir + '/result.json', 'w') as fp:
            json.dump(result, fp)