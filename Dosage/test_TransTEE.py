# Copyright (c) 2020, Ioana Bica

import argparse
import os
import numpy as np
import torch.nn.functional as F
import json

from data_simulation import get_dataset_splits, TCGA_Data, get_iter
from utils.evaluation_torch import compute_eval_metrics
import torch
from torch.autograd import Variable
from TransTEE import TransTEE
from utils.utils import get_optimizer_scheduler
from utils.DisCri import DisCri
from utils.tsne import plot_tnse_ihdp

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_treatments", default=3, type=int)
    parser.add_argument("--treatment_selection_bias", default=2.0, type=float)
    parser.add_argument("--dosage_selection_bias", default=2.0, type=float)
    parser.add_argument("--save_dataset", default=True)
    parser.add_argument("--validation_fraction", default=0.1, type=float)
    parser.add_argument("--test_fraction", default=0.2, type=float)
    parser.add_argument("--model_name", default="TransTEE_tr")
    parser.add_argument("--verbose", default=False)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--h_dim", default=48, type=int)
    parser.add_argument("--rep", default=1, type=int)
    parser.add_argument("--h_inv_eqv_dim", default=64, type=int)
    parser.add_argument("--num_dosage_samples", default=5, type=int)
    parser.add_argument("--alpha", default=1.0, type=float)

    #optimizer and scheduler
    parser.add_argument('--beta', type=float, default=0.5, help='tradeoff parameter for advesarial loss')
    parser.add_argument('--p', type=int, default=0, help='dim for outputs of treatments discriminator, 1 for value, 2 for mean and var')
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["adam", "amsgrad"]
    )
    parser.add_argument(
            "--log_interval",
            type=int,
            default=10,
            help="How many batches to wait before logging training status",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["exponential", "cosine", "cycle", "none"],
    )
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--cov_dim", type=int, default=100)
    parser.add_argument(
        "--initialiser",
        type=str,
        default="xavier",
        choices=["xavier", "orthogonal", "kaiming", "none"],
    )
    return parser.parse_args()

def neg_guassian_likelihood(d, u):
    """return: -N(u; mu, var)"""
    B, dim = u.shape[0], 1
    assert (d.shape[1] == dim * 2)
    mu, logvar = d[:, :dim], d[:, dim:]
    return 0.5 * (((u - mu) ** 2) / torch.exp(logvar) + logvar).mean()

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

    result = {}
    result['in'] = []
    result['out'] = []
    for r in range(args.rep):
        data_class = TCGA_Data(dataset_params)
        dataset = data_class.dataset
        dataset_train, dataset_val, dataset_test = get_dataset_splits(dataset)
        params = {'num_features': dataset_train['x'].shape[1], 'num_treatments': args.num_treatments,
            'num_dosage_samples': args.num_dosage_samples, 'export_dir': export_dir,
            'alpha': args.alpha, 'batch_size': args.batch_size, 'h_dim': args.h_dim,
            'h_inv_eqv_dim': args.h_inv_eqv_dim, 'cov_dim':args.cov_dim, 'initialiser':args.initialiser}

        if 'tr' in args.model_name:
            TargetReg = DisCri(args.h_dim,dim_hidden=50, dim_output=args.p)#args.num_treatments
            TargetReg.cuda()
            tr_optimizer = torch.optim.SGD(TargetReg.parameters(), lr=0.001, weight_decay=5e-3)

        model = TransTEE(params)
        print("model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
        model.cuda()
        optimizer, scheduler = get_optimizer_scheduler(args=args, model=model)
        dataset_train['x'] = torch.from_numpy(dataset_train['x']).float()
        dataset_train['t'] = torch.from_numpy(dataset_train['t']).float()
        dataset_train['d'] = torch.from_numpy(dataset_train['d']).float()
        dataset_train['y'] = torch.from_numpy(dataset_train['y']).float()#y_normalized

        train_loader = get_iter(dataset_train, batch_size=args.batch_size, shuffle=True)
        for it in range(args.max_epochs):
            for idx, (x, t, d, y) in enumerate(train_loader):
                X_mb = Variable(x).cuda().detach().float()
                T_mb = Variable(t).cuda().detach().float()
                D_mb = Variable(d).cuda().detach().float()
                Y_mb = Variable(y).cuda().detach().float()

                optimizer.zero_grad()
                pred_outcome = model(X_mb, T_mb, D_mb)
                if 'tr' in args.model_name:
                    set_requires_grad(TargetReg, True)
                    tr_optimizer.zero_grad()
                    trg = TargetReg(pred_outcome[0].detach())
                    if args.p == 1:
                        loss_D = F.mse_loss(trg.squeeze(), T_mb)
                    elif args.p == 2:
                        loss_D = neg_guassian_likelihood(trg.squeeze(), T_mb)
                    loss_D.backward()
                    tr_optimizer.step()

                    set_requires_grad(TargetReg, False)
                    trg = TargetReg(pred_outcome[0])
                    if args.p == 1:
                        loss_D = F.mse_loss(trg.squeeze(), T_mb)
                    elif args.p == 2:
                        loss_D = neg_guassian_likelihood(trg.squeeze(), T_mb)
                    loss = F.mse_loss(input=pred_outcome[1].squeeze(), target=Y_mb) - args.beta * loss_D
                    loss.backward()
                    optimizer.step()
                else:
                    loss = F.mse_loss(input=pred_outcome[1].squeeze(), target=Y_mb)
                    loss.backward()
                    optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if it % args.log_interval == 0:
                tr_mise, tr_dpe, tr_pe, tr_ite = compute_eval_metrics(dataset, X_mb.cpu().detach().numpy(), num_treatments=params['num_treatments'],num_dosage_samples=params['num_dosage_samples'], model_folder=export_dir, model=model, train=True)
                mise, dpe, pe, ite = compute_eval_metrics(dataset, dataset_test['x'], num_treatments=params['num_treatments'],num_dosage_samples=params['num_dosage_samples'], model_folder=export_dir, model=model, train=True)
                print( "Train Epoch: [{}/{}]\tLoss: {:.6f}, tr_mise: {:.6f}, tr_dpe: {:.6f}, tr_pe: {:.6f}, tr_ite: {:.6}, mise: {:.6f}, dpe: {:.6f}, pe: {:.6f}, ite: {:.6}".format(it, args.max_epochs, loss.item(),  tr_mise, tr_dpe, tr_pe, tr_ite, mise, dpe, pe, ite )) 
        
        print('-----------------Test------------------')
        out = model(dataset_train['x'].cuda().detach().float(), dataset_train['t'].cuda().detach().float(), dataset_train['d'].cuda().detach().float())
        plot_tnse_ihdp(out[0], dataset_train['t'], model_name=args.model_name)
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

        with open(export_dir + '/p' + str(args.p) + str(args.beta) + 'result.json', 'w') as fp:
            json.dump(result, fp)