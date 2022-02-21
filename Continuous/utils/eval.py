import torch
import numpy as np
import json
from datas.data import get_iter
from torch.autograd import Variable

def curve(model, test_matrix, t_grid, targetreg=None):
    n_test = t_grid.shape[1]
    t_grid_hat = torch.zeros(2, n_test)
    t_grid_hat[0, :] = t_grid[0, :]

    test_loader = get_iter(test_matrix, batch_size=test_matrix.shape[0], shuffle=False)

    if targetreg is None:
        for _ in range(n_test):
            for idx, (inputs, y) in enumerate(test_loader):
                inputs = Variable(inputs.cuda().detach())
                y = Variable(y.cuda().detach())
                t = inputs[:, 0]
                t *= 0
                t += t_grid[0, _]
                x = inputs[:, 1:]
                break
            out = model.forward(t, x) # apply the given t to the batch and calculate the mean
            out = out[1].data.squeeze()
            out = out.mean()
            t_grid_hat[1, _] = out
        mse = ((t_grid_hat[1, :].squeeze() - t_grid[1, :].squeeze()) ** 2).mean().data
        return t_grid_hat, mse
    else:
        for _ in range(n_test):
            for idx, (inputs, y) in enumerate(test_loader):
                t = inputs[:, 0]
                t *= 0
                t += t_grid[0, _]
                x = inputs[:, 1:]
                break
            out = model.forward(t, x)
            tr_out = targetreg(t).data
            g = out[0].data.squeeze()
            out = out[1].data.squeeze() + tr_out / (g + 1e-6)
            out = out.mean()
            t_grid_hat[1, _] = out
        mse = ((t_grid_hat[1, :].squeeze() - t_grid[1, :].squeeze()) ** 2).mean().data
        return t_grid_hat, mse

class Evaluator(object):
    def __init__(self, y, t, y_cf=None, mu0=None, mu1=None):
        self.y = y
        self.t = t
        self.y_cf = y_cf
        self.mu0 = mu0
        self.mu1 = mu1
        if mu0 is not None and mu1 is not None:
            self.true_ite = mu1 - mu0

    def rmse_ite(self, ypred1, ypred0):
        pred_ite = np.zeros_like(self.true_ite)
        idx1, idx0 = np.where(self.t == 1), np.where(self.t == 0)
        ite1, ite0 = self.y[idx1] - ypred0[idx1], ypred1[idx0] - self.y[idx0]
        pred_ite[idx1] = ite1
        pred_ite[idx0] = ite0
        return np.sqrt(np.mean(np.square(self.true_ite - pred_ite)))

    def abs_ate(self, ypred1, ypred0):
        return np.abs(np.mean(ypred1 - ypred0) - np.mean(self.true_ite))

    def pehe(self, ypred1, ypred0):
        return np.sqrt(np.mean(np.square((self.mu1 - self.mu0) - (ypred1 - ypred0))))

    def y_errors(self, y0, y1):
        ypred = (1 - self.t) * y0 + self.t * y1
        ypred_cf = self.t * y0 + (1 - self.t) * y1
        return self.y_errors_pcf(ypred, ypred_cf)

    def y_errors_pcf(self, ypred, ypred_cf):
        rmse_factual = np.sqrt(np.mean(np.square(ypred - self.y)))
        rmse_cfactual = np.sqrt(np.mean(np.square(ypred_cf - self.y_cf)))
        return rmse_factual, rmse_cfactual

    def calc_stats(self, ypred1, ypred0):        
        ite = self.rmse_ite(ypred1, ypred0)
        ate = self.abs_ate(ypred1, ypred0)
        pehe = self.pehe(ypred1, ypred0)
        return ite, ate, pehe