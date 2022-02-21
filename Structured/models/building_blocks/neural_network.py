import logging

from torch import nn
from torch_geometric.data.batch import Batch

import wandb


class NeuralNetworkEstimator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def train_step(self, device, train_loader, epoch: int, log_interval: int):
        self.train()
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            self.optimizer.zero_grad()
            prediction = self.forward(batch)
            loss = self.loss(prediction, batch)
            loss.backward()
            self.optimizer.step()
            if batch_idx % log_interval == 0:
                logging.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * self.args.batch_size,
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )

    def test_prediction(self, batch: Batch):
        return self.forward(batch).view(-1)
