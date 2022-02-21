from argparse import Namespace
from typing import Callable, Iterator, Optional, Union

from torch import nn, optim

def get_initialiser(name: str) -> Callable:
    if name == "orthogonal":
        return nn.init.orthogonal_
    elif name == "xavier":
        return nn.init.xavier_uniform_
    elif name == "kaiming":
        return nn.init.kaiming_uniform_
    elif name == "none":
        pass
    else:
        raise Exception("Unknown init method")


def get_optimizer(
    args: Namespace, params: Iterator[nn.Parameter], net: Optional[str] = None
) -> optim.Optimizer:
    weight_decay = args.weight_decay
    lr = args.lr

    optimizer = None
    if args.optimizer == "sgd":
        optimizer = optim.SGD(params, lr=lr, weight_decay=weight_decay)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif args.optimizer == "amsgrad":
        optimizer = optim.Adam(params, lr=lr, amsgrad=True, weight_decay=weight_decay)
    return optimizer


class NoneScheduler:
    def step(self):
        pass


def get_lr_scheduler(
    args: Namespace, optimizer: optim.Optimizer
) -> Union[
    optim.lr_scheduler.ExponentialLR,
    optim.lr_scheduler.CosineAnnealingLR,
    optim.lr_scheduler.CyclicLR,
    NoneScheduler,
]:
    if args.lr_scheduler == "exponential":
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=0)
    elif args.lr_scheduler == "cycle":
        return optim.lr_scheduler.CyclicLR(
            optimizer, 0, max_lr=args.lr, step_size_up=20, cycle_momentum=False
        )
    elif args.lr_scheduler == "none":
        return None


def get_optimizer_scheduler(
    args: Namespace, model: nn.Module
):
    optimizer = get_optimizer(args=args, params=model.parameters())
    lr_scheduler = get_lr_scheduler(args=args, optimizer=optimizer)
    return optimizer, lr_scheduler

def get_initialiser(name='xavier'):
    if name == "orthogonal":
        return nn.init.orthogonal_
    elif name == "xavier":
        return nn.init.xavier_uniform_
    elif name == "kaiming":
        return nn.init.kaiming_uniform_
    elif name == "none":
        pass
    else:
        raise Exception("Unknown init method")


def get_activation(name: str, leaky_relu: Optional[float] = 0.5) -> nn.Module:
    if name == "leaky_relu":
        return nn.LeakyReLU(leaky_relu)
    elif name == "rrelu":
        return nn.RReLU()
    elif name == "relu":
        return nn.ReLU()
    elif name == "elu":
        return nn.ELU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name == "selu":
        return nn.SELU()
    else:
        raise Exception("Unknown activation")

def get_iter(data_matrix, batch_size, shuffle=True):
    dataset = Dataset_from_matrix(data_matrix)
    iterator = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return iterator