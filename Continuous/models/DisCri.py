 from torch import Tensor, nn
import torch

def get_initialiser(name: str):
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

def get_activation(name: str, leaky_relu = 0.1) -> nn.Module:
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

def create_batch_norm_1d_layers(num_layers: int, dim_hidden: int):
    batch_norm_layers = nn.ModuleList()
    for i in range(num_layers - 1):
        batch_norm_layers.append(nn.BatchNorm1d(num_features=dim_hidden))
    return batch_norm_layers

def create_linear_layers(
    num_layers: int, dim_input: int, dim_hidden: int, dim_output: int
):
    linear_layers = nn.ModuleList()
    # Input layer
    linear_layers.append(nn.Linear(in_features=dim_input, out_features=dim_hidden))
    # Hidden layers
    for i in range(1, num_layers - 1):
        linear_layers.append(nn.Linear(in_features=dim_hidden, out_features=dim_hidden))
    # Output layer
    linear_layers.append(nn.Linear(dim_hidden, dim_output))
    return linear_layers

def init_layers(initialiser_name: str, layers: nn.ModuleList):
    initialiser = get_initialiser(initialiser_name)
    for layer in layers:
        initialiser(layer.weight)

class DisCri(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_hidden: int,
        dim_output=1,
        num_layers=1,
        batch_norm=False,
        initialiser='xavier',
        dropout=0.0,
        activation='relu',
        leaky_relu=0.1,
        is_output_activation=False,
    ):
        super().__init__()
        self.layers = create_linear_layers(
            num_layers=num_layers,
            dim_input=dim_input,
            dim_hidden=dim_hidden,
            dim_output=dim_output,
        )
        init_layers(initialiser_name=initialiser, layers=self.layers)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.batch_norm_layers = (
            create_batch_norm_1d_layers(num_layers=num_layers, dim_hidden=dim_hidden)
            if batch_norm
            else None
        )
        self.activation_function = get_activation(
            name=activation, leaky_relu=leaky_relu
        )
        self.is_output_activation = is_output_activation

    def forward(self, x: Tensor):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.activation_function(x)
            if self.batch_norm_layers:
                x = self.batch_norm_layers[i](x)
            if self.dropout:
                x = self.dropout(x)
        x = self.layers[-1](x)
        if self.is_output_activation:
            x = self.activation_function(x)
        return x

