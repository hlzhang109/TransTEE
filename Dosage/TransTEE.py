import torch
import torch.nn as nn
from utils.transformers import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder
from utils.utils import get_initialiser
from utils.mlp import MLP
from utils.trans_ci import TransformerModel, Embeddings
from tqdm import tqdm
import numpy as np
# replace the feature extractor of x by self-attention
# 0.015


class Linear(nn.Module):
    def __init__(self, ind, outd, act='relu', isbias=1):
        super(Linear, self).__init__()
        # ind does NOT include the extra concat treatment
        self.ind = ind
        self.outd = outd
        self.isbias = isbias

        self.weight = nn.Parameter(torch.rand(self.ind, self.outd), requires_grad=True)

        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd), requires_grad=True)
        else:
            self.bias = None

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = None

    def forward(self, x):
        # x: batch_size * (treatment, other feature)

        out = torch.matmul(x, self.weight)

        if self.isbias:
            out = out + self.bias

        if self.act is not None:
            out = self.act(out)

        return out



# Repalce dynamic-Q by feature embeddings, it works well
class TransTEE(nn.Module):
    def __init__(self, params, num_heads=2, att_layers=1, dropout=0.0, init_range_f=0.1, init_range_t=0.1):
        super(TransTEE, self).__init__()
        """
        cfg_density: cfg for the density estimator; [(ind1, outd1, isbias1), 'act', ....]; the cfg for density estimator head is not included
        num_grid: how many grid used for the density estimator head
        """

        # cfg/cfg_density = [(ind1, outd1, isbias1, activation),....]
        num_features = params['num_features']
        num_treatments = params['num_treatments']
        self.export_dir = params['export_dir']

        h_dim = params['h_dim']
        self.h_inv_eqv_dim = params['h_inv_eqv_dim']
        self.batch_size = params['batch_size']
        self.alpha = params['alpha']
        self.num_dosage_samples = params['num_dosage_samples']
        
        self.linear1 = nn.Linear(num_features, 100)

        self.feature_weight = Embeddings(h_dim, initrange=init_range_f)
        self.treat_emb = Embeddings(h_dim, act='id', initrange=init_range_t)
        self.dosage_emb = Embeddings(h_dim, act='id', initrange=init_range_t)
        self.linear2 = MLP(
            dim_input=h_dim * 2,
            dim_hidden=h_dim,
            dim_output=h_dim,
        )

        encoder_layers = TransformerEncoderLayer(h_dim, nhead=num_heads, dim_feedforward=h_dim, dropout=dropout, num_cov=params['cov_dim'])
        self.encoder = TransformerEncoder(encoder_layers, att_layers)

        decoder_layers = TransformerDecoderLayer(h_dim, nhead=num_heads, dim_feedforward=h_dim, dropout=dropout,num_t=1)
        self.decoder = TransformerDecoder(decoder_layers, att_layers)

        self.Q = MLP(
            dim_input=h_dim,
            dim_hidden=h_dim,
            dim_output=1,
            is_output_activation=False,
        )

    def forward(self, x, t, d):
        hidden = self.feature_weight(self.linear1(x))
        memory = self.encoder(hidden)

        t = t.view(t.shape[0], 1)
        d = d.view(d.shape[0], 1)
        tgt = torch.cat([self.treat_emb(t), self.dosage_emb(d)], dim=-1)
        tgt = self.linear2(tgt)
        if len(tgt.shape) < 3:
            tgt = tgt.unsqueeze(1)
        out = self.decoder(tgt.permute(1, 0, 2), memory.permute(1, 0, 2))
        if out.shape[0] != 1:
            out = torch.mean(out, dim=1)
        Q = self.Q(out.squeeze(0))
        return torch.mean(hidden, dim=1).squeeze(), Q

    def _initialize_weights(self, initialiser):
        # TODO: maybe add more distribution for initialization
        initialiser = get_initialiser(initialiser)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # if m.in_features == 1:
                #     continue
                initialiser(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
