import torch
import torch.nn as nn
from models.transformers import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder
from models.trans_ci import TransformerModel, Embeddings
from models.dynamic_net import Density_Block, Dynamic_FC, Truncated_power, Treat_Linear

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

class TransTEE(nn.Module):
    def __init__(self, embed_size = 48, dim_f=4000, num_t=1, num_cov=25, num_heads=2, att_layers=1, dropout=0.0,init_range_f=0.1, init_range_t=0.2):
        super(TransTEE, self).__init__()
        """
        cfg_density: cfg for the density estimator; [(ind1, outd1, isbias1), 'act', ....]; the cfg for density estimator head is not included
        num_grid: how many grid used for the density estimator head
        """

        self.att_layers = att_layers
        # construct the density estimator
        self.linear1 = Linear(dim_f, num_cov)
        self.feature_weight = Embeddings(embed_size, initrange=init_range_f)
        self.treat_emb = Embeddings(embed_size, initrange=init_range_t)
        self.dosage_emb = Embeddings(embed_size, initrange=init_range_t)
        self.linear2 = Linear(embed_size * 2, embed_size)

        encoder_layers = TransformerEncoderLayer(embed_size, nhead=num_heads, dim_feedforward=50, dropout=dropout, num_cov=num_cov)
        self.encoder = TransformerEncoder(encoder_layers, att_layers)

        decoder_layers = TransformerDecoderLayer(embed_size, nhead=num_heads, dim_feedforward=50, dropout=dropout,num_t=num_t)
        self.decoder = TransformerDecoder(decoder_layers, att_layers)

        self.Q = Linear(embed_size, 1, act='none')

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
        return Q

    def _initialize_weights(self):
        # TODO: maybe add more distribution for initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.in_features == 1:
                    continue
                m.weight.data.normal_(0, 0.1)
                if m.bias is not None:
                    m.bias.data.zero_()