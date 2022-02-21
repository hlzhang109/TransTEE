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

class VcnetAttV1(nn.Module):
    def __init__(self, cfg_density, num_grid, cfg, degree, knots, att_layers=1):
        super(VcnetAtt, self).__init__()
        """

        cfg_density: cfg for the density estimator; [(ind1, outd1, isbias1), 'act', ....]; the cfg for density estimator head is not included
        num_grid: how many grid used for the density estimator head
        """

        # cfg/cfg_density = [(ind1, outd1, isbias1, activation),....]
        self.cfg_density = cfg_density
        self.num_grid = num_grid

        self.cfg = cfg
        self.degree = degree
        self.knots = knots
        self.att_layers = att_layers

        # construct the density estimator
        density_blocks = []
        density_hidden_dim = -1
        
        embed_size = 10
        self.feature_weight = Embeddings(embed_size)
        encoder_layers = TransformerEncoderLayer(embed_size, 1, 50, 0.1)
        self.encoder = TransformerEncoder(encoder_layers, att_layers)


        density_hidden_dim = 50
        self.linear1 = nn.Linear(embed_size, 50)

        self.density_hidden_dim = density_hidden_dim
        self.density_estimator_head = Density_Block(self.num_grid, density_hidden_dim, isbias=1)

        # construct the dynamics network
        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg)-1: # last layer
                last_layer = Dynamic_FC(layer_cfg[0], layer_cfg[1], self.degree, self.knots, act=layer_cfg[3], isbias=layer_cfg[2], islastlayer=1)
            else:
                blocks.append(
                    Dynamic_FC(layer_cfg[0], layer_cfg[1], self.degree, self.knots, act=layer_cfg[3], isbias=layer_cfg[2], islastlayer=0))
        blocks.append(last_layer)

        self.Q = nn.Sequential(*blocks)

    def forward(self, t, x):
        hidden = self.feature_weight(x)
        hidden = self.encoder(hidden)
        hidden = torch.mean(hidden, dim=1).squeeze()
        hidden = self.linear1(hidden)
        t_hidden = torch.cat((torch.unsqueeze(t, 1), hidden), 1)
        g = self.density_estimator_head(t, hidden)
        Q = self.Q(t_hidden)

        return g, Q

    def _initialize_weights(self):
        # TODO: maybe add more distribution for initialization
        for m in self.modules():
            if isinstance(m, Dynamic_FC):
                m.weight.data.normal_(0, 1.)
                if m.isbias:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, Density_Block):
                m.weight.data.normal_(0, 0.01)
                if m.isbias:
                    m.bias.data.zero_()

class Att_FC(nn.Module):
    def __init__(self, ind, outd, degree, knots, act='relu', isbias=1, islastlayer=0, residual=0, treat_emb=None):
        super(Att_FC, self).__init__()
        self.ind = ind
        self.outd = outd

        self.islastlayer = islastlayer
        self.treat_emb = treat_emb

        self.isbias = isbias
        self.res = residual
        self.spb = Truncated_power(degree, knots)
        self.d = self.spb.num_of_basis # num of basis#
        self.d = treat_emb.emb_size # num of basis

        self.weight = nn.Parameter(torch.rand(self.ind, self.outd, self.d), requires_grad=True)

        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd, self.d), requires_grad=True)
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
        x_feature = x[:, 1:]
        x_treat = x[:, 0]

        x_feature_weight = torch.matmul(self.weight.T, x_feature.T).T # bs, outd, d

        x_treat_basis = self.treat_emb(x_treat)# bs, d
        x_treat_basis_ = torch.unsqueeze(x_treat_basis, 1)
        # x_treat_basis = self.spb.forward(x_treat).cuda() # bs, d
        # x_treat_basis_ = torch.unsqueeze(x_treat_basis, 1)

        # x_feature_weight * x_treat_basis; bs, outd, d
        out = torch.sum(x_feature_weight * x_treat_basis_, dim=2) # bs, outd

        if self.isbias:
            out_bias = torch.matmul(self.bias, x_treat_basis.T).T
            out = out + out_bias
        
        if self.res:
            out = out + x_feature

        if self.act is not None:
            out = self.act(out)

        # concat the treatment for intermediate layer
        if not self.islastlayer:
            out = torch.cat((torch.unsqueeze(x_treat, 1), out), 1)

        return out

# Repalce dynamic-Q by feature embeddings, it works well
class VcnetAtt(nn.Module):
    def __init__(self, cfg_density, num_grid, cfg, degree, knots, att_layers=1):
        super(VcnetAtt, self).__init__()
        """
        cfg_density: cfg for the density estimator; [(ind1, outd1, isbias1), 'act', ....]; the cfg for density estimator head is not included
        num_grid: how many grid used for the density estimator head
        """

        # cfg/cfg_density = [(ind1, outd1, isbias1, activation),....]
        self.cfg_density = cfg_density
        self.num_grid = num_grid

        self.cfg = cfg
        self.degree = degree
        self.knots = knots
        self.att_layers = att_layers

        # construct the density estimator
        density_blocks = []
        density_hidden_dim = -1
        
        embed_size = 10
        self.feature_weight = Embeddings(embed_size, initrange=0.01)
        self.treat_emb = Embeddings(embed_size-1, act='id',initrange=0.2, res=1)

        encoder_layers = TransformerEncoderLayer(embed_size, 2, 50, 0.3)
        self.encoder = TransformerEncoder(encoder_layers, att_layers)

        density_hidden_dim = 50
        self.linear1 = Linear(embed_size, 50)#nn.Linear(embed_size, 50)

        self.density_hidden_dim = density_hidden_dim
        self.density_estimator_head = Density_Block(self.num_grid, density_hidden_dim, isbias=1)

        # construct the density estimator
        density_blocks = []
        density_hidden_dim = -1
        for layer_idx, layer_cfg in enumerate(cfg_density):
            # fc layer
            if layer_idx == 0:
                # weight connected to feature
                self.feature_weight = nn.Linear(in_features=layer_cfg[0], out_features=layer_cfg[1], bias=layer_cfg[2])
                density_blocks.append(self.feature_weight)
            else:
                density_blocks.append(nn.Linear(in_features=layer_cfg[0], out_features=layer_cfg[1], bias=layer_cfg[2]))
            density_hidden_dim = layer_cfg[1]
            if layer_cfg[3] == 'relu':
                density_blocks.append(nn.ReLU(inplace=True))
            elif layer_cfg[3] == 'tanh':
                density_blocks.append(nn.Tanh())
            elif layer_cfg[3] == 'sigmoid':
                density_blocks.append(nn.Sigmoid())
            else:
                print('No activation')

        self.hidden_features = nn.Sequential(*density_blocks)

        # construct the dynamics network
        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg)-1: # last layer
                last_layer = Att_FC(layer_cfg[0], layer_cfg[1], self.degree, self.knots, act=layer_cfg[3], isbias=layer_cfg[2], islastlayer=1, treat_emb=self.treat_emb)
            else:
                blocks.append(
                    Att_FC(layer_cfg[0], layer_cfg[1], self.degree, self.knots, act=layer_cfg[3], isbias=layer_cfg[2], islastlayer=0, treat_emb=self.treat_emb))
        blocks.append(last_layer)

        self.Q = nn.Sequential(*blocks)

    def forward(self, t, x):
        # hidden = self.feature_weight(x)
        # hidden = self.encoder(hidden)
        # hidden = torch.mean(hidden, dim=1).squeeze()
        # hidden = self.linear1(hidden)
        hidden = self.hidden_features(x)
        t_hidden = torch.cat((t.view(hidden.shape[0],1), hidden), 1)
        g = None#self.density_estimator_head(t, hidden)
        Q = self.Q(t_hidden)

        return g, Q

    def _initialize_weights(self):
        # TODO: maybe add more distribution for initialization
        for m in self.modules():
            if isinstance(m, Att_FC):
                m.weight.data.normal_(0, 1.)
                if m.isbias:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.in_features == 1:
                    continue
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, Density_Block):
                m.weight.data.normal_(0, 0.01)
                if m.isbias:
                    m.bias.data.zero_()

class Att_FC_V3(nn.Module):
    def __init__(self, ind, outd, degree, knots, act='relu', isbias=1, islastlayer=0, treat_emb=None):
        super(Att_FC_V3, self).__init__()
        self.ind = ind
        self.outd = outd

        self.islastlayer = islastlayer
        self.treat_emb = treat_emb

        self.isbias = isbias

        self.d = treat_emb.emb_size # num of basis

        self.weight = nn.Parameter(torch.rand(self.ind, self.outd, self.d), requires_grad=True)

        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd, self.d), requires_grad=True)
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
        x_feature = x[:, 1:]
        x_treat = x[:, 0]

        x_feature_weight = torch.matmul(self.weight.T, x_feature.T).T # bs, outd, d

        x_treat_basis = self.treat_emb(x_treat)# bs, d
        x_treat_basis_ = torch.unsqueeze(x_treat_basis, 1)

        # x_feature_weight * x_treat_basis; bs, outd, d
        out = torch.sum(x_feature_weight * x_treat_basis_, dim=2) # bs, outd

        if self.isbias:
            out_bias = torch.matmul(self.bias, x_treat_basis.T).T
            out = out + out_bias

        if self.act is not None:
            out = self.act(out)

        # concat the treatment for intermediate layer
        if not self.islastlayer:
            out = torch.cat((torch.unsqueeze(x_treat, 1), out), 1)

        return out

# Repalce dynamic-Q by feature embeddings, it works well
class VcnetAttv2(nn.Module):
    def __init__(self, embed_size = 10, num_t=1, num_cov=25, num_heads=2, att_layers=1, dropout=0.1,init_range_f=0.1, init_range_t=0.2):
        super(VcnetAttv2, self).__init__()
        """
        cfg_density: cfg for the density estimator; [(ind1, outd1, isbias1), 'act', ....]; the cfg for density estimator head is not included
        num_grid: how many grid used for the density estimator head
        """

        # cfg/cfg_density = [(ind1, outd1, isbias1, activation),....]
        self.att_layers = att_layers

        # construct the density estimator
        density_blocks = []
        density_hidden_dim = -1
        
        
        self.feature_weight = Embeddings(embed_size, initrange=init_range_f)
        self.treat_emb = Embeddings(embed_size, act='id', initrange=init_range_t)

        encoder_layers = TransformerEncoderLayer(embed_size, nhead=num_heads, dim_feedforward=50, dropout=dropout, num_cov=num_cov)
        self.encoder = TransformerEncoder(encoder_layers, att_layers)

        decoder_layers = TransformerDecoderLayer(embed_size, nhead=num_heads, dim_feedforward=50, dropout=dropout,num_t=num_t)
        self.decoder = TransformerDecoder(decoder_layers, att_layers)


        self.Q = nn.Linear(embed_size, 1)

    def forward(self, t, x):
        hidden = self.feature_weight(x)
        memory = self.encoder(hidden)

        tgt = self.treat_emb(t)
        if len(tgt.shape) < 3:
            tgt = tgt.unsqueeze(1)
        out = self.decoder(tgt.permute(1, 0, 2), memory.permute(1, 0, 2))

        Q = self.Q(out.squeeze(0))
        
        return torch.mean(hidden, dim=1).squeeze(), Q

    def _initialize_weights(self):
        # TODO: maybe add more distribution for initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.in_features == 1:
                    continue
                m.weight.data.normal_(0, 0.1)
                if m.bias is not None:
                    m.bias.data.zero_()