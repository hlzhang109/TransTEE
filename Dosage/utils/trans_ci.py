import torch
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import nn, Tensor
import torch.nn.functional as F



# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class DosageEncoding(nn.Module):
    def __init__(self,
                 emb_size: int = 50,
                 dropout: float = 0.,
                 maxlen: int = 5000):
        super(DosageEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input covariate variables into corresponding tensor of covariate embeddings

class Embeddings(nn.Module):
    def __init__(self, emb_size, act=None, initrange=0.01, res=0):
        super(Embeddings, self).__init__()
        self.treat_weight = nn.Linear(1, emb_size)
        self.initrange = initrange
        self.res = res
        if res:
            self.emb_size = emb_size + 1
        else:
            self.emb_size = emb_size
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = None
        self.init_weights()

    def forward(self, tokens):
        ebd = self.treat_weight(tokens.unsqueeze(-1).to(torch.float32))
        if self.res:
            ebd = torch.cat([torch.ones(ebd.shape[0],1).cuda(), ebd], dim=-1)
        if self.act is None:
            return ebd
        return self.act(ebd)

    def init_weights(self) -> None:
        self.treat_weight.weight.data.normal_(-self.initrange, self.initrange)
        self.treat_weight.bias.data.zero_()

class TransformerModel(nn.Module):
    
    def __init__(self, ntoken: int, d_model: int = 50, nhead: int = 5, d_hid: int = 50, nlayers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.model_type = 'Transformer'

        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]
        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output