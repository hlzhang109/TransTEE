from torch import Tensor, nn

from models.building_blocks.mlp import MLP
from models.building_blocks.transformers import *

class OutcomeModel(nn.Module):
    def __init__(self, args):
        super(OutcomeModel, self).__init__()
        self.outcome_net = MLP(
            dim_input=args.dim_output_treatment + args.dim_output_covariates,
            dim_hidden=args.dim_hidden_covariates,
            dim_output=1,
            num_layers=args.num_final_ff_layer,
            batch_norm=args.mlp_batch_norm,
            initialiser=args.initialiser,
            dropout=args.dropout,
            activation=args.activation,
            leaky_relu=args.leaky_relu,
            is_output_activation=False,
        )

    def forward(self, treatment_and_unit_features: Tensor):
        return self.outcome_net(treatment_and_unit_features)

class OutcomeModelAtt(nn.Module):
    def __init__(self, args):
        super(OutcomeModelAtt, self).__init__()

        decoder_layers = TransformerDecoderLayer(args.embed_size, nhead=args.num_heads, dim_feedforward=args.dim_hidden_covariates, dropout=args.dropout,num_t=1)
        self.decoder = TransformerDecoder(decoder_layers, args.num_atten_layer)
        self.outcome_net = nn.Linear(args.embed_size, 1)

    def forward(self, treatment_f, covarities_f):
        out = self.decoder(treatment_f.permute(1, 0, 2), covarities_f.permute(1, 0, 2))
        return self.outcome_net(out.squeeze(0))
