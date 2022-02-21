from torch import nn

from models.building_blocks.mlp import MLP
from models.building_blocks.transformers import *


class CovariatesFeatureExtractor(nn.Module):
    def __init__(self, args):
        super(CovariatesFeatureExtractor, self).__init__()
        dim_output = (
            args.dim_output if args.model == "gin" else args.dim_output_covariates
        )
        self.covariates_net = MLP(
            dim_input=args.dim_covariates,
            dim_hidden=args.dim_hidden_covariates,
            dim_output=dim_output,
            num_layers=args.num_covariates_layer,
            batch_norm=args.mlp_batch_norm,
            initialiser=args.initialiser,
            dropout=args.dropout,
            activation=args.activation,
            leaky_relu=args.leaky_relu,
            is_output_activation=args.output_activation_treatment_features,
        )

    def forward(self, unit):
        return self.covariates_net(unit)

class CovariatesFeatureExtractorAtt(nn.Module):
    def __init__(self, args):
        super(CovariatesFeatureExtractorAtt, self).__init__()
        dim_output = (
            args.dim_covariates if args.task == "sw" else args.dim_output_covariates
        )
        self.task = args.task
        if self.task == 'tcga':
            self.linear = MLP(
                dim_input=args.dim_covariates,
                dim_hidden=args.dim_hidden_covariates,
                dim_output=dim_output,
                num_layers=1,
                batch_norm=args.mlp_batch_norm,
                initialiser=args.initialiser,
                dropout=args.dropout,
                activation=args.activation,
                leaky_relu=args.leaky_relu,
                is_output_activation=False,
            )
        self.feature_weight = Embeddings(args.embed_size, initrange=args.init_range_f)
        encoder_layers = TransformerEncoderLayer(args.embed_size, nhead=args.num_heads, dim_feedforward=args.dim_hidden_covariates, dropout=args.dropout, num_cov=dim_output)
        self.covariates_net = TransformerEncoder(encoder_layers, args.num_atten_layer)

    def forward(self, unit):
        if self.task == 'tcga':
            unit = self.linear(unit)
        hidden = self.feature_weight(unit)
        return self.covariates_net(hidden)