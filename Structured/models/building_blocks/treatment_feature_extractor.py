from torch import Tensor, nn

from models.building_blocks.gnn import GNN
from models.building_blocks.mlp import MLP

class TreatmentFeatureExtractor(nn.Module):
    def __init__(self, args):
        super(TreatmentFeatureExtractor, self).__init__()
        dim_output = (
            args.dim_output if args.model == "gin" else args.dim_output_covariates
        )
        self.treatment_net = GNN(
            gnn_conv=args.gnn_conv,
            dim_input=args.dim_node_features,
            dim_hidden=args.dim_hidden_treatment,
            dim_output=dim_output,
            num_layers=args.num_treatment_layer,
            batch_norm=args.gnn_batch_norm,
            initialiser=args.initialiser,
            dropout=args.gnn_dropout,
            activation=args.activation,
            leaky_relu=args.leaky_relu,
            is_output_activation=args.output_activation_treatment_features,
            num_relations=args.gnn_num_relations,
            num_bases=args.gnn_num_bases,
            is_multi_relational=args.gnn_multirelational,
        )

    def forward(
        self,
        treatment_node_features: Tensor,
        treatment_edges: Tensor,
        edge_types: Tensor,
        batch_assignments: Tensor,
    ):
        return self.treatment_net.forward(
            nodes=treatment_node_features,
            edges=treatment_edges,
            edge_types=edge_types,
            batch_assignments=batch_assignments,
        )

class TreatmentFeatureExtractorAtt(nn.Module):
    def __init__(self, args):
        super(TreatmentFeatureExtractorAtt, self).__init__()
        dim_output = args.dim_output_treatment
        self.treatment_net = GNN(
            gnn_conv=args.gnn_conv,
            dim_input=args.dim_node_features,
            dim_hidden=args.dim_hidden_treatment,
            dim_output=dim_output,
            num_layers=args.num_treatment_layer,
            batch_norm=args.gnn_batch_norm,
            initialiser=args.initialiser,
            dropout=args.gnn_dropout,
            activation=args.activation,
            leaky_relu=args.leaky_relu,
            is_output_activation=args.output_activation_treatment_features,
            num_relations=args.gnn_num_relations,
            num_bases=args.gnn_num_bases,
            is_multi_relational=args.gnn_multirelational,
        )
        self.linear =  MLP(
            dim_input=dim_output,
            dim_hidden=dim_output,
            dim_output=args.embed_size,
            num_layers=1,
            batch_norm=args.mlp_batch_norm,
            initialiser=args.initialiser,
            dropout=args.dropout,
            activation=args.activation,
            leaky_relu=args.leaky_relu,
            is_output_activation=args.output_activation_treatment_features,
        )

    def forward(
        self,
        treatment_node_features: Tensor,
        treatment_edges: Tensor,
        edge_types: Tensor,
        batch_assignments: Tensor,
    ):
        hidden = self.treatment_net.forward(
            nodes=treatment_node_features,
            edges=treatment_edges,
            edge_types=edge_types,
            batch_assignments=batch_assignments,
        )
        return self.linear(hidden).unsqueeze(1)