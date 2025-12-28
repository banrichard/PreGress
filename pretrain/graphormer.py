from torch import nn as nn
import torch
from torch.nn.init import trunc_normal_
from pretrain.matcher import Matcher
from utils.mask import make_mask
from pretrain.base import PreTrain
from model.mlp import Mlp
from model.attention import TransformerRegressor
import torch.nn.functional as F
from model.graphconv import Graphormer


class Gphormer(nn.Module):
    def __init__(self, num_layer,
                 input_node_dim=1,
                 node_dim=64,
                 input_edge_dim=1,
                 edge_dim=64,
                 output_dim=64,
                 n_heads=1,
                 ff_dim=128,
                 max_in_degree=5,
                 max_out_degree=5,
                 max_path_distance=3,
                 pretrain=False):
        super().__init__()
        self.num_layer = num_layer
        self.input_node_dim = input_node_dim
        self.output_dim = output_dim
        self.input_edge_dim = input_edge_dim
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        # self.init_emb = nn.Parameter(torch.randn(self.gnn.input_dim))
        self.gnn = Graphormer(num_layers=3, input_node_dim=self.input_node_dim, node_dim=self.node_dim,
                              input_edge_dim=self.input_edge_dim, edge_dim=self.edge_dim, output_dim=self.output_dim,
                              pretrain=True)
        self.projection_head = nn.Sequential(
            nn.Linear(self.node_dim, int(self.node_dim / 2)),
            nn.ReLU(inplace=True),
            nn.Linear(int(self.node_dim / 2), output_dim),
        )
        # from iclr20
        self.sim_loss = nn.MSELoss()

    def importance_loss(self, pred_importance, target_importance):
        return F.mse_loss(
            pred_importance.float(), target_importance.float(), reduction="mean"
        )

    def forward(self, data):
        importance = data.degree_centrality
        # generate embedding
        pred = self.gnn(data)
        pred_importance = self.projection_head(pred)
        importance_loss = self.importance_loss(pred_importance, importance)
        return pred, importance_loss
