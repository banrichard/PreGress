import torch.nn as nn
import torch
from torch_geometric.nn import global_mean_pool

from model.graphconv import Backbone
from model.attention import TransformerRegressor
import torch.nn.functional as F


class BaseGNN(nn.Module):
    def __init__(self, type="GCN", num_layer=5, input_dim=11, hid_dim=32, output_dim=16, dropout=0.2):
        super().__init__()
        self.num_layer = num_layer
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.type = type
        self.gnn = self.initialize_gnn()
        self.projection_head = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, 1),
        )
        self.prompt = nn.Parameter(torch.Tensor(1, self.hid_dim), requires_grad=True)
        self.prompt = nn.init.kaiming_uniform(self.prompt)

    def initialize_gnn(self):
        self.gnn = Backbone(self.type, self.num_layer, self.input_dim, self.hid_dim, self.output_dim)
        return self.gnn

    def forward(self, data):
        x, edge_index, edge_attr, importance, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.eigenvector_centrality,
            data.batch
        )
        # generate embedding
        pred = self.gnn(x, edge_index, edge_attr)
        pred = pred + self.prompt
        pred_importance = self.projection_head(pred)
        importance_loss = F.l1_loss(pred_importance.squeeze(), importance)

        return importance_loss
