import torch.nn as nn
import torch
from fvcore.nn import FlopCountAnalysis
from thop import profile
from torch_geometric.nn import global_mean_pool

from model.graphconv import Backbone, DeeperGCN, JKNet
from model.attention import TransformerRegressor
import torch.nn.functional as F


# from pthflops import count_ops

class BaseGNN(nn.Module):
    def __init__(self, type="GCN", num_layer=5, input_dim=11, hid_dim=32, output_dim=16, dropout=0.2,eval_term = "betweenness",task='regression',num_classes=7):
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
            nn.ReLU(inplace=True)
        )
        self.task = task
        if self.task == 'classification':
            self.class_head = nn.Sequential(
                nn.Linear(hid_dim, hid_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(hid_dim, num_classes)
            )
        self.prompt = nn.Parameter(torch.Tensor(1, self.hid_dim), requires_grad=True)
        self.prompt = nn.init.kaiming_uniform(self.prompt)
        self.eval_term = eval_term
        self.printed_flops_params = False
        self.task = task
    def initialize_gnn(self):
        if self.type in ["GAT", "GCN", "Transformer", "GIN", "SAGE", "DirGNN","APPNP","SG"]:
            self.gnn = Backbone(self.type, self.num_layer, self.input_dim, self.hid_dim, self.output_dim)
        elif self.type in ["DeepGCN"]:
            self.gnn = DeeperGCN(self.num_layer, self.input_dim, self.hid_dim, self.output_dim,edge_input_dim=1)
        elif self.type in ["JKNet"]:
            self.gnn = JKNet(self.num_layer, self.input_dim, self.hid_dim, self.output_dim)
        return self.gnn

    def forward(self, data):
        x, edge_index, edge_attr = (
            data.x,
            data.edge_index,
            data.edge_attr
        )
        # generate embedding
        x = x.unsqueeze(1) if x.dim() == 1 else x

        pred = self.gnn(x, edge_index, edge_attr)
        # pred = global_mean_pool(pred, batch)
        # print("FLOPs",FlopCountAnalysis(self.gnn, (x, edge_index)).total())
        # pred = pred + self.prompt
        if self.task == 'regression':
            pred_importance = self.projection_head(pred)
            if self.eval_term == "betweenness":
                importance = data.betweenness_centrality if hasattr(data, "betweenness_centrality") else data.pagerank
            elif self.eval_term == "eigenvector":
                importance = data.eigenvector_centrality if hasattr(data, "eigenvector_centrality") else data.pagerank
            else:
                importance = data.pagerank
            importance_loss = F.l1_loss(pred_importance, importance)

            return importance_loss, pred_importance
        else:
            logits = self.class_head(pred)  # (N, num_classes) or (N, 1)
            return logits
