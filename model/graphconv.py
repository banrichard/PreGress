import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
import torch.nn.functional as F


class Backbone(nn.Module):
    def __init__(self, type, num_layers, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.type = type
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.conv = self.build_conv_layers()
        self.conv_layers = nn.ModuleList()
        for l in range(self.num_layers):
            hidden_input_dim = self.input_dim if l == 0 else self.hidden_dim
            hidden_output_dim = self.hidden_dim
            self.conv_layers.append(self.conv(hidden_input_dim, hidden_output_dim))

    def build_conv_layers(self):
        if self.type == "GCN":
            return GCNConv
        elif self.type == "GAT":
            return GATConv
        elif self.type == "SAGE":
            return SAGEConv
        elif self.type == "GIN":
            return lambda in_ch, hid_ch: GINConv(nn=nn.Sequential(
                nn.Linear(in_ch, hid_ch), nn.ReLU(), nn.Linear(hid_ch, hid_ch)), train_eps=True)
        else:
            raise NotImplementedError("Current do not support!")

    def forward(self, x, edge_index, edge_attr):
        for i in range(self.num_layers):
            if self.model_type == "GIN" or self.model_type == "GINE" or self.model_type == "GAT" \
                    or self.model_type == "SAGE":
                x = self.convs[i](x, edge_index)  # for GIN and GINE
            elif self.model_type == "Graph" or self.model_type == "GCN":
                x = self.convs[i](x, edge_index, edge_weight=edge_attr)
            elif self.model_type == "NN" or self.model_type == "NNGIN" or self.model_type == "NNGINConcat":
                x = self.convs[i](x=x, edge_index=edge_index, edge_attr=edge_attr)
            else:
                print("Unsupported model type!")

            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(x)
        return x
