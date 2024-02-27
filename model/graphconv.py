import torch.nn as nn
from torch_geometric.nn import GCNConv,GATConv,SAGEConv,GINConv

class Backbone(nn.Module):
    def __init__(self,type,num_layers,input_dim,hidden_dim,output_dim):
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
    