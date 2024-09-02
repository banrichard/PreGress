import torch
from torch import nn
from torch import functional as F


class CCANet(torch.nn.Module):
    def __init__(self, pattern_dim, graph_dim, hidden_dim, dropout=0.5, lam=0.1) -> None:
        super(CCANet, self).__init__()
        self.pattern_dim = pattern_dim
        self.graph_dim = graph_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lam = lam
        self.graph_layer = nn.Linear(graph_dim, hidden_dim)
        self.motif_layer = nn.Linear(pattern_dim, hidden_dim)
        self.pred_layer1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        # self.pred_layer2 = nn.Linear(self.hidden_dim, 1)
        self.ln = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(self.hidden_dim, 1)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.reset_parameter()

    def reset_parameter(self):
        for layer in [self.graph_layer, self.motif_layer, self.pred_layer1]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        for layer in [self.pred_mean, self.pred_var]:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)

    def forward(self, graph, motif):
        graph_cca = self.ln(self.graph_layer(graph))
        motif_cca = self.ln(self.motif_layer(motif))
        c = torch.mm(graph_cca.T, motif_cca)
        c1 = torch.mm(graph_cca.T, graph_cca)
        c2 = torch.mm(motif_cca.T, motif_cca)
        loss_inv = -torch.diagonal(c).sum()
        iden = torch.eye(c.shape[0]).to(c.device)
        loss_dec1 = (iden - c1).pow(2).sum()
        loss_dec2 = (iden - c2).pow(2).sum()
        y = self.pred_layer1(torch.cat([graph_cca, motif_cca], dim=1))
        y = self.act(y)
        mean = self.output(y)
        mean = F.relu(mean)
        cca_reg = loss_inv + self.lam * (loss_dec1 + loss_dec2)
        return mean, cca_reg
