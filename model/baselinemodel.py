import torch.nn as nn
import torch
from torch_geometric.nn import global_mean_pool

from graphconv import Backbone
from model.attention import TransformerRegressor
import torch.nn.functional as F

class BaseGNN(Backbone):
    def __init__(self, num_layer=5, input_dim=11, hid_dim=32, output_dim=16, dropout=0.2):
        super().__init__("SAGE", dropout=dropout)
        self.num_layer = num_layer
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.initialize_gnn(self.input_dim, self.hid_dim)
        self.init_emb = nn.Parameter(torch.randn(self.gnn.input_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, self.hid_dim))
        self.projection_head = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, 1),
        )
        # from iclr20
        self.pos_decoder = nn.Linear(self.hid_dim, self.hid_dim)
        self.matcher = nn.Linear(self.hid_dim, self.input_dim)
        self.build_regressor()
        self.sim_loss = nn.MSELoss()

