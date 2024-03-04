import torch.nn as nn
from base import PreTrain


class GIN(PreTrain):
    def __init__(self, hid_dim):
        self.gnn_type = "GIN"
        self.load_graph_data()
        self.initialize_gnn(self.input_dim, hid_dim)
        self.hid_dim = hid_dim
        self.projection_head = nn.Sequential(nn.Linear(hid_dim, hid_dim),
                                             nn.ReLU(inplace=True),
                                             nn.Linear(hid_dim, hid_dim))
