import torch.nn as nn
import torch
from torch.nn.init import trunc_normal_
from utils.mask import make_mask
from base import PreTrain
from model.mlp import Mlp
from model.attention import TransformerRegressor
import torch.nn.functional as F


class GIN(PreTrain):
    def __init__(self, hid_dim):
        self.gnn_type = "GIN"
        self.load_graph_data()
        self.initialize_gnn(self.input_dim, hid_dim)
        self.hid_dim = hid_dim
        self.projection_head = nn.Sequential(nn.Linear(hid_dim, hid_dim),
                                             nn.ReLU(inplace=True),
                                             nn.Linear(hid_dim, hid_dim))

    @torch.no_grad()
    def momentum_update(self, base_momentum=0):
        """Momentum update of the teacher network."""
        for param_encoder, param_teacher in zip(self.student.parameters(),
                                                self.teacher.parameters()):
            param_teacher.data = param_teacher.data * base_momentum + \
                                 param_encoder.data * (1. - base_momentum)

    def build_regressor(self):
        self.mask_regressor = TransformerRegressor(embed_dim=self.hid_dim,
                                                   depth=self.regressor_depth,
                                                   drop_path_rate=0.1,
                                                   num_heads=self.regressor_num_heads)

    def build_masked_decoder(self):
        if self.mask_ratio > 0.:
            # print_log(f'[Point-RAE] build masked decoder for feature prediction ...', logger='Point-RAE')
            self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.decoder_pos_embed = nn.Sequential(
                nn.Linear(3, 128),
                nn.GELU(),
                nn.Linear(128, self.embed_dim)
            )
            dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
            self.RAE_decoder = Mlp(
                in_features=self.output_dim,
                out_features=self.output_dim
            )
            trunc_normal_(self.mask_token, std=.02)
        else:
            self.mask_token = None
            self.RAE_decoder = None

    def importance_loss(self, pred_importance, target_importance):
        return F.mse_loss(pred_importance, target_importance, reduction='mean')

    def similarity_loss(self, pred_feat, orig_feat):
        return F.cosine_similarity(pred_feat, orig_feat, dim=1)

    def forward(self, data, use_mask=None):
        x, edge_index, edge_attr, importance = data.x, data.edge_index, data.edge_attr, data.degree_centrality
        if use_mask is not None:
            mask = make_mask(x)
        else:
            mask = None
        # generate embedding
        pred = self.gnn(x, edge_index, edge_attr)
        pred_importance = self.projection_head(pred)
        importance_loss = self.importance_loss(pred_importance, importance)
        batch_size, num_nodes, channel = pred.shape
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.hid_dim))

        if self.mask_token is not None:
            pos_emd_vis = self.projection_head(pred[~mask]).reshape(batch_size, -1, channel)
            pos_emd_mask = self.projection_head(pred[mask]).reshape(batch_size, -1, channel)
            _, num_mask, _ = pos_emd_mask.shape
            mask_token = self.mask_token.expand(batch_size, num_mask, -1)
            x_full = torch.cat([pred, mask_token], dim=1)
            pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

            latent_pred = self.mask_regressor(mask_token, pred, pos_emd_mask, pos_emd_vis, mask)
            # temporarily can not find a good solution to solve the attr loss, current is cossimilarity
            attr_loss = self.attr_loss(latent_pred, pred)
