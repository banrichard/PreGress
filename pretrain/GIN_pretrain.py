import torch.nn as nn
import torch
from torch.nn import BCEWithLogitsLoss
from torch_geometric.nn import global_mean_pool

from utils.mask import make_mask
from pretrain.base import PreTrain
from model.attention import TransformerRegressor
import torch.nn.functional as F

from utils.preprocess import get_negative_edges


class GraphTrainer(PreTrain):
    def __init__(self, num_layer=5, input_dim=11, hid_dim=32, output_dim=16, dropout=0.2, classification=False,
                 enhance=False,attr_only=False):
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
            nn.ReLU(inplace=True)
        )
        # from iclr20
        self.pos_decoder = nn.Linear(self.hid_dim, self.hid_dim)
        self.matcher = nn.Linear(self.hid_dim, self.input_dim)
        self.build_regressor()
        self.sim_loss = nn.MSELoss()
        self.classification = classification
        self.enhance = enhance
        self.attr_only = attr_only
    def build_regressor(self):
        self.mask_regressor = TransformerRegressor(
            embed_dim=self.hid_dim,
            drop_path_rate=0.1,
        )

    def importance_loss(self, pred_importance, target_importance):
        return F.mse_loss(
            pred_importance.float(), target_importance.float(), reduction="mean"
        )

    def similarity_loss(self, pred_feat, orig_feat):
        return F.mse_loss(
            pred_feat.float(), orig_feat.float(), reduction="mean"
        )

    def classification_loss(self, pred_importance, target_importance):
        pred_importance = F.log_softmax(pred_importance)
        return F.cross_entropy(pred_importance, target_importance)

    def forward(self, data, edge_label=None, enhanced_edge_index=None, use_mask=True, attr_only=False):
        x, edge_index, edge_attr, importance, batch = (
            data.x.squeeze(1),
            data.edge_index,
            data.edge_attr,
            data.y_dc,
            data.batch if data.batch is not None else None
        )
        # generate embedding
        if not self.classification:
            if self.attr_only:
                mask = make_mask(x)
                pred = self.gnn(x, edge_index, edge_attr)
                pos_emd_vis = self.pos_decoder(pred[mask])
                pos_emd_mask = self.pos_decoder(pred[~mask])
                num_mask, _ = pos_emd_mask.shape
                mask_token = self.mask_token.expand(num_mask, -1)
                if mask_token.size(0) == 0:
                    return 0.0, 0.0
                pred_attr = self.mask_regressor(
                    mask_token, pred[mask], pos_emd_mask, pos_emd_vis, mask
                )
                pred_attr = self.matcher(pred_attr)
                attr_loss = self.similarity_loss(pred_attr, data.x[~mask])
                return None, attr_loss
            else:
                pred = self.gnn(x, edge_index, edge_attr)
                pred_pooled = global_mean_pool(pred, batch)
                pred_importance = self.projection_head(pred_pooled)
                importance_loss = self.importance_loss(pred_importance.squeeze(), importance)
                if self.enhance:
                    mask = make_mask(x)
                else:
                    mask = None
                    return importance_loss, None
                if mask is not None:
                    pos_emd_vis = self.pos_decoder(pred[mask])
                    pos_emd_mask = self.pos_decoder(pred[~mask])
                    num_mask, _ = pos_emd_mask.shape
                    mask_token = self.mask_token.expand(num_mask, -1)
                    if mask_token.size(0) == 0:
                        return 0.0, 0.0
                    pred_attr = self.mask_regressor(
                        mask_token, pred[mask], pos_emd_mask, pos_emd_vis, mask
                    )
                    pred_attr = self.matcher(pred_attr)
                    attr_loss = self.similarity_loss(pred_attr, data.x[~mask])

                    return importance_loss, attr_loss
        else:
            # pos_edge_index = edge_index
            # neg_edge_index = get_negative_edges(pos_edge_index, len(x), len(edge_index))
            pred = self.gnn(x, edge_index, edge_attr)
            enhanced_edge_index = enhanced_edge_index.transpose(0, 1)
            pos_score = (pred[enhanced_edge_index[0]] * pred[enhanced_edge_index[1]]).sum(dim=-1).view(-1)
            importance_loss = F.binary_cross_entropy_with_logits(pos_score, edge_label)
            if self.enhance:
                mask = make_mask(x)
                pos_emd_vis = self.pos_decoder(pred[mask])
                pos_emd_mask = self.pos_decoder(pred[~mask])
                num_mask, _ = pos_emd_mask.shape
                mask_token = self.mask_token.expand(num_mask, -1)
                pred_attr = self.mask_regressor(
                    mask_token, pred[mask], pos_emd_mask, pos_emd_vis, mask
                )
                pred_attr = self.matcher(pred_attr)
                attr_loss = self.similarity_loss(pred_attr, data.x[~mask])

                return importance_loss, attr_loss
            else:
                return importance_loss, None
