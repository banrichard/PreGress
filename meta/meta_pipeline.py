import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, TransformerConv
from torch_geometric.utils import sort_edge_index, add_self_loops, to_undirected
from meta.motifNN import MotifGNN
from meta.maml_learner import MAML
from model.graphconv import Backbone
from torch_geometric.data import Batch


def model_components(args, round=1, pre_train_path='', gnn_type='GIN', project_head_path=None):
    if round == 1:
        model = Pipeline(input_dim=11, pre_train_path="../saved_model/best_epoch_GIN.pt")
    elif round == 2 or round == 3:
        if project_head_path is None:
            raise ValueError("project_head_path is None! it should be a specific path when round=2 or 3")
        model = Pipeline(input_dim=11, pre_train_path="../saved_model/best_epoch_GIN.pt",
                         project_head_path=project_head_path)
    else:
        raise ValueError('round value wrong! (it should be 1,2,3)')

    maml = MAML(model, lr=args.adapt_lr, first_order=False, allow_nograd=True)
    opt = optim.Adam(filter(lambda p: p.requires_grad, maml.parameters()), args.meta_lr)
    lossfn = nn.MSELoss()

    return maml, opt, lossfn


class Pipeline(torch.nn.Module):
    def __init__(self, input_dim, pre_train_path=None, layer_num=3, hid_dim=128, frozen_gnn='all',
                 frozen_project_head=False, pool_mode=0, gnn_type='GIN', project_head_path=None):

        super().__init__()
        self.pool_mode = pool_mode
        self.gnn = Backbone(type=gnn_type, num_layers=layer_num, input_dim=input_dim, hidden_dim=hid_dim,
                            output_dim=hid_dim)
        self.motifnn = MotifGNN(num_layers=layer_num, num_g_hid=1, num_e_hid=1, out_g_ch=hid_dim, model_type="NNGINConcat",
                                dropout=0.2)
        self.project_head = torch.nn.Sequential(
            torch.nn.Linear(hid_dim * 2, 1),
            torch.nn.ReLU())
        self.with_prompt = False
        self.set_gnn_project_head(pre_train_path, frozen_gnn, frozen_project_head, project_head_path)

    def set_gnn_project_head(self, pre_train_path, frozen_gnn, frozen_project_head, project_head_path=None):
        if pre_train_path:
            self.gnn.load_state_dict(torch.load(pre_train_path), strict=False)
            print("successfully load pre-trained weights for gnn! @ {}".format(pre_train_path))

        if project_head_path:
            self.project_head.load_state_dict(torch.load(project_head_path))
            print("successfully load project_head! @ {}".format(project_head_path))

        if frozen_gnn == 'all':
            for p in self.gnn.parameters():
                p.requires_grad = False
        elif frozen_gnn == 'none':
            for p in self.gnn.parameters():
                p.requires_grad = True
        else:
            pass

        if frozen_project_head:
            for p in self.project_head.parameters():
                p.requires_grad = False

    def forward(self, graph_batch: Batch, motif_batch):
        # num_graphs = graph_batch.num_graphs
        if self.with_prompt:
            xp, xp_edge_index, batch_one, batch_two = self.prompt(graph_batch)

            if self.pool_mode == 1:
                graph_emb = self.gnn(xp, xp_edge_index, batch_one)
                pre = self.project_head(graph_emb)
                return pre
            # elif self.pool_mode == 2:
            #     emb = self.gnn(xp, xp_edge_index, batch_two)
            #     graph_emb = emb[0:num_graphs, :]
            #     prompt_emb = emb[num_graphs:, :]
            #     com_emb = graph_emb - prompt_emb
            #     pre = self.project_head(com_emb)
            #     return pre
        else:
            graph_emb = self.gnn(graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr)
            motif_emb = self.motifnn(motif_batch.x, motif_batch.edge_index, motif_batch.edge_attr)
            final_emb = torch.cat([graph_emb, motif_emb], dim=1)
            pre = self.project_head(final_emb)
            return pre


if __name__ == "__main__":
    model = Pipeline(input_dim=11, pre_train_path="../saved_model/best_epoch_GIN.pt")
    print(model)
