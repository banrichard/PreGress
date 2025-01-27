import torch
import torch.nn as nn
from torch import optim
# from torch_geometric.utils import sort_edge_index, add_self_loops, to_undirected
from model.motifNN import MotifGNN
from meta.maml_learner import MAML
from model.graphconv import Backbone, Graphormer
from torch_geometric.data import Batch


def model_components(args, round=1, pre_train_path='', gnn_type='GIN', project_head_path=None,task='counting'):
    if round == 1:
        if task == "counting":
            model = Pipeline(input_dim=11, layer_num=args.layer_num, pre_train_path="../saved_model/best_epoch_GIN.pt")
        elif task == 'importance':
            model = ImportancePipeline(input_dim=11, layer_num=args.layer_num, pre_train_path="../saved_model/best_epoch_GIN.pt")
    elif round == 2 or round == 3:
        if project_head_path is None:
            raise ValueError("project_head_path is None! it should be a specific path when round=2 or 3")
        model = Pipeline(input_dim=11, layer_num=args.layer_num, pre_train_path="../saved_model/best_epoch_GIN.pt",
                         project_head_path=project_head_path)
    else:
        raise ValueError('round value wrong! (it should be 1,2,3)')

    maml = MAML(model, lr=args.adapt_lr, first_order=False, allow_nograd=True)
    opt = optim.Adam(filter(lambda p: p.requires_grad, maml.parameters()), args.meta_lr)
    lossfn = nn.SmoothL1Loss(reduction="sum")

    return maml, opt, lossfn


class Pipeline(torch.nn.Module):
    def __init__(self, input_dim, pre_train_path=None, layer_num=3, hid_dim=64, frozen_gnn='all',
                 frozen_project_head=False, pool_mode=0, gnn_type='GIN', mnn_type="graphormer", project_head_path=None,
                 m_layer_num=3):

        super().__init__()
        self.pool_mode = pool_mode
        self.mnn_type = mnn_type
        self.norm = nn.LayerNorm(hid_dim)
        self.gnn = Backbone(type=gnn_type, num_layers=layer_num, input_dim=input_dim, hidden_dim=hid_dim,
                            output_dim=hid_dim)
        if self.mnn_type == "graphormer":
            self.motifnn = Graphormer(num_layers=m_layer_num)
        else:
            self.motifnn = MotifGNN(num_layers=m_layer_num, num_g_hid=64, num_e_hid=64, out_g_ch=hid_dim,
                                    model_type="NNGINConcat",
                                    dropout=0.2)
        self.project_head = torch.nn.Sequential(
            torch.nn.Linear(hid_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
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

    def to_cuda(self):
        self.motifnn = self.motifnn.cuda()
        self.project_head = self.project_head.cuda()

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
            if self.mnn_type == "graphormer":
                motif_emb = self.motifnn(motif_batch)
            else:
                motif_emb = self.motifnn(motif_batch.x, motif_batch.edge_index, motif_batch.edge_attr)
            # graph_emb = global_mean_pool(graph_emb, batch=graph_batch.batch)
            graph_emb = self.norm(graph_emb)
            # final_emb = torch.cat([graph_emb, motif_emb], dim=1)
            graph_emb += motif_emb  # x = x + p
            pre = self.project_head(graph_emb)
            return pre


class ImportancePipeline(torch.nn.Module):
    def __init__(self, input_dim, pre_train_path=None, layer_num=3, hid_dim=64, frozen_gnn='all',
                 frozen_project_head=False, gnn_type='GIN', project_head_path=None):
        super().__init__()
        self.norm = nn.LayerNorm(hid_dim)
        self.gnn = Backbone(type=gnn_type, num_layers=layer_num, input_dim=input_dim, hidden_dim=hid_dim,
                            output_dim=hid_dim)
        self.project_head = torch.nn.Sequential(
            torch.nn.Linear(hid_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
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

    def forward(self, graph_batch: Batch):
        graph_emb = self.gnn(graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr)
        # graph_emb = global_mean_pool(graph_emb, batch=graph_batch.batch)
        graph_emb = self.norm(graph_emb)
        pre = self.project_head(graph_emb)
        return pre


if __name__ == "__main__":
    model = ImportancePipeline(input_dim=11, layer_num=5, pre_train_path="../saved_model/best_epoch_GIN.pt")
    print(model)
