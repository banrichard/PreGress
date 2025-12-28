import torch
from dgl import graph
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

from model.adapterGNN import AdapterGNN
from model.graphconv import Backbone, Graphormer
from model.mlp import Mlp
from model.motifNN import MotifGNN
from model.regularization import CCANet
from thop import profile
from fvcore.nn import FlopCountAnalysis


class Pipeline(torch.nn.Module):
    def __init__(self, input_dim, pre_train_path=None, layer_num=3, hid_dim=64, frozen_gnn='first_layer',
                 frozen_project_head=False, frozen_prompt=False, pool_mode=0, gnn_type='GIN', mnn_type="ginconcat",
                 project_head_path=None,
                 m_layer_num=3,
                 without_snn=False,
                 without_prompt=False,
                 gnn_name="NGNN"):

        super().__init__()
        self.pool_mode = pool_mode
        self.mnn_type = mnn_type
        self.frozen_prompt = frozen_prompt
        self.norm = nn.LayerNorm(hid_dim)
        self.without_prompt = without_prompt
        self.gnn_name = gnn_name
        self.task_head_used = False
        if without_snn:
            self.gnn = MotifGNN(num_layers=layer_num, num_g_hid=hid_dim,
                                num_e_hid=32, out_g_ch=hid_dim, model_type="GIN",
                                dropout=0.2)
        else:
            if self.gnn_name == "adapter":
                self.task_head = torch.nn.Linear(input_dim, hid_dim)
                self.gnn = AdapterGNN(num_layer=layer_num, emb_dim=hid_dim)
            else:
                self.gnn = Backbone(type=gnn_type, num_layers=layer_num, input_dim=input_dim, hidden_dim=hid_dim,
                                    output_dim=hid_dim)
        if self.mnn_type == "graphormer":
            self.motifnn = Graphormer(num_layers=m_layer_num)
        elif self.mnn_type == "vector":
            self.motifnn = nn.Parameter(torch.Tensor(1, hid_dim))
        else:
            self.motifnn = MotifGNN(num_layers=m_layer_num, num_g_hid=32, num_e_hid=32, out_g_ch=hid_dim,
                                    model_type="NNGINConcat",
                                    dropout=0.2)
        if frozen_prompt:
            for p in self.motifnn.parameters():
                p.requires_grad = False
        if without_prompt:
            self.projection_head = Mlp(hid_dim, hid_dim, 1, drop=0.5)
        else:
            self.projection_head = CCANet(hid_dim, hid_dim, 32)
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
        elif frozen_gnn == 'first_layer':
            for name, param in self.gnn.named_parameters():
                if 'convs.0' in name:  # Assuming the first layer is the first parameter set
                    param.requires_grad = True
                else:
                    param.requires_grad = False
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
        if self.gnn_name == "adapter" and self.training and not self.task_head_used:
            graph_batch.x = self.task_head(graph_batch.x.to(torch.float32))
            self.task_head_used = True
            # graph_batch.x = torch.zeros_like(graph_batch.x)
        graph_emb = self.gnn(graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr)
        if self.without_prompt:
            graph_emb = global_mean_pool(graph_emb, batch=graph_batch.batch)
            reg = 0
            pre = self.projection_head(graph_emb)
        else:
            if self.mnn_type == "graphormer":
                motif_emb = self.motifnn(motif_batch)
            elif self.mnn_type == "vector":
                motif_emb = self.motifnn
            else:
                motif_emb = self.motifnn(motif_batch.x, motif_batch.edge_index, motif_batch.edge_attr)
            # graph_emb = global_mean_pool(graph_emb, batch=graph_batch.batch)
            graph_emb = self.norm(graph_emb)
            # final_emb = torch.cat([graph_emb, motif_emb], dim=1)
            graph_emb += motif_emb  # x = x + p
            pre, reg = self.projection_head(graph_emb, motif_emb)
        return pre, reg


class Prompt(torch.nn.Module):
    def __init__(self, hid_dim=64):
        super().__init__()
        self.prompt = nn.Parameter(torch.Tensor(1, hid_dim))
        self.reset_parameters()

    def reset_parameters(self):
        self.prompt = torch.nn.init.kaiming_uniform_(self.prompt)

    def forward(self, graph_emb):
        graph_emb = self.prompt + graph_emb
        return graph_emb


class ImportancePipeline(torch.nn.Module):
    def __init__(self, input_dim, pre_train_path=None, layer_num=3, hid_dim=32, frozen_gnn='first_layer',
                 frozen_prompt=False, frozen_project_head=False, gnn_type='GIN', project_head_path=None,
                 gnn_name="NGNN"):
        super().__init__()
        self.norm = nn.LayerNorm(hid_dim)
        self.gnn_name = gnn_name
        if self.gnn_name == "adapter":
            self.task_head = torch.nn.Linear(input_dim, hid_dim)
            self.gnn = AdapterGNN(num_layer=layer_num, emb_dim=hid_dim)
        else:
            self.gnn = Backbone(type=gnn_type, num_layers=layer_num, input_dim=input_dim, hidden_dim=hid_dim,
                                output_dim=hid_dim)
        self.prompt = Prompt(hid_dim)
        self.project_head = torch.nn.Sequential(
            torch.nn.Linear(hid_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.ReLU())
        self.set_gnn_project_head(pre_train_path, frozen_gnn, frozen_project_head, frozen_prompt, project_head_path)

    def set_gnn_project_head(self, pre_train_path, frozen_gnn, frozen_project_head, frozen_prompt,
                             project_head_path=None):
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
        elif frozen_gnn == 'first_layer':
            for name, param in self.gnn.named_parameters():
                if 'convs.0' in name:  # Assuming the first layer is the first parameter set
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            pass

        if frozen_project_head:
            for p in self.project_head.parameters():
                p.requires_grad = False
        if frozen_prompt:
            self.prompt.requires_grad = False

    def forward(self, graph_batch: Batch, print_verbose=False):
        graph_batch.x = self.task_head(graph_batch.x.to(torch.float32)) if self.gnn_name == "adapter" else graph_batch.x
        graph_emb = self.gnn(graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr)
        # graph_emb = global_mean_pool(graph_emb, batch=graph_batch.batch)
        graph_emb = self.norm(graph_emb)
        graph_emb = self.prompt(graph_emb)
        # if not print_verbose:
        #     print("FLOPS:", FlopCountAnalysis(self.prompt, inputs=(graph_emb,)).total())
        #     print_verbose = True
        # graph_emb = global_mean_pool(graph_emb, batch=graph_batch.batch)
        pre = self.project_head(graph_emb)
        pre = global_mean_pool(pre, batch=graph_batch.batch)
        # print("FLOPS:{:f}, Params:{:f}".format(*profile(self.project_head, inputs=(graph_emb,))))
        return pre


class ClassificationPipeline(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 num_classes=2,
                 pre_train_path=None,
                 layer_num=3,
                 hid_dim=64,
                 frozen_gnn='first_layer',
                 frozen_prompt=False,
                 frozen_project_head=False,
                 gnn_type='GIN',
                 project_head_path=None,
                 gnn_name="NGNN",
                 dropout=0.2,
                 pool='mean'):
        super().__init__()
        self.gnn_name = gnn_name
        self.num_classes = num_classes
        self.pool = pool
        self.norm = nn.LayerNorm(hid_dim)

        if self.gnn_name == "adapter":
            self.task_head = torch.nn.Linear(input_dim, hid_dim)
            self.gnn = AdapterGNN(num_layer=layer_num, emb_dim=hid_dim)
        else:
            self.gnn = Backbone(type=gnn_type,
                                num_layers=layer_num,
                                input_dim=input_dim,
                                hidden_dim=hid_dim,
                                output_dim=hid_dim)

        self.prompt = Prompt(hid_dim)

        self.project_head = torch.nn.Sequential(
            torch.nn.Linear(hid_dim, hid_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hid_dim, num_classes)
        )

        self.set_gnn_project_head(pre_train_path,
                                  frozen_gnn,
                                  frozen_project_head,
                                  frozen_prompt,
                                  project_head_path)

    def set_gnn_project_head(self,
                             pre_train_path,
                             frozen_gnn,
                             frozen_project_head,
                             frozen_prompt,
                             project_head_path=None):
        if pre_train_path:
            self.gnn.load_state_dict(torch.load(pre_train_path), strict=False)
            print(f"successfully load pre-trained weights for gnn! @ {pre_train_path}")
        if project_head_path:
            self.project_head.load_state_dict(torch.load(project_head_path))
            print(f"successfully load project_head! @ {project_head_path}")

        if frozen_gnn == 'all':
            for p in self.gnn.parameters():
                p.requires_grad = False
        elif frozen_gnn == 'none':
            for p in self.gnn.parameters():
                p.requires_grad = True
        elif frozen_gnn == 'first_layer':
            for name, param in self.gnn.named_parameters():
                if 'convs.0' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        if frozen_project_head:
            for p in self.project_head.parameters():
                p.requires_grad = False
        if frozen_prompt:
            for p in self.prompt.parameters():
                p.requires_grad = False

    def _pool(self, x, batch):
        if self.pool == 'mean':
            return global_mean_pool(x, batch)
        elif self.pool == 'sum':
            from torch_geometric.nn import global_add_pool
            return global_add_pool(x, batch)
        elif self.pool == 'max':
            from torch_geometric.nn import global_max_pool
            return global_max_pool(x, batch)
        else:
            raise ValueError(f"Unsupported pool: {self.pool}")

    def forward(self, graph_batch: Batch):
        x = graph_batch.x
        if self.gnn_name == "adapter":
            x = self.task_head(x.to(torch.float32))
        graph_emb = self.gnn(x, graph_batch.edge_index, graph_batch.edge_attr)
        graph_emb = self.norm(graph_emb)
        graph_emb = self.prompt(graph_emb)  # node-level prompt
        # graph_emb = self._pool(graph_emb, graph_batch.batch)  # graph-level embedding
        logits = self.project_head(graph_emb)  # shape: (num_graphs, num_classes)

        return logits


class GlobalCountingPipeline(torch.nn.Module):

    def __init__(self, input_dim, pre_train_path=None, layer_num=3, hid_dim=64, frozen_gnn='first_layer',
                 frozen_project_head=False, frozen_prompt=False, pool_mode=0, gnn_type='GIN', mnn_type="ginconcat",
                 project_head_path=None,
                 m_layer_num=3,
                 without_snn=False,
                 without_prompt=False,
                 gnn_name="NGNN"):

        super().__init__()
        self.pool_mode = pool_mode
        self.mnn_type = mnn_type
        self.frozen_prompt = frozen_prompt
        self.norm = nn.LayerNorm(hid_dim)
        self.without_prompt = without_prompt
        self.gnn_name = gnn_name
        self.task_head_used = False
        if without_snn:
            self.gnn = MotifGNN(num_layers=layer_num, num_g_hid=hid_dim,
                                num_e_hid=32, out_g_ch=hid_dim, model_type="GIN",
                                dropout=0.2)
        else:
            if self.gnn_name == "adapter":
                self.task_head = torch.nn.Linear(input_dim, hid_dim)
                self.gnn = AdapterGNN(num_layer=layer_num, emb_dim=hid_dim)
            else:
                self.gnn = Backbone(type=gnn_type, num_layers=layer_num, input_dim=input_dim, hidden_dim=hid_dim,
                                    output_dim=hid_dim)
        self.projection_head = Mlp(hid_dim, hid_dim, 1, drop=0.5)
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
        elif frozen_gnn == 'first_layer':
            for name, param in self.gnn.named_parameters():
                if 'convs.0' in name:  # Assuming the first layer is the first parameter set
                    param.requires_grad = True
                else:
                    param.requires_grad = False
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
        if self.gnn_name == "adapter" and self.training and not self.task_head_used:
            graph_batch.x = self.task_head(graph_batch.x.to(torch.float32))
            self.task_head_used = True
            # graph_batch.x = torch.zeros_like(graph_batch.x)
        graph_emb = self.gnn(graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr)
        if self.without_prompt:
            graph_emb = global_mean_pool(graph_emb, batch=graph_batch.batch)
            reg = 0
            pre = self.projection_head(graph_emb)
        else:
            if self.mnn_type == "graphormer":
                motif_emb = self.motifnn(motif_batch)
            elif self.mnn_type == "vector":
                motif_emb = self.motifnn
            else:
                motif_emb = self.motifnn(motif_batch.x, motif_batch.edge_index, motif_batch.edge_attr)
            # graph_emb = global_mean_pool(graph_emb, batch=graph_batch.batch)
            graph_emb = self.norm(graph_emb)
            # final_emb = torch.cat([graph_emb, motif_emb], dim=1)
            graph_emb += motif_emb  # x = x + p
            pre, reg = self.projection_head(graph_emb, motif_emb)
        return pre, reg
