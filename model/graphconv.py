from typing import Union

import torch
from torch import nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, global_mean_pool
import torch.nn.functional as F

from model.layers import CentralityEncoding, SpatialEncoding, GraphormerEncoderLayer, shortest_path_distance, \
    batched_shortest_path_distance
from .normalization import PairNorm


class Backbone(nn.Module):
    def __init__(self, type, num_layers, input_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.model_type = type
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hid_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.pair_norm = PairNorm()
        self.conv = self.build_conv_layers()
        self.convs = nn.ModuleList()
        for l in range(self.num_layers):
            hidden_input_dim = self.input_dim if l == 0 else self.hid_dim
            hidden_output_dim = self.hid_dim
            self.convs.append(self.conv(hidden_input_dim, hidden_output_dim))

    def build_conv_layers(self):
        if self.model_type == "GCN":
            return GCNConv
        elif self.model_type == "GAT":
            return GATConv
        elif self.model_type == "GraphSage":
            return SAGEConv
        elif self.model_type == "GIN":
            return lambda in_ch, hid_ch: GINConv(nn=nn.Sequential(
                nn.Linear(in_ch, hid_ch), nn.ReLU(), nn.Linear(hid_ch, hid_ch)), train_eps=True)
        elif self.model_type == "Graphormer":
            return
        else:
            raise NotImplementedError("Current do not support!")

    def forward(self, x, edge_index, edge_attr=None):
        for i in range(self.num_layers):
            if self.model_type == "GIN" or self.model_type == "GINE" or self.model_type == "GAT" \
                    or self.model_type == "GraphSage":
                x = self.convs[i](x, edge_index)  # for GIN and GINE
            elif self.model_type == "Graph" or self.model_type == "GCN":
                x = self.convs[i](x, edge_index)
            elif self.model_type == "NN" or self.model_type == "NNGIN" or self.model_type == "NNGINConcat":
                x = self.convs[i](x=x, edge_index=edge_index, edge_attr=edge_attr)
            else:
                print("Unsupported model type!")
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.pair_norm(x)
        # x = global_mean_pool(x, batch)
        return x


class Graphormer(nn.Module):
    def __init__(self,
                 num_layers: int,
                 input_node_dim=1,
                 node_dim=64,
                 input_edge_dim=1,
                 edge_dim=64,
                 output_dim=64,
                 n_heads=1,
                 ff_dim=128,
                 max_in_degree=5,
                 max_out_degree=5,
                 max_path_distance=3,
                 pretrain=False):
        """
        :param num_layers: number of Graphormer layers
        :param input_node_dim: input dimension of node features
        :param node_dim: hidden dimensions of node features
        :param input_edge_dim: input dimension of edge features
        :param edge_dim: hidden dimensions of edge features
        :param output_dim: number of output node features
        :param n_heads: number of attention heads
        :param max_in_degree: max in degree of nodes
        :param max_out_degree: max out degree of nodes
        :param max_path_distance: max pairwise distance between two nodes
        :param mode: when used in pretraining, the pooling layer will be used for subgraph aggregation;
        otherwise pooling will be ignored.
        """
        super().__init__()  # 调用父类的构造函数

        # 初始化参数
        self.num_layers = num_layers
        self.input_node_dim = input_node_dim
        self.node_dim = node_dim
        self.input_edge_dim = input_edge_dim
        self.edge_dim = edge_dim
        self.output_dim = output_dim
        self.ff_dim = ff_dim
        self.num_heads = n_heads
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.max_path_distance = max_path_distance
        self.pretrain = pretrain
        # 创建节点特征的输入线性层和边特征的输入线性层
        self.node_in_lin = nn.Linear(self.input_node_dim, self.node_dim)
        self.edge_in_lin = nn.Linear(self.input_edge_dim, self.edge_dim)

        # 创建中心性编码和空间编码
        self.centrality_encoding = CentralityEncoding(
            max_in_degree=self.max_in_degree,
            max_out_degree=self.max_out_degree,
            node_dim=self.node_dim,
        )
        self.spatial_encoding = SpatialEncoding(
            max_path_distance=self.max_path_distance,
        )

        # 创建Graphormer注意力层
        self.layers = nn.ModuleList(
            [
                GraphormerEncoderLayer(
                    node_dim=self.node_dim,
                    edge_dim=self.edge_dim,
                    num_heads=self.num_heads,
                    ff_dim=self.ff_dim,
                    max_path_distance=self.max_path_distance,
                )
                for _ in range(self.num_layers)
            ]
        )

        # 初始化节点输出线性层
        self.node_out_lin = nn.Linear(self.node_dim, self.output_dim)

    # 前向传播,data是一个Data对象,包含了图的信息
    # 返回值是一个torch.Tensor,表示节点的输出特征
    def forward(self, data: Union[Data]) -> torch.Tensor:
        """
        :param data: input graph of batch of graphs
        :return: torch.Tensor, output node embeddings
        """
        x = data.x.float()
        edge_index = data.edge_index.long()  # 边索引
        edge_attr = data.edge_attr.float()  # 边特征
        batch = data.batch
        if type(data) == Data:  # 如果data是单个图
            ptr = None
            # 最短路径特征
            node_paths, edge_paths = shortest_path_distance(data)
        else:  # 如果data是一个batch of graphs
            ptr = data.ptr
            # 最短路径特征
            node_paths, edge_paths = batched_shortest_path_distance(data)

        # 输入特征线性变换
        x = self.node_in_lin(x)
        edge_attr = self.edge_in_lin(edge_attr)

        # 中心性编码和空间编码
        x = self.centrality_encoding(x, edge_index)
        b = self.spatial_encoding(x, node_paths)

        # Graphormer层,多层堆叠,每一层的输入是上一层的输出
        for layer in self.layers:
            x = layer(x, edge_attr, b, edge_paths, ptr)
        # 输出特征线性变换
        x = self.node_out_lin(x)

        # 全局平均池化,因为encoder的输出是节点的特征,需要将其池化为图的特征
        if not self.pretrain:
            x = global_mean_pool(x, batch)
        return x
