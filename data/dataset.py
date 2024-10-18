import networkx as nx
import torch
from networkx import eigenvector_centrality
from torch_geometric.data.remote_backend_utils import num_nodes
from torch_geometric.datasets import ZINC, QM9
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import FakeDataset
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import os.path as osp

from data.data_load import meta_graph_load



class PretrainDataset(InMemoryDataset):
    def __init__(self, root="/mnt/data/dataset", name="flixster", pre_transform=None, transform=None,
                 pre_filter=None,
                 use_edge_attr=False, filepath="flixster", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        self.root = root
        self.use_edge_attr = use_edge_attr
        self.filepath = filepath
        self.name = name
        self.pre_transform = pre_transform
        self.transform = transform
        self.pre_filter = pre_filter
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        super().__init__(root, pre_transform, transform, pre_filter)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.filepath)

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.filepath, self.name)

    @property
    def raw_file_names(self):
        return ['./dataset/krogan/label', "./dataset/krogan/queryset"]

    @property
    def processed_file_names(self):
        return [self.name + '.pt']

    def dataset_split(self, data):
        assert self.train_ratio + self.val_ratio <= 1.0, "Error split ratios!"
        num_nodes = data.num_nodes
        indices = torch.randperm(num_nodes)  # Shuffle indices

        train_size = int(self.train_ratio * num_nodes)
        val_size = int(self.val_ratio * num_nodes)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        # Create masks
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        data.train_mask[train_indices] = True
        data.val_mask[val_indices] = True
        data.test_mask[test_indices] = True
        return data.train_mask, data.val_mask, data.test_mask

    def process(self):
        # Read data into huge `Data` list.
        graph, _, _ = meta_graph_load(osp.join(self.root, self.filepath, self.name + ".txt"))
        data = from_networkx(graph)
        data.x = data.x.unsqueeze(1)
        data.edge_attr = data.edge_attr.unsqueeze(1)
        data.degree_centrality = data.degree_centrality.unsqueeze(1)
        data.pagerank = data.pagerank.unsqueeze(1)
        data.eigenvector_centrality = data.eigenvector_centrality.unsqueeze(1)
        data.train_mask, data.val_mask, data.test_mask = self.dataset_split(data)
        torch.save(data, self.processed_paths[0])


