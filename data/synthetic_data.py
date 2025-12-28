import time

import networkx as nx
import torch
from pylibcugraph import betweenness_centrality
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx
import os.path as osp

from data_load import subgraph_construction


class SynDataset(InMemoryDataset):
    def __init__(self, root="/mnt/data/banlujie/dataset", name="synthetic1", pre_transform=None, transform=None,
                 pre_filter=None,
                 use_edge_attr=False, filepath="synthetic1", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                 num_nodes=1000, density=5,force_reload=False):
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
        self.num_nodes = num_nodes
        self.density = density
        super().__init__(root, pre_transform, transform, pre_filter,force_reload=force_reload)
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
        num_edges = self.num_nodes * self.density
        graph = nx.gnm_random_graph(self.num_nodes, num_edges)

        eigenvector_centrality = nx.eigenvector_centrality(graph, backend="cugraph")
        print("Begin to compute betweenness centrality...")
        st = time.time()
        betweenness_centrality = nx.betweenness_centrality(graph)
        et = time.time()
        print(f"Betweenness centrality computation time: {(et - st) * 1000}  ms")
        for i in range(len(graph.nodes)):
            graph.nodes[i]['x'] = 0.0
            graph.nodes[i]['eigenvector_centrality'] = eigenvector_centrality[i]
            graph.nodes[i]['betweenness_centrality'] = betweenness_centrality[i]
        data = from_networkx(graph)
        data.x = torch.zeros(self.num_nodes, 1).to(torch.float32)
        data.eigenvector_centrality = data.eigenvector_centrality.unsqueeze(1)
        data.betweenness_centrality = data.betweenness_centrality.unsqueeze(1)
        data.train_mask, data.val_mask, data.test_mask = self.dataset_split(data)
        st1 = time.time()
        subgraph_construction(graph, data_file=self.name, k=2)
        et1 = time.time()
        print(f"Subgraph construction time: {(et1 - st1) * 1000}  ms")
        torch.save(data, self.processed_paths[0])


if __name__ == "__main__":
    synthetic1 = SynDataset(name="synthetic10", filepath="synthetic10", num_nodes=100000, density=5,force_reload=True)
    # synthetic1.process()
