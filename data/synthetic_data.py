import networkx as nx
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx
import os.path as osp


class SynDataset(InMemoryDataset):
    def __init__(self, root="/mnt/data/banlujie/dataset", name="synthetic1", pre_transform=None, transform=None,
                 pre_filter=None,
                 use_edge_attr=False, filepath="synthetic1", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                 num_nodes=1000, density=5):
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
        num_edges = self.num_nodes * self.density
        graph = nx.gnm_random_graph(self.num_nodes, num_edges)
        eigenvector_centrality = nx.eigenvector_centrality(graph, backend="cugraph")
        for i in range(len(graph.nodes)):
            graph.nodes[i]['eigenvector_centrality'] = eigenvector_centrality[i]
        data = from_networkx(graph)
        data.x = torch.zeros(self.num_nodes, 1).to(torch.float32)
        data.eigenvector_centrality = data.eigenvector_centrality.unsqueeze(1)
        data.train_mask, data.val_mask, data.test_mask = self.dataset_split(data)
        torch.save(data, self.processed_paths[0])

if __name__ == "__main__":
    synthetic1 = SynDataset(name="synthetic50", filepath="synthetic50", num_nodes=10000, density=50)
    print(synthetic1[0].x.size())