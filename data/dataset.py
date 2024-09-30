import networkx as nx
import torch
from torch_geometric.datasets import ZINC, QM9
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.data import Data, InMemoryDataset
import torch_geometric.transforms as T
import os
import os.path as osp

from data.data_load import meta_graph_load


def dataset_load(name="ZINC", type="train"):
    if name == "ZINC":
        data = ZINC(root=os.path.join("/mnt/data/lujie/metacounting_dataset", name), split=type)
    elif name == "QM9":
        data = QM9(root=os.path.join("/mnt/data/lujie/metacounting_dataset", name))
    importance_cal(data)
    # transform the PyG data into networkX
    # if not os.path.exists(os.path.join("/mnt/data/lujie/metacounting_dataset", "networkx", name)):
    #     os.mkdir(os.path.join("/mnt/data/lujie/metacounting_dataset", "networkx", name))
    # for i in range(len(data)):
    #     graph_to_file(data[i], name, i)
    return data


def importance_cal(data):
    for cnt, d in enumerate(data):
        graph = to_networkx(d, node_attrs=['x'], edge_attrs=['edge_attr'], to_undirected=True)
        # the centralities are dictionaries
        degree_centrality = nx.degree_centrality(graph)
        betweenness_centrality = nx.betweenness_centrality(graph)
        try:
            eigenvector_centrality = nx.eigenvector_centrality(graph)
        except nx.PowerIterationFailedConvergence:
            eigenvector_centrality = -1
        # add them to original node feature as labels
        for i in range(len(graph.nodes)):
            graph.nodes[i]['degree_centrality'] = degree_centrality[i]
            graph.nodes[i]['betweenness_centrality'] = betweenness_centrality[i]
            graph.nodes[i]['eigenvector_centrality'] = eigenvector_centrality[i] if type(
                eigenvector_centrality) == dict else -1
        graph_to_file(graph, "QM9", cnt)


def graph_to_file(graph, name, i):
    if not os.path.exists(os.path.join("/mnt/data/lujie/metacounting_dataset", name, "networkx")):
        os.mkdir(os.path.join("/mnt/data/lujie/metacounting_dataset", name, "networkx"))
    with open("/mnt/data/lujie/metacounting_dataset/" + name + "/networkx/" + str(i) + ".txt", "w") as file:
        for node in graph.nodes(data=True):
            file.write("v,{},{},{},{},{}\n".format(node[0], node[1]['x'], node[1]['degree_centrality'],
                                                   node[1]['eigenvector_centrality'],
                                                   node[1]['betweenness_centrality']))
        for edge in graph.edges(data=True):
            file.write("e,{},{},{}\n".format(edge[0], edge[1], edge[2]['edge_attr']))


class PretrainDataset(InMemoryDataset):
    def __init__(self, root="/mnt/data/banlujie/dataset", name="flixster", pre_transform=None, transform=None,
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
        data.degree_centrality = data.degree_centrality.unsqueeze(1)
        data.betweenness_centrality = data.degree_centrality.unsqueeze(1)
        data.eigenvector_centrality = data.eigenvector_centrality.unsqueeze(1)
        data.train_mask, data.val_mask, data.test_mask = self.dataset_split(data)
        torch.save(data, self.processed_paths[0])


if __name__ == "__main__":
    data = PretrainDataset(name="youtube", filepath="youtube")
    print(len(data[0].x))
