import networkx as nx
import torch
from torch_geometric.datasets import ZINC, QM9
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import NeighborLoader
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
    def __init__(self, root="./dataset", name="krogan_core", transform=None, pre_transform=None, pre_filter=None,
                 use_edge_attr=True, filepath="krogan", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        self.root = root
        self.use_edge_attr = use_edge_attr
        self.filepath = filepath
        self.name = name
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.cnt = 0
        self.edge_batch = torch.zeros(1).to(torch.int64)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices, self.edge_batch = torch.load(self.processed_paths[0])

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

    def dataset_split(self):
        assert self.train_ratio + self.val_ratio <= 1.0, "Error split ratios!"


    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        graph,_,_ = meta_graph_load(osp.join(self.root, self.filepath, self.name + ".txt"))
        data = from_networkx(graph)
        graph.x = graph.x.unsqueeze(1)
        graph.degree_centrality = graph.degree_centrality.unsqueeze(1)
        graph.betweenness_centrality = graph.degree_centrality.unsqueeze(1)
        graph.eigen
        torch.save(data, self.processed_paths[0])


if __name__ == "__main__":
    dataset_load("QM9")
