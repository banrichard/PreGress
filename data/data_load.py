import random
import os

import numpy as np
import torch
import networkx as nx
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected, from_networkx
from torch_geometric.data.batch import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.loader.cluster import ClusterData
from torch_geometric.data import Data
from torch_geometric.datasets import QM9, Planetoid
import ast


def single_graph_load(file):
    file = open(file)

    nodes_list = []
    edges_list = []
    degree_centrality_list = []
    betweenness_centrality_list = []
    eigenvector_centrality_list = []
    for line in file:
        pattern = f"[{line}]"
        if line.strip().startswith("v"):
            tokens = ast.literal_eval(pattern)
            # v nodeID labelID degree
            id = int(tokens[1])
            x = tokens[2]
            degree_centraility = float(tokens[3])
            eigenvector_centrality = float(tokens[4])
            betweenness_centrality = float(tokens[5])
            nodes_list.append((id, {"x": np.array(x)}, {"degree_centrality": degree_centraility},
                               {"eigenvector_centrality": eigenvector_centrality},
                               {"betweenness_centrality": betweenness_centrality}))
            degree_centrality_list.append(degree_centraility)
            betweenness_centrality_list.append(betweenness_centrality)
            eigenvector_centrality_list.append(eigenvector_centrality)
        if line.strip().startswith("e"):
            tokens = ast.literal_eval(pattern)
            src, dst = int(tokens[1]), int(tokens[2])
            edge_attr = tokens[3]  # tokens[3:]
            edges_list.append((src, dst, {"edge_attr": np.array(edge_attr)}))

    graph = nx.Graph()
    graph.add_nodes_from(nodes_list)
    graph.add_edges_from(edges_list)

    print('number of nodes: {}'.format(graph.number_of_nodes()))
    print('number of edges: {}'.format(graph.number_of_edges()))
    file.close()
    return graph, np.array(degree_centrality_list), np.array(betweenness_centrality_list), np.array(
        eigenvector_centrality_list)


def dataset_load(dataset_name):
    if dataset_name == "QM9":
        data_dir = "/mnt/data/lujie/metacounting_dataset/QM9/networkx"
        graphs = []
        for file in os.listdir(data_dir):
            graph, degree_centrality, betweenness_centrality, eigenvector_centrality = single_graph_load(
                os.path.join(data_dir, file))
            data = from_networkx(graph, group_node_attrs=['x'],
                                 group_edge_attrs=['edge_attr'])
            data.degree_centrality = torch.tensor(degree_centrality)
            data.betweenness_centrality = torch.tensor(betweenness_centrality)
            data.eigenvector_centrality = torch.tensor(eigenvector_centrality)
            graphs.append(data)
    trainsets, val_sets, test_sets = data_split(graphs, 0.8, 0.1)
    train_loader = to_dataloader(trainsets)
    val_loader = to_dataloader(val_sets)
    test_loader = to_dataloader(test_sets)
    return train_loader, val_loader, test_loader


def to_dataloader(dataset, batch_size=1, shuffle=True, num_workers=16):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def data_split(graphs, train_ratio, val_ratio, seed=1):
    assert train_ratio + val_ratio <= 1.0, "Error data split ratio!"
    random.seed(seed)
    train_sets, val_sets, test_sets = [], [], []

    num_instances = len(graphs)
    random.shuffle(graphs)
    train_sets = graphs[: int(num_instances * train_ratio)]
    # merge to all_train_sets
    val_sets = graphs[int(num_instances * train_ratio): int(num_instances * (train_ratio + val_ratio))]
    test_sets = graphs[int(num_instances * (train_ratio + val_ratio)):]
    return train_sets, val_sets, test_sets


def load4graph(dataset_name, shot_num=10, num_parts=None):
    if dataset_name in ['QM9']:
        dataset = QM9(root='/mnt/data/lujie/metacounting_dataset/QM9')
        torch.manual_seed(42)
        dataset = dataset.shuffle()
        graph_list = [data for data in dataset]

        # 分类并选择每个类别的图
        class_datasets = {}
        for data in dataset:
            label = data.y.item()
            if label not in class_datasets:
                class_datasets[label] = []
            class_datasets[label].append(data)

        train_data = []
        remaining_data = []
        for label, data_list in class_datasets.items():
            train_data.extend(data_list[:shot_num])
            random.shuffle(train_data)
            remaining_data.extend(data_list[shot_num:])

        # 将剩余的数据 1：9 划分为测试集和验证集
        random.shuffle(remaining_data)
        val_dataset_size = len(remaining_data) // 9
        val_dataset = remaining_data[:val_dataset_size]
        test_dataset = remaining_data[val_dataset_size:]

        input_dim = dataset.num_features
        out_dim = dataset.num_classes

        return input_dim, out_dim, train_data, test_dataset, val_dataset, graph_list

    if dataset_name in ['PubMed', 'CiteSeer', 'Cora']:
        dataset = Planetoid(root='data/Planetoid', name=dataset_name, transform=NormalizeFeatures())
        data = dataset[0]
        num_parts = 200

        x = data.x.detach()
        edge_index = data.edge_index
        edge_index = to_undirected(edge_index)
        data = Data(x=x, edge_index=edge_index)
        input_dim = dataset.num_features
        out_dim = dataset.num_classes

        dataset = list(ClusterData(data=data, num_parts=num_parts))
        graph_list = dataset
        # 这里的图没有标签

        return input_dim, out_dim, None, None, None, graph_list


if __name__ == "__main__":
    single_graph_load("/mnt/data/lujie/metacounting_dataset/QM9/networkx/0.txt")
