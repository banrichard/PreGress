import random
import os
from tqdm import tqdm
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
from torch_geometric.data import InMemoryDataset as IMD
import re

def find_numbers_and_lists(string):
    # Find numbers outside lists
    if string[0] == "v": 
        number_after_v_or_e_pattern = r"(?<=[ve],)(\d+)?" #^[ve],(\d+)(?:,(\d+))?$
    else:
        number_after_v_or_e_pattern = r"(?<=[ve],)(\d+),(\d+)?" 
    number_after_list_pattern = r"\,(-?\d+\.\d+)" # (?:,(-?\d+\.\d+))?(?:,(-?\d+\.\d+))?
    former_parts = string.split('[', 1)
    later_parts = string.split(']', 1)
    # Extract the number following "v" or "e"
    number_after_v_or_e = re.findall(number_after_v_or_e_pattern, former_parts[0])
    if string[0] == "v":
        number_after_v_or_e = [int(num) for num in number_after_v_or_e] 
    else:
        number_after_v_or_e = [int(num) for num in number_after_v_or_e[0]]
    list_pattern = r"\[(.*?)\]"
    list = re.findall(list_pattern, string)
    list = list[0].split(",")
    list = [float(num) for num in list]
    # Confirming the approach with adjusted focus
    if string[0] == "v":
        number_after_list = re.findall(number_after_list_pattern, later_parts[1])
        if len(number_after_list) == 0 or number_after_list[0] == '':
            number_after_list = []
        else:
            try:
                number_after_list = [float(num) for num in number_after_list]
            except ValueError:
                print(string)
        return number_after_v_or_e, list, number_after_list
    else:
        return number_after_v_or_e,list


def single_graph_load(file):
    file = open(file)

    nodes_list = []
    edges_list = []
    degree_centrality_list = []
    for line in file:
        if line.strip().startswith("v"):
            ids,x,importance = find_numbers_and_lists(line)
            # v nodeID labelID degree
            id = ids[0]
            importance = importance[0]
            # eigenvector_centrality = float(tokens[4])
            # betweenness_centrality = float(tokens[5])
            nodes_list.append([id, {"x": np.array(x),"degree_centrality": importance}])
                               # {"eigenvector_centrality": eigenvector_centrality},
                               # {"betweenness_centrality": betweenness_centrality}))
            degree_centrality_list.append(importance)
            # betweenness_centrality_list.append(betweenness_centrality)
            # eigenvector_centrality_list.append(eigenvector_centrality)
        if line.strip().startswith("e"):
            ids,edge_attr = find_numbers_and_lists(line)
            src, dst = ids[0],ids[1]
            # edge_attr = tokens[2]  # tokens[3:]
            edges_list.append([src, dst, {"edge_attr": np.array(edge_attr)}])

    graph = nx.Graph()
    graph.add_nodes_from(nodes_list)
    graph.add_edges_from(edges_list)

    file.close()
    return graph, np.array(degree_centrality_list)


def dataset_load(dataset_name):
    if os.path.exists(os.path.join("/mnt","data","lujie","metacounting_dataset",dataset_name,dataset_name+".pt")):
        graphs = torch.load(os.path.join("/mnt","data","lujie","metacounting_dataset",dataset_name,dataset_name+".pt"))
    else:
        if dataset_name == "QM9":
            data_dir = "/mnt/data/lujie/metacounting_dataset/QM9/networkx"
            graphs = []
            pbar = tqdm(os.listdir(data_dir))
            for file in pbar:
                graph, degree_centrality = single_graph_load(
                    os.path.join(data_dir, file))
                data = from_networkx(graph, group_node_attrs=['x'],
                                    group_edge_attrs=['edge_attr'])
                data.degree_centrality = torch.tensor(degree_centrality).reshape(-1,1)
                graphs.append(data)
            torch.save(graphs,os.path.join("/mnt","data","lujie","metacounting_dataset",dataset_name,dataset_name+".pt"))
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
    dataset_load("QM9")
