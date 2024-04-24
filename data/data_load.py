import functools
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
        number_after_v_or_e_pattern = r"(?<=[ve],)(\d+)?"  # ^[ve],(\d+)(?:,(\d+))?$
    else:
        number_after_v_or_e_pattern = r"(?<=[ve],)(\d+),(\d+)?"
    number_after_list_pattern = r"\,(-?\d+\.\d+)"  # (?:,(-?\d+\.\d+))?(?:,(-?\d+\.\d+))?
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
        return number_after_v_or_e, list


def single_graph_load(file_name):
    file = open(file_name)

    nodes_list = []
    edges_list = []
    degree_centrality_list = []
    for line in file:
        if line.strip().startswith("v"):
            ids, x, importance = find_numbers_and_lists(line)
            # v nodeID labelID degree
            id = ids[0]
            importance = importance[0]
            nodes_list.append([id, {"x": np.array(x), "degree_centrality": importance}])
            degree_centrality_list.append(importance)

        if line.strip().startswith("e"):
            ids, edge_attr = find_numbers_and_lists(line)
            src, dst = ids[0], ids[1]
            # edge_attr = tokens[2]  # tokens[3:]
            edges_list.append([src, dst, {"edge_attr": np.array(edge_attr)}])

    graph = nx.Graph()
    graph.add_nodes_from(nodes_list)
    graph.add_edges_from(edges_list)

    file.close()
    return graph, np.array(degree_centrality_list)


def meta_graph_load(file_name):
    """
     This is for loading the graphs used in meta-learning
    """

    file = open(file_name)
    nodes_list = []
    edges_list = []
    degree_centrality_list = []
    for line in file:
        tokens = line.strip().split(" ")
        if line.strip().startswith("v"):
            # v nodeID labelID degree
            id = int(tokens[1])
            label = int(tokens[2])
            nodes_list.append([id, {"x": label}])
        if line.strip().startswith("e"):
            src = int(tokens[1])
            dst = int(tokens[2])
            edge_attr = list(tokens[3:])
            # edge_attr = tokens[2]  # tokens[3:]
            edges_list.append([src, dst, {"edge_attr": edge_attr}])

    graph = nx.Graph()
    graph.add_nodes_from(nodes_list)
    graph.add_edges_from(edges_list)

    file.close()
    return graph


def load_queries(queryset_load_path, true_card_load_path):
    upper_card = 1e17
    lower_card = 0
    all_queries = {}
    num_queries = 0
    gt_file = open(true_card_load_path, "r")
    for line in gt_file:
        query_name, ground_truth = line.strip().split()
        ground_truth = int(ground_truth)
        pattern, size = query_name.split("_")[1], int(query_name.split("_")[2])  # query_pattern_size_no.graph
        if size not in all_queries.keys():
            all_queries[size] = []
        query_load_path = os.path.join(queryset_load_path, query_name)
        if not os.path.isfile(query_load_path):
            continue
        # load the query /mnt/8t_data/lujie/dataset/yeast/query_graph/query_pattern_size_no.graph
        query_graph_to_be_load = os.path.join(queryset_load_path, query_name)
        query = meta_graph_load(query_graph_to_be_load)
        query_tensor = from_networkx(query)
        if ground_truth >= upper_card or ground_truth < lower_card:
            continue
        true_card = ground_truth + 1 if ground_truth == 0 else ground_truth

        all_queries[size].append((query_tensor, true_card))

        num_queries += 1
    return all_queries, num_queries


def transform_query_to_tensors(all_subsets):
    tmp_subsets = []
    num_queries = 0
    for (pattern, size), graphs_card_pairs in all_subsets.items():
        for (graphs, card) in graphs_card_pairs:
            tmp_graph = from_networkx(graphs, group_node_attrs=['x'], group_edge_attrs=['edge_attr'])
            tmp_subsets.append((tmp_graph, card, size))
            num_queries += 1
    return tmp_subsets


def meta_dataset_load(dataset_name):
    graph_path = os.path.join("/mnt", "8t_data", "banlujie", "dataset", dataset_name, dataset_name + ".pt")
    if os.path.exists(graph_path):
        graph = torch.load(graph_path)
    else:
        if dataset_name == "yeast":
            org_graph = meta_graph_load(
                os.path.join("/mnt", "8t_data", "banlujie", "dataset", dataset_name, "data_graph",
                             dataset_name + ".graph"))
            graph = from_networkx(org_graph, group_node_attrs=['x'],
                                     group_edge_attrs=['edge_attr'])
            graph.x = graph.x.repeat(0,11) # align with pretrained GNN
            torch.save(graph,
                       os.path.join("/mnt", "8t_data", "banlujie", "dataset", dataset_name, dataset_name + ".pt"))
    return graph


def dataset_load(dataset_name, batch_size=256):
    if os.path.exists(
            os.path.join("/mnt", "data", "lujie", "metacounting_dataset", dataset_name, dataset_name + ".pt")):
        graphs = torch.load(
            os.path.join("/mnt", "data", "lujie", "metacounting_dataset", dataset_name, dataset_name + ".pt"))
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
                data.degree_centrality = torch.tensor(degree_centrality).reshape(-1, 1)
                graphs.append(data)
            torch.save(graphs, os.path.join("/mnt", "data", "lujie", "metacounting_dataset", dataset_name,
                                            dataset_name + ".pt"))
    trainsets, val_sets, test_sets = data_split(graphs, 0.8, 0.1)
    train_loader = to_dataloader(trainsets, batch_size=batch_size)
    val_loader = to_dataloader(val_sets, batch_size=batch_size)
    test_loader = to_dataloader(test_sets, batch_size=batch_size)
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
    if dataset_name in ['yeast']:
        train_file_name = os.path.join("/mnt/8t_data/banlujie/dataset", dataset_name, "meta_set", "train.pt")
        val_file_name = os.path.join("/mnt/8t_data/banlujie/dataset", dataset_name, "meta_set", "val.pt")
        test_file_name = os.path.join("/mnt/8t_data/banlujie/dataset", dataset_name, "meta_set", "test.pt")
        if os.path.exists(train_file_name) and os.path.exists(val_file_name) and os.path.exists(test_file_name):
            train_set = torch.load(train_file_name)
            val_dataset = torch.load(val_file_name)
            test_dataset = torch.load(test_file_name)
            return train_set, val_dataset, test_dataset
        query_graphs, num_queries = load_queries("/mnt/8t_data/banlujie/dataset/yeast/query_graph",
                                                 "/mnt/8t_data/banlujie/dataset/yeast/yeast_ans.txt")
        train_set, remaining_data = [], []
        # select each pattern and size
        cnt = 0
        for size, data in query_graphs.items():
            train_set.extend(data[:shot_num])
            random.shuffle(train_set)
            remaining_data.extend(data[shot_num:])
        random.shuffle(remaining_data)
        val_dataset_size = len(remaining_data) // 9
        val_dataset = remaining_data[:val_dataset_size]
        test_dataset = remaining_data[val_dataset_size:]
        train_set = raw_meta_set2pyg(train_set, "train")
        val_dataset = raw_meta_set2pyg(val_dataset, "val")
        test_dataset = raw_meta_set2pyg(test_dataset, "test")
        return train_set, val_dataset, test_dataset


def raw_meta_set2pyg(dataset, type, dataset_name="yeast"):
    data = []
    label = []
    for d in dataset:
        data.append(d[0])
        label.append(d[1])
    pyg_dataset = Batch.from_data_list(data)
    pyg_dataset.y = label
    file_name = type + ".pt"
    torch.save(pyg_dataset, os.path.join("/mnt/8t_data/banlujie/dataset", dataset_name, "meta_set", file_name))
    return pyg_dataset


if __name__ == "__main__":
    load4graph("yeast")
