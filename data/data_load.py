import math
import random
import os
import sys

from data.synthetic_data import SynDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import pickle
import time

from tqdm import tqdm
import numpy as np
import torch
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data.batch import Batch
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.datasets import QM9
from gensim.models import KeyedVectors
from sklearn.preprocessing import MinMaxScaler
import re

from utils.extraction import k_hop_induced_subgraph


def load_embeddings(embeddings_file):
    # load embeddings from word2vec format file
    model = KeyedVectors.load_word2vec_format(embeddings_file, binary=False)
    features_matrix = np.asarray([model[str(node)] for node in range(len(model.index_to_key))])
    return features_matrix


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
    dir_name = os.path.dirname(file_name)
    name_file = os.path.basename(file_name)
    nodes_list = []
    edges_list = []
    if os.path.exists(
            os.path.join("/mnt", "data", "dataset", dir_name, name_file + '.pickle')):
        graph = pickle.load(
            open(os.path.join("/mnt", "data",  "dataset", dir_name, name_file + '.pickle'), "rb"))
        degree_centrality = np.array([graph.nodes[i]['degree_centrality'] for i in graph.nodes()])
        eigenvector_centrality = np.array([graph.nodes[i]['eigenvector_centrality'] for i in graph.nodes()])
        print("successfully load pickle!\n")
        return graph, degree_centrality, eigenvector_centrality
    elif file_name.endswith("web-spam.txt"):
        node_fea_np = load_embeddings(os.path.join("/mnt", "data",  "dataset", "web-spam",
                                                   "web-spam.emb"))  # align with pretrained GNN
        graph = graph_file_reader(edges_list, file_name)
    else:
        graph = graph_file_reader(edges_list, file_name)

    graph, degree_centrality, betweenness_centrality, eigenvector_centrality, pagerank = centrality_cal(graph)
    for i in range(len(graph.nodes)):
        graph.nodes[i]['x'] = node_fea_np[i] if file_name.endswith("web-spam.txt") else 0
        graph.nodes[i]['degree_centrality'] = degree_centrality[i]
        graph.nodes[i]['betweenness_centrality'] = betweenness_centrality[i]
        graph.nodes[i]['pagerank'] = pagerank[i]
        graph.nodes[i]['eigenvector_centrality'] = eigenvector_centrality[i] if type(
            eigenvector_centrality) == dict else -1
    pickle.dump(graph,
                open(os.path.join("/mnt", "data",  "dataset", dir_name, name_file + '.pickle'), 'wb'))
    return graph, np.array(degree_centrality), np.array(eigenvector_centrality)


def centrality_cal(graph):
    degree_centrality = nx.degree_centrality(graph, backend="cugraph")
    betweenness_centrality = nx.betweenness_centrality(graph, backend="cugraph")
    print("Calculating eigenvector...\n")
    s = time.time()
    try:
        eigenvector_centrality = nx.eigenvector_centrality(graph, backend="cugraph")
    except nx.PowerIterationFailedConvergence:
        eigenvector_centrality = -1
    et = time.time()
    print(" eigenvector done!\n")
    eigen_cal_time = et - s
    print("Calculating pagerank...\n")
    ps = time.time()
    pagerank = nx.pagerank(graph, backend="cugraph")
    pet = time.time()
    print(" pagerank done...\n")
    pg_cal_time = pet - ps
    print("eigenvector centrality cal time:{:.3f}\n pagerank cal time:{:.3f}\n".format(eigen_cal_time, pg_cal_time))
    return graph, degree_centrality, betweenness_centrality, eigenvector_centrality, pagerank


def graph_file_reader(edges_list, file):
    dir_name = os.path.dirname(file)
    name_file = os.path.basename(file)
    graph = nx.Graph()
    with open(file, "r") as file:
        next(file)
        for line in file:
            tokens = line.strip().split(" ")
            src = int(tokens[0])
            dst = int(tokens[1])
            graph.add_edge(src, dst, edge_attr=0.0)
    return graph


def load_queries(queryset_load_path, true_card_load_path, dataname="yeast", submode=False):
    """

    :param queryset_load_path:  /mnt/8t_data/lujie/dataset/web-spam/query_graph/
    :param true_card_load_path: /mnt/8t_data/lujie/dataset/web-spam/label/no.txt
    :param dataname:
    :return:
    """
    upper_card = 1e17
    lower_card = 0
    all_queries = {}
    num_queries = 0
    size_num = 0
    pattern_set = set()
    if dataname == "yeast":
        gt_file = open(true_card_load_path, "r")
        for line in gt_file:
            query_name, ground_truth = line.strip().split()
            ground_truth = int(ground_truth)
            pattern, size = query_name.split("_")[1], int(query_name.split("_")[2])  # query_pattern_size_no.graph
            pattern_set.add(pattern)
            if size not in all_queries.keys():
                all_queries[size] = []
                size_num += 1
            query_load_path = os.path.join(queryset_load_path, query_name)
            if not os.path.isfile(query_load_path):
                continue
            # load the query /mnt/8t_data/lujie/dataset/yeast/query_graph/query_pattern_size_no.graph
            query_graph_to_be_load = os.path.join(queryset_load_path, query_name)
            query, _, _ = meta_graph_load(query_graph_to_be_load)
            query_tensor = from_networkx(query, group_edge_attrs=['edge_attr'])
            if ground_truth >= upper_card or ground_truth < lower_card:
                continue
            true_card = ground_truth + 1 if ground_truth == 0 else ground_truth

            all_queries[size].append((query_tensor, math.log2(true_card)))
            num_queries += 1
    else:
        sign = 0
        test_set = {}
        for subdir in os.listdir(queryset_load_path):
            subdir_path = os.path.join(queryset_load_path, subdir)
            for filename in os.listdir(subdir_path):
                if os.path.exists(os.path.join(true_card_load_path, subdir, filename)):
                    true_card = read_ground_truth_from_file(os.path.join(true_card_load_path, subdir, filename))
                else:
                    continue
                file_path = os.path.join(subdir_path, filename)
                size, orbit = extract_size_from_directory_name(file_path)
                sign = size * 10 + orbit  # 5_3_1.txt -> 53, last number is an id and does not have any effect
                if sign not in all_queries.keys():
                    all_queries[sign] = []
                    size_num += 1
                pattern_set.add(file_path)
                query = load_local_query(file_path)
                query_tensor = from_networkx(query, group_node_attrs=['x'], group_edge_attrs=['edge_attr'])
                query_tensor.y = true_card
                all_queries[sign].append(query_tensor)
                if submode and sign > 70:
                    test_set[sign].append(query_tensor)
                    all_queries.pop(sign)
                num_queries += 1

    return all_queries, num_queries, size_num, len(pattern_set), test_set


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
    graph_path = os.path.join("/mnt", "data",  "dataset", dataset_name, dataset_name + ".pt")
    # if os.path.exists(graph_path):
    #     graph = torch.load(graph_path)
    # else:
    if dataset_name == "yeast":
        org_graph, _, _ = meta_graph_load(
            os.path.join("/mnt", "8t_data", "dataset", dataset_name, "data_graph",
                         dataset_name + ".graph"))
        graph = from_networkx(org_graph, group_node_attrs=['x'],
                              group_edge_attrs=['edge_attr'])
        graph.x = graph.x.repeat(graph.x.shape[0], 11)  # align with pretrained GNN
        graph = Batch.from_data_list([graph])
        torch.save(graph,
                   os.path.join("/mnt", "data", "dataset", dataset_name, dataset_name + ".pt"))
    elif dataset_name == "web-spam":
        org_graph, _, centrality = meta_graph_load(
            os.path.join("/mnt", "data",  "dataset", dataset_name,
                         dataset_name + ".txt"))
        node_fea_np = load_embeddings(os.path.join("/mnt", "data",  "dataset", "web-spam",
                                                   "web-spam.emb"))  # align with pretrained GNN

        graph = from_networkx(org_graph)
        graph.x = torch.from_numpy(node_fea_np)
        graph = Batch.from_data_list([graph])
        graph.y = torch.tensor(centrality).to(torch.float32)
        torch.save(graph,
                   os.path.join("/mnt", "data",  "dataset", dataset_name, dataset_name + ".pt"))
    return graph


def single_graph_loader(graph=None, data_file=None, train_ratio=0.8, val_ratio=0.1, few_shot=False, shot_num=10, k=1,
                        ana_mode=False):
    if os.path.exists(os.path.dirname(data_file) + "/subgraph.pt") and ana_mode == False:
        subgraph_sets = torch.load(os.path.dirname(data_file) + "/subgraph.pt")
    else:
        subgraph_sets = subgraph_construction(graph, data_file, k=k)
    trainsets, val_sets, test_sets = data_split(subgraph_sets, train_ratio, val_ratio, few_shot=False, shot_num=10)
    return trainsets, val_sets, test_sets


def subgraph_construction(graph=None, data_file=None, k=1):
    subgraph_sets = []
    num_nodes = graph.number_of_nodes()
    for node in graph.nodes():
        sub_graph = k_hop_induced_subgraph(graph, node, k=k)
        if sub_graph.number_of_nodes() == 0:
            continue
        sub_graph = from_networkx(sub_graph, group_node_attrs=['x'])
        # this importance is y (label for downstream)
        sub_graph.y_dc = graph.nodes[node]['degree_centrality']
        sub_graph.y_eigen = graph.nodes[node]['eigenvector_centrality']
        subgraph_sets.append(sub_graph)
    torch.save(subgraph_sets, os.path.dirname(data_file) + "/subgraph.pt")
    # batch = Batch.from_data_list(subgraph_sets)
    return subgraph_sets


def dataset_load(dataset_name, batch_size=256):
    if os.path.exists(
            os.path.join("/mnt", "data", "dataset", dataset_name, dataset_name + ".pt")):
        graphs = torch.load(
            os.path.join("/mnt", "data", "dataset", dataset_name, dataset_name + ".pt"))
    else:
        if dataset_name == "QM9":
            data_dir = "/mnt/data//dataset/QM9/networkx"
            graphs = []
            pbar = tqdm(os.listdir(data_dir))
            for file in pbar:
                graph, degree_centrality = single_graph_load(
                    os.path.join(data_dir, file))
                data = from_networkx(graph, group_node_attrs=['x'],
                                     group_edge_attrs=['edge_attr'])
                data.degree_centrality = torch.tensor(degree_centrality).reshape(-1, 1)
                graphs.append(data)
            torch.save(graphs, os.path.join("/mnt", "data",  "dataset", dataset_name,
                                            dataset_name + ".pt"))
            trainsets, val_sets, test_sets = data_split(graphs, 0.8, 0.1)
            train_loader = to_dataloader(trainsets, batch_size=batch_size)
            val_loader = to_dataloader(val_sets, batch_size=batch_size)
            test_loader = to_dataloader(test_sets, batch_size=batch_size)
        else:
            if os.path.exists(os.path.join("/mnt", "data",  "dataset", dataset_name, "subgraph.pt")):
                train_set, val_set, test_set = single_graph_loader(
                    data_file=os.path.join("/mnt", "data",  "dataset", dataset_name, dataset_name + ".txt"),
                )
            else:
                # node_fea_np = load_embeddings(os.path.join("/mnt", "8t_data",  "dataset", "web-spam",
                #                                            "web-spam.emb"))  # align with pretrained GNN
                data_file = os.path.join("/mnt", "data",  "dataset", dataset_name, dataset_name + ".txt")
                graph, importance, eigenvector_importance = meta_graph_load(
                    data_file)
                train_set, val_set, test_set = single_graph_loader(graph, data_file)
    return train_set, val_set, test_set


def synthetic_graph_load(dataset_name, batch_size=1024, train_ratio=0.8, val_ratio=0.1):
    data = SynDataset(name=dataset_name, filepath=dataset_name)[0]
    train_loader = NeighborLoader(data, num_neighbors=[-1], input_nodes=data.train_mask, batch_size=batch_size,
                                  shuffle=True)
    val_loader = NeighborLoader(data, num_neighbors=[-1], input_nodes=data.val_mask, shuffle=False, batch_size=1024)
    test_loader = NeighborLoader(data, num_neighbors=[-1], input_nodes=data.test_mask, shuffle=False, batch_size=1024)
    return train_loader, val_loader, test_loader


def importance_graph_load(dataset_name, batch_size=16, task="importance", train_ratio=0.8, val_ratio=0.1,
                          few_shot=False, shot_num=10, k=1, ana_mode=True):
    data_file = os.path.join("/mnt", "data",  "dataset", dataset_name, dataset_name + ".txt")
    if os.path.exists(
            os.path.join("/mnt", "data",  "dataset", dataset_name, "subgraph.pt")) and ana_mode == False:
        train_set, val_set, test_set = single_graph_loader(
            data_file=os.path.join("/mnt", "data",  "dataset", dataset_name, dataset_name + ".txt"))
    else:
        graph, importance, eigenvector_importance = meta_graph_load(
            data_file)
        # node_fea_np = load_embeddings(os.path.join("/mnt", "8t_data",  "dataset", "web-spam",
        #                                            "web-spam.emb"))  # align with pretrained GNN
        if few_shot:
            train_set, val_set, test_set = single_graph_loader(graph, data_file, train_ratio, val_ratio,
                                                               few_shot=few_shot, shot_num=10)
        else:
            train_set, val_set, test_set = single_graph_loader(graph, data_file,
                                                               train_ratio=train_ratio, val_ratio=val_ratio, k=k,
                                                               ana_mode=ana_mode)

    train_loader, val_loader, test_loader = (to_dataloader(train_set, batch_size=batch_size),
                                             to_dataloader(val_set, shuffle=False),
                                             to_dataloader(test_set, shuffle=False))
    return train_loader, val_loader, test_loader


def graph_with_motif_loader(query_graphs, subgraph_sets, batch_size, train_ratio, val_ratio):
    train_loader, val_loader, test_loader = [], [], []
    # extract all the motifs: (motif,label)
    queries = list(query_graphs.values())

    return train_loader, val_loader, test_loader


def graph_split(graph_batch, train_ratio, val_ratio):
    num_instances = len(graph_batch)
    random.shuffle(graph_batch)
    train_sets = graph_batch[: int(num_instances * train_ratio)]
    # merge to all_train_sets
    val_sets = graph_batch[int(num_instances * train_ratio): int(num_instances * (train_ratio + val_ratio))]
    test_sets = graph_batch[int(num_instances * (train_ratio + val_ratio)):]
    return train_sets, val_sets, test_sets


def counting_graph_load(dataset_name, batch_size=16, task="localcounting", train_ratio=0.8, val_ratio=0.1,
                        k=2, ana_mode=False):
    data_file = os.path.join("/mnt", "data",  "dataset", dataset_name, dataset_name + ".txt")
    pt_file = os.path.join("/mnt", "data",  "dataset", dataset_name, "subgraph.pt")
    if os.path.exists(pt_file) and ana_mode == False:
        subgraph_sets = torch.load(pt_file)
    else:
        graph, _, _ = meta_graph_load(data_file)

        subgraph_sets = subgraph_construction(graph, data_file, k=k)
    graph_batch = Batch.from_data_list(subgraph_sets)
    train_graph_set, val_graph_set, test_graph_set = graph_split(graph_batch, train_ratio=0.8, val_ratio=0.1)

    query_load_path = os.path.join("/mnt/data//dataset", dataset_name, "query_graph")
    true_card_load_path = os.path.join("/mnt/data//dataset", dataset_name, "label")
    query_graphs, num_queries, size_num, pattern_num, _ = load_queries(
        query_load_path, true_card_load_path, dataname=dataset_name, submode=False)
    motif_list = [item for sublist in list(query_graphs.values()) for item in
                  sublist]  # list(query_graphs.values())
    motif_batch = Batch.from_data_list(motif_list)
    motif_batch.edge_attr = motif_batch.edge_attr.to(torch.float32)
    num_instances = len(motif_list)
    random.shuffle(motif_list)

    train_sets = motif_batch[: int(num_instances * train_ratio)]
    val_sets = motif_batch[int(num_instances * train_ratio): int(num_instances * (train_ratio + val_ratio))]
    test_sets = motif_batch[int(num_instances * (train_ratio + val_ratio)):]
    train_loader, val_loader, test_loader = to_dataloader(train_sets, batch_size=batch_size), to_dataloader(
        val_sets, batch_size=batch_size), to_dataloader(test_sets, batch_size=batch_size)
    return Batch.from_data_list(train_graph_set), Batch.from_data_list(val_graph_set), Batch.from_data_list(
        test_graph_set), train_loader, val_loader, test_loader


def to_dataloader(dataset, batch_size=1, shuffle=True, num_workers=16):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def data_split(graphs, train_ratio, val_ratio, few_shot=False, shot_num=10, seed=1):
    assert train_ratio + val_ratio <= 1.0, "Error data split ratio!"
    random.seed(seed)
    train_sets, val_sets, test_sets = [], [], []

    num_instances = len(graphs)
    random.shuffle(graphs)
    if few_shot:
        train_sets = graphs[:shot_num]
        val_sets = graphs[shot_num:int(num_instances * 0.1) + shot_num]
        test_sets = graphs[int(num_instances * 0.1) + shot_num:]
    else:
        train_sets = graphs[: int(num_instances * train_ratio)]
        val_sets = graphs[int(num_instances * train_ratio): int(num_instances * (train_ratio + val_ratio))]
        test_sets = graphs[int(num_instances * (train_ratio + val_ratio)):]
    return train_sets, val_sets, test_sets


def load4graph(dataset_name, shot_num=10, num_parts=None):
    if dataset_name in ['QM9']:
        dataset = QM9(root='/mnt/data/metacounting_dataset/QM9')
        torch.manual_seed(42)
        dataset = dataset.shuffle()
        graph_list = [data for data in dataset]

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

        random.shuffle(remaining_data)
        val_dataset_size = len(remaining_data) // 9
        val_dataset = remaining_data[:val_dataset_size]
        test_dataset = remaining_data[val_dataset_size:]

        input_dim = dataset.num_features
        out_dim = dataset.num_classes

        return input_dim, out_dim, train_data, test_dataset, val_dataset, graph_list


def meta_motif_load(dataset_name, shot_num=5, task_pairs=None):
    if dataset_name in ['yeast']:

        query_graphs, num_queries, size_num, pattern_num, _ = load_queries(
            "/mnt/8t_data//dataset/yeast/query_graph",
            "/mnt/8t_data//dataset/yeast/yeast_ans.txt", dataname="web-spam")
        train_set, remaining_data = [], []
        # select each pattern and size, here use 4 node and 8 node as example
        i = 0
        max_iter = 100
        while i < len(task_pairs) and i < max_iter:
            support = []
            query = []
            # query_size = 1
            task_1, task_2 = task_pairs[i]
            support.extend(query_graphs[task_1][:shot_num])
            remaining_data.extend(query_graphs[task_1][shot_num:])
            support.extend(query_graphs[task_2][:shot_num])
            remaining_data.extend(query_graphs[task_2][shot_num:])
            random.shuffle(support)
            query = remaining_data[random.randint(0, len(remaining_data) - 1)]
            label = query[1]
            support = raw_meta_set2pyg(support, "train")
            query[0].x = query[0].x.reshape(-1, 1).to(torch.float32)
            query[0].edge_attr = query[0].edge_attr.to(torch.float32)
            query[0].y = label.reshape(-1, 1)
            del label
            # pyg_dataset = Batch.from_data_list(data)
            # pyg_dataset.x = pyg_dataset.x.reshape(-1, 1)
            # pyg_dataset.y = label
            i += 1
            yield task_1, task_2, support, query[0], len(task_pairs), query[0].y
    if dataset_name in ["web-spam"]:
        query_graphs, num_queries, size_num, pattern_num, _ = load_queries(
            "/mnt/data/dataset/web-spam/query_graph",
            "/mnt/data/dataset/web-spam/label", dataname="web-spam")
        train_set, remaining_data = [], []
        # select each pattern and size, here use 4 node and 8 node as example
        i = 0
        max_iter = 100
        while i < len(task_pairs) and i < max_iter:
            support = []
            query = []
            # query_size = 1
            task_1, task_2 = task_pairs[i]
            support.extend(query_graphs[task_1][:shot_num])
            remaining_data.extend(query_graphs[task_1][shot_num:])
            support.extend(query_graphs[task_2][:shot_num])
            remaining_data.extend(query_graphs[task_2][shot_num:])
            random.shuffle(support)
            query = remaining_data[random.randint(0, len(remaining_data) - 1)]
            label = query[1]
            support = raw_meta_set2pyg(support, "train")
            query[0].x = query[0].x.reshape(-1, 1).to(torch.float32)
            query[0].edge_attr = query[0].edge_attr.to(torch.float32)
            query[0].y = label.reshape(-1, 1)
            del label
            # pyg_dataset = Batch.from_data_list(data)
            # pyg_dataset.x = pyg_dataset.x.reshape(-1, 1)
            # pyg_dataset.y = label
            i += 1
            yield task_1, task_2, support, query[0], len(task_pairs), query[0].y


def raw_meta_set2pyg(dataset, type, dataset_name="yeast"):
    # dataset = query_graphs[task_1][:shot_num] + query_graphs[task_2][:shot_num]
    data = []
    label = []
    if len(dataset) == 1:
        data.append(dataset[0][0])
        label.append(dataset[0][1])
    else:
        for d in dataset:
            # d[0] is a PyG graph tensor
            d[0].y = d[1]
            data.append(d[0])
            # label.append(d[1])
    pyg_dataset = Batch.from_data_list(data)
    pyg_dataset.x = pyg_dataset.x.reshape(-1, 1).to(torch.float32)
    pyg_dataset.edge_attr = pyg_dataset.edge_attr.to(torch.float32)
    # else:
    #     num_nodes = 0
    #     pyg_dataset.x = torch.zeros(num_nodes, 1)
    # pyg_dataset.y = torch.tensor(label)
    pyg_dataset.y = pyg_dataset.y.reshape(-1, 1)
    # file_name = type + ".pt"
    # torch.save(pyg_dataset, os.path.join("/mnt/8t_data//dataset", dataset_name, "meta_set", file_name))
    return pyg_dataset


def load_local_query(query_path):
    G = nx.Graph()

    # Read the file and parse the lines
    with open(query_path, 'r') as file:
        lines = file.readlines()

    # Skip the first and last lines
    edges = lines[1:-1]

    # Add edges to the graph
    for edge in edges:
        # Assuming the format "node1 node2" per line
        node1, node2 = edge.strip().split()
        G.add_edge(node1, node2, edge_attr=0)
    for node in G.nodes():
        G.nodes[node]['x'] = 0
    return G


def read_ground_truth_from_file(file_path):
    """Reads ground truth data from a file, where each row contains a single value."""
    ground_truths = []
    with open(file_path, 'r') as file:
        ground_truths = [int(line.strip()) for line in file.readlines()]
        gt = np.array(ground_truths)
        scaler = MinMaxScaler()
        normalized_gt = scaler.fit_transform(gt.reshape(-1, 1))
        ground_truths = torch.tensor(normalized_gt)
        ground_truths = ground_truths.to(torch.float32)
    return ground_truths


def extract_size_from_directory_name(file_name):
    """Extracts the first number from the directory name."""
    file = os.path.basename(file_name)
    size, orbit = int(file.split("_")[0]), int(file.split("_")[1])
    return size, orbit

