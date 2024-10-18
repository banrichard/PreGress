import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import networkx as nx
import argparse
import random
from tqdm import tqdm
import nx_cugraph as nxcg
from networkx.algorithms import isomorphism
import numpy as np
from data.data_load import load_local_query, extract_size_from_directory_name, graph_file_reader


def subgraph_match(graph1, graph2):
    matcher = isomorphism.GraphMatcher(graph1,graph2) or isomorphism.GraphMatcher(graph2,graph1)

    return matcher.subgraph_is_isomorphic()


def load_queries(queryset_load_path="/mnt/data/banlujie/dataset/web-spam/query_graph",
                 true_card_load_path="/mnt/data/banlujie/dataset/web-spam/label",
                 dataname="web-spam"):
    size_num = 0
    all_queries = {}
    for subdir in os.listdir(queryset_load_path):
        subdir_path = os.path.join(queryset_load_path, subdir)
        for filename in os.listdir(subdir_path):
            if not os.path.exists(os.path.join(true_card_load_path, subdir, filename)):
                continue
            file_path = os.path.join(subdir_path, filename)
            size, orbit = extract_size_from_directory_name(file_path)
            if size not in all_queries.keys():
                all_queries[size] = []
                size_num += 1
            query = load_local_query(file_path)
            all_queries[size].append(query)
    return all_queries


def get_subgraph(graph, radius=1):
    subgraphs = []
    for node in tqdm(graph.nodes):
        ego_net = nx.ego_graph(graph, node, radius=radius, undirected=True, backend="cugraph")
        subgraphs.append(ego_net)
    return subgraphs


def random_select_subgraph(graph, radius=1):
    rand_node = random.choice(list(graph.nodes))
    ego_net = nx.ego_graph(graph, rand_node, radius=radius, undirected=True, backend="cugraph")
    ego_net = nx.Graph(ego_net)
    return ego_net


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument("--dataset",type=str,default="web-spam")
    parser.add_argument("--radius",type=int,default=1)
    args = parser.parse_args()
    dataset_name = args.dataset
    radius = args.radius
    query_path = os.path.join("/mnt/data/banlujie/dataset/", dataset_name, "query_graph")
    label_path = os.path.join("/mnt/data/banlujie/dataset", dataset_name, "label")
    graph_path = os.path.join("/mnt/data/banlujie/dataset", dataset_name, dataset_name + ".txt")
    queries = load_queries(queryset_load_path=query_path,
                           true_card_load_path=label_path,
                           dataname=dataset_name)
    edge_list = []
    node_five_query = queries.get(5)
    num_five_query = len(node_five_query)
    node_six_query = queries.get(6)
    num_six_query = len(node_six_query)
    node_seven_query = queries.get(7)
    num_seven_query = len(node_seven_query)
    graph = graph_file_reader(edge_list, graph_path)
    subgraphs = []
    fail_rate = {5: [], 6: [], 7: []}
    for i in range(2):
        subgraphs.append(random_select_subgraph(graph, radius))
    for subgraph in subgraphs:
        hit_num = {5: 0, 6: 0, 7: 0}
        for query in node_five_query:
            if subgraph_match(subgraph, query):
                hit_num[5] += 1
        for query in node_six_query:
            if subgraph_match(subgraph, query):
                hit_num[6] += 1
        for query in node_seven_query:
            if subgraph_match(subgraph, query):
                hit_num[7] += 1
        fail_rate_five = (1 - hit_num.get(5) / num_five_query) * 100
        fail_rate_six = (1 - hit_num.get(6) / num_six_query) * 100
        fail_rate_seven = (1 - hit_num.get(7) / num_seven_query) * 100
        fail_rate.get(5).append(fail_rate_five)
        fail_rate.get(6).append(fail_rate_six)
        fail_rate.get(7).append(fail_rate_seven)
    fail_rate_mean_5 = np.mean(fail_rate.get(5))
    fail_rate_mean_6 = np.mean(fail_rate.get(6))
    fail_rate_mean_7 = np.mean(fail_rate.get(7))

    with open("/home/banlujie/metaCounting/non_meta/" + dataset_name + "_fail_{:d}.txt".format(radius), "w") as f:
        f.write(
            "fail rate 5:{:.4f}\n fail rate 6:{:.4f}\n fail rate 7:{:.4f}".format(fail_rate_mean_5, fail_rate_mean_6,
                                                                                  fail_rate_mean_7))
