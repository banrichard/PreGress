import networkx as nx
from torch_geometric.datasets import ZINC, QM9
from torch_geometric.utils import to_networkx
import os


def dataset_load(name="ZINC", type="train"):
    if name == "ZINC":
        data = ZINC(root=os.path.join("/mnt/data/lujie/metacounting_dataset", name), split=type)
    else:
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

if __name__ == "__main__":
    dataset_load("QM9")
