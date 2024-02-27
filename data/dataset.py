from torch_geometric.datasets import ZINC, QM9
from torch_geometric.utils import to_networkx
import os


def dataset_load(name="ZINC", type="train"):
    if name == "ZINC":
        data = ZINC(root=os.path.join("/mnt/data/lujie/metacounting_dataset", name), split=type)
    else:
        data = QM9(root=os.path.join("/mnt/data/lujie/metacounting_dataset", name))
    # transform the PyG data into networkX
    if not os.path.exists(os.path.join("/mnt/data/lujie/metacounting_dataset", "networkx", name)):
        os.mkdir(os.path.join("/mnt/data/lujie/metacounting_dataset", "networkx", name))
    for i in range(len(data)):
        graph_to_file(data[i], name, i)
    return data

def graph_to_file(graph, name, i):
    graph = to_networkx(graph)
    with open(os.path.join("/mnt/data/lujie/metacounting_dataset", "networkx", name, str(i) + ".txt")) as f:
        for node in graph.nodes(data=True):
            f.write("v {} {}\n".format(node[0], node[1]['label']))
        for edge in graph.edges(data=True):
            f.write("e {} {} {}\n".format(edge[0], edge[1], edge[2]['label']))
    # with open(os.path.join(query_dir, "{}.txt".format(i)), 'w') as f1:
    #     f1.write("t # {}\n".format(i))
    #     for node in sample.nodes(data=True):
    #         f1.write("v {} {} {}\n".format(node[0], node[1]["label"], node[1]['dvid']))
    #     for edge in sample.edges(data=True):
    #         f1.write("e {} {} {:.2f}\n".format(edge[0], edge[1], edge[2]['prob']))


if __name__ == "__main__":
    data = dataset_load(name="QM9")
    print(data[0])
