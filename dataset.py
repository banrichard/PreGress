from torch_geometric.datasets import ZINC, QM9
import os


def dataset_load(name="ZINC", type="train"):
    if name == "ZINC":
        data = ZINC(root=os.path.join("/mnt/data/lujie/metacounting_dataset", name), split=type)
    else:
        data = QM9(root=os.path.join("/mnt/data/lujie/metacounting_dataset", name))
    return data


if __name__ == "__main__":
    data = dataset_load(name="QM9")
    print(data[0])
