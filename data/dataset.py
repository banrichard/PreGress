import time

import networkx as nx
from networkx import eigenvector_centrality
from pylibcugraph import betweenness_centrality
from torch_geometric.io import read_planetoid_data, fs

import json
from typing import Callable, List, Optional, Union

import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset, download_google_url, download_url
from torch_geometric.utils import to_networkx, from_networkx, coalesce
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import os.path as osp
from torch_geometric.datasets import Flickr
from data.data_load import meta_graph_load


def wiki_load(root, name, geom_gcn_preprocess=True, transform=None, pre_transform=None, force_reload=False):
    data = MyWikipediaNetwork(root=root, name=name, geom_gcn_preprocess=geom_gcn_preprocess, transform=transform,
                              pre_transform=pre_transform, force_reload=force_reload)[0]
    return data


def centrality_cal(graph):
    degree_centrality = nx.degree_centrality(graph, backend="cugraph")
    print("Calculating pagerank...\n")
    ps = time.time()
    pagerank = nx.pagerank(graph, backend="cugraph")
    eigenvector_centrality = nx.eigenvector_centrality(G=graph, tol=1e-2, backend="cugraph")
    pet = time.time()
    print(" pagerank done...\n")
    pg_cal_time = pet - ps
    print(" pagerank cal time:{:.3f}\n".format(pg_cal_time))
    bet = time.time()
    betweenness_centrality = nx.betweenness_centrality(graph, backend="cugraph")
    bed = time.time()
    bet_cal_time = bed - bet
    print(" betweenness cal time:{:.3f}\n".format(bet_cal_time))
    return degree_centrality, pagerank, eigenvector_centrality, betweenness_centrality


class PretrainDataset(InMemoryDataset):
    def __init__(self, root="/mnt/data/banlujie/dataset", name="flixster", pre_transform=None, transform=None,
                 pre_filter=None,
                 use_edge_attr=False, filepath="flixster", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                 force_reload=False):
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
        super().__init__(root, pre_transform, transform, pre_filter, force_reload=force_reload)
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
        data.x = data.x.unsqueeze(1) if self.name != "web-spam" else data.x
        data.edge_attr = data.edge_attr.unsqueeze(1)
        data.degree_centrality = data.degree_centrality.unsqueeze(1)
        data.betweenness_centrality = data.betweenness_centrality.unsqueeze(1) if hasattr(data,
                                                                                          'betweenness_centrality') else torch.zeros(
            (data.num_nodes, 1), dtype=torch.float32)
        data.pagerank = data.pagerank.unsqueeze(1)
        data.eigenvector_centrality = data.eigenvector_centrality.unsqueeze(1)
        data.train_mask, data.val_mask, data.test_mask = self.dataset_split(data)
        torch.save(data, self.processed_paths[0])

    def centrality_cal(self, data):
        graph = to_networkx(data)
        degree_centrality = nx.degree_centrality(graph, backend="cugraph")
        degree_centrality = torch.tensor(list(degree_centrality.values())).unsqueeze(1)
        betweenness_centrality = nx.betweenness_centrality(graph, backend="cugraph")
        print("Calculating pagerank...\n")
        ps = time.time()
        pagerank = nx.pagerank(graph, backend="cugraph")
        pagerank = torch.tensor(list(pagerank.values())).unsqueeze(1)
        pet = time.time()
        print(" pagerank done...\n")
        pg_cal_time = pet - ps
        print(" pagerank cal time:{:.3f}\n".format(pg_cal_time))

        return degree_centrality, pagerank, betweenness_centrality


class MyReddit2(InMemoryDataset):
    r"""The Reddit dataset from the `"GraphSAINT: Graph Sampling Based
    Inductive Learning Method" <https://arxiv.org/abs/1907.04931>`_ paper,
    containing Reddit posts belonging to different communities.

    .. note::

        This is a sparser version of the original
        :obj:`~torch_geometric.datasets.Reddit` dataset (~23M edges instead of
        ~114M edges), and is used in papers such as
        `SGC <https://arxiv.org/abs/1902.07153>`_ and
        `GraphSAINT <https://arxiv.org/abs/1907.04931>`_.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10
        :header-rows: 1

        * - #nodes
          - #edges
          - #features
          - #classes
        * - 232,965
          - 23,213,838
          - 602
          - 41
    """
    adj_full_id = '1sncK996BM5lpuDf75lDFqCiDZyErc1c2'
    feats_id = '1ZsHaJ0ussP1W722krmEIp_8pwKAoi5b3'
    class_map_id = '1JF3Pjv9OboMNYs2aXRQGbJbc4t_nDd5u'
    role_id = '1nJIKd77lcAGU4j-kVNx_AIGEkveIKz3A'

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['adj_full.npz', 'feats.npy', 'class_map.json', 'role.json']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        download_google_url(self.adj_full_id, self.raw_dir, 'adj_full.npz')
        download_google_url(self.feats_id, self.raw_dir, 'feats.npy')
        download_google_url(self.class_map_id, self.raw_dir, 'class_map.json')
        download_google_url(self.role_id, self.raw_dir, 'role.json')

    def process(self) -> None:
        import scipy.sparse as sp

        f = np.load(osp.join(self.raw_dir, 'adj_full.npz'))
        adj = sp.csr_matrix((f['data'], f['indices'], f['indptr']), f['shape'])
        adj = adj.tocoo()
        row = torch.from_numpy(adj.row).to(torch.long)
        col = torch.from_numpy(adj.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        x = np.load(osp.join(self.raw_dir, 'feats.npy'))
        x = torch.from_numpy(x).to(torch.float)

        ys = [-1] * x.size(0)
        with open(osp.join(self.raw_dir, 'class_map.json')) as f:
            class_map = json.load(f)
            for key, item in class_map.items():
                ys[int(key)] = item
        y = torch.tensor(ys)

        with open(osp.join(self.raw_dir, 'role.json')) as f:
            role = json.load(f)

        train_mask = torch.zeros(x.size(0), dtype=torch.bool)
        train_mask[torch.tensor(role['tr'])] = True

        val_mask = torch.zeros(x.size(0), dtype=torch.bool)
        val_mask[torch.tensor(role['va'])] = True

        test_mask = torch.zeros(x.size(0), dtype=torch.bool)
        test_mask[torch.tensor(role['te'])] = True

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)
        graph = to_networkx(data, to_undirected=True)
        degree_centrality, pagerank, eigenvector_centrality = centrality_cal(graph)
        data.degree_centrality = torch.tensor([degree_centrality[i] for i in range(len(degree_centrality))]).unsqueeze(
            1)
        data.pagerank = torch.tensor([pagerank[i] for i in range(len(pagerank))]).unsqueeze(1)
        data = data if self.pre_transform is None else self.pre_transform(data)

        self.save([data], self.processed_paths[0])


class MyFlickr(InMemoryDataset):
    r"""The Flickr dataset from the `"GraphSAINT: Graph Sampling Based
    Inductive Learning Method" <https://arxiv.org/abs/1907.04931>`_ paper,
    containing descriptions and common properties of images.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10
        :header-rows: 1

        * - #nodes
          - #edges
          - #features
          - #classes
        * - 89,250
          - 899,756
          - 500
          - 7
    """
    adj_full_id = '1crmsTbd1-2sEXsGwa2IKnIB7Zd3TmUsy'
    feats_id = '1join-XdvX3anJU_MLVtick7MgeAQiWIZ'
    class_map_id = '1uxIkbtg5drHTsKt-PAsZZ4_yJmgFmle9'
    role_id = '1htXCtuktuCW8TR8KiKfrFDAxUgekQoV7'

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['adj_full.npz', 'feats.npy', 'class_map.json', 'role.json']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        download_google_url(self.adj_full_id, self.raw_dir, 'adj_full.npz')
        download_google_url(self.feats_id, self.raw_dir, 'feats.npy')
        download_google_url(self.class_map_id, self.raw_dir, 'class_map.json')
        download_google_url(self.role_id, self.raw_dir, 'role.json')

    def process(self) -> None:
        import scipy.sparse as sp

        f = np.load(osp.join(self.raw_dir, 'adj_full.npz'))
        adj = sp.csr_matrix((f['data'], f['indices'], f['indptr']), f['shape'])
        adj = adj.tocoo()
        row = torch.from_numpy(adj.row).to(torch.long)
        col = torch.from_numpy(adj.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        x = np.load(osp.join(self.raw_dir, 'feats.npy'))
        x = torch.from_numpy(x).to(torch.float)

        ys = [-1] * x.size(0)
        with open(osp.join(self.raw_dir, 'class_map.json')) as f:
            class_map = json.load(f)
            for key, item in class_map.items():
                ys[int(key)] = item
        y = torch.tensor(ys)

        with open(osp.join(self.raw_dir, 'role.json')) as f:
            role = json.load(f)

        train_mask = torch.zeros(x.size(0), dtype=torch.bool)
        train_mask[torch.tensor(role['tr'])] = True

        val_mask = torch.zeros(x.size(0), dtype=torch.bool)
        val_mask[torch.tensor(role['va'])] = True

        test_mask = torch.zeros(x.size(0), dtype=torch.bool)
        test_mask[torch.tensor(role['te'])] = True

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)

        data = data if self.pre_transform is None else self.pre_transform(data)
        graph = to_networkx(data, to_undirected=True)
        degree_centrality, pagerank, eigenvector_centrality, betweenness_centrality = centrality_cal(graph)
        data.degree_centrality = torch.tensor([degree_centrality[i] for i in range(len(degree_centrality))]).unsqueeze(
            1)
        data.edge_attr = torch.zeros(size=(data.num_edges, 1), dtype=torch.float32)
        data.eigenvector_centrality = torch.tensor(
            [eigenvector_centrality[i] for i in range(len(eigenvector_centrality))]).unsqueeze(1)
        data.betweenness_centrality = torch.tensor(
            [betweenness_centrality[i] for i in range(len(betweenness_centrality))]).unsqueeze(1)
        data.pagerank = torch.tensor([pagerank[i] for i in range(len(pagerank))]).unsqueeze(1)
        self.save([data], self.processed_paths[0])


class MyWikipediaNetwork(InMemoryDataset):
    r"""The Wikipedia networks introduced in the
    `"Multi-scale Attributed Node Embedding"
    <https://arxiv.org/abs/1909.13021>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features represent several informative nouns in the Wikipedia pages.
    The task is to predict the average daily traffic of the web page.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"chameleon"`,
            :obj:`"crocodile"`, :obj:`"squirrel"`).
        geom_gcn_preprocess (bool): If set to :obj:`True`, will load the
            pre-processed data as introduced in the `"Geom-GCN: Geometric
            Graph Convolutional Networks" <https://arxiv.org/abs/2002.05287>_`,
            in which the average monthly traffic of the web page is converted
            into five categories to predict.
            If set to :obj:`True`, the dataset :obj:`"crocodile"` is not
            available.
            If set to :obj:`True`, train/validation/test splits will be
            available as masks for multiple splits with shape
            :obj:`[num_nodes, num_splits]`. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)

    """

    raw_url = 'https://graphmining.ai/datasets/ptg/wiki'
    processed_url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/'
                     'geom-gcn/f1fc0d14b3b019c562737240d06ec83b07d16a8f')

    def __init__(
            self,
            root: str,
            name: str,
            geom_gcn_preprocess: bool = True,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            force_reload: bool = False,
    ) -> None:
        self.name = name.lower()
        self.geom_gcn_preprocess = geom_gcn_preprocess
        assert self.name in ['chameleon', 'crocodile', 'squirrel']
        if geom_gcn_preprocess and self.name == 'crocodile':
            raise AttributeError("The dataset 'crocodile' is not available in "
                                 "case 'geom_gcn_preprocess=True'")
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        if self.geom_gcn_preprocess:
            return osp.join(self.root, self.name, 'geom_gcn', 'raw')
        else:
            return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        if self.geom_gcn_preprocess:
            return osp.join(self.root, self.name, 'geom_gcn', 'processed')
        else:
            return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> Union[List[str], str]:
        if self.geom_gcn_preprocess:
            return (['out1_node_feature_label.txt', 'out1_graph_edges.txt'] +
                    [f'{self.name}_split_0.6_0.2_{i}.npz' for i in range(10)])
        else:
            return f'{self.name}.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        if self.geom_gcn_preprocess:
            for filename in self.raw_file_names[:2]:
                url = f'{self.processed_url}/new_data/{self.name}/{filename}'
                download_url(url, self.raw_dir)
            for filename in self.raw_file_names[2:]:
                url = f'{self.processed_url}/splits/{filename}'
                download_url(url, self.raw_dir)
        else:
            download_url(f'{self.raw_url}/{self.name}.npz', self.raw_dir)

    def process(self) -> None:
        if self.geom_gcn_preprocess:
            with open(self.raw_paths[0]) as f:
                lines = f.read().split('\n')[1:-1]
            xs = [[float(value) for value in line.split('\t')[1].split(',')]
                  for line in lines]
            x = torch.tensor(xs, dtype=torch.float)
            ys = [int(line.split('\t')[2]) for line in lines]
            y = torch.tensor(ys, dtype=torch.long)

            with open(self.raw_paths[1]) as f:
                lines = f.read().split('\n')[1:-1]
                edge_indices = [[int(value) for value in line.split('\t')]
                                for line in lines]
            edge_index = torch.tensor(edge_indices).t().contiguous()
            edge_index = coalesce(edge_index, num_nodes=x.size(0))

            train_masks, val_masks, test_masks = [], [], []
            for filepath in self.raw_paths[2:]:
                masks = np.load(filepath)
                train_masks += [torch.from_numpy(masks['train_mask'])]
                val_masks += [torch.from_numpy(masks['val_mask'])]
                test_masks += [torch.from_numpy(masks['test_mask'])]
            train_mask = torch.stack(train_masks, dim=1).to(torch.bool)
            val_mask = torch.stack(val_masks, dim=1).to(torch.bool)
            test_mask = torch.stack(test_masks, dim=1).to(torch.bool)

            data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                        val_mask=val_mask, test_mask=test_mask)

        else:
            raw_data = np.load(self.raw_paths[0], 'r', allow_pickle=True)
            x = torch.from_numpy(raw_data['features']).to(torch.float)
            edge_index = torch.from_numpy(raw_data['edges']).to(torch.long)
            edge_index = edge_index.t().contiguous()
            edge_index = coalesce(edge_index, num_nodes=x.size(0))
            y = torch.from_numpy(raw_data['target']).to(torch.float)

            data = Data(x=x, edge_index=edge_index, y=y)

            self.save([data], self.processed_paths[0])
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        graph = to_networkx(data, to_undirected=True)
        degree_centrality, pagerank, eigen = centrality_cal(graph)
        data.degree_centrality = torch.tensor(
            [degree_centrality[i] for i in range(len(degree_centrality))]).unsqueeze(
            1)
        data.pagerank = torch.tensor([pagerank[i] for i in range(len(pagerank))]).unsqueeze(1)
        data.eigenvector_centrality = torch.tensor([eigen[i] for i in range(len(eigen))]).unsqueeze(1)
        self.save([data], self.processed_paths[0])


if __name__ == "__main__":
    data = MyWikipediaNetwork(root="/mnt/data/banlujie/dataset/wiki", name="squirrel", geom_gcn_preprocess=True)[0]
    graph = to_networkx(data, to_undirected=True)
    with open("/mnt/data/banlujie/dataset/wiki/squirrel/squirrel.txt", "w") as f:
        f.write(f"{graph.number_of_nodes()} {graph.number_of_edges()}\n")
        for edge in sorted(graph.edges):
            f.write(f"{edge[0]} {edge[1]}\n")
