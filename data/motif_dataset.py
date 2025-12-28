import os.path

import numpy as np
import torch
from torch_geometric.data import Data, Dataset, DataLoader
from torch.utils.data import Sampler


class MotifDataset(Dataset):

    def __init__(self, motif_gt_data, subgraphs, dataset_name):
        super().__init__()
        self.motif_gt_data = motif_gt_data
        self.subgraphs = subgraphs
        self.dataset_name = dataset_name
        self.data_list = self._prepare_data()

    def _prepare_data(self):
        if os.path.exists(os.path.join("/mnt/data/banlujie/dataset", self.dataset_name, self.dataset_name + "counting.pt")):
            data_list = torch.load(
                os.path.join("/mnt/data/banlujie", self.dataset_name, self.dataset_name + "counting.pt"))
        else:
            data_list = []
            for key, motif_pairs in self.motif_gt_data.items():
                for pair in motif_pairs:
                    motif, gt_tensor = pair
                    gt_list = gt_tensor.tolist()
                    for i, gt_value in enumerate(gt_list):
                        # Create a tuple with the subgraph, motif, and ground truth value
                        data_list.append((self.subgraphs[i], motif, torch.tensor([gt_value])))
            torch.save(data_list,
                       os.path.join("/mnt/data/banlujie/dataset", self.dataset_name, self.dataset_name + "counting.pt"))
        return data_list


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, idx):
        return self.data_list[idx]


class RandomBatchSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):
        num_batches = len(self.data_source) // self.batch_size
        indices = np.arange(len(self.data_source))
        np.random.shuffle(indices)
        for i in range(num_batches):
            batch_indices = indices[i * self.batch_size:(i + 1) * self.batch_size]
            yield batch_indices.tolist()

    def __len__(self):
        return len(self.data_source) // self.batch_size
