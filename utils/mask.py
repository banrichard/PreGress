import numpy as np
import torch
from torch_geometric.utils import index_to_mask


def make_mask(node_index, mask_ratio=0.3, no_aug=False):
    """
    mask the selected node to
    :param node_index: the node index of the selected subgraph [N,1]
    :return: masked index
    """
    N, D = node_index.shape
    if no_aug or mask_ratio == 0:
        return torch.zeros(node_index).bool()
    overall_mask = np.zeros(N, D)
    num_mask = mask_ratio * N
    for i in range(N):
        mask = np.hstack([
            np.zeros(N - num_mask),
            np.ones(num_mask)
        ])
        np.random.shuffle(mask)
        overall_mask[i, :] = mask
    overall_mask = torch.from_numpy(overall_mask).to(torch.bool)
    return overall_mask.to(node_index.device)
