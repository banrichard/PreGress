import numpy as np
import torch
from torch_geometric.utils import index_to_mask


def make_mask(node_index, mask_ratio=0.3, no_aug=False):
    """
    mask the selected node to
    :param node_index: the node index of the selected subgraph [N,1]
    :return: masked index
    """
    cnt = 0
    N, D = node_index.shape
    if no_aug or mask_ratio == 0:
        return torch.zeros((N, 1)).bool()
    num_mask = int(mask_ratio * N)
    overall_mask = np.ones(N)
    for i in range(len(overall_mask)):
        if cnt >= num_mask:
            continue
        elif np.random.random() >= 0.5:
            overall_mask[i] = 0
            cnt += 1
    
    overall_mask = torch.tensor(overall_mask).bool()
    return overall_mask.to(node_index.device)
